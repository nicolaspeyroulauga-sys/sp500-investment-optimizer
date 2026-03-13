import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier
from datetime import datetime
import time
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum Portfolio Terminal", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .news-card {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #0047bb;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🏛️ Institutional Quantitative Research Terminal")
st.markdown("**Lead Developer: Nicolas Peyrou-Lauga** | Portfolio Engineering & Management")

# --- SIDEBAR ---
st.sidebar.header("1. Portfolio Capitalization")
total_capital = st.sidebar.number_input("Total Investment Capital ($)", min_value=1.0, value=10000.0, step=100.0)

st.sidebar.header("2. Strategy Settings")
# ADDED BACK: The number of assets slider
n_stocks_slider = st.sidebar.slider("Number of Assets to Analyze", 5, 30, 15)
max_weight = st.sidebar.slider("Concentration Limit (%)", 5, 20, 10) / 100

st.sidebar.header("3. Risk & Optimization")
risk_level = st.sidebar.select_slider(
    "Asset Allocation Strategy",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

# FULL UNIVERSE
FULL_UNIVERSE = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]
# Sliced Universe based on slider
SELECTED_UNIVERSE = FULL_UNIVERSE[:n_stocks_slider]

# --- SESSION SETUP (To avoid Rate Limits) ---
@st.cache_resource
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_market_data(tickers):
    # Download prices in bulk (1 request)
    data = yf.download(tickers, start="2021-01-01", progress=False, session=get_session())["Close"]
    return data.dropna(axis=1)

@st.cache_data(ttl=3600)
def fetch_metadata_and_news(tickers):
    meta = {}
    news_data = {}
    session = get_session()
    
    # We loop with a tiny delay to avoid being flagged
    for t in tickers:
        ticker_obj = yf.Ticker(t, session=session)
        try:
            # Fetch Info
            info = ticker_obj.info
            meta[t] = {
                "Sector": info.get('sector', 'Other'),
                "Name": info.get('longName', t),
                "Summary": info.get('longBusinessSummary', 'No summary available.')[:400] + "...",
                "PE": info.get('trailingPE', 'N/A')
            }
            # Fetch News (Directly in-app)
            news_data[t] = ticker_obj.news[:2] 
            time.sleep(0.1) # Politeness delay
        except:
            meta[t] = {"Sector": "Other", "Name": t, "Summary": "N/A", "PE": "N/A"}
            news_data[t] = []
            
    return meta, news_data

# --- EXECUTION ---
if st.button("🚀 Execute Full Institutional Analysis"):
    with st.spinner(f"Analyzing {n_stocks_slider} assets..."):
        
        # 1. Prices
        prices = fetch_market_data(SELECTED_UNIVERSE)
        current_prices = prices.iloc[-1]
        returns = prices.pct_change().dropna()
        
        # 2. Metadata
        metadata, all_news = fetch_metadata_and_news(list(prices.columns))
        
        # 3. Optimization
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        if "Low" in risk_level:
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.min_volatility())
        elif "High" in risk_level:
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.max_sharpe())
        else:
            hrp = HRPOpt(returns)
            weights = pd.Series(hrp.optimize())
            weights = (weights.clip(upper=max_weight) / (weights.clip(upper=max_weight).sum()))
        
        weights = weights.sort_values(ascending=False)

        # 4. Metrics
        port_daily_ret = (returns[weights.index] * weights).sum(axis=1)
        ann_ret = port_daily_ret.mean() * 252
        ann_vol = port_daily_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / ann_vol
        
        # --- DASHBOARD ---
        st.write(f"### 🔑 Tactical Portfolio KPIs")
        k1, k2, k3 = st.columns(3)
        k1.metric("Exp. Annual Return", f"{ann_ret:.2%}")
        k2.metric("Annual Volatility", f"{ann_vol:.2%}")
        k3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # --- TABS ---
        t_alloc, t_sector, t_intel = st.tabs(["📊 Allocation", "🗺️ Sector Map", "🔍 Asset Intelligence"])

        with t_alloc:
            st.write("### 📋 Trade Execution & Fractional Shares")
            trade_df = pd.DataFrame(weights, columns=["Weight"])
            trade_df["Investment ($)"] = trade_df["Weight"] * total_capital
            trade_df["Current Price"] = current_prices[weights.index]
            trade_df["Shares to Buy"] = trade_df["Investment ($)"] / trade_df["Current Price"]
            
            st.table(trade_df.style.format({
                "Weight": "{:.2%}", "Investment ($)": "${:,.2f}", 
                "Current Price": "${:,.2f}", "Shares to Buy": "{:.4f}"
            }))

        with t_sector:
            # Build Sector Data
            sector_data = []
            for t in weights.index:
                sector_data.append({
                    "Ticker": t,
                    "Weight": weights[t],
                    "Sector": metadata[t]["Sector"]
                })
            sector_df = pd.DataFrame(sector_data).sort_values("Sector")
            
            # Treemap color-coded by Sector and Ticker
            fig_sector = px.treemap(
                sector_df, 
                path=['Sector', 'Ticker'], 
                values='Weight',
                color='Sector',
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Hierarchical Sector Allocation"
            )
            st.plotly_chart(fig_sector, use_container_width=True)

        with t_intel:
            st.write("### 🔍 Individual Asset Deep-Dive & Live News")
            for t in weights.index:
                with st.expander(f"{t} - {metadata[t]['Name']} ({weights[t]:.2%})"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.write(f"**Sector:** {metadata[t]['Sector']}")
                        st.write(f"**P/E Ratio:** {metadata[t]['PE']}")
                        st.write("**Business Summary:**")
                        st.caption(metadata[t]['Summary'])
                        st.line_chart(prices[t].tail(60))
                    
                    with c2:
                        st.write("**Latest Market News (Direct)**")
                        news_items = all_news.get(t, [])
                        if not news_items:
                            st.write("No recent news found.")
                        for n in news_items:
                            st.markdown(f"""
                            <div class="news-card">
                                <strong>{n['title']}</strong><br>
                                <small>{n['publisher']} | {datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d')}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            st.button(f"Link: {n['title'][:20]}...", key=n['uuid'], on_click=lambda url=n['link']: st.write(url))

        st.success("Strategy executed successfully.")
else:
    st.info("👈 System Ready. Select asset count and click Execute.")
