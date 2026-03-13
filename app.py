import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier
from datetime import datetime
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum Portfolio Terminal", layout="wide")

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

# ELITE UNIVERSE
FULL_UNIVERSE = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]
SELECTED_UNIVERSE = FULL_UNIVERSE[:n_stocks_slider]

# --- OPTIMIZED DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_market_data(tickers):
    # We no longer pass a custom 'session'. YFinance handles its own browser impersonation now.
    data = yf.download(tickers, start="2021-01-01", progress=False)["Close"]
    return data.dropna(axis=1)

@st.cache_data(ttl=3600)
def fetch_metadata_and_news(tickers):
    meta = {}
    news_data = {}
    
    for t in tickers:
        ticker_obj = yf.Ticker(t)
        try:
            # We fetch minimal info to avoid hitting rate limits too hard
            info = ticker_obj.info
            meta[t] = {
                "Sector": info.get('sector', 'Other'),
                "Name": info.get('longName', t),
                "Summary": info.get('longBusinessSummary', 'No summary available.')[:400] + "...",
                "PE": info.get('trailingPE', 'N/A')
            }
            news_data[t] = ticker_obj.news[:2] 
            # Subtle delay for server politeness
            time.sleep(0.2) 
        except Exception:
            meta[t] = {"Sector": "Other", "Name": t, "Summary": "N/A", "PE": "N/A"}
            news_data[t] = []
            
    return meta, news_data

# --- EXECUTION ---
if st.button("🚀 Execute Full Institutional Analysis"):
    with st.spinner(f"Simulating Institutional Allocation for {n_stocks_slider} Assets..."):
        
        # 1. Historical Data
        prices = fetch_market_data(SELECTED_UNIVERSE)
        current_prices = prices.iloc[-1]
        returns = prices.pct_change().dropna()
        
        # 2. Intelligence & News
        metadata, all_news = fetch_metadata_and_news(list(prices.columns))
        
        # 3. Optimization Logic
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

        # 4. Global Metrics
        port_daily_ret = (returns[weights.index] * weights).sum(axis=1)
        ann_ret = port_daily_ret.mean() * 252
        ann_vol = port_daily_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / ann_vol

        # --- KPI GRID ---
        st.write(f"### 🔑 Strategic KPI Dashboard")
        k1, k2, k3 = st.columns(3)
        k1.metric("Expected Annual Return", f"{ann_ret:.2%}")
        k2.metric("Annual Volatility", f"{ann_vol:.2%}")
        k3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # --- MAIN OUTPUT TABS ---
        t_alloc, t_sector, t_intel = st.tabs(["📊 Allocation Strategy", "🗺️ Sector Distribution", "🔍 Asset Analysis & News"])

        with t_alloc:
            st.write("### 📋 Trade Execution List")
            trade_df = pd.DataFrame(weights, columns=["Weight"])
            trade_df["Investment ($)"] = trade_df["Weight"] * total_capital
            trade_df["Current Price"] = current_prices[weights.index]
            trade_df["Shares to Buy"] = trade_df["Investment ($)"] / trade_df["Current Price"]
            
            st.table(trade_df.style.format({
                "Weight": "{:.2%}", "Investment ($)": "${:,.2f}", 
                "Current Price": "${:,.2f}", "Shares to Buy": "{:.4f}"
            }))

        with t_sector:
            # Build Sector Analysis Data
            sector_list = []
            for t in weights.index:
                sector_list.append({
                    "Ticker": t,
                    "Weight": weights[t],
                    "Sector": metadata[t]["Sector"]
                })
            sector_df = pd.DataFrame(sector_list)
            
            # Map sectors to colors for discrete coding
            fig_sector = px.treemap(
                sector_df, 
                path=['Sector', 'Ticker'], 
                values='Weight',
                color='Sector',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Sectoral Concentration Architecture"
            )
            st.plotly_chart(fig_sector, use_container_width=True)

        with t_intel:
            st.write("### 🔍 Live Intelligence & News Summaries")
            for t in weights.index:
                with st.expander(f"{t} | {metadata[t]['Name']} ({weights[t]:.2%})"):
                    col_meta, col_news = st.columns([1, 2])
                    with col_meta:
                        st.markdown(f"**Sector:** {metadata[t]['Sector']}")
                        st.markdown(f"**P/E Ratio:** {metadata[t]['PE']}")
                        st.write("**Asset Profile:**")
                        st.caption(metadata[t]['Summary'])
                        # Performance Sparkline
                        st.line_chart(prices[t].tail(45))
                    
                    with col_news:
                        st.write("**Direct News Feed**")
                        asset_news = all_news.get(t, [])
                        if not asset_news:
                            st.write("No recent strategic news detected.")
                        for article in asset_news:
                            # Cleanly formatted news without external link clutter
                            pub_date = datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d')
                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #0047bb;">
                                <small>{pub_date} | {article['publisher']}</small><br>
                                <strong>{article['title']}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            st.caption(f"[Read full article on {article['publisher']}]({article['link']})")

        st.success("Quant Execution Complete.")
else:
    st.info("👈 System Standby. Click 'Execute' to run the institutional models.")
