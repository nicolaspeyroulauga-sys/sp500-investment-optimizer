import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum Portfolio Terminal", layout="wide")

# --- CUSTOM CSS FOR NEWS ---
st.markdown("""
    <style>
    .news-card {
        background-color: #f1f3f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0047bb;
        margin-bottom: 10px;
    }
    .stock-header {
        color: #0047bb;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🏛️ Institutional Quantitative Research Terminal")
st.markdown("**Lead Developer: Nicolas Peyrou-Lauga** | Portfolio Engineering & Management")

# --- UNIVERSES & SECTOR MAPPING ---
UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]

# --- SIDEBAR ---
st.sidebar.header("1. Portfolio Capitalization")
total_capital = st.sidebar.number_input("Total Investment Capital ($)", min_value=1.0, value=10000.0, step=100.0)

st.sidebar.header("2. Strategy Settings")
max_weight = st.sidebar.slider("Concentration Limit (%)", 5, 20, 10) / 100

st.sidebar.header("3. Risk & Optimization")
risk_level = st.sidebar.select_slider(
    "Asset Allocation Strategy",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_stock_data(tickers):
    data = yf.download(tickers, start="2020-01-01", progress=False)["Close"]
    return data.dropna(axis=1)

@st.cache_data(ttl=3600)
def get_asset_info(tickers):
    info_dict = {}
    for t in tickers:
        ticker_obj = yf.Ticker(t)
        s_info = ticker_obj.info
        info_dict[t] = {
            "Sector": s_info.get('sector', 'Other'),
            "Full Name": s_info.get('longName', t),
            "PE": s_info.get('trailingPE', 'N/A'),
            "DivYield": s_info.get('dividendYield', 0) * 100 if s_info.get('dividendYield') else 0,
            "Summary": s_info.get('longBusinessSummary', 'No summary available.')[:500] + "..."
        }
    return info_dict

def get_stock_news(ticker_symbol):
    t = yf.Ticker(ticker_symbol)
    news = t.news[:3]  # Get latest 3 news items
    return news

# --- MAIN EXECUTION ---
if st.button("🚀 Execute Full Institutional Analysis"):
    with st.spinner("Synchronizing Market Intelligence..."):
        
        # 1. DATA CORE
        prices = get_stock_data(UNIVERSE)
        current_prices = prices.iloc[-1]
        returns = prices.pct_change().dropna()
        asset_metadata = get_asset_info(UNIVERSE)
        
        # 2. OPTIMIZATION ENGINE
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

        # 3. PERFORMANCE METRICS
        port_daily_ret = (returns[weights.index] * weights).sum(axis=1)
        ann_return = port_daily_ret.mean() * 252
        ann_vol = port_daily_ret.std() * np.sqrt(252)
        sharpe = (ann_return - 0.02) / ann_vol
        
        spy_raw = yf.download("SPY", start="2020-01-01", progress=False)["Close"]
        spy_rets = spy_raw.pct_change().dropna()
        combined_df = pd.concat([port_daily_ret, spy_rets], axis=1).dropna()
        combined_df.columns = ['portfolio', 'benchmark']
        matrix = np.cov(combined_df['portfolio'], combined_df['benchmark'])
        beta = matrix[0, 1] / matrix[1, 1]

        # --- KPI DASHBOARD ---
        st.write(f"### 🔑 Tactical KPIs (Basis: ${total_capital:,.2f})")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Annual Return", f"{ann_return:.2%}")
        kpi2.metric("Annual Volatility", f"{ann_vol:.2%}")
        kpi3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        kpi4.metric("Beta vs S&P500", f"{beta:.2f}")

        # --- MAIN TABS ---
        t_allocation, t_risk, t_deep_dive = st.tabs(["📋 Allocation & Performance", "🌩️ Risk Analysis", "🔍 Asset Intelligence Deep-Dive"])

        with t_allocation:
            st.write("### 📋 Trade Execution")
            alloc_df = pd.DataFrame(weights, columns=["Weight"])
            alloc_df["Investment ($)"] = alloc_df["Weight"] * total_capital
            alloc_df["Current Price ($)"] = current_prices[weights.index]
            alloc_df["Shares"] = alloc_df["Investment ($)"] / alloc_df["Current Price ($)"]
            
            # Formatted display
            f_df = alloc_df.copy()
            f_df["Weight"] = f_df["Weight"].map("{:.2%}".format)
            f_df["Investment ($)"] = f_df["Investment ($)"].map("${:,.2f}".format)
            f_df["Current Price ($)"] = f_df["Current Price ($)"].map("${:,.2f}".format)
            st.table(f_df)

            st.divider()
            st.write("### 📈 Cumulative Growth")
            cum_port = (1 + combined_df['portfolio']).cumprod()
            cum_spy = (1 + combined_df['benchmark']).cumprod()
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Strategic Portfolio", line=dict(color="#0047bb", width=3)))
            fig_hist.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, name="S&P 500 Index", line=dict(color="#d1d1d1", width=2)))
            st.plotly_chart(fig_hist, use_container_width=True)

        with t_risk:
            st.write("### 🏛️ Sectoral Architecture")
            # Build tree_df with metadata
            tree_data = []
            for t in weights.index:
                tree_data.append({
                    "Asset": t,
                    "Weight": weights[t],
                    "Sector": asset_metadata[t]["Sector"]
                })
            tree_df = pd.DataFrame(tree_data).sort_values(by="Sector")
            
            fig_tree = px.treemap(
                tree_df, 
                path=['Sector', 'Asset'], 
                values='Weight',
                color='Sector', # Color coded by sector
                title="Portfolio Sector Concentration",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            st.plotly_chart(fig_tree, use_container_width=True)
            
            st.divider()
            st.write("### 🌩️ Systematic Stress Testing")
            var_95 = np.percentile(port_daily_ret, 5)
            s1, s2 = st.columns(2)
            s1.metric("Market Crash Impact (-20%)", f"-${total_capital * abs(beta * 0.20):,.2f}")
            s2.metric("95% Daily Value-at-Risk", f"-${total_capital * abs(var_95):,.2f}")

        with t_deep_dive:
            st.write("### 🔍 Individual Asset Intelligence & Live News")
            st.info("Directly pulling latest SEC filings and market news summaries for your portfolio.")
            
            # Using expanders to keep 30 stocks manageable
            for ticker in weights.index:
                with st.expander(f"{ticker} - {asset_metadata[ticker]['Full Name']} ({weights[ticker]:.2%})"):
                    col_info, col_news = st.columns([1, 2])
                    
                    with col_info:
                        st.markdown(f"**Sector:** {asset_metadata[ticker]['Sector']}")
                        st.markdown(f"**P/E Ratio:** {asset_metadata[ticker]['PE']}")
                        st.markdown(f"**Div. Yield:** {asset_metadata[ticker]['DivYield']:.2f}%")
                        st.write("**Business Summary:**")
                        st.caption(asset_metadata[ticker]['Summary'])
                        
                        # Sparkline mini-chart
                        st.write("**30-Day Trend**")
                        spark_data = prices[ticker].tail(30)
                        st.line_chart(spark_data)

                    with col_news:
                        st.write("**Latest Strategic News Headlines**")
                        news_items = get_stock_news(ticker)
                        if not news_items:
                            st.write("No recent news found.")
                        for item in news_items:
                            pub_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
                            st.markdown(f"""
                            <div class="news-card">
                                <strong>{item['title']}</strong><br>
                                <small>Source: {item['publisher']} | {pub_time}</small><br>
                                <p style='font-size: 13px; margin-top: 5px;'>{item.get('summary', 'Click link for full coverage...')[:300]}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.link_button(f"View Full Story: {item['publisher']}", item['link'])

        st.success("Analysis complete. All 30 assets mapped and analyzed.")
else:
    st.info("👈 System Standby. Click Execute to generate interactive analysis.")
