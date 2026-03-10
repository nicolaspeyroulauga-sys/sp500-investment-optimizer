import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum Portfolio Research Terminal", layout="wide")

# --- HEADER ---
st.title("🏛️ Quantitative Portfolio Research Terminal")
st.markdown("**Lead Developer: Nicolas Peyrou-Lauga** | Master's in Finance Strategy")

# --- SIDEBAR ---
st.sidebar.header("1. Analysis Parameters")
n_stocks = st.sidebar.slider("Number of Top Assets", 10, 30, 25)
max_weight = st.sidebar.slider("Max Weight per Asset (%)", 5, 20, 10) / 100

st.sidebar.header("2. Risk Configuration")
risk_level = st.sidebar.select_slider(
    "Optimization Strategy",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

# Constants for Consistency
START_DATE = "2020-01-01"
RISK_FREE_RATE = 0.02
UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]

if st.button("🚀 Execute Research & Optimization"):
    with st.spinner("Processing Market Data, Optimizing, and Simulating..."):
        
        # 1. DATA ACQUISITION
        all_tickers = UNIVERSE + ["SPY"]
        data = yf.download(all_tickers, start=START_DATE, progress=False)["Close"].dropna(axis=1)
        prices = data[UNIVERSE]
        spy_prices = data["SPY"]
        
        # 2. FACTOR BRAIN (Momentum/Quality Ranking)
        returns = prices.pct_change().dropna()
        momentum = prices.shift(21).pct_change(252).iloc[-1]
        vol_vec = returns.std() * np.sqrt(252)
        factor_df = pd.DataFrame({
            "mom": momentum.rank(ascending=False),
            "qual": (momentum/vol_vec).rank(ascending=False)
        }).dropna()
        factor_df["avg_rank"] = (factor_df["mom"] + factor_df["qual"]) / 2
        selected = factor_df.sort_values("avg_rank").head(n_stocks).index.tolist()
        
        # 3. OPTIMIZATION ENGINE
        mu = expected_returns.mean_historical_return(prices[selected])
        S = risk_models.CovarianceShrinkage(prices[selected]).ledoit_wolf()

        if "Low" in risk_level:
            ef = EfficientFrontier(mu, S); ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.min_volatility())
        elif "High" in risk_level:
            ef = EfficientFrontier(mu, S); ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.max_sharpe())
        else:
            hrp = HRPOpt(returns[selected]); weights = pd.Series(hrp.optimize())
            weights = (weights.clip(upper=max_weight) / (weights.clip(upper=max_weight).sum()))

        weights = weights.sort_values(ascending=False)

        # 4. PORTFOLIO METRICS (KPIs)
        port_daily_ret = (returns[selected] * weights).sum(axis=1)
        spy_daily_ret = spy_prices.pct_change().dropna()
        common_idx = port_daily_ret.index.intersection(spy_daily_ret.index)
        
        # Performance Calculations
        ann_return = np.dot(weights, mu)
        ann_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol
        
        # Beta Calculation (Covariance of Port with Market / Var of Market)
        covariance_matrix = np.cov(port_daily_ret.loc[common_idx], spy_daily_ret.loc[common_idx])
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]

        # --- DISPLAY KPI SECTION ---
        st.write("### 🔑 Key Performance Indicators")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Annual Expected Return", f"{ann_return:.2%}")
        k2.metric("Annual Volatility", f"{ann_vol:.2%}")
        k3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        k4.metric("Portfolio Beta", f"{beta:.2f}", help="Sensitivity to the S&P 500. A beta of 1.0 moves exactly with the market.")

        # 5. HISTORICAL GROWTH VS SPY
        st.divider()
        st.write("### 📈 Historical Growth vs. S&P 500 (Base $1)")
        cum_port = (1 + port_daily_ret.loc[common_idx]).cumprod()
        cum_spy = (1 + spy_daily_ret.loc[common_idx]).cumprod()
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Your Optimized Portfolio", line=dict(color="#0047bb", width=3)))
        fig_hist.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, name="S&P 500 (SPY)", line=dict(color="#d1d1d1", width=2)))
        st.plotly_chart(fig_hist, use_container_width=True)

        # 6. MONTE CARLO PROBABILITY CONE
        st.divider()
        st.write("### 🔮 1-Year Monte Carlo Probability Projection")
        n_sims, n_days = 1000, 252
        mean_ret, std_dev = port_daily_ret.mean(), port_daily_ret.std()
        
        sim_paths = np.zeros((n_days, n_sims))
        for i in range(n_sims):
            daily_rets = np.random.normal(mean_ret, std_dev, n_days)
            sim_paths[:, i] = 100 * (1 + daily_rets).cumprod()
        
        p5, p50, p95 = np.percentile(sim_paths, 5, axis=1), np.percentile(sim_paths, 50, axis=1), np.percentile(sim_paths, 95, axis=1)
        
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p95, mode='lines', line=dict(width=0), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 71, 187, 0.1)', name="90% Confidence Interval"))
        fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p50, name="Median Expectation", line=dict(color="#0047bb", width=3)))
        fig_mc.update_layout(xaxis_title="Future Trading Days", yaxis_title="Growth of $100")
        st.plotly_chart(fig_mc, use_container_width=True)

        # 7. ASSET DEEP-DIVE & INDIVIDUAL GRAPHS
        st.divider()
        st.write("### 🔍 Fundamental Insights: Top Portfolio Holdings")
        top_assets = weights.head(5).index.tolist()
        
        for ticker in top_assets:
            col_text, col_graph = st.columns([1, 2])
            with col_text:
                try:
                    t_obj = yf.Ticker(ticker)
                    info = t_obj.info
                    st.subheader(f"{ticker}")
                    st.write(f"**Current Weight:** {weights[ticker]*100:.2f}%")
                    st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                    st.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%")
                    st.caption(info.get('longBusinessSummary', 'Description Unavailable.')[:350] + "...")
                except:
                    st.write("Data sync pending for this asset.")
            
            with col_graph:
                stock_hist = prices[ticker].tail(252)
                fig_stock = px.line(stock_hist, title=f"{ticker} - LTM Price Action", labels={"value": "Price", "Date": ""})
                fig_stock.update_layout(height=300, showlegend=False, margin=dict(t=40, b=10))
                st.plotly_chart(fig_stock, use_container_width=True)
            st.markdown("---")

        st.success(f"Analysis Finalized. Strategy: {risk_level}")
else:
    st.info("👈 Set your parameters in the sidebar and click the button to generate the Research Terminal.")
