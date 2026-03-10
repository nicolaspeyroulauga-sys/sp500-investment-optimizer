import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum Portfolio Terminal", layout="wide")

# --- HEADER ---
st.title("🏛️ Institutional Quantitative Research Terminal")
st.markdown("**Lead Developer: Nicolas Peyrou-Lauga** | Portfolio Engineering & Management")

# --- SIDEBAR ---
st.sidebar.header("1. Portfolio Capitalization")
total_capital = st.sidebar.number_input(
    "Total Investment Capital ($)", 
    min_value=1.0, 
    value=10000.0, 
    step=100.0
)

st.sidebar.header("2. Strategy Settings")
n_stocks = st.sidebar.slider("Number of Assets", 10, 30, 30)
max_weight = st.sidebar.slider("Concentration Limit (%)", 5, 20, 10) / 100

st.sidebar.header("3. Risk & Optimization")
risk_level = st.sidebar.select_slider(
    "Asset Allocation Strategy",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

# FIXED UNIVERSE - AS PER RESEARCH
START_DATE = "2020-01-01"
RISK_FREE_RATE = 0.02
UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]

if st.button("🚀 Execute Full Institutional Analysis"):
    with st.spinner("Processing Market Data..."):
        
        # 1. DATA CORE
        data = yf.download(UNIVERSE, start=START_DATE, progress=False)["Close"]
        prices = data.dropna(axis=1)
        current_prices = prices.iloc[-1]
        returns = prices.pct_change().dropna()
        
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
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol
        
        # --- ROBUST BETA CALCULATION ---
        spy_raw = yf.download("SPY", start=START_DATE, progress=False)["Close"]
        spy_rets = spy_raw.pct_change().dropna()
        
        # Aligning the two series to solve the ValueError
        combined_df = pd.concat([port_daily_ret, spy_rets], axis=1).dropna()
        combined_df.columns = ['portfolio', 'benchmark']
        
        beta = 1.0 # Default
        if not combined_df.empty and len(combined_df) > 2:
            matrix = np.cov(combined_df['portfolio'], combined_df['benchmark'])
            beta = matrix[0, 1] / matrix[1, 1]

        # --- KPI DASHBOARD ---
        st.write(f"### 🔑 Tactical KPIs (Basis: ${total_capital:,.2f})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Capital Basis", f"${total_capital:,.2f}")
        m2.metric("Annual Return", f"{ann_return:.2%}")
        m3.metric("Annual Volatility", f"{ann_vol:.2%}")
        m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # --- FRACTIONAL TRADE TABLE ---
        st.divider()
        st.write("### 📋 Trade Execution & Asset Allocation")
        alloc_df = pd.DataFrame(weights, columns=["Weight"])
        alloc_df["Investment ($)"] = alloc_df["Weight"] * total_capital
        alloc_df["Current Price ($)"] = current_prices[weights.index]
        alloc_df["Shares to Buy"] = alloc_df["Investment ($)"] / alloc_df["Current Price ($)"]
        
        display_df = alloc_df.copy()
        display_df["Weight"] = display_df["Weight"].map("{:.2%}".format)
        display_df["Investment ($)"] = display_df["Investment ($)"].map("${:,.2f}".format)
        display_df["Current Price ($)"] = display_df["Current Price ($)"].map("${:,.2f}".format)
        display_df["Shares to Buy"] = display_df["Shares to Buy"].map("{:.3f}".format)
        st.table(display_df)

        # 4. STRESS TESTING
        st.divider()
        st.write("### 🌩️ Strategic Stress Tests: Capital at Risk")
        sc1, sc2 = st.columns(2)
        crash_impact = beta * -0.20
        var_95 = np.percentile(port_daily_ret, 5)

        with sc1:
            st.info("**Scenario: Market Crash (-20%)**")
            st.metric("Estimated Impact", f"{crash_impact:.2%}", f"-${total_capital * abs(crash_impact):,.2f}", delta_color="inverse")
            st.caption(f"Portfolio Beta: {beta:.2f}")

        with sc2:
            st.warning("**Scenario: Daily Value-at-Risk (95% CI)**")
            st.metric("95% Daily Floor", f"{var_95:.2%}", f"-${total_capital * abs(var_95):,.2f}", delta_color="inverse")

        # 5. RISK VISUALS
        t_heat, t_dist, t_tree = st.tabs(["Correlation Heatmap", "Return Distribution", "Sector Map"])
        with t_heat:
            corr_matrix = returns[weights.index].corr()
            fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_heat, use_container_width=True)
        with t_dist:
            fig_risk = px.histogram(port_daily_ret, nbins=100, title="Return Frequency", color_discrete_sequence=['#0047bb'])
            fig_risk.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="95% VaR")
            st.plotly_chart(fig_risk, use_container_width=True)
        with t_tree:
            sectors = []
            for t in weights.index:
                try: sectors.append(yf.Ticker(t).info.get('sector', 'Other'))
                except: sectors.append('Other')
            tree_df = pd.DataFrame({"Asset": weights.index, "Weight": weights.values, "Sector": sectors})
            fig_tree = px.treemap(tree_df, path=['Sector', 'Asset'], values='Weight', color='Weight', color_continuous_scale='RdBu')
            st.plotly_chart(fig_tree, use_container_width=True)

        # 6. HISTORICAL GROWTH
        st.divider()
        st.write("### 📈 Cumulative Performance Analysis")
        if not combined_df.empty:
            cum_port = (1 + combined_df['portfolio']).cumprod()
            cum_spy = (1 + combined_df['benchmark']).cumprod()
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="HRP Strategy", line=dict(color="#0047bb", width=3)))
            fig_hist.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, name="S&P 500", line=dict(color="#d1d1d1", width=2)))
            st.plotly_chart(fig_hist, use_container_width=True)

        # 7. MONTE CARLO PROJECTION
        st.divider()
        st.write(f"### 🔮 1-Year Wealth Projection (Basis: ${total_capital:,.2f})")
        n_sims, n_days = 1000, 252
        sim_paths = np.zeros((n_days, n_sims))
        for i in range(n_sims):
            daily_rets = np.random.normal(port_daily_ret.mean(), port_daily_ret.std(), n_days)
            sim_paths[:, i] = total_capital * (1 + daily_rets).cumprod()
        
        p5, p50, p95 = np.percentile(sim_paths, 5, axis=1), np.percentile(sim_paths, 50, axis=1), np.percentile(sim_paths, 95, axis=1)
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p95, line=dict(width=0), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p5, line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 71, 187, 0.1)', name="90% Confidence Interval"))
        fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p50, name="Median Projection", line=dict(color="#0047bb", width=3)))
        st.plotly_chart(fig_mc, use_container_width=True)

        st.success("Analysis complete. Beta calculated using synchronized time-series.")
else:
    st.info("👈 System Standby. Click Execute to generate interactive analysis.")
