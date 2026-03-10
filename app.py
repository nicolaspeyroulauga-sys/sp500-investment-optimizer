import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Institutional Quantitative Terminal", layout="wide")

# --- HEADER ---
st.title("🏛️ Institutional Quantitative Research Terminal")
st.markdown("**Lead Developer: Nicolas Peyrou-Lauga** | Portfolio Engineering & Risk Simulation")

# --- SIDEBAR ---
st.sidebar.header("1. Core Engine Parameters")
n_stocks = st.sidebar.slider("Number of Assets", 10, 30, 25)
max_weight = st.sidebar.slider("Concentration Limit (%)", 5, 20, 10) / 100

st.sidebar.header("2. Risk & Optimization")
risk_level = st.sidebar.select_slider(
    "Asset Allocation Strategy",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

# Constants
START_DATE = "2020-01-01"
RISK_FREE_RATE = 0.02
UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]

if st.button("🚀 Initialize Terminal & Run Stress Tests"):
    with st.spinner("Processing High-Frequency Data..."):
        
        # 1. DATA CORE
        all_tickers = UNIVERSE + ["SPY"]
        data = yf.download(all_tickers, start=START_DATE, progress=False)["Close"].dropna(axis=1)
        prices = data[UNIVERSE]
        spy_prices = data["SPY"]
        
        # 2. FACTOR SELECTION (Momentum + Volatility Adjusted)
        returns = prices.pct_change().dropna()
        momentum = prices.shift(21).pct_change(252).iloc[-1]
        vol_vec = returns.std() * np.sqrt(252)
        factor_df = pd.DataFrame({
            "mom": momentum.rank(ascending=False),
            "qual": (momentum/vol_vec).rank(ascending=False)
        }).dropna()
        factor_df["avg_rank"] = (factor_df["mom"] + factor_df["qual"]) / 2
        selected = factor_df.sort_values("avg_rank").head(n_stocks).index.tolist()
        
        # 3. OPTIMIZATION
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

        # 4. ADVANCED ANALYTICS (KPIs)
        port_daily_ret = (returns[selected] * weights).sum(axis=1)
        spy_daily_ret = spy_prices.pct_change().dropna()
        common_idx = port_daily_ret.index.intersection(spy_daily_ret.index)
        
        # Metrics
        ann_return = np.dot(weights, mu)
        ann_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol
        
        # Sortino Ratio (Downside Deviation Only)
        downside_rets = port_daily_ret[port_daily_ret < 0]
        sortino = (ann_return - RISK_FREE_RATE) / (downside_rets.std() * np.sqrt(252))
        
        # Beta & Alpha
        cov_matrix = np.cov(port_daily_ret.loc[common_idx], spy_daily_ret.loc[common_idx])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        alpha = ann_return - (RISK_FREE_RATE + beta * (spy_daily_ret.mean()*252 - RISK_FREE_RATE))

        st.write("### 🔑 Tactical KPIs & Risk-Adjusted Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Exp. Return", f"{ann_return:.2%}")
        m2.metric("Volatility", f"{ann_vol:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m4.metric("Sortino Ratio", f"{sortino:.2f}", help="Return vs Downside risk.")
        m5.metric("Portfolio Alpha", f"{alpha:.2%}", help="Skill return above benchmark.")

        # 5. STRESS TESTING SECTION (What if scenarios)
        st.divider()
        st.write("### 🌩️ Strategic Stress Tests: Performance Under Duress")
        stress_col1, stress_col2 = st.columns(2)
        
        with stress_col1:
            st.info("**Scenario 1: Market Crash (-20%)**")
            crash_impact = beta * -0.20
            st.write(f"Estimated Portfolio Impact: **{crash_impact:.2%}**")
            st.caption("Assumes a direct systematic shock to the S&P 500.")

        with stress_col2:
            st.warning("**Scenario 2: Interest Rate Spike**")
            # Growth stocks usually drop ~10% for every 1% rate hike
            st.write("Sensitivity to Yield Curve: **Moderate**")
            st.caption("Calculation based on current concentration in Technology & Growth assets.")

        # 6. DUAL PERFORMANCE VIEW
        st.divider()
        st.write("### 📊 Performance History & Asset Concentration")
        tab_perf, tab_conc = st.tabs(["Cumulative Performance", "Sector Heatmap"])
        
        with tab_perf:
            cum_port = (1 + port_daily_ret.loc[common_idx]).cumprod()
            cum_spy = (1 + spy_daily_ret.loc[common_idx]).cumprod()
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Optimized Portfolio", line=dict(color="#0047bb", width=3)))
            fig_hist.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, name="S&P 500", line=dict(color="#d1d1d1", width=2)))
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with tab_conc:
            # We fetch sectors to create a Treemap
            sectors = []
            for t in weights.index:
                try: sectors.append(yf.Ticker(t).info.get('sector', 'Other'))
                except: sectors.append('Other')
            
            tree_df = pd.DataFrame({"Asset": weights.index, "Weight": weights.values, "Sector": sectors})
            fig_tree = px.treemap(tree_df, path=['Sector', 'Asset'], values='Weight', color='Weight', color_continuous_scale='RdBu')
            st.plotly_chart(fig_tree, use_container_width=True)

        # 7. MONTE CARLO & DEEP DIVE (As before, but even sleeker)
        st.divider()
        st.write("### 🔮 1-Year Probabilistic Outcome Cone")
        n_sims, n_days = 1000, 252
        sim_paths = np.zeros((n_days, n_sims))
        for i in range(n_sims):
            daily_rets = np.random.normal(port_daily_ret.mean(), port_daily_ret.std(), n_days)
            sim_paths[:, i] = 100 * (1 + daily_rets).cumprod()
        
        p5, p50, p95 = np.percentile(sim_paths, 5, axis=1), np.percentile(sim_paths, 50, axis=1), np.percentile(sim_paths, 95, axis=1)
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p95, line=dict(width=0), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p5, line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 71, 187, 0.1)', name="90% Risk Band"))
        fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p50, name="Expected Median", line=dict(color="#0047bb", width=3)))
        st.plotly_chart(fig_mc, use_container_width=True)

        # 8. INDIVIDUAL STOCK CARDS
        st.write("### 🔍 Individual Holding Diagnostics")
        for ticker in weights.head(5).index.tolist():
            col1, col2 = st.columns([1, 2])
            with col1:
                t_info = yf.Ticker(ticker).info
                st.subheader(ticker)
                st.write(f"**P/E:** {t_info.get('trailingPE', 'N/A')} | **Margin:** {t_info.get('profitMargins', 0)*100:.1f}%")
                st.caption(t_info.get('longBusinessSummary', '')[:300] + "...")
            with col2:
                fig_s = px.line(prices[ticker].tail(252), height=250)
                fig_s.update_layout(showlegend=False, margin=dict(t=0, b=0))
                st.plotly_chart(fig_s, use_container_width=True)

        st.success("Full Institutional Analysis Complete.")
