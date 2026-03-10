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
st.markdown("**Lead Developer: Nicolas Peyrou-Lauga** | Portfolio Engineering & Risk Management")

# --- SIDEBAR ---
st.sidebar.header("1. Portfolio Capitalization")
total_capital = st.sidebar.number_input(
    "Total Investment Capital ($)", 
    min_value=1.0, 
    value=10000.0, 
    step=100.0, 
    help="Enter the total fiat value of the current portfolio for rebalancing and projection scaling."
)

st.sidebar.header("2. Strategy Settings")
n_stocks = st.sidebar.slider("Number of Assets", 10, 30, 25)
max_weight = st.sidebar.slider("Concentration Limit (%)", 5, 20, 10) / 100

st.sidebar.header("3. Risk & Optimization")
risk_level = st.sidebar.select_slider(
    "Asset Allocation Strategy",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

# Constants
START_DATE = "2020-01-01"
RISK_FREE_RATE = 0.02
UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]

if st.button("🚀 Execute Full Institutional Analysis"):
    with st.spinner("Processing Market Data, Optimizing, and Simulating Scenarios..."):
        
        # 1. DATA CORE (CRASH-PROOFED)
        all_tickers = UNIVERSE + ["SPY"]
        data = yf.download(all_tickers, start=START_DATE, progress=False)["Close"]
        
        available_tickers = [t for t in UNIVERSE if t in data.columns]
        prices = data[available_tickers].dropna(axis=1)
        spy_prices = data["SPY"].dropna()
        
        # 2. FACTOR SELECTION (MOMENTUM & QUALITY)
        returns = prices.pct_change().dropna()
        momentum = prices.shift(21).pct_change(252).iloc[-1]
        vol_vec = returns.std() * np.sqrt(252)
        factor_df = pd.DataFrame({
            "mom": momentum.rank(ascending=False),
            "qual": (momentum/vol_vec).rank(ascending=False)
        }).dropna()
        factor_df["avg_rank"] = (factor_df["mom"] + factor_df["qual"]) / 2
        
        actual_n = min(n_stocks, len(factor_df))
        selected = factor_df.sort_values("avg_rank").head(actual_n).index.tolist()
        
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

        # 4. ADVANCED ANALYTICS (KPIs)
        port_daily_ret = (returns[selected] * weights).sum(axis=1)
        spy_daily_ret = spy_prices.pct_change().dropna()
        common_idx = port_daily_ret.index.intersection(spy_daily_ret.index)
        
        ann_return = np.dot(weights, mu)
        ann_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol
        
        cov_matrix = np.cov(port_daily_ret.loc[common_idx], spy_daily_ret.loc[common_idx])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        # --- KPI DASHBOARD ---
        st.write(f"### 🔑 Tactical KPIs (Basis: ${total_capital:,.2f})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Asset Value", f"${total_capital:,.2f}")
        m2.metric("Exp. Annual Growth", f"${total_capital * ann_return:,.2f}", f"{ann_return:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m4.metric("Portfolio Beta", f"{beta:.2f}")

        # 5. EFFICIENT FRONTIER VISUALIZATION
        st.divider()
        st.write("### 🌌 The Efficient Frontier & Strategy Placement")
        n_portfolios = 500
        p_ret, p_vol = [], []
        for _ in range(n_portfolios):
            w = np.random.random(len(selected))
            w /= np.sum(w)
            p_ret.append(np.dot(w, mu))
            p_vol.append(np.sqrt(np.dot(w.T, np.dot(S, w))))
        
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=p_vol, y=p_ret, mode='markers', name='Random Portfolios', marker=dict(color='lightgrey', size=5, opacity=0.5)))
        fig_ef.add_trace(go.Scatter(x=[ann_vol], y=[ann_return], mode='markers', name='Optimized Portfolio', marker=dict(color='#0047bb', size=15, symbol='star')))
        fig_ef.update_layout(xaxis_title="Annualized Volatility", yaxis_title="Annualized Return")
        st.plotly_chart(fig_ef, use_container_width=True)

        # 6. STRESS TESTS & RISK DISTRIBUTION
        st.divider()
        st.write("### 🌩️ Strategic Stress Tests: Capital at Risk")
        sc1, sc2 = st.columns(2)
        crash_pct = beta * -0.20
        var_95 = np.percentile(port_daily_ret, 5)

        with sc1:
            st.info("**Scenario: Market Crash (-20%)**")
            st.metric("Estimated Impact", f"{crash_pct:.2%}", f"-${total_capital * abs(crash_pct):,.2f}", delta_color="inverse")
            st.caption(f"Portfolio Sensitivity (Beta): {beta:.2f}")

        with sc2:
            st.warning("**Scenario: Daily Value-at-Risk (95% CI)**")
            st.metric("95% Daily Floor", f"{var_95:.2%}", f"-${total_capital * abs(var_95):,.2f}", delta_color="inverse")
            st.caption("Maximum likely loss in a single trading day.")

        fig_risk = px.histogram(port_daily_ret, nbins=100, title="Historical Daily Returns & Tail Risk", labels={'value': 'Daily Return %'}, color_discrete_sequence=['#0047bb'])
        fig_risk.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text=f"VaR Line")
        st.plotly_chart(fig_risk, use_container_width=True)

        # 7. CORRELATION & SECTOR ANALYSIS
        st.divider()
        t1, t2, t3 = st.tabs(["Performance History", "Correlation Heatmap", "Sector Exposure"])
        
        with t1:
            cum_port = (1 + port_daily_ret.loc[common_idx]).cumprod()
            cum_spy = (1 + spy_daily_ret.loc[common_idx]).cumprod()
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Strategy", line=dict(color="#0047bb", width=3)))
            fig_hist.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, name="S&P 500", line=dict(color="#d1d1d1", width=2)))
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Drawdown Calculation
            rolling_max = cum_port.expanding().max()
            drawdowns = (cum_port - rolling_max) / rolling_max
            st.write(f"**Historical Maximum Drawdown:** {drawdowns.min():.2%}")

        with t2:
            corr_matrix = returns[selected].corr()
            fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_heat, use_container_width=True)

        with t3:
            sectors = []
            for t in weights.index:
                try: sectors.append(yf.Ticker(t).info.get('sector', 'Other'))
                except: sectors.append('Other')
            tree_df = pd.DataFrame({"Asset": weights.index, "Weight": weights.values, "Sector": sectors})
            fig_tree = px.treemap(tree_df, path=['Sector', 'Asset'], values='Weight', color='Weight', color_continuous_scale='RdBu')
            st.plotly_chart(fig_tree, use_container_width=True)

        # 8. MONTE CARLO PROJECTION
        st.divider()
        st.write(f"### 🔮 1-Year Wealth Projection (Scaling from ${total_capital:,.2f})")
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

        # 9. INDIVIDUAL ASSET CARDS
        st.divider()
        st.write("### 🔍 Fundamental Holding Diagnostics")
        for ticker in weights.head(5).index.tolist():
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader(ticker)
                st.write(f"**Allocated:** ${total_capital * weights[ticker]:,.2f}")
                try:
                    t_info = yf.Ticker(ticker).info
                    st.write(f"**P/E:** {t_info.get('trailingPE', 'N/A')}")
                    st.caption(t_info.get('longBusinessSummary', '')[:300] + "...")
                except: pass
            with c2:
                fig_s = px.line(prices[ticker].tail(252), height=200)
                fig_s.update_layout(showlegend=False, margin=dict(t=0, b=0), yaxis_title="Price")
                st.plotly_chart(fig_s, use_container_width=True)

        st.success("Analysis Complete.")
else:
    st.info("👈 Configuration ready. Execute analysis to begin.")
