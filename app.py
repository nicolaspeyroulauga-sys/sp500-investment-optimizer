import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum Portfolio Terminal Pro", layout="wide")

# --- HEADER ---
st.title("🏛️ Institutional Portfolio Terminal")
st.markdown("**Lead Developer: Nicolas Peyrou-Lauga** | Quantitative Investment Analysis")

# --- SIDEBAR ---
st.sidebar.header("1. Strategy Settings")
investment_amount = st.sidebar.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000)
n_stocks = st.sidebar.slider("Number of Stocks", 10, 30, 20)
max_weight = st.sidebar.slider("Max Weight (%)", 5, 20, 10) / 100

st.sidebar.header("2. Risk Profile")
risk_level = st.sidebar.select_slider(
    "Select Strategy Profile",
    options=["Conservative (Min Vol)", "Balanced (HRP)", "Aggressive (Max Sharpe)"],
    value="Balanced (HRP)"
)

# Asset Universe
UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]

if st.button("🚀 Execute Quantitative Analysis"):
    with st.spinner("Analyzing Market Dynamics..."):
        
        # 1. DATA & OPTIMIZATION
        all_tickers = UNIVERSE + ["SPY"]
        data = yf.download(all_tickers, start="2021-01-01", progress=False)["Close"].dropna(axis=1)
        prices, spy_prices = data[UNIVERSE], data["SPY"]
        returns = prices.pct_change().dropna()
        
        # Factor selection logic
        mom = prices.shift(21).pct_change(252).iloc[-1]
        vol = returns.std() * np.sqrt(252)
        factor_df = pd.DataFrame({"rank": (mom.rank(ascending=False) + (mom/vol).rank(ascending=False))/2})
        selected = factor_df.sort_values("rank").head(n_stocks).index.tolist()
        
        mu = expected_returns.mean_historical_return(prices[selected])
        S = risk_models.CovarianceShrinkage(prices[selected]).ledoit_wolf()

        if "Conservative" in risk_level:
            ef = EfficientFrontier(mu, S); ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.min_volatility())
        elif "Aggressive" in risk_level:
            ef = EfficientFrontier(mu, S); ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.max_sharpe())
        else:
            hrp = HRPOpt(returns[selected]); weights = pd.Series(hrp.optimize())
            weights = (weights.clip(upper=max_weight) / (weights.clip(upper=max_weight).sum()))

        weights = weights.sort_values(ascending=False)

        # 2. MONTE CARLO SIMULATION (The "Pro" Feature)
        st.write("### 🔮 Monte Carlo Risk Projection (1-Year)")
        
        # Simulating 100 paths
        n_sims = 100
        n_days = 252
        port_ret_avg = np.dot(weights, mu) / 252
        port_vol_avg = np.sqrt(np.dot(weights.T, np.dot(S, weights))) / np.sqrt(252)
        
        sim_results = np.zeros((n_days, n_sims))
        for i in range(n_sims):
            daily_sim_rets = np.random.normal(port_ret_avg, port_vol_avg, n_days)
            sim_results[:, i] = investment_amount * (1 + daily_sim_rets).cumprod()
        
        # Charting MC
        fig_mc = go.Figure()
        for i in range(n_sims):
            fig_mc.add_trace(go.Scatter(y=sim_results[:, i], mode='lines', line=dict(width=1), opacity=0.1, showlegend=False))
        
        # Highlighting the Median Path
        fig_mc.add_trace(go.Scatter(y=np.median(sim_results, axis=1), name="Median Expectation", line=dict(color='red', width=3)))
        fig_mc.update_layout(xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig_mc, use_container_width=True)

        # 3. PERFORMANCE DASHBOARD
        st.divider()
        c1, c2, c3 = st.columns(3)
        
        # Calculate VaR (Value at Risk 95%)
        daily_rets = (returns[selected] * weights).sum(axis=1)
        var_95 = np.percentile(daily_rets, 5)
        
        c1.metric("Expected Annual Return", f"{np.dot(weights, mu):.2%}")
        c2.metric("Annual Volatility", f"{np.sqrt(np.dot(weights.T, np.dot(S, weights))):.2%}")
        c3.metric("Daily Value-at-Risk (95%)", f"{var_95:.2%}", help="The potential loss in a single day with 95% confidence.")

        # 4. REBALANCING TABLE (The "Actionable" Feature)
        st.write("### 📝 Execution: Buy List & Rebalancing")
        latest_prices = prices[selected].iloc[-1]
        rebalance_df = pd.DataFrame({
            "Ticker": weights.index,
            "Target Weight (%)": (weights.values * 100).round(2),
            "Allocation ($)": (weights.values * investment_amount).round(2),
            "Shares to Buy": ((weights.values * investment_amount) / latest_prices).astype(int)
        })
        st.table(rebalance_df)

        # 5. FUNDAMENTAL DEEP-DIVE
        st.divider()
        st.write("### 🔍 Fundamental Insights")
        top_3 = weights.head(3).index.tolist()
        f_cols = st.columns(3)
        for i, t in enumerate(top_3):
            with f_cols[i]:
                info = yf.Ticker(t).info
                st.subheader(f"{t}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                st.write(f"**Profit Margin:** {info.get('profitMargins', 0)*100:.1f}%")

        st.success("Analysis Complete. Terminal Ready for Execution.")
