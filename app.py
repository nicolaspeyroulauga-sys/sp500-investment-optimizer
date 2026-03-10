import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum-Logic Portfolio Terminal", layout="wide")

# --- HEADER ---
st.title("📊 Multi-Strategy Portfolio Terminal")
st.markdown(f"**Lead Developer: Nicolas Peyrou-Lauga**")

# --- SIDEBAR ---
st.sidebar.header("1. Strategy Settings")
n_stocks = st.sidebar.slider("Number of Top Stocks", 10, 30, 25)
max_weight = st.sidebar.slider("Max Individual Weight (%)", 5, 20, 10) / 100

st.sidebar.header("2. Risk Profile")
risk_level = st.sidebar.select_slider(
    "Select your Risk Appetite",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

start_date = st.sidebar.date_input("Analysis Start Date", pd.to_datetime("2020-01-01"))

UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]

if st.button("🚀 Optimize & Analyze"):
    with st.spinner("Processing Market Data..."):
        
        # 1. Data Fetching
        all_tickers = UNIVERSE + ["SPY"]
        data = yf.download(all_tickers, start=start_date, progress=False)["Close"].dropna(axis=1)
        prices = data[UNIVERSE]
        spy_prices = data["SPY"]
        
        # 2. Factor Ranking
        returns = prices.pct_change().dropna()
        momentum = prices.shift(21).pct_change(252).iloc[-1]
        volatility_vec = returns.std() * np.sqrt(252)
        quality = momentum / volatility_vec
        factor_df = pd.DataFrame({"momentum": momentum, "quality": quality}).dropna()
        factor_df["avg_rank"] = (factor_df["momentum"].rank(ascending=False) + factor_df["quality"].rank(ascending=False)) / 2
        selected_tickers = factor_df.sort_values("avg_rank").head(n_stocks).index.tolist()
        
        prices_sel = prices[selected_tickers]
        returns_sel = returns[selected_tickers]

        # 3. Optimization
        mu = expected_returns.mean_historical_return(prices_sel)
        S = risk_models.CovarianceShrinkage(prices_sel).ledoit_wolf()

        if "Low" in risk_level:
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.min_volatility())
        elif "High" in risk_level:
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.max_sharpe())
        else:
            hrp = HRPOpt(returns_sel)
            weights = pd.Series(hrp.optimize())
            weights = (weights.clip(upper=max_weight) / (weights.clip(upper=max_weight).sum()))

        # 4. Performance Metrics (The Math from your code)
        port_return = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        sharpe = (port_return - 0.02) / port_vol

        # --- DISPLAY METRICS ---
        st.write("### 🔑 Key Performance Indicators")
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Annual Return", f"{port_return:.2%}")
        m2.metric("Annual Volatility", f"{port_vol:.2%}")
        m3.metric("Sharpe Ratio (Risk-Adjusted)", f"{sharpe:.2f}")

        # --- CHARTS ---
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("### 📈 Cumulative Performance vs. S&P 500")
            port_daily_ret = (returns_sel * weights).sum(axis=1)
            spy_daily_ret = spy_prices.pct_change().dropna()
            common_dates = port_daily_ret.index.intersection(spy_daily_ret.index)
            cum_port = (1 + port_daily_ret.loc[common_dates]).cumprod()
            cum_spy = (1 + spy_daily_ret.loc[common_dates]).cumprod()
            comp_df = pd.DataFrame({"Strategy": cum_port, "S&P 500": cum_spy})
            st.plotly_chart(px.line(comp_df), use_container_width=True)

        with c2:
            st.write("### 🏆 Top Allocations")
            weight_df = pd.DataFrame({"Ticker": weights.index, "Weight (%)": (weights.values * 100).round(2)}).sort_values("Weight (%)", ascending=False)
            st.dataframe(weight_df, height=400)

        # --- NEWS SECTION ---
        st.write("### 📰 Latest News for your Portfolio")
        # We take the top 3 holdings to keep it clean
        top_3 = weights.sort_values(ascending=False).head(3).index.tolist()
        for ticker in top_3:
            st.markdown(f"**{ticker} News Headlines:**")
            news = yf.Ticker(ticker).news
            for item in news[:2]: # Show top 2 headlines per stock
                st.write(f"- [{item['title']}]({item['link']})")

        st.success("Analysis and News Sync Complete.")
