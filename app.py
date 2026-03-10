import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum-Logic Terminal v2", layout="wide")

# --- HEADER ---
st.title("📊 Multi-Strategy Portfolio Terminal")
st.markdown("**Lead Developer: Nicolas Peyrou-Lauga**")

# --- SIDEBAR ---
st.sidebar.header("1. Strategy Settings")
n_stocks = st.sidebar.slider("Number of Top Stocks", 10, 30, 25)
max_weight = st.sidebar.slider("Max Individual Weight (%)", 5, 20, 10) / 100

st.sidebar.header("2. Risk Profile")
risk_level = st.sidebar.select_slider(
    "Select Risk Appetite",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

start_date = st.sidebar.date_input("Analysis Start Date", pd.to_datetime("2020-01-01"))

UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]

if st.button("🚀 Run Portfolio Analysis"):
    with st.spinner("Fetching Data & Forecasting..."):
        
        # 1. Data Logic
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
            ef = EfficientFrontier(mu, S); ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.min_volatility())
        elif "High" in risk_level:
            ef = EfficientFrontier(mu, S); ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.max_sharpe())
        else:
            hrp = HRPOpt(returns_sel); weights = pd.Series(hrp.optimize())
            weights = (weights.clip(upper=max_weight) / (weights.clip(upper=max_weight).sum()))

        # 4. Performance & Projection Math
        port_daily_ret = (returns_sel * weights).sum(axis=1)
        cum_port = (1 + port_daily_ret).cumprod()
        
        # Projection (Next 90 Days)
        avg_daily_ret = port_daily_ret.mean()
        last_val = cum_port.iloc[-1]
        future_dates = pd.date_range(cum_port.index[-1], periods=90, freq='D')
        projection = [last_val * (1 + avg_daily_ret)**i for i in range(len(future_dates))]
        proj_series = pd.Series(projection, index=future_dates)

        # --- CHARTS ---
        st.write("### 📈 Portfolio Growth & 90-Day Projection")
        fig = go.Figure()
        # Historical Line
        fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Historical Performance", line=dict(color="#0047bb", width=3)))
        # Dotted Projection Line
        fig.add_trace(go.Scatter(x=proj_series.index, y=proj_series, name="90-Day Projection", line=dict(color="#0047bb", width=2, dash='dash')))
        
        fig.update_layout(hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True)

        # --- NEWS CARDS (TOP 5) ---
        st.divider()
        st.write("### 📰 Intelligence Feed: Top 5 Holdings")
        top_5 = weights.sort_values(ascending=False).head(5).index.tolist()
        
        cols = st.columns(5)
        for i, ticker in enumerate(top_5):
            with cols[i]:
                st.subheader(ticker)
                try:
                    news = yf.Ticker(ticker).news[:2]
                    for n in news:
                        st.markdown(f"**{n.get('publisher', 'News')}**")
                        st.write(n.get('title', 'No Headline'))
                        st.caption(f"[Read Article]({n.get('link','#')})")
                        st.markdown("---")
                except:
                    st.write("Feed unavailable")

        st.success("Analysis, Projection, and News Feed Updated.")
