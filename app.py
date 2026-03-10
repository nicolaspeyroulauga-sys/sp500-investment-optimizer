import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum-Logic Portfolio Terminal", layout="wide")

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

# S&P 500 Leading Tickers
UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]

if st.button("🚀 Run Portfolio Analysis"):
    with st.spinner("Processing Data, Optimizing Weights & Fetching Intelligence..."):
        
        # 1. Data Logic
        all_tickers = UNIVERSE + ["SPY"]
        data = yf.download(all_tickers, start=start_date, progress=False)["Close"].dropna(axis=1)
        prices = data[UNIVERSE]
        spy_prices = data["SPY"]
        
        # 2. Factor Ranking (Your Momentum/Quality logic)
        returns = prices.pct_change().dropna()
        momentum = prices.shift(21).pct_change(252).iloc[-1]
        vol_vec = returns.std() * np.sqrt(252)
        quality = momentum / vol_vec
        factor_df = pd.DataFrame({"momentum": momentum, "quality": quality}).dropna()
        factor_df["avg_rank"] = (factor_df["momentum"].rank(ascending=False) + factor_df["quality"].rank(ascending=False)) / 2
        selected_tickers = factor_df.sort_values("avg_rank").head(n_stocks).index.tolist()
        
        prices_sel = prices[selected_tickers]
        returns_sel = returns[selected_tickers]

        # 3. Optimization Logic
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

        weights = weights.sort_values(ascending=False)

        # 4. Performance & Projection Math
        port_daily_ret = (returns_sel * weights).sum(axis=1)
        spy_daily_ret = spy_prices.pct_change().dropna()
        common_dates = port_daily_ret.index.intersection(spy_daily_ret.index)
        
        cum_port = (1 + port_daily_ret.loc[common_dates]).cumprod()
        cum_spy = (1 + spy_daily_ret.loc[common_dates]).cumprod()
        
        # 90-Day Projection
        avg_daily_ret = port_daily_ret.mean()
        last_val = cum_port.iloc[-1]
        future_dates = pd.date_range(cum_port.index[-1], periods=90, freq='D')
        projection = [last_val * (1 + avg_daily_ret)**i for i in range(len(future_dates))]
        proj_series = pd.Series(projection, index=future_dates)

        # --- VISUALIZATION SECTION ---
        col_main, col_side = st.columns([2, 1])

        with col_main:
            st.write("### 📈 Cumulative Performance & 90-Day Forecast")
            fig = go.Figure()
            # Portfolio Line
            fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Your Strategy", line=dict(color="#0047bb", width=3)))
            # SPY Line
            fig.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, name="S&P 500 (SPY)", line=dict(color="#d1d1d1", width=2)))
            # Dotted Projection
            fig.add_trace(go.Scatter(x=proj_series.index, y=proj_series, name="90-Day Projection", line=dict(color="#0047bb", width=2, dash='dash')))
            
            fig.update_layout(hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig, use_container_width=True)

        with col_side:
            st.write("### 🏆 Asset Allocation")
            weight_df = pd.DataFrame({"Ticker": weights.index, "Weight": weights.values})
            fig_pie = px.pie(weight_df, values='Weight', names='Ticker', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- NEWS INTELLIGENCE FEED ---
        st.divider()
        st.write("### 📰 Intelligence Feed: Top 5 Holdings")
        top_5 = weights.head(5).index.tolist()
        
        news_cols = st.columns(5)
        for i, ticker in enumerate(top_5):
            with news_cols[i]:
                st.subheader(ticker)
                try:
                    t_obj = yf.Ticker(ticker)
                    news_list = t_obj.news
                    
                    if news_list and len(news_list) > 0:
                        for n in news_list[:2]:
                            title = n.get('title', 'Headline Unavailable')
                            publisher = n.get('publisher', 'Financial News')
                            link = n.get('link', '#')
                            st.markdown(f"**{publisher}**")
                            st.write(title)
                            st.caption(f"[Full Story]({link})")
                            st.markdown("---")
                    else:
                        # Fallback: Show Company Description if News is empty
                        summary = t_obj.info.get('longBusinessSummary', 'No news or summary available.')
                        st.write("📊 *Market Context:*")
                        st.caption(summary[:250] + "...")
                except:
                    st.write("Live feed sync pending...")

        st.success(f"Strategy Deployed: {risk_level}")
else:
    st.info("👈 Adjust parameters and click Run to generate your quantitative terminal.")
