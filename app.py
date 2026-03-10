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
st.markdown("**Lead Developer: Nicolas Peyrou-Lauga** | Quantitative Investment Pipeline")

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
    with st.spinner("Processing Data, Optimizing Weights & Forecasting..."):
        
        # 1. Data Logic
        all_tickers = UNIVERSE + ["SPY"]
        data = yf.download(all_tickers, start=start_date, progress=False)["Close"].dropna(axis=1)
        prices = data[UNIVERSE]
        spy_prices = data["SPY"]
        
        # 2. Factor Ranking
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

        # 4. Performance & 1-Year Forecast Math
        port_daily_ret = (returns_sel * weights).sum(axis=1)
        spy_daily_ret = spy_prices.pct_change().dropna()
        common_dates = port_daily_ret.index.intersection(spy_daily_ret.index)
        
        cum_port = (1 + port_daily_ret.loc[common_dates]).cumprod()
        cum_spy = (1 + spy_daily_ret.loc[common_dates]).cumprod()
        
        # 365-Day Projection
        avg_daily_ret = port_daily_ret.mean()
        avg_spy_ret = spy_daily_ret.mean()
        last_val_port = cum_port.iloc[-1]
        last_val_spy = cum_spy.iloc[-1]
        
        future_dates = pd.date_range(cum_port.index[-1], periods=365, freq='D')
        
        proj_port = [last_val_port * (1 + avg_daily_ret)**i for i in range(len(future_dates))]
        proj_spy = [last_val_spy * (1 + avg_spy_ret)**i for i in range(len(future_dates))]
        
        proj_port_ser = pd.Series(proj_port, index=future_dates)
        proj_spy_ser = pd.Series(proj_spy, index=future_dates)

        # --- VISUALIZATION SECTION ---
        col_main, col_side = st.columns([2, 1])

        with col_main:
            st.write("### 📈 Performance & 1-Year Forecast")
            fig = go.Figure()
            # Historical Lines
            fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio (Historical)", line=dict(color="#0047bb", width=3)))
            fig.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, name="S&P 500 (Historical)", line=dict(color="#d1d1d1", width=2)))
            
            # Forecast Lines (Dotted)
            fig.add_trace(go.Scatter(x=proj_port_ser.index, y=proj_port_ser, name="Portfolio (1Y Forecast)", line=dict(color="#0047bb", width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=proj_spy_ser.index, y=proj_spy_ser, name="S&P 500 (1Y Forecast)", line=dict(color="#d1d1d1", width=2, dash='dash')))
            
            fig.update_layout(hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig, use_container_width=True)

        with col_side:
            st.write("### 🏆 Asset Allocation")
            weight_df = pd.DataFrame({"Ticker": weights.index, "Weight": weights.values})
            fig_pie = px.pie(weight_df, values='Weight', names='Ticker', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- QUANTITATIVE DEEP-DIVE (REPLACING NEWS) ---
        st.divider()
        st.write("### 🔍 Fundamental Snapshot: Top 5 Holdings")
        top_5 = weights.head(5).index.tolist()
        
        info_cols = st.columns(5)
        for i, ticker in enumerate(top_5):
            with info_cols[i]:
                try:
                    t_obj = yf.Ticker(ticker)
                    info = t_obj.info
                    st.subheader(ticker)
                    st.metric("Price", f"${info.get('currentPrice', 'N/A')}")
                    st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                    st.write(f"**Div Yield:** {info.get('dividendYield', 0)*100:.2f}%")
                    
                    # 52 Week Range Progress Bar (Simulated)
                    low = info.get('fiftyTwoWeekLow', 0)
                    high = info.get('fiftyTwoWeekHigh', 1)
                    current = info.get('currentPrice', 0)
                    if high > low:
                        progress = (current - low) / (high - low)
                        st.write(f"**52W Range:**")
                        st.progress(min(max(progress, 0.0), 1.0))
                        st.caption(f"L: ${low} | H: ${high}")
                except:
                    st.write(f"Data unavailable for {ticker}")

        st.success(f"Full Analysis Deployed: {risk_level}")
else:
    st.info("👈 Adjust parameters and click Run to generate your quantitative terminal.")
