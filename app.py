import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, HRPOpt

# --- PAGE CONFIG ---
st.set_page_config(page_title="S&P 500 Quantum-Logic Optimizer", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; background-color: #0047bb; color: white; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 S&P 500 Portfolio Optimizer")
st.subheader("Developed by Nicolas Peyrou-Lauga")
st.info("This engine uses Factor Ranking (Momentum/Quality) combined with Hierarchical Risk Parity (HRP) for stable asset allocation.")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Strategy Settings")
n_stocks = st.sidebar.slider("Number of Top Stocks to Select", 10, 30, 25)
max_weight = st.sidebar.slider("Max Weight per Individual Stock (%)", 5, 20, 10) / 100
start_date = st.sidebar.date_input("Analysis Start Date", pd.to_datetime("2020-01-01"))

# Hard-coded Universe (Top S&P Components)
UNIVERSE = [
    "AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA",
    "UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK",
    "BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD",
    "QCOM","ORCL","TXN","INTC"
]

# --- THE "ONE BUTTON" ENGINE ---
if st.button("🚀 Run Optimization"):
    with st.spinner("Fetching market data and calculating factors..."):
        
        # 1. Download Data
        all_tickers = UNIVERSE + ["SPY"]
        data = yf.download(all_tickers, start=start_date, progress=False)["Close"].dropna(axis=1)
        prices = data[UNIVERSE]
        spy_prices = data["SPY"]
        
        # 2. Factor Definitions (Momentum & Quality)
        returns = prices.pct_change().dropna()
        momentum = prices.shift(21).pct_change(252).iloc[-1]
        volatility = returns.std() * np.sqrt(252)
        quality = momentum / volatility
        
        # 3. Ranking
        factor_df = pd.DataFrame({"momentum": momentum, "quality": quality}).dropna()
        factor_df["avg_rank"] = (factor_df["momentum"].rank(ascending=False) + 
                                 factor_df["quality"].rank(ascending=False)) / 2
        
        selected_tickers = factor_df.sort_values("avg_rank").head(n_stocks).index.tolist()
        prices_sel = prices[selected_tickers]
        returns_sel = returns[selected_tickers]

        # 4. HRP Optimization
        hrp = HRPOpt(returns_sel)
        weights = pd.Series(hrp.optimize())
        
        # Apply Capping & Normalization
        weights = weights.clip(upper=max_weight)
        weights = weights / weights.sum()
        weights = weights.sort_values(ascending=False)

        # --- RESULTS LAYOUT ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("### 🏆 Optimal Allocations")
            weight_df = pd.DataFrame({"Ticker": weights.index, "Weight (%)": (weights.values * 100).round(2)})
            st.dataframe(weight_df, height=400)

        with col2:
            st.write("### 📈 Cumulative Performance vs. S&P 500")
            # Calculate Backtest
            port_daily_ret = (returns_sel * weights).sum(axis=1)
            spy_daily_ret = spy_prices.pct_change().dropna()
            
            # Align dates
            common_dates = port_daily_ret.index.intersection(spy_daily_ret.index)
            cum_port = (1 + port_daily_ret.loc[common_dates]).cumprod()
            cum_spy = (1 + spy_daily_ret.loc[common_dates]).cumprod()
            
            comparison_df = pd.DataFrame({
                "Strategy (Your Algo)": cum_port,
                "Benchmark (SPY)": cum_spy
            })
            
            fig = px.line(comparison_df, labels={"value": "Growth of $1", "index": "Date"})
            fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig, use_container_width=True)

        # --- SECTOR DISTRIBUTION ---
        st.write("### 🏗️ Portfolio Concentration")
        fig_pie = px.pie(weight_df, values='Weight (%)', names='Ticker', hole=0.4)
        st.plotly_chart(fig_pie)

        st.success("Analysis Complete. This portfolio is optimized for risk-adjusted returns using Hierarchical Clustering.")

else:
    st.write("👈 Adjust the parameters in the sidebar and click **Run Optimization** to see the results.")
