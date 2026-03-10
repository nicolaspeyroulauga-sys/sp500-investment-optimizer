import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from pypfopt import expected_returns, risk_models, HRPOpt, EfficientFrontier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum-Logic Portfolio Terminal", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 Multi-Strategy Portfolio Terminal")
st.markdown(f"**Lead Developer: Nicolas Peyrou-Lauga** | Finance & Quantitative Analysis")

# --- SIDEBAR ---
st.sidebar.header("1. Strategy Settings")
n_stocks = st.sidebar.slider("Number of Top Stocks to Rank", 10, 30, 25)
max_weight = st.sidebar.slider("Max Individual Weight (%)", 5, 20, 10) / 100

st.sidebar.header("2. Risk Profile")
risk_level = st.sidebar.select_slider(
    "Select Risk Appetite",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

start_date = st.sidebar.date_input("Analysis Start Date", pd.to_datetime("2020-01-01"))

# Asset Universe (S&P 500 Leaders)
UNIVERSE = [
    "AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA",
    "UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK",
    "BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD",
    "QCOM","ORCL","TXN","INTC"
]

if st.button("🚀 Optimize & Analyze"):
    with st.spinner("Processing Market Data and Running Optimization..."):
        
        # 1. Data Fetching
        all_tickers = UNIVERSE + ["SPY"]
        data = yf.download(all_tickers, start=start_date, progress=False)["Close"].dropna(axis=1)
        prices = data[UNIVERSE]
        spy_prices = data["SPY"]
        
        # 2. Factor Ranking (Momentum & Quality Logic)
        returns = prices.pct_change().dropna()
        momentum = prices.shift(21).pct_change(252).iloc[-1]
        volatility_vec = returns.std() * np.sqrt(252)
        quality = momentum / volatility_vec
        
        factor_df = pd.DataFrame({"momentum": momentum, "quality": quality}).dropna()
        factor_df["avg_rank"] = (factor_df["momentum"].rank(ascending=False) + 
                                 factor_df["quality"].rank(ascending=False)) / 2
        
        selected_tickers = factor_df.sort_values("avg_rank").head(n_stocks).index.tolist()
        prices_sel = prices[selected_tickers]
        returns_sel = returns[selected_tickers]

        # 3. Optimization Engine
        mu = expected_returns.mean_historical_return(prices_sel)
        S = risk_models.CovarianceShrinkage(prices_sel).ledoit_wolf()

        if "Low" in risk_level:
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= max_weight)
            weights = pd.Series(ef.min_volatility())
        elif "High" in risk_level:
            ef = EfficientFrontier(mu, S
