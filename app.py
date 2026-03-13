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
total_capital = st.sidebar.number_input("Total Investment Capital ($)", min_value=1.0, value=10000.0, step=100.0)

st.sidebar.header("2. Strategy Settings")
n_stocks = st.sidebar.slider("Number of Assets", 5, 30, 30)
max_weight = st.sidebar.slider("Concentration Limit (%)", 5, 20, 10) / 100

st.sidebar.header("3. Risk & Optimization")
risk_level = st.sidebar.select_slider(
    "Asset Allocation Strategy",
    options=["Low (Min Variance)", "Medium (Balanced HRP)", "High (Max Sharpe)"],
    value="Medium (Balanced HRP)"
)

# FIXED UNIVERSE
START_DATE = "2020-01-01"
RISK_FREE_RATE = 0.02
FULL_UNIVERSE = ["AAPL","MSFT","GOOG","AMZN","NVDA","META","JPM","V","MA","UNH","HD","XOM","AVGO","COST","PEP","ABBV","KO","MRK","BAC","PFE","TMO","CSCO","ADBE","CRM","WMT","MCD","QCOM","ORCL","TXN","INTC"]
UNIVERSE = FULL_UNIVERSE[:n_stocks]

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
        
        spy_raw = yf.download("SPY", start=START_DATE, progress=False)["Close"]
        spy_rets = spy_raw.pct_change().dropna()
        combined_df = pd.concat([port_daily_ret, spy_rets], axis=1).dropna()
        combined_df.columns = ['portfolio', 'benchmark']
        
        beta = 1.0
        if not combined_df.empty and len(combined_df) > 2:
            matrix = np.cov(combined_df['portfolio'], combined_df['benchmark'])
            beta = matrix[0, 1] / matrix[1, 1]

        # --- KPI DASHBOARD ---
        st.write(f"### 🔑 Tactical KPIs (Basis: ${total_capital:,.2f})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ann. Return", f"{ann_return:.2%}")
        m2.metric("Ann. Volatility", f"{ann_vol:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m4.metric("Portfolio Beta", f"{beta:.2f}")

        # --- FRACTIONAL TRADE TABLE ---
        st.divider()
        st.write("### 📋 Trade Execution & Asset Allocation")
        alloc_df = pd.DataFrame(weights, columns=["Weight"])
        alloc_df["Investment ($)"] = alloc_df["Weight"] * total_capital
        alloc_df["Current Price ($)"] = current_prices[weights.index]
        alloc_df["Shares to Buy"] = alloc_df["Investment ($)"] / alloc_df["Current Price ($)"]
        
        st.dataframe(alloc_df.style.format({
            "Weight": "{:.2%}", "Investment ($)": "${:,.2f}", 
            "Current Price ($)": "${:,.2f}", "Shares to Buy": "{:.3f}"
        }), use_container_width=True)

        # 4. ENHANCED VISUALS (THE "MORE STUFF" SECTION)
        st.divider()
        t_risk, t_intel, t_growth = st.tabs(["🌩️ Risk & Sector Map", "🔍 Individual Asset Intel", "📈 Growth Projections"])

        with t_risk:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.write("#### Sector Concentration Architecture")
                sectors = []
                # Faster metadata fetching
                for t in weights.index:
                    try: sectors.append(yf.Ticker(t).fast_info['lastPrice'] if False else "Sector Loading...") # Placeholder for speed
                    except: sectors.append('Other')
                
                # Manual sector mapping for speed (prevents RateLimitError)
                sector_map = {
                    "AAPL":"Tech", "MSFT":"Tech", "NVDA":"Tech", "GOOG":"Comm", "META":"Comm",
                    "AMZN":"Cons Disc", "HD":"Cons Disc", "MCD":"Cons Disc",
                    "JPM":"Finance", "V":"Finance", "MA":"Finance", "BAC":"Finance",
                    "XOM":"Energy", "CVX":"Energy", "UNH":"Health", "ABBV":"Health"
                }
                mapped_sectors = [sector_map.get(t, "Other") for t in weights.index]
                
                tree_df = pd.DataFrame({"Asset": weights.index, "Weight": weights.values, "Sector": mapped_sectors})
                fig_tree = px.treemap(tree_df, path=['Sector', 'Asset'], values='Weight', color='Sector',
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_tree, use_container_width=True)

            with c2:
                st.write("#### Correlation Matrix")
                corr_matrix = returns[weights.index].corr()
                fig_heat = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', aspect="auto")
                st.plotly_chart(fig_heat, use_container_width=True)

        with t_intel:
            st.write("#### 📰 Asset Deep-Dive & News Feed")
            selected_ticker = st.selectbox("Select Asset for Real-Time Intel", weights.index)
            
            # This only runs when a stock is selected, preventing blocks
            ticker_obj = yf.Ticker(selected_ticker)
            i1, i2 = st.columns([1, 2])
            with i1:
                st.write(f"**{selected_ticker} Key Metrics**")
                info = ticker_obj.fast_info
                st.metric("Market Cap", f"${info['marketCap']/1e12:.2f}T")
                st.metric("Day Change", f"{((info['lastPrice']/info['previousClose'])-1):.2%}")
                st.write("**Asset Summary**")
                st.caption(ticker_obj.info.get('longBusinessSummary', 'N/A')[:500] + "...")
            
            with i2:
                st.write(f"**Latest {selected_ticker} News**")
                news = ticker_obj.news[:5]
                for item in news:
                    st.markdown(f"**{item['title']}**")
                    st.caption(f"Source: {item['publisher']} | [Link]({item['link']})")
                    st.divider()

        with t_growth:
            st.write("#### 🔮 1-Year Wealth & Monte Carlo Projection")
            n_sims, n_days = 1000, 252
            sim_paths = np.zeros((n_days, n_sims))
            for i in range(n_sims):
                daily_rets = np.random.normal(port_daily_ret.mean(), port_daily_ret.std(), n_days)
                sim_paths[:, i] = total_capital * (1 + daily_rets).cumprod()
            
            p5, p50, p95 = np.percentile(sim_paths, 5, axis=1), np.percentile(sim_paths, 50, axis=1), np.percentile(sim_paths, 95, axis=1)
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p95, line=dict(width=0), name="Upper Bound (95%)"))
            fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p5, line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 71, 187, 0.1)', name="Lower Bound (5%)"))
            fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p50, name="Median Path", line=dict(color="#0047bb", width=3)))
            st.plotly_chart(fig_mc, use_container_width=True)

        st.success("Analysis complete. Assets synchronized and optimized.")
else:
    st.info("👈 System Standby. Click Execute to generate interactive analysis.")
