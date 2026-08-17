import streamlit as st
import pandas as pd
import numpy as np

# Set base configuration
st.set_page_config(page_title="Trading Platform", layout="wide")


# ---------------------------------------------------------
# PAGE 1: Main Overview Dashboard
# ---------------------------------------------------------
def main_dashboard():
    st.title("📈 Main Trading Overview")
    st.write("Welcome to the trading dashboard. Use the button below to jump to detailed analytics.")
    
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quick Status")
        st.success("System Status: Operational")
        st.info("Active Strategy: Momentum Alpha v2")

    with col2:
        st.subheader("Navigation")
        # Trigger button that navigates directly to the Analytics page
        if st.button("🚀 Open Market Analytics & Strategy Summary", type="primary"):
            st.switch_page(analytics_page)


# ---------------------------------------------------------
# PAGE 2: Summarized Data & Analytics
# ---------------------------------------------------------
def summary_analytics():
    st.title("📊 Market Analytics & Strategy Summary")
    
    # Navigation link back to the main page
    st.page_link(main_page, label="⬅ Back to Dashboard", icon="🏠")
    
    st.divider()

    # --- TOP METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="BTC/USDT Price", value="$64,250.00", delta="+2.4%")
    with col2:
        st.metric(label="Total 24h Liquidity", value="$1.42B", delta="-0.8%")
    with col3:
        st.metric(label="Strategy Direction", value="BULLISH", delta="84% Confidence")
    with col4:
        st.metric(label="High/Low Liquidity Imbalance", value="1.85", delta="High Ratio")

    st.divider()

    # --- ROW 2: Price Movement & Prediction ---
    chart_col, insight_col = st.columns([2, 1])

    with chart_col:
        st.subheader("📈 Price Movement & Model Forecast")
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
        historical = np.cumsum(np.random.randn(20)) + 63000
        forecast = np.cumsum(np.random.randn(10)) + historical[-1]
        
        df_hist = pd.DataFrame({"Date": dates[:20], "Historical Price": historical}).set_index("Date")
        df_fore = pd.DataFrame({"Date": dates[20:], "Forecast": forecast}).set_index("Date")
        
        st.line_chart(pd.concat([df_hist, df_fore]))

    with insight_col:
        st.subheader("🤖 Strategy Prediction Insights")
        st.write("**Model Signal:** Strong Buy")
        st.write("**Target Range:** $65,500 – $66,200")
        st.write("**Stop Loss Level:** $62,800")
        st.write("**Recommended Exposure:** 65% Long")
        st.info("Prediction is based on order book imbalance, funding rate shifts, and VWAP deviation.")

    st.divider()

    # --- ROW 3: Liquidity Pools & Winners / Losers ---
    liq_col, wl_col = st.columns(2)

    with liq_col:
        st.subheader("🌊 Places of High & Low Liquidity")
        
        liquidity_df = pd.DataFrame({
            "Asset": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT"],
            "Zone Price": ["$62,000", "$3,400", "$140", "$580", "$28"],
            "Liquidity Pool Depth": ["High ($45M)", "High ($32M)", "Low ($4M)", "Medium ($12M)", "Low ($2M)"],
            "Type": ["Order Wall Support", "Resistance Cluster", "Thin Liquidity (Slippage)", "Balanced", "Breakout Target"]
        })
        st.dataframe(liquidity_df, use_container_width=True, hide_index=True)

    with wl_col:
        st.subheader("🏆 Current Winners & Losers (24h)")
        
        tab_gainers, tab_losers = st.tabs(["🔥 Top Gainers", "📉 Top Losers"])
        
        with tab_gainers:
            gainers = pd.DataFrame({
                "Asset": ["SOL", "NEAR", "RENDER", "SUI"],
                "Price": ["$145.20", "$5.12", "$6.40", "$1.85"],
                "24h Change": ["+12.4%", "+9.8%", "+8.2%", "+7.1%"]
            })
            st.dataframe(gainers, hide_index=True, use_container_width=True)
            
        with tab_losers:
            losers = pd.DataFrame({
                "Asset": ["DOGE", "SHIB", "PEPE", "WIF"],
                "Price": ["$0.112", "$0.000017", "$0.000008", "$1.92"],
                "24h Change": ["-6.5%", "-5.2%", "-4.8%", "-3.9%"]
            })
            st.dataframe(losers, hide_index=True, use_container_width=True)


# ---------------------------------------------------------
# ROUTING & NAVIGATION MANAGEMENT
# ---------------------------------------------------------
main_page = st.Page(main_dashboard, title="Main Overview", icon="📈", default=True)
analytics_page = st.Page(summary_analytics, title="Market Summary & Analytics", icon="📊")

# Run navigation
pg = st.navigation([main_page, analytics_page])
pg.run()
    
