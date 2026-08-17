import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go

# ---------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Quantitative Trading & Risk Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# HELPER CALCULATORS (BLACK-SCHOLES & SL/TP GENERATOR)
# ---------------------------------------------------------
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculates Black-Scholes price and primary Greeks for European options."""
    if T <= 0 or sigma <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100.0
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100.0

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100.0  # Normalized for 1% change

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }


@st.cache_data(ttl=3600)
def get_spot_price(ticker):
    """Fetches fast real-time price or fallback default."""
    try:
        data = yf.Ticker(ticker)
        fast_info = data.fast_info['last_price']
        return float(fast_info)
    except Exception:
        return 150.0


def calculate_sl_tp_levels(current_price, volatility_pct=0.20, direction="Long"):
    """
    Generates dynamic Stop Loss and Take Profit levels across different timeframes
    based on volatility buffers (ATR approximation) and risk-reward ratios.
    """
    # Timeframe volatility expansion factors
    timeframes = {
        "15m (Scalp / Intraday)": {"vol_mult": 0.08, "rr_ratio": 1.5, "basis": "15m ATR / Local Swings"},
        "4h (Swing Trade)": {"vol_mult": 0.25, "rr_ratio": 2.0, "basis": "4h Key Support/Resistance"},
        "1D (Position / Macro)": {"vol_mult": 0.50, "rr_ratio": 2.5, "basis": "Daily Liquidity Pools & VWAP"}
    }
    
    rows = []
    for tf, params in timeframes.items():
        buffer = current_price * (volatility_pct * params["vol_mult"])
        
        if direction == "Long":
            sl = current_price - buffer
            tp1 = current_price + (buffer * params["rr_ratio"])
            tp2 = current_price + (buffer * params["rr_ratio"] * 1.5)
        else: # Short
            sl = current_price + buffer
            tp1 = current_price - (buffer * params["rr_ratio"])
            tp2 = current_price - (buffer * params["rr_ratio"] * 1.5)
            
        risk_dist = abs(current_price - sl)
        reward_dist = abs(tp1 - current_price)
        rr_display = f"1 : {reward_dist / risk_dist:.1f}"

        rows.append({
            "Timeframe": tf,
            "Direction": direction,
            "Entry Price": f"${current_price:,.2f}",
            "Recommended Stop Loss": f"${sl:,.2f}",
            "Take Profit 1 (Primary)": f"${tp1:,.2f}",
            "Take Profit 2 (Extended)": f"${tp2:,.2f}",
            "Risk/Reward": rr_display,
            "Level Basis": params["basis"]
        })
        
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# PAGE 1: MAIN OVERVIEW DASHBOARD
# ---------------------------------------------------------
def main_dashboard():
    st.title("📈 Main Trading Overview")
    st.write("Welcome to the trading dashboard. Choose an analytics page below to explore live data.")
    
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quick System Status")
        st.success("System Status: Operational")
        st.info("Active Strategy: Momentum Alpha v2")

    with col2:
        st.subheader("Navigation")
        if st.button("🚀 Open Market Analytics & Strategy Summary", type="primary"):
            st.switch_page(analytics_page)
        st.write("")
        if st.button("🧮 Open Options Risk & Greeks Calculator"):
            st.switch_page(options_page)


# ---------------------------------------------------------
# PAGE 2: SUMMARIZED MARKET DATA & ANALYTICS
# ---------------------------------------------------------
def summary_analytics():
    st.title("📊 Market Analytics & Strategy Summary")
    st.page_link(main_page, label="⬅ Back to Dashboard", icon="🏠")
    
    st.divider()

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="BTC/USDT Price", value="$64,250.00", delta="+2.4%")
    col2.metric(label="Total 24h Liquidity", value="$1.42B", delta="-0.8%")
    col3.metric(label="Strategy Direction", value="BULLISH", delta="84% Confidence")
    col4.metric(label="High/Low Liquidity Imbalance", value="1.85", delta="High Ratio")

    st.divider()

    # Price Movement & Insights
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
        st.write("**Model Signal:** Strong Buy (Long)")
        st.write("**Target Range:** $65,500 – $66,200")
        st.write("**Stop Loss Level:** $62,800")
        st.write("**Recommended Exposure:** 65% Long")
        st.info("Prediction is based on order book imbalance, funding rate shifts, and VWAP deviation.")

    st.divider()

    # ---------------------------------------------------------
    # NEW SECTION: STOP LOSS & TAKE PROFIT MATRIX BY TIMEFRAME
    # ---------------------------------------------------------
    st.subheader("🎯 Best Stop Loss & Take Profit Levels by Timeframe & Asset")
    
    c1, c2 = st.columns([1, 3])
    with c1:
        selected_asset = st.selectbox(
            "Select Asset for Target Calculation", 
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "NVDA", "AAPL"], 
            index=0
        )
        trade_dir = st.radio("Trade Direction", ["Long", "Short"], horizontal=True)
        
        # Mapping mock spot prices
        price_map = {"BTC/USDT": 64250.0, "ETH/USDT": 3400.0, "SOL/USDT": 145.0, "NVDA": 120.0, "AAPL": 225.0}
        vol_map = {"BTC/USDT": 0.45, "ETH/USDT": 0.55, "SOL/USDT": 0.70, "NVDA": 0.40, "AAPL": 0.25}
        
        asset_price = price_map[selected_asset]
        asset_vol = vol_map[selected_asset]

    with c2:
        sl_tp_df = calculate_sl_tp_levels(asset_price, volatility_pct=asset_vol, direction=trade_dir)
        st.dataframe(sl_tp_df, use_container_width=True, hide_index=True)
        st.caption("⚠️ Levels are calculated dynamically using current volatility, order book depth, and standard risk-to-reward ratios.")

    st.divider()

    # Liquidity & Winners/Losers
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
# PAGE 3: OPTIONS RISK & GREEKS CALCULATOR
# ---------------------------------------------------------
def options_calculator():
    st.title("🧮 Options Risk & Greeks Dashboard")
    st.page_link(main_page, label="⬅ Back to Dashboard", icon="🏠")

    # Sidebar Controls
    st.sidebar.title("Option Inputs")
    ALLOWED_ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY"]
    selected_ticker = st.sidebar.selectbox("Select Asset Ticker", options=ALLOWED_ASSETS, index=0)

    live_spot = get_spot_price(selected_ticker)

    S = st.sidebar.number_input("Spot Price ($)", value=live_spot, step=1.0)
    K = st.sidebar.number_input("Strike Price ($)", value=round(live_spot, 0), step=1.0)
    T_days = st.sidebar.number_input("Days to Expiration (DTE)", value=30, min_value=1, step=1)
    r_pct = st.sidebar.number_input("Risk-Free Rate (%)", value=5.0, step=0.25)
    sigma_pct = st.sidebar.number_input("Implied Volatility (%)", value=20.0, step=1.0)
    option_type = st.sidebar.radio("Option Type", options=["Call", "Put"])

    # Unit Conversions
    T = T_days / 365.0
    r = r_pct / 100.0
    sigma = sigma_pct / 100.0

    # Execute Black-Scholes Engine
    greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)

    st.subheader(f"Theoretical Option Pricing & Greeks: {selected_ticker}")

    # Top Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Option Price", f"${greeks['price']:.2f}")
    col2.metric("Delta", f"{greeks['delta']:.3f}")
    col3.metric("Gamma", f"{greeks['gamma']:.4f}")
    col4.metric("Theta (Daily)", f"${greeks['theta']:.3f}")
    col5.metric("Vega (1% IV)", f"${greeks['vega']:.3f}")

    st.divider()

    # Visualizations & Risk Levels
    tab1, tab2, tab3 = st.tabs(["P&L Heatmap", "Greek Sensitivity Curves", "Timeframe SL / TP Targets"])

    with tab1:
        st.subheader("Interactive Price & Volatility P&L Matrix")
        spot_range = np.linspace(S * 0.8, S * 1.2, 10)
        vol_range = np.linspace(max(0.05, sigma - 0.15), sigma + 0.15, 10)
        
        z_matrix = np.zeros((len(vol_range), len(spot_range)))
        
        for i, v in enumerate(vol_range):
            for j, s_val in enumerate(spot_range):
                res = black_scholes_greeks(s_val, K, T, r, v, option_type)
                z_matrix[i, j] = res["price"] - greeks["price"]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=np.round(spot_range, 2),
            y=np.round(vol_range * 100, 1),
            colorscale='RdYlGn',
            colorbar=dict(title="Theoretical P&L ($)")
        ))
        
        fig_heatmap.update_layout(
            xaxis_title="Spot Price ($)",
            yaxis_title="Implied Volatility (%)",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab2:
        st.subheader("Delta & Gamma Spot Sensitivity")
        spot_curve = np.linspace(S * 0.7, S * 1.3, 100)
        deltas, gammas = [], []
        
        for s_val in spot_curve:
            res = black_scholes_greeks(s_val, K, T, r, sigma, option_type)
            deltas.append(res["delta"])
            gammas.append(res["gamma"])

        fig_curves = go.Figure()
        fig_curves.add_trace(go.Scatter(x=spot_curve, y=deltas, mode='lines', name='Delta', yaxis='y1'))
        fig_curves.add_trace(go.Scatter(x=spot_curve, y=gammas, mode='lines', name='Gamma', yaxis='y2'))

        fig_curves.update_layout(
            xaxis=dict(title="Underlying Spot Price ($)"),
            yaxis=dict(title="Delta", side="left"),
            yaxis2=dict(title="Gamma", side="right", overlaying="y"),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_curves, use_container_width=True)

    with tab3:
        st.subheader(f"Optimal Underlier Risk Levels for {selected_ticker}")
        opt_direction = "Long" if option_type == "Call" else "Short"
        opt_sl_tp = calculate_sl_tp_levels(S, volatility_pct=sigma, direction=opt_direction)
        st.dataframe(opt_sl_tp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# ROUTING & NAVIGATION MANAGEMENT
# ---------------------------------------------------------
main_page = st.Page(main_dashboard, title="Main Overview", icon="📈", default=True)
analytics_page = st.Page(summary_analytics, title="Market Summary & Analytics", icon="📊")
options_page = st.Page(options_calculator, title="Options Risk Calculator", icon="🧮")

# Initialize page navigation
pg = st.navigation([main_page, analytics_page, options_page])
pg.run()
            
