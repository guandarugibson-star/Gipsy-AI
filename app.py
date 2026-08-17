import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Monte Carlo Risk & Trade Level Simulator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Monte Carlo Risk & Optimal Trade Level Simulator")
st.caption("Quantitative risk estimation and optimal Entry / Take Profit / Stop Loss levels for liquid assets.")

# ---------------------------------------------------------
# Educational Expander
# ---------------------------------------------------------
with st.expander("❓ How are Entry, TP, and SL calculated?"):
    st.markdown("""
    * **Optimal Entry:** Calculated using Average True Range (ATR) to identify optimal pullback/retest levels instead of chasing the current market price.
    * **Take Profit (TP1 & TP2):** Set at the **68th percentile** (conservative) and **80th percentile** (aggressive) upside targets from the Monte Carlo paths.
    * **Stop Loss (SL):** Positioned below the **ATR market noise buffer** and aligned with the **5th percentile VaR downside limit** to protect capital without tight stop-outs.
    * **Risk-to-Reward (R:R):** Automatically calculated based on TP1 vs. SL distance. Look for a ratio of **1:1.5 or better**.
    """)

# ---------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------
st.sidebar.header("🛠️ Controls")

experience_mode = st.sidebar.radio(
    "Select Mode",
    ["🐣 Simple Mode (Beginner)", "⚙️ Advanced Mode (Expert)"],
    help="Simple Mode picks the best risk settings automatically. Advanced Mode lets you tweak the raw math."
)

st.sidebar.markdown("---")
st.sidebar.subheader("1. Asset & Timeline")

ticker = st.sidebar.text_input(
    "Asset Ticker Symbol",
    value="EURUSD=X",
    help="Enter high-liquidity Forex pairs (EURUSD=X, GBPUSD=X), Stocks (AAPL, NVDA), or Crypto (BTC-USD)."
).strip().upper()

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=14,
    max_value=365,
    value=60,
    help="Trading timeframe horizon."
)

# ---------------------------------------------------------
# Data Fetching & Technical Indicators
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_market_data(symbol):
    try:
        df = yf.download(symbol, period="1y", progress=False)
        if df.empty or 'Close' not in df.columns:
            return None, None, None
            
        close = df['Close'].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
            
        high = df['High'].squeeze() if 'High' in df.columns else close
        low = df['Low'].squeeze() if 'Low' in df.columns else close
        
        # Calculate ATR (14-period Average True Range) for volatility noise
        tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        returns = np.log(close / close.shift(1)).dropna()
        return close, returns, atr
    except Exception:
        return None, None, None

close_data, log_returns, atr_14 = load_market_data(ticker)

if close_data is None or log_returns.empty:
    st.error(f"❌ Could not load price data for **{ticker}**. Please check the ticker symbol (e.g., EURUSD=X, AAPL, BTC-USD).")
    st.stop()

S0 = float(close_data.iloc[-1])
mu = float(log_returns.mean())
sigma = float(log_returns.std())
annualized_vol = sigma * np.sqrt(252)

# ---------------------------------------------------------
# Model Selection Logic
# ---------------------------------------------------------
if experience_mode == "🐣 Simple Mode (Beginner)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Market Profile")
    
    asset_profile = st.sidebar.selectbox(
        "Select Market Type",
        ["Forex / High-Liquidity Currency (e.g., EURUSD=X)", "Major Stock / Index ETF (e.g., AAPL, SPY)", "High-Volatility / Crypto (e.g., BTC-USD)"],
        help="Automatically adapts simulation parameters to match liquidity and tail behavior."
    )
    
    if "Forex" in asset_profile or "Major Stock" in asset_profile:
        model_type = "Standard (Normal Distribution)"
    elif "Crypto" in asset_profile:
        model_type = "Fat-Tail (Student's t)"
    else:
        model_type = "Jump-Diffusion (Merton)"

    n_simulations = 1000
    jump_lambda, jump_mu, jump_sigma = 4.0 / 252.0, -0.05, 0.05

else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Model Selection")
    
    model_type = st.sidebar.radio(
        "Simulation Engine",
        ["Standard (Normal Distribution)", "Fat-Tail (Student's t)", "Jump-Diffusion (Merton)"]
    )
    n_simulations = st.sidebar.slider("Number of Simulations", 500, 5000, 1000, step=500)
    
    jump_lambda, jump_mu, jump_sigma = 0.0, 0.0, 0.0
    if model_type == "Jump-Diffusion (Merton)":
        st.sidebar.markdown("**Jump Shocks Settings**")
        jump_lambda = st.sidebar.slider("Expected Market Shocks / Year", 1, 12, 4) / 252.0
        jump_mu = st.sidebar.slider("Average Shock Drop (%)", -20.0, 5.0, -5.0) / 100.0
        jump_sigma = st.sidebar.slider("Shock Uncertainty (%)", 1.0, 15.0, 5.0) / 100.0

# ---------------------------------------------------------
# Simulation Engine Execution
# ---------------------------------------------------------
dt = 1.0
sim_matrix = np.zeros((forecast_days, n_simulations))
sim_matrix[0] = S0

if model_type == "Standard (Normal Distribution)":
    shocks = np.random.normal(0, 1, size=(forecast_days - 1, n_simulations))
    for step in range(1, forecast_days):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * shocks[step - 1]
        sim_matrix[step] = sim_matrix[step - 1] * np.exp(drift + diffusion)

elif model_type == "Fat-Tail (Student's t)":
    df_fit, _, _ = t.fit(log_returns)
    df_fit = max(df_fit, 3.0)
    t_shocks = t.rvs(df_fit, size=(forecast_days - 1, n_simulations)) / np.sqrt(df_fit / (df_fit - 2.0))
    for step in range(1, forecast_days):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * t_shocks[step - 1]
        sim_matrix[step] = sim_matrix[step - 1] * np.exp(drift + diffusion)

elif model_type == "Jump-Diffusion (Merton)":
    shocks = np.random.normal(0, 1, size=(forecast_days - 1, n_simulations))
    for step in range(1, forecast_days):
        n_jumps = np.random.poisson(jump_lambda, size=n_simulations)
        jump_factor = np.random.normal(jump_mu, jump_sigma, size=n_simulations) * n_jumps
        drift = (mu - 0.5 * sigma**2 - jump_lambda * (np.exp(jump_mu + 0.5 * jump_sigma**2) - 1)) * dt
        diffusion = sigma * np.sqrt(dt) * shocks[step - 1]
        sim_matrix[step] = sim_matrix[step - 1] * np.exp(drift + diffusion + jump_factor)

# ---------------------------------------------------------
# Quantitative Entry / TP / SL Calculations
# ---------------------------------------------------------
final_prices = sim_matrix[-1, :]
p5_var = np.percentile(final_prices, 5)
p50_median = np.percentile(final_prices, 50)
p68_tp1 = np.percentile(final_prices, 68)
p80_tp2 = np.percentile(final_prices, 80)

# Optimal Retest / Entry Level (Current Price - 0.5 * ATR or near current market)
optimal_entry = max(S0 - (0.5 * atr_14), S0 * 0.995)
stop_loss = min(p5_var, optimal_entry - (1.5 * atr_14))
tp_1 = max(p68_tp1, optimal_entry + (1.5 * atr_14))
tp_2 = max(p80_tp2, optimal_entry + (3.0 * atr_14))

# Risk-to-Reward Ratio (R:R)
risk_per_unit = optimal_entry - stop_loss
reward_per_unit = tp_1 - optimal_entry
rr_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

# ---------------------------------------------------------
# Main Page Dashboard Metrics
# ---------------------------------------------------------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Current Market Price", f"${S0:.4f}" if S0 < 10 else f"${S0:.2f}")
col_m2.metric("Optimal Limit Entry", f"${optimal_entry:.4f}" if S0 < 10 else f"${optimal_entry:.2f}", help="Pullback level factoring ATR volatility.")
col_m3.metric("Take Profit 1 (Conservative)", f"${tp_1:.4f}" if S0 < 10 else f"${tp_1:.2f}", delta=f"+{((tp_1/optimal_entry)-1)*100:.1f}%")
col_m4.metric("Stop Loss (SL)", f"${stop_loss:.4f}" if S0 < 10 else f"${stop_loss:.2f}", delta=f"{((stop_loss/optimal_entry)-1)*100:.1f}%", delta_color="inverse")

st.markdown("---")

# ---------------------------------------------------------
# Trade Strategy Card
# ---------------------------------------------------------
col_strat, col_rr = st.columns([3, 1])

with col_strat:
    st.subheader("🎯 Trade Execution Card")
    st.markdown(f"""
    * **Asset:** `{ticker}` | **14-Day ATR Volatility Buffer:** `${atr_14:.4f}`
    * **Optimal Entry Zone:** Buy Limit at **`${optimal_entry:.4f}`** *(Wait for key intraday dip/retest)*
    * **Take Profit 1 (TP1):** **`${tp_1:.4f}`** *(68th percentile distribution target)*
    * **Take Profit 2 (TP2 - Extended):** **`${tp_2:.4f}`** *(80th percentile distribution target)*
    * **Stop Loss (SL):** **`${stop_loss:.4f}`** *(Placed below 1.5x ATR noise buffer & 95% VaR floor)*
    """)

with col_rr:
    st.subheader("⚖️ Risk : Reward")
    st.metric("R:R Ratio (TP1 / SL)", f"1 : {rr_ratio:.2f}")
    if rr_ratio >= 1.5:
        st.success("✅ Excellent Trade Setup (R:R > 1:1.5)")
    elif rr_ratio >= 1.0:
        st.warning("⚠️ Moderate Setup (R:R 1:1.0)")
    else:
        st.error("🚫 Low R:R Setup - Consider Skipping")

# ---------------------------------------------------------
# Simulation Chart with Entry/TP/SL Lines
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Price Path Projections & Key Trade Levels")

fig = go.Figure()

# Plot path lines
display_paths = min(80, n_simulations)
for i in range(display_paths):
    fig.add_trace(go.Scatter(
        y=sim_matrix[:, i],
        mode='lines',
        line=dict(width=0.7),
        showlegend=False,
        opacity=0.20
    ))

# Highlight Median Path
fig.add_trace(go.Scatter(
    y=np.median(sim_matrix, axis=1),
    mode='lines',
    name='Median Path',
    line=dict(color='gold', width=2.5)
))

# Add Horizontal Trade Level Markers
fig.add_hline(y=optimal_entry, line_dash="dash", line_color="blue", annotation_text="Optimal Entry", annotation_position="bottom right")
fig.add_hline(y=tp_1, line_dash="dash", line_color="green", annotation_text="TP 1 (Conservative)", annotation_position="top right")
fig.add_hline(y=tp_2, line_dash="dot", line_color="emerald", annotation_text="TP 2 (Aggressive)", annotation_position="top right")
fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="Stop Loss (SL)", annotation_position="bottom right")

fig.update_layout(
    xaxis_title="Trading Days Ahead",
    yaxis_title="Asset Price ($)",
    template="plotly_white",
    height=500,
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)
