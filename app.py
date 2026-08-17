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

st.title("📈 Monte Carlo Risk & Trade Level Simulator")
st.caption("Quantitative risk estimation & optimal trade targets for stocks, forex, and crypto.")

# ---------------------------------------------------------
# Sidebar - Beginner Controls
# ---------------------------------------------------------
st.sidebar.header("🛠️ Trading Controls")

experience_mode = st.sidebar.radio(
    "Experience Level",
    ["🐣 Beginner (Presets)", "⚙️ Advanced (Custom Math)"],
    help="Beginner Mode uses smart risk presets. Advanced Mode unlocks raw statistical controls."
)

st.sidebar.markdown("---")
st.sidebar.subheader("1. Asset & Timeline")

ticker = st.sidebar.text_input(
    "Asset Ticker Symbol",
    value="EURUSD=X",
    help="Enter ticker symbols like EURUSD=X (Forex), AAPL or NVDA (Stocks), or BTC-USD (Crypto)."
).strip().upper()

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=14,
    max_value=365,
    value=60,
    help="How many days into the future you want to project."
)

# ---------------------------------------------------------
# Preset Logic & Model Selection
# ---------------------------------------------------------
if experience_mode == "🐣 Beginner (Presets)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Risk Strategy Preset")
    
    risk_preset = st.sidebar.select_slider(
        "Select Profile",
        options=["🛡️ Conservative", "⚖️ Balanced", "🚀 Aggressive"],
        value="⚖️ Balanced",
        help="Adjusts target percentiles and volatility buffers automatically based on your risk tolerance."
    )
    
    if risk_preset == "🛡️ Conservative":
        tp1_pct, tp2_pct, sl_atr_mult = 60, 75, 1.5
    elif risk_preset == "⚖️ Balanced":
        tp1_pct, tp2_pct, sl_atr_mult = 68, 80, 1.0
    else:  # Aggressive
        tp1_pct, tp2_pct, sl_atr_mult = 75, 90, 0.75

    asset_profile = st.sidebar.selectbox(
        "Market Category",
        ["Forex / Low Volatility", "Major Stocks / ETFs", "Crypto / High Volatility"]
    )
    
    model_type = "Fat-Tail (Student's t)" if "Crypto" in asset_profile else "Standard (Normal Distribution)"
    n_simulations = 1000
    jump_lambda, jump_mu, jump_sigma = 0.0, 0.0, 0.0

else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Model Selection")
    
    model_type = st.sidebar.radio(
        "Simulation Engine",
        ["Standard (Normal Distribution)", "Fat-Tail (Student's t)", "Jump-Diffusion (Merton)"]
    )
    n_simulations = st.sidebar.slider("Simulations", 500, 5000, 1000, step=500)
    
    tp1_pct = st.sidebar.slider("TP1 Target Percentile", 50, 80, 68)
    tp2_pct = st.sidebar.slider("TP2 Target Percentile", 70, 95, 80)
    sl_atr_mult = st.sidebar.slider("SL ATR Buffer Multiplier", 0.5, 3.0, 1.0, step=0.25)

    jump_lambda, jump_mu, jump_sigma = 0.0, 0.0, 0.0
    if model_type == "Jump-Diffusion (Merton)":
        st.sidebar.markdown("**Jump Shocks Settings**")
        jump_lambda = st.sidebar.slider("Expected Market Shocks / Year", 1, 12, 4) / 252.0
        jump_mu = st.sidebar.slider("Average Shock Drop (%)", -20.0, 5.0, -5.0) / 100.0
        jump_sigma = st.sidebar.slider("Shock Uncertainty (%)", 1.0, 15.0, 5.0) / 100.0

# ---------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_market_data(symbol):
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period="1y")
        
        if df.empty or 'Close' not in df.columns:
            return None, None, None
            
        close = df['Close'].dropna().astype(float)
        high = df['High'].dropna().astype(float)
        low = df['Low'].dropna().astype(float)
        
        if len(close) < 30:
            return None, None, None
            
        close_prev = close.shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = float(tr.rolling(14).mean().iloc[-1])
        returns = np.log(close / close_prev).dropna()
        
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

decimals = 4 if S0 < 10 else 2
fmt = f":.{decimals}f"

# ---------------------------------------------------------
# Simulation Engine
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
# Trade Targets
# ---------------------------------------------------------
final_prices = sim_matrix[-1, :]
p5_var = np.percentile(final_prices, 5)
p_tp1 = np.percentile(final_prices, tp1_pct)
p_tp2 = np.percentile(final_prices, tp2_pct)

optimal_entry = max(S0 - (0.5 * atr_14), S0 * 0.95)
stop_loss = max(optimal_entry - (sl_atr_mult * atr_14), p5_var)
stop_loss = max(stop_loss, S0 * 0.01)

tp_1 = max(p_tp1, optimal_entry + (1.5 * atr_14))
tp_2 = max(p_tp2, optimal_entry + (3.0 * atr_14))

risk_per_unit = optimal_entry - stop_loss
reward_per_unit = tp_1 - optimal_entry
rr_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

# ---------------------------------------------------------
# Beginner-Friendly Visual Setup Banner
# ---------------------------------------------------------
st.markdown("### 🚦 Trade Snapshot")

col1, col2, col3, col4 = st.columns(4)
col1.metric("1. Current Price", f"${S0:{fmt}}")
col2.metric("2. Limit Entry Target", f"${optimal_entry:{fmt}}", help="Suggested entry price to wait for a dip.")
col3.metric("3. Conservative TP1", f"${tp_1:{fmt}}", delta=f"+{((tp_1/optimal_entry)-1)*100:.1f}%")
col4.metric("4. Protection Stop Loss", f"${stop_loss:{fmt}}", delta=f"{((stop_loss/optimal_entry)-1)*100:.1f}%", delta_color="inverse")

st.markdown("---")

# ---------------------------------------------------------
# Interactive Calculator & Strategy Card
# ---------------------------------------------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🎯 Trade Plan Breakdown")
    st.markdown(f"""
    * **Asset:** `{ticker}`
    * **Daily Volatility Noise Buffer (14-Day ATR):** `${atr_14:{fmt}}`
    * **Optimal Buy Level:** Place a Limit Order at **`${optimal_entry:{fmt}}`** *(Avoid buying at current peak)*
    * **Take Profit Target (TP1):** **`${tp_1:{fmt}}`**
    * **Extended Profit Target (TP2):** **`${tp_2:{fmt}}`**
    * **Stop Loss Boundary:** **`${stop_loss:{fmt}}`**
    """)

with col_right:
    st.subheader("⚖️ Risk Setup Rating")
    st.metric("Risk-to-Reward (R:R)", f"1 : {rr_ratio:.2f}")
    if rr_ratio >= 1.5:
        st.success("🟢 **Great Setup** (R:R ≥ 1:1.5)")
    elif rr_ratio >= 1.0:
        st.warning("🟡 **Acceptable** (R:R ≥ 1:1.0)")
    else:
        st.error("🔴 **High Risk** — Risk outweighs potential reward.")

# ---------------------------------------------------------
# Beginner Friendly Position Sizing Calculator
# ---------------------------------------------------------
with st.expander("🧮 Position Size Calculator (How much should I buy?)"):
    c1, c2 = st.columns(2)
    account_size = c1.number_input("Account Balance ($)", value=10000, step=1000)
    risk_pct = c2.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, step=0.5)
    
    max_risk_dollars = account_size * (risk_pct / 100.0)
    risk_per_share = optimal_entry - stop_loss
    
    if risk_per_share > 0:
        shares_to_buy = max_risk_dollars / risk_per_share
        total_position_val = shares_to_buy * optimal_entry
        
        st.markdown(f"""
        * **Maximum Capital at Risk:** `${max_risk_dollars:,.2f}` ({risk_pct}% of account)
        * **Recommended Units/Shares to Buy:** `{shares_to_buy:,.2f}`
        * **Total Outlay / Position Value:** `${total_position_val:,.2f}`
        """)

# ---------------------------------------------------------
# Plotly Projection Chart
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Projected Price Outcomes")

fig = go.Figure()

display_paths = min(80, n_simulations)
for i in range(display_paths):
    fig.add_trace(go.Scatter(
        y=sim_matrix[:, i],
        mode='lines',
        line=dict(width=0.7),
        showlegend=False,
        opacity=0.15
    ))

fig.add_trace(go.Scatter(
    y=np.median(sim_matrix, axis=1),
    mode='lines',
    name='Median Path',
    line=dict(color="rgb(234, 179, 8)", width=2.5)
))

fig.add_hline(y=optimal_entry, line_dash="dash", line_color="rgb(59, 130, 246)", annotation_text="Limit Entry", annotation_position="bottom right")
fig.add_hline(y=tp_1, line_dash="dash", line_color="rgb(16, 185, 129)", annotation_text="TP 1", annotation_position="top right")
fig.add_hline(y=tp_2, line_dash="dot", line_color="rgb(5, 150, 105)", annotation_text="TP 2", annotation_position="top right")
fig.add_hline(y=stop_loss, line_dash="dash", line_color="rgb(239, 68, 68)", annotation_text="Stop Loss", annotation_position="bottom right")

fig.update_layout(
    xaxis_title="Days Ahead",
    yaxis_title="Price ($)",
    template="plotly_white",
    height=500,
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)
        
