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
    page_title="Monte Carlo Risk Simulator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Monte Carlo Risk Simulator")
st.caption("A smart risk-estimation tool for traders and investors.")

# ---------------------------------------------------------
# Beginner-Friendly Educational Expander
# ---------------------------------------------------------
with st.expander("❓ New to Monte Carlo? Click here for a 1-minute breakdown"):
    st.markdown("""
    ### What is this app doing?
    Instead of giving you a single "magic prediction" for a stock, this tool simulates **thousands of possible future price paths** based on history.
    
    * **VaR (Value at Risk / 5th Percentile):** The "warning threshold." In 95% of simulated paths, your stock stays *above* this return level.
    * **CVaR (Conditional VaR / Tail Loss):** The "worst-case average." If a major market crash *does* happen, this is the average loss you should prepare for.
    """)

# ---------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------
st.sidebar.header("🛠️ Controls")

# Mode Toggle
experience_mode = st.sidebar.radio(
    "Select Mode",
    ["🐣 Simple Mode (Beginner)", "⚙️ Advanced Mode (Expert)"],
    help="Simple Mode picks the best risk settings automatically. Advanced Mode lets you tweak the raw math."
)

st.sidebar.markdown("---")
st.sidebar.subheader("1. Stock & Timeline")

ticker = st.sidebar.text_input(
    "Stock Ticker Symbol",
    value="AAPL",
    help="Enter any public stock, ETF, or crypto ticker (e.g., AAPL, TSLA, SPY, BTC-USD)."
).strip().upper()

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=30,
    max_value=365,
    value=252,
    help="252 trading days is roughly equal to 1 calendar year."
)

# ---------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_market_data(symbol):
    try:
        df = yf.download(symbol, period="2y", progress=False)
        if df.empty or 'Close' not in df.columns:
            return None, None
        close = df['Close'].dropna()
        # Handle multi-index columns if yfinance returns them
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        returns = np.log(close / close.shift(1)).dropna()
        return close, returns
    except Exception:
        return None, None

close_data, log_returns = load_market_data(ticker)

if close_data is None or log_returns.empty:
    st.error(f"❌ Could not load price data for **{ticker}**. Please check the ticker symbol and try again.")
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
    st.sidebar.subheader("2. Market Environment")
    
    asset_profile = st.sidebar.selectbox(
        "What type of asset are you analyzing?",
        ["Stable Stock / Index ETF (e.g., AAPL, SPY)", "High-Volatility / Crypto (e.g., TSLA, BTC)", "Earnings Event / High Uncertainty"],
        help="The app will automatically select the best mathematical engine based on your choice."
    )
    
    if "Stable" in asset_profile:
        model_type = "Standard (Normal Distribution)"
        model_desc = "Uses a standard bell curve. Best for low-volatility, predictable assets."
    elif "High-Volatility" in asset_profile:
        model_type = "Fat-Tail (Student's t)"
        model_desc = "Accounts for extreme market drops and sudden spikes that happen more often in volatile assets."
    else:
        model_type = "Jump-Diffusion (Merton)"
        model_desc = "Simulates sudden price gaps caused by surprise news, earnings reports, or market shocks."

    n_simulations = 1000
    jump_lambda = 4.0 / 252.0
    jump_mu = -0.05
    jump_sigma = 0.05

else:
    # Advanced Mode Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Model Selection")
    
    model_type = st.sidebar.radio(
        "Simulation Engine",
        ["Standard (Normal Distribution)", "Fat-Tail (Student's t)", "Jump-Diffusion (Merton)"],
        help="• Standard: Regular Brownian Motion.\n• Fat-Tail: Captures heavy extreme risk.\n• Jump-Diffusion: Adds sudden Poisson shocks."
    )
    
    n_simulations = st.sidebar.slider("Number of Simulations", 500, 5000, 1000, step=500)
    
    jump_lambda = 0.0
    jump_mu = 0.0
    jump_sigma = 0.0
    
    if model_type == "Jump-Diffusion (Merton)":
        st.sidebar.markdown("**Jump Shocks Settings**")
        jump_lambda = st.sidebar.slider("Expected Market Shocks / Year", 1, 12, 4) / 252.0
        jump_mu = st.sidebar.slider("Average Shock Drop (%)", -20.0, 5.0, -5.0) / 100.0
        jump_sigma = st.sidebar.slider("Shock Uncertainty (%)", 1.0, 15.0, 5.0) / 100.0

# ---------------------------------------------------------
# Main Page Header Metrics
# ---------------------------------------------------------
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Current Price", f"${S0:.2f}")
col_m2.metric("Annualized Volatility", f"{annualized_vol * 100:.1f}%", help="Higher volatility means wider price swings.")
col_m3.metric("Selected Engine", model_type.split(" ")[0])

if experience_mode == "🐣 Simple Mode (Beginner)":
    st.info(f"💡 **Simple Mode Setting:** {model_desc}")

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
    df_fit = max(df_fit, 3.0)  # Ensure variance remains finite
    
    t_shocks = t.rvs(df_fit, size=(forecast_days - 1, n_simulations))
    t_shocks = t_shocks / np.sqrt(df_fit / (df_fit - 2.0))
    
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
# Visualization & Risk Dashboard
# ---------------------------------------------------------
st.markdown("---")
col_chart, col_risk = st.columns([2, 1])

with col_chart:
    st.subheader("📊 Simulated Price Paths")
    fig = go.Figure()
    
    # Render first 80 paths for fast UI rendering
    display_paths = min(80, n_simulations)
    for i in range(display_paths):
        fig.add_trace(go.Scatter(
            y=sim_matrix[:, i],
            mode='lines',
            line=dict(width=0.8),
            showlegend=False,
            opacity=0.25
        ))
    
    # Highlight median outcome path
    median_path = np.median(sim_matrix, axis=1)
    fig.add_trace(go.Scatter(
        y=median_path,
        mode='lines',
        name='Expected Median Path',
        line=dict(color='bold yellow', width=2.5)
    ))

    fig.update_layout(
        xaxis_title="Days into Future",
        yaxis_title="Stock Price ($)",
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_risk:
    st.subheader("🛡️ Risk Analysis Summary")
    
    final_prices = sim_matrix[-1, :]
    returns_pct = (final_prices - S0) / S0
    
    var_95 = np.percentile(returns_pct, 5)
    cvar_95 = returns_pct[returns_pct <= var_95].mean()
    median_return = np.median(returns_pct)
    
    st.metric(
        label="🎯 Median Price Outcome",
        value=f"${S0 * (1 + median_return):.2f}",
        delta=f"{median_return * 100:+.1f}% overall"
    )
    
    st.metric(
        label="⚠️ 5th Percentile Warning (VaR 95%)",
        value=f"${S0 * (1 + var_95):.2f}",
        delta=f"{var_95 * 100:.1f}% bad case",
        help="95% of the simulated paths ended ABOVE this price point."
    )
    
    st.metric(
        label="🚨 Catastrophic Crash Average (CVaR / ES)",
        value=f"${S0 * (1 + cvar_95):.2f}",
        delta=f"{cvar_95 * 100:.1f}% worst-case avg",
        help="If the stock drops into its worst 5% scenario, this is the average expected price."
    )

    if experience_mode == "🐣 Simple Mode (Beginner)":
        st.caption(
            "💡 **Takeaway:** Focus on the **Catastrophic Crash Average**. "
            "Make sure your portfolio can handle this maximum loss level before investing!"
    )
