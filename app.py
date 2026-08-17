import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration & Navigation
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gipsy Trading AI",
    page_icon="📈",
    layout="wide"
)

page = st.sidebar.radio(
    "📌 Navigation Menu",
    ["📖 Beginner's Guide & Single Simulator", "📊 Multi-Asset Summary Dashboard"]
)

# ---------------------------------------------------------
# Shared Helper Functions
# ---------------------------------------------------------
def sanitize_ticker(symbol: str) -> str:
    """Sanitizes user input tickers for yfinance compatibility."""
    symbol = symbol.strip().upper()
    crypto_bases = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "AVAX", "DOT", "LINK"]
    for coin in crypto_bases:
        if symbol.startswith(coin) and ("USD" in symbol or "=X" in symbol or "/" in symbol):
            return f"{coin}-USD"
    return symbol

@st.cache_data(ttl=3600)
def load_market_data(symbol: str):
    """Fetches historical price data and calculates ATR and returns vector."""
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

def run_monte_carlo(S0: float, mu: float, sigma: float, days: int, n_sims: int = 1000, model_type: str = "Standard (Normal Distribution)", log_returns=None, jump_params=None):
    """Executes Monte Carlo matrix simulation across specified distribution types."""
    dt = 1.0
    sim_matrix = np.zeros((days, n_sims))
    sim_matrix[0] = S0

    if model_type == "Standard (Normal Distribution)":
        shocks = np.random.normal(0, 1, size=(days - 1, n_sims))
        for step in range(1, days):
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * shocks[step - 1]
            sim_matrix[step] = sim_matrix[step - 1] * np.exp(drift + diffusion)

    elif model_type == "Fat-Tail (Student's t)" and log_returns is not None:
        df_fit, _, _ = t.fit(log_returns)
        df_fit = max(df_fit, 3.0)
        t_shocks = t.rvs(df_fit, size=(days - 1, n_sims)) / np.sqrt(df_fit / (df_fit - 2.0))
        for step in range(1, days):
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * t_shocks[step - 1]
            sim_matrix[step] = sim_matrix[step - 1] * np.exp(drift + diffusion)

    elif model_type == "Jump-Diffusion (Merton)" and jump_params is not None:
        j_lambda, j_mu, j_sigma = jump_params
        shocks = np.random.normal(0, 1, size=(days - 1, n_sims))
        for step in range(1, days):
            n_jumps = np.random.poisson(j_lambda, size=n_sims)
            jump_factor = np.random.normal(j_mu, j_sigma, size=n_sims) * n_jumps
            drift = (mu - 0.5 * sigma**2 - j_lambda * (np.exp(j_mu + 0.5 * j_sigma**2) - 1)) * dt
            diffusion = sigma * np.sqrt(dt) * shocks[step - 1]
            sim_matrix[step] = sim_matrix[step - 1] * np.exp(drift + diffusion + jump_factor)

    return sim_matrix


# =========================================================
# PAGE 1: BEGINNER GUIDE & SINGLE SIMULATOR
# =========================================================
if page == "📖 Beginner's Guide & Single Simulator":
    st.title("📈 Gipsy Trading AI")
    st.caption("Quantitative risk estimation & optimal trade targets for stocks, forex, and crypto.")
    
    # --- Beginner-Friendly Guide Accordion ---
    with st.expander("📖 New to Trading & Risk Modeling? Click here for the Beginner's Guide", expanded=False):
        st.markdown("""
        ### 👋 Welcome to Gipsy Trading AI!
        This tool uses mathematical statistical models to project thousands of potential future price paths for any asset. Here is how to read and use the dashboard:

        #### 1. How Gipsy Trading AI Models Prices
        Instead of predicting a single "exact price," Gipsy Trading AI generates thousands of random daily price movements based on an asset's **historical volatility** and **average daily drift**.
        * **Median Path:** Represents the central expected outcome (50th percentile).
        * **Outer Lines:** Represent extreme upside and downside conditions.

        #### 2. Key Terminology Explained
        * **ATR (Average True Range):** Measures daily market volatility noise. A high ATR means large daily swings.
        * **Limit Entry:** The recommended price to buy. It factors in ATR noise to help you buy on a small pull-back instead of buying at peak prices.
        * **Take Profit (TP1 / TP2):** Target prices where you lock in gains based on statistical probability horizons.
        * **Stop Loss (SL):** Your protective exit price. If market price hits this level, close your position to prevent severe losses.
        * **Risk-to-Reward Ratio (R:R):** Compares how much money you risk against how much you stand to gain. Aim for **1 : 1.5 or better**.

        #### 3. Quick-Start Steps
        1. Choose an **Asset Ticker** in the sidebar (e.g., `AAPL`, `BTC-USD`, `EURUSD=X`).
        2. Pick a **Risk Strategy Preset** that fits your personal comfort level.
        3. Review the **Trade Snapshot** cards for your entry, target, and exit levels.
        4. Use the **Position Size Calculator** to determine how many units/shares to purchase safely.
        """)

    st.markdown("---")

    # Sidebar Parameters
    st.sidebar.header("🛠️ Trading Controls")
    
    experience_mode = st.sidebar.radio(
        "Experience Level",
        ["🐣 Beginner (Presets)", "⚙️ Advanced (Custom Math)"],
        help="Beginner Mode uses smart risk presets. Advanced Mode unlocks raw statistical controls."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Asset & Timeline")

    user_ticker_input = st.sidebar.text_input(
        "Asset Ticker Symbol",
        value="EURUSD=X",
        help="Enter symbols like EURUSD=X (Forex), AAPL (Stocks), or BTC-USD / BTCUSD=X (Crypto)."
    )

    ticker = sanitize_ticker(user_ticker_input)
    if ticker != user_ticker_input.strip().upper():
        st.sidebar.info(f"💡 Auto-corrected ticker to **{ticker}** for yfinance compatibility.")

    forecast_days = st.sidebar.slider(
        "Forecast Horizon (Days)",
        min_value=14,
        max_value=365,
        value=60
    )

    # Controls Logic
    if experience_mode == "🐣 Beginner (Presets)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("2. Risk Strategy Preset")
        
        risk_preset = st.sidebar.select_slider(
            "Select Profile",
            options=["🛡️ Conservative", "⚖️ Balanced", "🚀 Aggressive"],
            value="⚖️ Balanced"
        )
        
        if risk_preset == "🛡️ Conservative":
            tp1_pct, tp2_pct, sl_atr_mult = 60, 75, 1.5
        elif risk_preset == "⚖️ Balanced":
            tp1_pct, tp2_pct, sl_atr_mult = 68, 80, 1.0
        else:
            tp1_pct, tp2_pct, sl_atr_mult = 75, 90, 0.75

        asset_profile = st.sidebar.selectbox(
            "Market Category",
            ["Forex / Low Volatility", "Major Stocks / ETFs", "Crypto / High Volatility"]
        )
        
        model_type = "Fat-Tail (Student's t)" if "Crypto" in asset_profile else "Standard (Normal Distribution)"
        n_simulations = 1000
        jump_params = None

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

        jump_params = None
        if model_type == "Jump-Diffusion (Merton)":
            st.sidebar.markdown("**Jump Shocks Settings**")
            j_lambda = st.sidebar.slider("Expected Market Shocks / Year", 1, 12, 4) / 252.0
            j_mu = st.sidebar.slider("Average Shock Drop (%)", -20.0, 5.0, -5.0) / 100.0
            j_sigma = st.sidebar.slider("Shock Uncertainty (%)", 1.0, 15.0, 5.0) / 100.0
            jump_params = (j_lambda, j_mu, j_sigma)

    # Data Fetching & Calculations
    close_data, log_returns, atr_14 = load_market_data(ticker)

    if close_data is None or log_returns.empty:
        st.error(f"❌ Could not load price data for **{ticker}**. Please check the symbol (e.g., EURUSD=X, AAPL, BTC-USD).")
        st.stop()

    S0 = float(close_data.iloc[-1])
    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    decimals = 4 if S0 < 10 else 2

    # Run Engine
    sim_matrix = run_monte_carlo(
        S0=S0, mu=mu, sigma=sigma, days=forecast_days, 
        n_sims=n_simulations, model_type=model_type, 
        log_returns=log_returns, jump_params=jump_params
    )

    # Targets & Level Setup
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

    # Display Metrics Banner
    st.markdown("### 🚦 Trade Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1. Current Price", f"${S0:.{decimals}f}")
    col2.metric("2. Limit Entry Target", f"${optimal_entry:.{decimals}f}")
    col3.metric("3. Conservative TP1", f"${tp_1:.{decimals}f}", delta=f"+{((tp_1/optimal_entry)-1)*100:.1f}%")
    col4.metric("4. Protection Stop Loss", f"${stop_loss:.{decimals}f}", delta=f"{((stop_loss/optimal_entry)-1)*100:.1f}%", delta_color="inverse")

    st.markdown("---")

    # Trade Breakdown & Rating
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🎯 Trade Plan Breakdown")
        st.markdown(f"""
        * **Asset:** `{ticker}`
        * **Daily Volatility Noise Buffer (14-Day ATR):** `${atr_14:.{decimals}f}`
        * **Optimal Buy Level:** Place a Limit Order at **`${optimal_entry:.{decimals}f}`** *(Avoid buying at current peak)*
        * **Take Profit Target (TP1):** **`${tp_1:.{decimals}f}`**
        * **Extended Profit Target (TP2):** **`${tp_2:.{decimals}f}`**
        * **Stop Loss Boundary:** **`${stop_loss:.{decimals}f}`**
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

    # Position Size Calculator
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

    # Chart
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


# =========================================================
# PAGE 2: MULTI-ASSET & TIMELINE SUMMARY DASHBOARD
# =========================================================
elif page == "📊 Multi-Asset Summary Dashboard":
    st.title("📊 Multi-Asset & Timeline Summary Dashboard")
    st.caption("Compare expected returns, risk levels, and stop-loss boundaries across multiple assets and timeframes side-by-side.")

    st.sidebar.header("⚙️ Dashboard Settings")
    
    # Preset Basket Selection
    default_tickers_text = st.sidebar.text_area(
        "Assets to Compare (Comma-Separated)",
        value="AAPL, MSFT, BTC-USD, ETH-USD, EURUSD=X",
        help="Enter valid Yahoo Finance tickers separated by commas."
    )
    
    selected_timelines = st.sidebar.multiselect(
        "Select Forecast Timelines (Days)",
        options=[14, 30, 60, 90, 180, 365],
        default=[30, 60, 90]
    )

    if not selected_timelines:
        st.warning("⚠️ Please select at least one forecast timeline in the sidebar.")
        st.stop()

    ticker_list = [sanitize_ticker(t) for t in default_tickers_text.split(",") if t.strip()]

    if st.button("🚀 Run Comparative Analysis") or "summary_df" not in st.session_state:
        summary_rows = []
        progress_bar = st.progress(0)
        total_tasks = len(ticker_list) * len(selected_timelines)
        task_count = 0

        for t_symbol in ticker_list:
            close_data, log_returns, atr_14 = load_market_data(t_symbol)
            
            if close_data is None or log_returns.empty:
                continue
                
            S0 = float(close_data.iloc[-1])
            mu = float(log_returns.mean())
            sigma = float(log_returns.std())
            ann_vol = sigma * np.sqrt(252) * 100

            for days in selected_timelines:
                sim_matrix = run_monte_carlo(S0, mu, sigma, days, n_sims=1000)
                final_prices = sim_matrix[-1, :]

                median_final = float(np.median(final_prices))
                exp_return_pct = ((median_final / S0) - 1) * 100
                p5_val = float(np.percentile(final_prices, 5))
                p95_val = float(np.percentile(final_prices, 95))
                var_5_pct = ((p5_val / S0) - 1) * 100

                optimal_entry = max(S0 - (0.5 * atr_14), S0 * 0.95)
                recommended_sl = max(optimal_entry - atr_14, p5_val)
                recommended_tp = max(np.percentile(final_prices, 68), optimal_entry + (1.5 * atr_14))

                summary_rows.append({
                    "Ticker": t_symbol,
                    "Current Price ($)": round(S0, 4 if S0 < 10 else 2),
                    "Timeline (Days)": days,
                    "Ann. Volatility (%)": round(ann_vol, 1),
                    "Median Outcome ($)": round(median_final, 4 if S0 < 10 else 2),
                    "Expected Return (%)": round(exp_return_pct, 2),
                    "5% VaR / Downside Risk (%)": round(var_5_pct, 2),
                    "Rec. Limit Entry ($)": round(optimal_entry, 4 if S0 < 10 else 2),
                    "Rec. Stop Loss ($)": round(recommended_sl, 4 if S0 < 10 else 2),
                    "Rec. Take Profit ($)": round(recommended_tp, 4 if S0 < 10 else 2)
                })

                task_count += 1
                progress_bar.progress(task_count / total_tasks)

        progress_bar.empty()
        st.session_state["summary_df"] = pd.DataFrame(summary_rows)

    summary_df = st.session_state.get("summary_df", pd.DataFrame())

    if summary_df.empty:
        st.error("No data could be retrieved for the specified tickers. Please check your inputs.")
        st.stop()

    # --- Section 1: Summary Table ---
    st.subheader("📋 Comprehensive Asset & Horizon Matrix")
    
    col_f1, col_f2 = st.columns(2)
    filter_ticker = col_f1.multiselect("Filter by Ticker", options=summary_df["Ticker"].unique(), default=summary_df["Ticker"].unique())
    filter_timeline = col_f2.multiselect("Filter by Timeline", options=summary_df["Timeline (Days)"].unique(), default=summary_df["Timeline (Days)"].unique())

    filtered_df = summary_df[
        (summary_df["Ticker"].isin(filter_ticker)) & 
        (summary_df["Timeline (Days)"].isin(filter_timeline))
    ]

    st.dataframe(
        filtered_df.style.background_gradient(subset=["Expected Return (%)"], cmap="RdYlGn")
                         .background_gradient(subset=["5% VaR / Downside Risk (%)"], cmap="Reds_r"),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --- Section 2: Visual Comparison Charts ---
    st.subheader("📊 Visual Comparisons Across Timelines")
    
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### Expected Median Return (%)")
        fig_ret = px.bar(
            filtered_df,
            x="Ticker",
            y="Expected Return (%)",
            color="Timeline (Days)",
            barmode="group",
            text_auto=".1f",
            template="plotly_white"
        )
        fig_ret.update_layout(height=400)
        st.plotly_chart(fig_ret, use_container_width=True)

    with chart_col2:
        st.markdown("##### Downside Value at Risk (5th Percentile Drop %)")
        fig_var = px.bar(
            filtered_df,
            x="Ticker",
            y="5% VaR / Downside Risk (%)",
            color="Timeline (Days)",
            barmode="group",
            text_auto=".1f",
            template="plotly_white"
        )
        fig_var.update_layout(height=400)
        st.plotly_chart(fig_var, use_container_width=True)

    st.sub
