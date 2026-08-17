import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------
# Page Configuration & Navigation
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gipsy Trading AI - Live Quantitative Risk",
    page_icon="📈",
    layout="wide"
)

page = st.sidebar.radio(
    "📌 Navigation Menu",
    ["📖 Beginner's Guide & Single Simulator", "📊 Multi-Asset Summary Dashboard"]
)

# ---------------------------------------------------------
# Helper Functions & Optimized Live Engine
# ---------------------------------------------------------
def sanitize_ticker(symbol: str) -> str:
    """Sanitizes user input tickers for yfinance compatibility."""
    symbol = symbol.strip().upper()
    crypto_bases = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "AVAX", "DOT", "LINK"]
    for coin in crypto_bases:
        if symbol.startswith(coin) and ("USD" in symbol or "=X" in symbol or "/" in symbol):
            return f"{coin}-USD"
    return symbol

@st.cache_data(ttl=600, show_spinner=False)
def get_historical_baseline(symbol: str):
    """
    Heavy Network Call: Cached for 10 minutes (600s).
    Fetches 1-year daily data for ATR, log returns, and distribution stats.
    """
    try:
        tk = yf.Ticker(symbol)
        df_hist = tk.history(period="1y")
        if df_hist.empty:
            return None, None, None
            
        if isinstance(df_hist.columns, pd.MultiIndex):
            df_hist.columns = df_hist.columns.get_level_values(0)

        df_hist = df_hist.rename(columns={col: col.capitalize() for col in df_hist.columns})

        if 'Close' not in df_hist.columns or 'High' not in df_hist.columns or 'Low' not in df_hist.columns:
            return None, None, None

        close = df_hist['Close'].dropna().astype(float)
        high = df_hist['High'].dropna().astype(float)
        low = df_hist['Low'].dropna().astype(float)
        
        if len(close) < 30:
            return None, None, None

        # Wilder's Exponential Smoothing for ATR
        close_prev = close.shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
        
        returns = np.log(close / close_prev).dropna()

        return close, returns, atr
    except Exception:
        return None, None, None


def get_live_quote_fast(symbol: str, fallback_price: float) -> float:
    """
    Ultra-Fast Network Call: Uncached.
    Only fetches the single latest spot quote tick.
    """
    try:
        tk = yf.Ticker(symbol)
        price = float(tk.fast_info.last_price)
        if not np.isnan(price) and price > 0:
            return price
    except Exception:
        pass
        
    try:
        df_intraday = tk.history(period="1d", interval="1m")
        if not df_intraday.empty and 'Close' in df_intraday.columns:
            return float(df_intraday['Close'].iloc[-1])
    except Exception:
        pass

    return fallback_price


def run_monte_carlo_fast(S0: float, mu: float, sigma: float, days: int, n_sims: int = 1000, model_type: str = "Standard (Normal Distribution)", log_returns=None, jump_params=None):
    """Vectorized Monte Carlo engine using cumulative matrix operations for fast simulation."""
    dt = 1.0
    
    if model_type == "Standard (Normal Distribution)":
        shocks = np.random.normal(0, 1, size=(days - 1, n_sims))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * shocks
        daily_returns = drift + diffusion

    elif model_type == "Fat-Tail (Student's t)" and log_returns is not None:
        df_fit, _, _ = t.fit(log_returns)
        df_fit = max(df_fit, 3.0)
        t_shocks = t.rvs(df_fit, size=(days - 1, n_sims)) / np.sqrt(df_fit / (df_fit - 2.0))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * t_shocks
        daily_returns = drift + diffusion

    elif model_type == "Jump-Diffusion (Merton)" and jump_params is not None:
        j_lambda, j_mu, j_sigma = jump_params
        shocks = np.random.normal(0, 1, size=(days - 1, n_sims))
        n_jumps = np.random.poisson(j_lambda, size=(days - 1, n_sims))
        jump_factor = np.random.normal(j_mu, j_sigma, size=(days - 1, n_sims)) * n_jumps
        drift = (mu - 0.5 * sigma**2 - j_lambda * (np.exp(j_mu + 0.5 * j_sigma**2) - 1)) * dt
        diffusion = sigma * np.sqrt(dt) * shocks
        daily_returns = drift + diffusion + jump_factor
    else:
        daily_returns = np.zeros((days - 1, n_sims))

    zero_day = np.zeros((1, n_sims))
    cum_returns = np.vstack([zero_day, np.cumsum(daily_returns, axis=0)])
    
    return S0 * np.exp(cum_returns)


# =========================================================
# PAGE 1: BEGINNER GUIDE & SINGLE SIMULATOR
# =========================================================
if page == "📖 Beginner's Guide & Single Simulator":
    st.title("📈 Gipsy Trading AI")
    st.caption("Live quantitative risk estimation & optimal trade targets for stocks, forex, and crypto.")
    
    # --- Ultra-Fast Auto-Refresh Controls ---
    st.sidebar.header("⚡ Live Data Feed Settings")
    auto_refresh_active = st.sidebar.toggle("Enable Live Auto-Refresh", value=True)
    refresh_rate_sec = st.sidebar.slider("Refresh Interval (Seconds)", 2, 30, 3, step=1)

    if st.sidebar.button("🔄 Force Refresh All Data"):
        st.cache_data.clear()

    # --- Beginner Accordion ---
    with st.expander("📖 New to Trading & Risk Modeling? Click here for the Beginner's Guide", expanded=False):
        st.markdown("""
        ### 👋 Welcome to Gipsy Trading AI!
        This tool uses mathematical statistical models to project thousands of potential future price paths for any asset in real time.

        #### 1. Key Terminology Explained
        * **Live Spot Price:** The last traded market quote fetched directly from exchange feeds.
        * **ATR (Average True Range):** Measures daily market volatility noise. A high ATR means large daily swings.
        * **Limit Entry:** Recommended buy level. It factors in volatility noise so you buy on a pull-back instead of chasing peaks.
        * **Take Profit (TP1 / TP2):** Target prices to lock in profits based on statistical probability horizons.
        * **Stop Loss (SL):** Protective exit level to prevent severe losses if the market moves against you.
        * **Risk-to-Reward Ratio (R:R):** Compares potential loss against potential gain. Target **1 : 1.5 or higher**.
        """)

    st.markdown("---")

    # Controls
    st.sidebar.header("🛠️ Simulation Setup")
    experience_mode = st.sidebar.radio(
        "Experience Level",
        ["🐣 Beginner (Presets)", "⚙️ Advanced (Custom Math)"]
    )

    st.sidebar.markdown("---")
    user_ticker_input = st.sidebar.text_input("Asset Ticker Symbol", value="BTC-USD")
    ticker = sanitize_ticker(user_ticker_input)

    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 14, 365, 60)

    if experience_mode == "🐣 Beginner (Presets)":
        risk_preset = st.sidebar.select_slider(
            "Risk Profile",
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
            ["Crypto / High Volatility", "Major Stocks / ETFs", "Forex / Low Volatility"]
        )
        model_type = "Fat-Tail (Student's t)" if "Crypto" in asset_profile else "Standard (Normal Distribution)"
        n_simulations = 1000
        jump_params = None
    else:
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
            j_lambda = st.sidebar.slider("Expected Market Shocks / Year", 1, 12, 4) / 252.0
            j_mu = st.sidebar.slider("Average Shock Drop (%)", -20.0, 5.0, -5.0) / 100.0
            j_sigma = st.sidebar.slider("Shock Uncertainty (%)", 1.0, 15.0, 5.0) / 100.0
            jump_params = (j_lambda, j_mu, j_sigma)

    # Fetch Baseline Data outside fragment (cached for 10 min)
    close_data, log_returns, atr_14 = get_historical_baseline(ticker)

    if close_data is None:
        st.error(f"❌ Could not load price data for **{ticker}**. Verify symbol formatting (e.g. BTC-USD, AAPL, EURUSD=X).")
        st.stop()

    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    fallback_last_price = float(close_data.iloc[-1])

    # ---------------------------------------------------------
    # ULTRA-FAST SMOOTH FRAGMENT
    # ---------------------------------------------------------
    @st.fragment(run_every=refresh_rate_sec if auto_refresh_active else None)
    def render_live_simulator_dashboard():
        # Fetch ONLY the quick live tick
        S0 = get_live_quote_fast(ticker, fallback_price=fallback_last_price)
        decimals = 4 if S0 < 10 else 2

        # Fast simulation run
        sim_matrix = run_monte_carlo_fast(
            S0=S0, mu=mu, sigma=sigma, days=forecast_days, 
            n_sims=n_simulations, model_type=model_type, 
            log_returns=log_returns, jump_params=jump_params
        )

        final_prices = sim_matrix[-1, :]
        p5_var = np.percentile(final_prices, 5)
        p25 = np.percentile(sim_matrix, 25, axis=1)
        p75 = np.percentile(sim_matrix, 75, axis=1)
        p95 = np.percentile(sim_matrix, 95, axis=1)
        p5 = np.percentile(sim_matrix, 5, axis=1)

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
        st.caption(f"🟢 **Live Data Stream Active:** `${S0:,.{decimals}f}` | Last Check: `{pd.Timestamp.now().strftime('%H:%M:%S UTC')}`")
        st.markdown("### 🚦 Trade Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Live Spot Price", f"${S0:.{decimals}f}")
        col2.metric("2. Recommended Limit Entry", f"${optimal_entry:.{decimals}f}")
        col3.metric("3. Target Take Profit (TP1)", f"${tp_1:.{decimals}f}", delta=f"+{((tp_1/optimal_entry)-1)*100:.1f}%")
        col4.metric("4. Protection Stop Loss", f"${stop_loss:.{decimals}f}", delta=f"{((stop_loss/optimal_entry)-1)*100:.1f}%", delta_color="inverse")

        st.markdown("---")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("🎯 Trade Setup Breakdown")
            st.markdown(f"""
            * **Asset:** `{ticker}`
            * **Daily Volatility Noise (14-Day ATR):** `${atr_14:.{decimals}f}`
            * **Optimal Limit Order Entry:** **`${optimal_entry:.{decimals}f}`** *(Avoids buying at peaks)*
            * **Conservative Target (TP1):** **`${tp_1:.{decimals}f}`**
            * **Extended Target (TP2):** **`${tp_2:.{decimals}f}`**
            * **Stop Loss Boundary:** **`${stop_loss:.{decimals}f}`**
            """)

        with col_right:
            st.subheader("⚖️ Risk Setup Rating")
            st.metric("Risk-to-Reward (R:R)", f"1 : {rr_ratio:.2f}")
            if rr_ratio >= 1.5:
                st.success("🟢 **Great Setup** (R:R ≥ 1:1.5)")
            elif rr_ratio >= 1.0:
                st.warning("🟡 **Acceptable Setup** (R:R ≥ 1:1.0)")
            else:
                st.error("🔴 **High Risk** — Risk outweighs potential reward.")

        # Position Size Calculator
        with st.expander("🧮 Position Size Calculator"):
            c1_pos, c2_pos = st.columns(2)
            account_size = c1_pos.number_input("Account Balance ($)", value=10000, step=1000)
            risk_pct = c2_pos.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, step=0.5)
            
            max_risk_dollars = account_size * (risk_pct / 100.0)
            risk_per_share = optimal_entry - stop_loss
            
            if risk_per_share > 0:
                shares_to_buy = max_risk_dollars / risk_per_share
                total_position_val = shares_to_buy * optimal_entry
                st.markdown(f"""
                * **Maximum Capital at Risk:** `${max_risk_dollars:,.2f}` ({risk_pct}% of balance)
                * **Recommended Quantity to Buy:** `{shares_to_buy:,.4f}` units
                * **Total Position Outlay:** `${total_position_val:,.2f}`
                """)

        # Interactive Chart
        st.markdown("---")
        st.subheader("📊 Projected Price Cone & Probability Horizon")

        days_axis = np.arange(forecast_days)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.concatenate([days_axis, days_axis[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='90% Probability Band'
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([days_axis, days_axis[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.25)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='50% Probability Band'
        ))

        fig.add_trace(go.Scatter(
            x=days_axis,
            y=np.median(sim_matrix, axis=1),
            mode='lines',
            name='Median Path',
            line=dict(color="rgb(234, 179, 8)", width=3)
        ))

        fig.add_hline(y=optimal_entry, line_dash="dash", line_color="rgb(59, 130, 246)", annotation_text="Limit Entry")
        fig.add_hline(y=tp_1, line_dash="dash", line_color="rgb(16, 185, 129)", annotation_text="TP 1")
        fig.add_hline(y=tp_2, line_dash="dot", line_color="rgb(5, 150, 105)", annotation_text="TP 2")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="rgb(239, 68, 68)", annotation_text="Stop Loss")

        fig.update_layout(
            xaxis_title="Days Ahead",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Execute fragment
    render_live_simulator_dashboard()


# =========================================================
# PAGE 2: MULTI-ASSET DASHBOARD WITH CONCURRENT FETCH
# =========================================================
elif page == "📊 Multi-Asset Summary Dashboard":
    st.title("📊 Multi-Asset & Horizon Summary Dashboard")
    st.caption("Compare returns, risk metrics, and trade boundaries across assets and horizons concurrently.")

    st.sidebar.header("⚙️ Dashboard Controls")
    default_tickers_text = st.sidebar.text_area(
        "Assets to Compare",
        value="BTC-USD, ETH-USD, AAPL, MSFT, EURUSD=X"
    )
    
    selected_timelines = st.sidebar.multiselect(
        "Forecast Timelines (Days)",
        options=[14, 30, 60, 90, 180, 365],
        default=[30, 60, 90]
    )

    if not selected_timelines:
        st.warning("⚠️ Select at least one timeline.")
        st.stop()

    ticker_list = [sanitize_ticker(t) for t in default_tickers_text.split(",") if t.strip()]

    if st.button("🚀 Run Comparative Analysis") or "summary_df" not in st.session_state:
        summary_rows = []
        
        def process_ticker(t_symbol):
            close_data, log_returns, atr_14 = get_historical_baseline(t_symbol)
            if close_data is None or log_returns.empty:
                return None
            S0 = get_live_quote_fast(t_symbol, fallback_price=float(close_data.iloc[-1]))
            return (t_symbol, close_data, log_returns, atr_14, S0)

        with ThreadPoolExecutor(max_workers=min(10, len(ticker_list))) as executor:
            data_results = list(executor.map(process_ticker, ticker_list))

        for res in data_results:
            if res is None:
                continue
            t_symbol, close_data, log_returns, atr_14, S0 = res
            
            mu = float(log_returns.mean())
            sigma = float(log_returns.std())
            ann_vol = sigma * np.sqrt(252) * 100

            for days in selected_timelines:
                sim_matrix = run_monte_carlo_fast(S0, mu, sigma, days, n_sims=1000)
                final_prices = sim_matrix[-1, :]

                median_final = float(np.median(final_prices))
                exp_return_pct = ((median_final / S0) - 1) * 100
                p5_val = float(np.percentile(final_prices, 5))
                var_5_pct = ((p5_val / S0) - 1) * 100

                optimal_entry = max(S0 - (0.5 * atr_14), S0 * 0.95)
                recommended_sl = max(optimal_entry - atr_14, p5_val)
                recommended_tp = max(np.percentile(final_prices, 68), optimal_entry + (1.5 * atr_14))

                summary_rows.append({
                    "Ticker": t_symbol,
                    "Live Price ($)": round(S0, 4 if S0 < 10 else 2),
                    "Timeline (Days)": days,
                    "Ann. Volatility (%)": round(ann_vol, 1),
                    "Median Outcome ($)": round(median_final, 4 if S0 < 10 else 2),
                    "Expected Return (%)": round(exp_return_pct, 2),
                    "5% Downside Risk (%)": round(var_5_pct, 2),
                    "Limit Entry ($)": round(optimal_entry, 4 if S0 < 10 else 2),
                    "Stop Loss ($)": round(recommended_sl, 4 if S0 < 10 else 2),
                    "Take Profit ($)": round(recommended_tp, 4 if S0 < 10 else 2)
                })

        st.session_state["summary_df"] = pd.DataFrame(summary_rows)

    summary_df = st.session_state.get("summary_df", pd.DataFrame())

    if summary_df.empty:
        st.error("No data could be retrieved. Check ticker symbols.")
        st.stop()

    st.subheader("📋 Comprehensive Asset Matrix")
    st.dataframe(
        summary_df.style.background_gradient(subset=["Expected Return (%)"], cmap="RdYlGn")
                        .background_gradient(subset=["5% Downside Risk (%)"], cmap="Reds_r"),
        use_container_width=True,
        hide_index=True
    )

    csv_data = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Matrix as CSV",
        data=csv_data,
        file_name="gipsy_trading_ai_summary.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("📊 Visual Horizon Comparisons")
    c1, c2 = st.columns(2)

    with c1:
        fig_returns = px.bar(
            summary_df,
            x="Ticker",
         
