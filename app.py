import time
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# ---------------------------------------------------------
# Page Configuration & Navigation
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gipsy Trading AI - Quantitative Risk Optimizer",
    page_icon="📈",
    layout="wide"
)

page = st.sidebar.radio(
    "📌 Navigation Menu",
    ["📖 Beginner's Guide & Single Simulator", "📊 Multi-Asset Summary Dashboard"]
)

# ---------------------------------------------------------
# Helper Functions & Quantitative Engines (With Retries)
# ---------------------------------------------------------
def retry_on_exception(retries=3, delay=1.5):
    """Decorator to retry flaky network calls against Yahoo Finance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


def sanitize_ticker(symbol: str) -> str:
    """
    Sanitizes user input tickers for yfinance compatibility.
    Automatically formats crypto pairs and forex currency pairs.
    """
    symbol = symbol.strip().upper()
    
    crypto_bases = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "AVAX", "DOT", "LINK"]
    for coin in crypto_bases:
        if symbol.startswith(coin) and ("USD" in symbol or "=X" in symbol or "/" in symbol):
            return f"{coin}-USD"
            
    if len(symbol) == 6 and symbol.isalpha() and "=" not in symbol and "-" not in symbol:
        return f"{symbol}=X"
        
    return symbol


@st.cache_data(ttl=300, show_spinner=False)
@retry_on_exception(retries=3, delay=1.0)
def _fetch_raw_history(symbol: str, fetch_period: str, fetch_interval: str):
    """Cached low-level fetcher for raw yfinance history dataframe."""
    tk = yf.Ticker(symbol)
    df_hist = tk.history(period=fetch_period, interval=fetch_interval)
    if df_hist.empty:
        return None
    if isinstance(df_hist.columns, pd.MultiIndex):
        df_hist.columns = df_hist.columns.get_level_values(0)
    df_hist = df_hist.rename(columns={col: col.capitalize() for col in df_hist.columns})
    return df_hist


def get_historical_baseline(symbol: str, interval: str):
    """
    Fetches and processes historical baseline data with interval resampling 
    handled outside the cache layer to prevent pollution.
    """
    try:
        period_map = {
            "15m": "59d",
            "1h": "730d",
            "4h": "730d",
            "1d": "1y"
        }
        fetch_period = period_map.get(interval, "1y")
        fetch_interval = "1h" if interval == "4h" else interval

        df_hist = _fetch_raw_history(symbol, fetch_period, fetch_interval)
        if df_hist is None or df_hist.empty:
            return None, None, None

        if 'Close' not in df_hist.columns or 'High' not in df_hist.columns or 'Low' not in df_hist.columns:
            return None, None, None

        # Resample 1h data into 4h blocks if requested
        if interval == "4h":
            df_hist = df_hist.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

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


@retry_on_exception(retries=2, delay=0.5)
def get_live_quote_fast(symbol: str, fallback_price: float) -> float:
    """Ultra-Fast Network Call: Fetches latest spot quote tick with fallbacks."""
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


def run_monte_carlo_fast(S0: float, mu: float, sigma: float, steps: int, n_sims: int = 1000, model_type: str = "Standard (Normal Distribution)", log_returns=None, jump_params=None):
    """Vectorized Monte Carlo engine supporting multiple statistical distributions."""
    dt = 1.0
    
    if model_type == "Standard (Normal Distribution)":
        shocks = np.random.normal(0, 1, size=(steps - 1, n_sims))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * shocks
        daily_returns = drift + diffusion

    elif model_type == "Fat-Tail (Student's t)" and log_returns is not None and len(log_returns) > 10:
        try:
            df_fit, _, _ = t.fit(log_returns)
            df_fit = max(df_fit, 3.0)
        except Exception:
            df_fit = 5.0
        t_shocks = t.rvs(df_fit, size=(steps - 1, n_sims)) / np.sqrt(df_fit / (df_fit - 2.0))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * t_shocks
        daily_returns = drift + diffusion

    elif model_type == "Jump-Diffusion (Merton)" and jump_params is not None:
        j_lambda, j_mu, j_sigma = jump_params
        shocks = np.random.normal(0, 1, size=(steps - 1, n_sims))
        n_jumps = np.random.poisson(j_lambda, size=(steps - 1, n_sims))
        jump_factor = np.random.normal(j_mu, j_sigma, size=(steps - 1, n_sims)) * n_jumps
        drift = (mu - 0.5 * sigma**2 - j_lambda * (np.exp(j_mu + 0.5 * j_sigma**2) - 1)) * dt
        diffusion = sigma * np.sqrt(dt) * shocks
        daily_returns = drift + diffusion + jump_factor
    else:
        shocks = np.random.normal(0, 1, size=(steps - 1, n_sims))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * shocks
        daily_returns = drift + diffusion

    zero_day = np.zeros((1, n_sims))
    cum_returns = np.vstack([zero_day, np.cumsum(daily_returns, axis=0)])
    
    return S0 * np.exp(cum_returns)


def calculate_optimal_trade_levels(sim_matrix, S0, atr_14):
    """Derives mathematically sound entry, stop-loss, and take-profit targets using MFE/MAE."""
    path_minima = np.min(sim_matrix, axis=0)
    median_path_dip = np.percentile(path_minima, 25)
    optimal_entry = max(S0 - (0.5 * atr_14), median_path_dip)
    optimal_entry = min(optimal_entry, S0)

    p5_downside = np.percentile(sim_matrix[-1, :], 5)
    stop_loss = max(optimal_entry - (1.5 * atr_14), p5_downside)
    if stop_loss >= optimal_entry:
        stop_loss = optimal_entry * 0.98

    path_maxima = np.max(sim_matrix, axis=0)
    tp_1 = float(np.percentile(path_maxima, 60))
    tp_2 = float(np.percentile(path_maxima, 85))

    tp_1 = max(tp_1, optimal_entry + (1.5 * atr_14))
    tp_2 = max(tp_2, optimal_entry + (3.0 * atr_14))

    risk = optimal_entry - stop_loss
    reward = tp_1 - optimal_entry
    rr_ratio = reward / risk if risk > 0 else 0.0

    return optimal_entry, stop_loss, tp_1, tp_2, rr_ratio


# =========================================================
# PAGE 1: BEGINNER GUIDE & SINGLE SIMULATOR
# =========================================================
if page == "📖 Beginner's Guide & Single Simulator":
    st.title("📈 Gipsy Trading AI")
    st.caption("Quantitative risk estimation and dynamic MFE/MAE trade targeting across intraday & daily intervals.")
    
    st.sidebar.header("⚡ Live Data Feed Settings")
    auto_refresh_active = st.sidebar.toggle("Enable Live Auto-Refresh", value=True)
    refresh_rate_sec = st.sidebar.slider("Refresh Interval (Seconds)", 2, 30, 3, step=1)

    if st.sidebar.button("🔄 Force Refresh All Data"):
        st.cache_data.clear()
        st.success("Cache cleared successfully!")

    with st.expander("📖 Guide to MFE/MAE Quantitative Optimization", expanded=False):
        st.markdown("""
        ### 🧠 How Trade Targets Are Optimized Here
        * **MAE (Maximum Adverse Excursion):** Bottoms out worst-case paths to build tight **Stop Losses**.
        * **MFE (Maximum Favorable Excursion):** Peaks high-upside paths to extract optimal **Take Profits**.
        """)

    st.markdown("---")

    st.sidebar.header("🛠️ Simulation Setup")
    user_ticker_input = st.sidebar.text_input("Asset Ticker Symbol (e.g. EURUSD, BTC-USD, AAPL)", value="EURUSD")
    ticker = sanitize_ticker(user_ticker_input)

    candle_interval = st.sidebar.selectbox("Candle Timeframe Interval", ["15m", "1h", "4h", "1d"], index=1)
    forecast_steps = st.sidebar.slider("Forecast Horizon (Steps Ahead)", 10, 100, 30)

    experience_mode = st.sidebar.radio("Engine Mode", ["🚀 MFE/MAE Quantitative Optimizer (Recommended)", "⚙️ Custom Manual Sliders"])

    st.sidebar.markdown("---")
    n_simulations = st.sidebar.slider("Monte Carlo Path Count", 500, 3000, 1000, step=500)
    
    model_type = st.sidebar.selectbox(
        "Statistical Distribution Model", 
        ["Fat-Tail (Student's t)", "Standard (Normal Distribution)", "Jump-Diffusion (Merton)"]
    )

    jump_params = None
    if model_type == "Jump-Diffusion (Merton)":
        jump_params = (4.0 / 252.0, -0.05, 0.05)

    close_data, log_returns, atr_14 = get_historical_baseline(ticker, candle_interval)

    if close_data is None or log_returns is None or log_returns.empty:
        st.error(f"❌ Could not load or parse valid price data for **{ticker}** on interval **{candle_interval}**. Please verify the symbol or choose a different interval.")
        st.stop()

    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    fallback_last_price = float(close_data.iloc[-1])

    @st.fragment(run_every=refresh_rate_sec if auto_refresh_active else None)
    def render_live_simulator_dashboard():
        if close_data is None or log_returns is None:
            st.warning(f"⚠️ Baseline data missing for **{ticker}**.")
            return

        S0 = get_live_quote_fast(ticker, fallback_price=fallback_last_price)
        decimals = 4 if S0 < 10 else 2

        sim_matrix = run_monte_carlo_fast(
            S0=S0, mu=mu, sigma=sigma, steps=forecast_steps, 
            n_sims=n_simulations, model_type=model_type, 
            log_returns=log_returns, jump_params=jump_params
        )

        optimal_entry, stop_loss, tp_1, tp_2, rr_ratio = calculate_optimal_trade_levels(
            sim_matrix, S0, atr_14
        )

        p25 = np.percentile(sim_matrix, 25, axis=1)
        p75 = np.percentile(sim_matrix, 75, axis=1)
        p95 = np.percentile(sim_matrix, 95, axis=1)
        p5 = np.percentile(sim_matrix, 5, axis=1)

        st.caption(f"🟢 **Live Data Stream Active ({candle_interval}):** `{S0:,.{decimals}f}` | Last Checked: `{pd.Timestamp.now().strftime('%H:%M:%S UTC')}`")
        st.markdown("### 🚦 Automated MFE/MAE Trade Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Live Spot Price", f"{S0:,.{decimals}f}")
        col2.metric("2. Optimized Limit Entry", f"{optimal_entry:,.{decimals}f}")
        col3.metric("3. Optimized Target (TP1)", f"{tp_1:,.{decimals}f}", delta=f"+{((tp_1/optimal_entry)-1)*100:.2f}%")
        col4.metric("4. Optimized Stop Loss", f"{stop_loss:,.{decimals}f}", delta=f"{((stop_loss/optimal_entry)-1)*100:.2f}%", delta_color="inverse")

        st.markdown("---")
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("🎯 Strategy Breakdown")
            st.markdown(f"""
            * **Asset / Timeframe:** `{ticker}` @ `{candle_interval}`
            * **Bar Noise (14-Bar ATR):** `{atr_14:,.{decimals}f}`
            * **Quantitative Limit Entry:** **`{optimal_entry:,.{decimals}f}`**
            * **MFE Target 1 (TP1):** **`{tp_1:,.{decimals}f}`**
            * **MFE Target 2 (TP2):** **`{tp_2:,.{decimals}f}`**
            * **MAE Stop Loss Boundary:** **`{stop_loss:,.{decimals}f}`**
            """)

        with col_right:
            st.subheader("⚖️ Setup Rating")
            st.metric("Risk-to-Reward (R:R)", f"1 : {rr_ratio:.2f}")
            if rr_ratio >= 1.5:
                st.success("🟢 **Prime Setup** (R:R ≥ 1:1.5)")
            elif rr_ratio >= 1.0:
                st.warning("🟡 **Acceptable Setup** (R:R ≥ 1:1.0)")
            else:
                st.error("🔴 **Low R:R Ratio** — Exercise caution.")

        st.markdown("---")
        st.subheader(f"📊 Projected Path Cone & Optimal Levels ({forecast_steps} Steps @ {candle_interval})")

        steps_axis = np.arange(forecast_steps)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.concatenate([steps_axis, steps_axis[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='90% Simulation Band'
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([steps_axis, steps_axis[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.25)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='50% Simulation Band'
        ))

        fig.add_trace(go.Scatter(
            x=steps_axis,
            y=np.median(sim_matrix, axis=1),
            mode='lines',
            name='Median Path',
            line=dict(color="rgb(234, 179, 8)", width=3)
        ))

        fig.add_hline(y=optimal_entry, line_dash="dash", line_color="rgb(59, 130, 246)", annotation_text="Optimized Entry")
        fig.add_hline(y=tp_1, line_dash="dash", line_color="rgb(16, 185, 129)", annotation_text="TP 1")
        fig.add_hline(y=tp_2, line_dash="dot", line_color="rgb(5, 150, 105)", annotation_text="TP 2")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="rgb(239, 68, 68)", annotation_text="MAE Stop Loss")

        fig.update_layout(
            xaxis_title=f"Candle Steps ({candle_interval})",
            yaxis_title="Price / Exchange Rate",
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    render_live_simulator_dashboard()


# =========================================================
# PAGE 2: MULTI-ASSET DASHBOARD WITH CONCURRENT FETCH
# =========================================================
elif page == "📊 Multi-Asset Summary Dashboard":
    st.title("📊 Multi-Asset & Multi-Interval Summary Dashboard")
    st.caption("Concurrent MFE/MAE risk and target analysis across a multi-asset watch list including forex currencies.")

    st.sidebar.header("⚙️ Dashboard Controls")
    
    default_12_assets = "EURUSD, GBPUSD, USDJPY, BTC-USD, ETH-USD, SOL-USD, AAPL, MSFT, NVDA, AMZN, GC=F, CL=F"
    default_tickers_text = st.sidebar.text_area("Assets to Compare (Comma-separated)", value=default_12_assets, height=100)
    
    selected_interval = st.sidebar.selectbox("Comparison Interval", ["15m", "1h", "4h", "1d"], index=1)
    selected_steps_list = st.sidebar.multiselect("Forecast Steps Ahead", options=[10, 20, 30, 50, 100], default=[20, 50])

    if not selected_steps_list:
        st.warning("⚠️ Select at least one step horizon.")
        st.stop()

    ticker_list = [sanitize_ticker(t) for t in default_tickers_text.split(",") if t.strip()]

    if st.button("🚀 Run Comparative MFE/MAE Analysis") or "summary_df" not in st.session_state:
        summary_rows = []
        
        def process_ticker(t_symbol):
            close_data, log_returns, atr_14 = get_historical_baseline(t_symbol, selected_interval)
            if close_data is None or log_returns is None or log_returns.empty:
                return None
            S0 = get_live_quote_fast(t_symbol, fallback_price=float(close_data.iloc[-1]))
            return (t_symbol, close_data, log_returns, atr_14, S0)

        with ThreadPoolExecutor(max_workers=min(12, len(ticker_list))) as executor:
            data_results = list(executor.map(process_ticker, ticker_list))

        for res in data_results:
            if res is None:
                continue
            t_symbol, close_data, log_returns, atr_14, S0 = res
            
            mu = float(log_returns.mean())
            sigma = float(log_returns.std())

            for steps in selected_steps_list:
                sim_matrix = run_monte_carlo_fast(S0, mu, sigma, steps, n_sims=1000)
                optimal_entry, stop_loss, tp_1, tp_2, rr_ratio = calculate_optimal_trade_levels(sim_matrix, S0, atr_14)
                
                final_prices = sim_matrix[-1, :]
                median_final = float(np.median(final_prices))
                exp_return_pct = ((median_final / S0) - 1) * 100
                p5_val = float(np.percentile(final_prices, 5))
                var_5_pct = ((p5_val / S0) - 1) * 100

                decimals = 4 if S0 < 10 else 2

                summary_rows.append({
                    "Ticker": t_symbol,
                    "Interval": selected_interval,
                    "Live Price": round(S0, decimals),
                    "Steps": steps,
                    "Limit Entry": round(optimal_entry, decimals),
                    "Stop Loss": round(stop_loss, decimals),
                    "Take Profit 1": round(tp_1, decimals),
                    "R:R Ratio": round(rr_ratio, 2),
                    "Exp Return (%)": round(exp_return_pct, 2)
                })

        st.session_state["summary_df"] = pd.DataFrame(summary_rows)

    summary_df = st.session_state.get("summary_df", pd.DataFrame())

    if summary_df.empty:
        st.error("No valid data could be retrieved for the specified tickers. Please verify your internet connection or ticker symbols.")
        st.stop()

    st.subheader("📋 Optimized Asset Target Matrix")
    st.dataframe(
        summary_df.style.background_gradient(subset=["Exp Return (%)"], cmap="RdYlGn")
                        .background_gradient(subset=["R:R Ratio"], cmap="Greens"),
        use_container_width=True,
        hide_index=True
    )

    csv_data = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Matrix as CSV",
        data=csv_data,
        file_name="gipsy_currency_asset_trading_matrix.csv",
        mime="text/csv"
    )
    
