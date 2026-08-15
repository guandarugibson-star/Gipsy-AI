import asyncio
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure local module path resolution
sys.path.append(str(Path(__file__).parent.resolve()))

try:
    from scanner import PredictiveScanner
except ImportError:
    st.error(
        "Could not find `scanner.py`. Please make sure `scanner.py` is in the same directory as `app.py`."
    )
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Predictive Market Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Predictive Market Scanner")
st.caption("Scan Binance USDT pairs for ADX trend strength, volume expansion, and moving average alignment.")

# Sidebar Settings
st.sidebar.header("Scanner Settings")

interval = st.sidebar.selectbox(
    "Timeframe Interval",
    options=["15m", "1h", "4h", "1d"],
    index=1,
)

min_vol_millions = st.sidebar.slider(
    "Min 24h Volume (USDT Millions)",
    min_value=1.0,
    max_value=50.0,
    value=10.0,
    step=1.0,
)

volume_multiplier = st.sidebar.slider(
    "Volume Gate Multiplier (vs SMA)",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
)

adx_cutoff = st.sidebar.slider(
    "Min ADX Threshold",
    min_value=15.0,
    max_value=50.0,
    value=25.0,
    step=1.0,
)

col1, col2 = st.sidebar.columns(2)
with col1:
    short_lb = st.number_input("Short Lookback", min_value=5, max_value=50, value=20)
with col2:
    long_lb = st.number_input("Long Lookback", min_value=20, max_value=200, value=50)

max_concurrency = st.sidebar.slider(
    "Max Concurrent API Requests",
    min_value=5,
    max_value=30,
    value=15,
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run Market Scan", use_container_width=True)


# Core Async Execution Wrapper
def execute_scan():
    scanner = PredictiveScanner(
        min_24h_volume_usdt=min_vol_millions * 1_000_000.0,
        volume_gate_multiplier=volume_multiplier,
        adx_threshold=adx_cutoff,
        short_lookback=int(short_lb),
        long_lookback=int(long_lb),
    )
    # Handle event loop execution safely
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        scanner.run_scan(interval=interval, max_concurrent=int(max_concurrency))
    )


# Dashboard Main View
if run_button:
    with st.spinner(f"Scanning Binance USDT pairs on {interval} timeframe..."):
        try:
            results = execute_scan()
            st.session_state["scan_results"] = results
            st.session_state["last_interval"] = interval
        except Exception as e:
            st.error(f"Error executing scan: {e}")

if "scan_results" in st.session_state:
    results = st.session_state["scan_results"]
    selected_tf = st.session_state.get("last_interval", interval)

    st.subheader(f"Results for {selected_tf} Timeframe")

    if not results:
        st.warning("No market setups passed all criteria with the current settings.")
    else:
        df_results = pd.DataFrame(results)

        # Top Summary Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Qualified Setups", len(df_results))
        m2.metric("Highest ADX Pair", df_results.iloc[0]["symbol"])
        m3.metric("Peak ADX Value", f"{df_results.iloc[0]['adx']:.2f}")

        st.markdown("### Actionable Setups")

        # Format dataframe for presentation
        display_df = df_results[
            [
                "symbol",
                "close",
                "adx",
                "current_volume",
                "volume_threshold",
                "sma_short",
                "sma_long",
            ]
        ].copy()

        display_df.columns = [
            "Symbol",
            "Price (USDT)",
            "ADX",
            "Current Vol",
            "Vol Threshold",
            "Short SMA",
            "Long SMA",
        ]

        st.dataframe(
            display_df.style.highlight_max(axis=0, subset=["ADX"], color="#1f77b422"),
            use_container_width=True,
        )

        st.download_button(
            label="📥 Download CSV Report",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name=f"scan_results_{selected_tf}.csv",
            mime="text/csv",
        )
else:
    st.info("Click **Run Market Scan** in the sidebar to fetch real-time market opportunities.")
