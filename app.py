import asyncio
import logging
import aiohttp
import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Core Predictive Scanner Engine
# ------------------------------------------------------------------

class PredictiveScanner:
    """Predictive market scanner equipped with ADX trend filtering,
    volume gating, and dynamic lookbacks to detect high-probability setups.
    """

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        min_24h_volume_usdt: float = 5_000_000.0,
        volume_gate_multiplier: float = 1.5,
        adx_threshold: float = 25.0,
        adx_period: int = 14,
        short_lookback: int = 20,
        long_lookback: int = 50,
        kline_limit: int = 100,
    ):
        self.base_url = base_url.rstrip("/")
        self.min_24h_volume_usdt = min_24h_volume_usdt
        self.volume_gate_multiplier = volume_gate_multiplier
        self.adx_threshold = adx_threshold
        self.adx_period = adx_period
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
        self.kline_limit = max(kline_limit, long_lookback + adx_period + 10)

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        df = df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range (TR)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement (+DM, -DM)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where(
            (down_move > up_move) & (down_move > 0), down_move, 0.0
        )

        plus_dm_s = pd.Series(plus_dm, index=df.index)
        minus_dm_s = pd.Series(minus_dm, index=df.index)

        # Wilder's Smoothing
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        smoothed_plus_dm = plus_dm_s.ewm(alpha=1 / period, adjust=False).mean()
        smoothed_minus_dm = minus_dm_s.ewm(
            alpha=1 / period, adjust=False
        ).mean()

        # Directional Indicators (+DI, -DI)
        plus_di = 100 * (smoothed_plus_dm / atr)
        minus_di = 100 * (smoothed_minus_dm / atr)

        # Directional Index (DX) & ADX
        di_diff = (plus_di - minus_di).abs()
        di_sum = plus_di + minus_di
        dx = 100 * (di_diff / np.where(di_sum == 0, 1, di_sum))

        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        return adx

    def apply_technical_filters(self, df: pd.DataFrame) -> dict:
        if len(df) < self.long_lookback + self.adx_period:
            return {"passed": False, "reason": "Insufficient data length"}

        # 1. ADX Filter
        df["adx"] = self.calculate_adx(df, period=self.adx_period)
        latest_adx = df["adx"].iloc[-1]
        adx_passed = latest_adx >= self.adx_threshold

        # 2. Volume Gating
        df["vol_sma"] = (
            df["volume"].rolling(window=self.short_lookback).mean()
        )
        current_volume = df["volume"].iloc[-1]
        volume_threshold = (
            df["vol_sma"].iloc[-1] * self.volume_gate_multiplier
        )
        volume_passed = current_volume >= volume_threshold

        # 3. Dynamic Lookbacks (SMA Trend Check)
        df["sma_short"] = (
            df["close"].rolling(window=self.short_lookback).mean()
        )
        df["sma_long"] = (
            df["close"].rolling(window=self.long_lookback).mean()
        )

        current_close = df["close"].iloc[-1]
        sma_short_val = df["sma_short"].iloc[-1]
        sma_long_val = df["sma_long"].iloc[-1]

        trend_aligned = (current_close > sma_short_val) and (
            sma_short_val > sma_long_val
        )
        overall_passed = adx_passed and volume_passed and trend_aligned

        return {
            "passed": overall_passed,
            "metrics": {
                "close": current_close,
                "adx": round(float(latest_adx), 2),
                "adx_passed": bool(adx_passed),
                "current_volume": round(float(current_volume), 2),
                "volume_threshold": round(float(volume_threshold), 2),
                "volume_passed": bool(volume_passed),
                "sma_short": round(float(sma_short_val), 4),
                "sma_long": round(float(sma_long_val), 4),
                "trend_aligned": bool(trend_aligned),
            },
        }

    async def _fetch_json(
        self, session: aiohttp.ClientSession, url: str
    ) -> dict | list:
        headers = {"User-Agent": "PredictiveScannerApp/1.0"}
        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                text = await response.text()
                logger.error(f"API Error ({response.status}) on {url}: {text}")
                return []
        except Exception as e:
            logger.error(f"Network error fetching {url}: {e}")
            return []

    async def fetch_active_usdt_pairs(
        self, session: aiohttp.ClientSession
    ) -> list[str]:
        endpoint = f"{self.base_url}/api/v3/ticker/24hr"
        tickers = await self._fetch_json(session, endpoint)

        valid_symbols = []

        # Safe guard to handle unexpected string/dict error responses from Binance
        if not isinstance(tickers, list):
            logger.error(
                f"Expected list from Binance 24hr ticker, got {type(tickers)}: {tickers}"
            )
            return valid_symbols

        for t in tickers:
            if not isinstance(t, dict):
                continue

            symbol = t.get("symbol", "")
            try:
                quote_vol = float(t.get("quoteVolume", 0.0))
            except (ValueError, TypeError):
                quote_vol = 0.0

            # Filter valid spot USDT pairs and exclude leveraged tokens
            if (
                symbol.endswith("USDT")
                and not any(symbol.endswith(x) for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])
                and quote_vol >= self.min_24h_volume_usdt
            ):
                valid_symbols.append(symbol)

        return valid_symbols

    async def fetch_klines(
        self, session: aiohttp.ClientSession, symbol: str, interval: str = "1h"
    ) -> pd.DataFrame:
        endpoint = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit={self.kline_limit}"
        raw_klines = await self._fetch_json(session, endpoint)

        if not isinstance(raw_klines, list) or len(raw_klines) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(
            raw_klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df

    async def scan_symbol(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        interval: str = "1h",
    ) -> dict | None:
        try:
            df = await self.fetch_klines(session, symbol, interval=interval)
            if df.empty:
                return None

            filter_res = self.apply_technical_filters(df)

            if filter_res.get("passed", False):
                return {"symbol": symbol, **filter_res["metrics"]}
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
        return None

    async def run_scan(
        self,
        interval: str = "1h",
        max_concurrent: int = 15,
        progress_callback=None,
    ) -> list[dict]:
        async with aiohttp.ClientSession() as session:
            symbols = await self.fetch_active_usdt_pairs(session)
            total = len(symbols)

            if total == 0:
                return []

            semaphore = asyncio.Semaphore(max_concurrent)
            counter = [0]

            async def _bounded_scan(sym):
                async with semaphore:
                    res = await self.scan_symbol(
                        session, sym, interval=interval
                    )
                    counter[0] += 1
                    if progress_callback:
                        progress_callback(counter[0], total)
                    return res

            tasks = [_bounded_scan(sym) for sym in symbols]
            results = await asyncio.gather(*tasks)

            actionable_signals = [res for res in results if res is not None]
            return sorted(
                actionable_signals, key=lambda x: x["adx"], reverse=True
            )


# ------------------------------------------------------------------
# Streamlit Dashboard UI
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Predictive Market Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Predictive Market Scanner")
st.markdown(
    "Detect high-probability crypto setups on Binance using **ADX trend filtering**, **volume gating**, and **moving average alignment**."
)

# Sidebar Configuration Controls
st.sidebar.header("⚙️ Scanner Settings")

kline_interval = st.sidebar.selectbox(
    "Candle Interval",
    options=["15m", "1h", "4h", "1d"],
    index=1,
)

min_volume = st.sidebar.number_input(
    "Min 24h Volume (USDT)",
    min_value=1_000_000.0,
    max_value=500_000_000.0,
    value=10_000_000.0,
    step=1_000_000.0,
    format="%.0f",
)

volume_mult = st.sidebar.slider(
    "Volume Gate Multiplier (x SMA)",
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

col_lookbacks = st.sidebar.columns(2)
short_lb = col_lookbacks[0].number_input(
    "Short Lookback", min_value=5, max_value=50, value=20
)
long_lb = col_lookbacks[1].number_input(
    "Long Lookback", min_value=20, max_value=200, value=50
)

max_threads = st.sidebar.slider(
    "Concurrency Limit", min_value=5, max_value=30, value=15
)

# Run Scan Trigger
if st.sidebar.button("🚀 Start Market Scan", type="primary"):
    scanner = PredictiveScanner(
        min_24h_volume_usdt=min_volume,
        volume_gate_multiplier=volume_mult,
        adx_threshold=adx_cutoff,
        short_lookback=short_lb,
        long_lookback=long_lb,
    )

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(current, total):
        pct = current / total if total > 0 else 0.0
        progress_bar.progress(pct)
        status_text.text(f"Scanning market pairs... {current}/{total}")

    # Standard asyncio run execution
    signals = asyncio.run(
        scanner.run_scan(
            interval=kline_interval,
            max_concurrent=max_threads,
            progress_callback=update_progress,
        )
    )

    progress_bar.empty()
    status_text.empty()

    # Render Results Section
    if signals:
        st.success(f"Found **{len(signals)}** actionable setups!")

        df_results = pd.DataFrame(signals)

        # Display Summary Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Top Setup", df_results.iloc[0]["symbol"])
        m2.metric("Highest ADX", f"{df_results.iloc[0]['adx']}")
        m3.metric("Total Matches", len(df_results))

        st.subheader("Filtered Setups")

        # Format dataframe columns for display
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
            "Vol Cutoff",
            f"SMA ({short_lb})",
            f"SMA ({long_lb})",
        ]

        st.dataframe(
            display_df.style.highlight_max(subset=["ADX"], color="#1f77b4"),
            use_container_width=True,
        )

        # CSV Download Option
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"predictive_scan_{kline_interval}.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            "No market pairs met all criteria or the API response returned no valid symbols. Try lowering thresholds in the sidebar."
        )
        
