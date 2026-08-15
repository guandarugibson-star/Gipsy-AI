import asyncio
import json
import logging
import math
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Configure Streamlit Page
st.set_page_config(
    page_title="Predictive Crypto Scanner",
    page_icon="📈",
    layout="wide"
)

# ------------------------------------------------------------------
# 1. PredictiveScanner Class Definition
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
        """Calculates Average Directional Index (ADX) to quantify trend strength."""
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
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_s = pd.Series(plus_dm, index=df.index)
        minus_dm_s = pd.Series(minus_dm, index=df.index)

        # Wilder's Smoothing
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        smoothed_plus_dm = plus_dm_s.ewm(alpha=1 / period, adjust=False).mean()
        smoothed_minus_dm = minus_dm_s.ewm(alpha=1 / period, adjust=False).mean()

        # Directional Indicators
        plus_di = 100 * (smoothed_plus_dm / atr)
        minus_di = 100 * (smoothed_minus_dm / atr)

        # DX & ADX
        di_diff = (plus_di - minus_di).abs()
        di_sum = plus_di + minus_di
        dx = 100 * (di_diff / np.where(di_sum == 0, 1, di_sum))

        return dx.ewm(alpha=1 / period, adjust=False).mean()

    def apply_technical_filters(self, df: pd.DataFrame) -> dict:
        """Applies dynamic lookbacks, ADX filter, and volume gating checks."""
        if len(df) < self.long_lookback + self.adx_period:
            return {"passed": False, "reason": "Insufficient data length"}

        # 1. ADX Filter
        df["adx"] = self.calculate_adx(df, period=self.adx_period)
        latest_adx = df["adx"].iloc[-1]
        adx_passed = latest_adx >= self.adx_threshold

        # 2. Volume Gating
        df["vol_sma"] = df["volume"].rolling(window=self.short_lookback).mean()
        current_volume = df["volume"].iloc[-1]
        volume_threshold = df["vol_sma"].iloc[-1] * self.volume_gate_multiplier
        volume_passed = current_volume >= volume_threshold

        # 3. Dynamic Lookback SMAs
        df["sma_short"] = df["close"].rolling(window=self.short_lookback).mean()
        df["sma_long"] = df["close"].rolling(window=self.long_lookback).mean()

        current_close = df["close"].iloc[-1]
        sma_short_val = df["sma_short"].iloc[-1]
        sma_long_val = df["sma_long"].iloc[-1]

        # Trend alignment (Price > SMA Fast > SMA Slow)
        trend_aligned = (current_close > sma_short_val) and (sma_short_val > sma_long_val)

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

    async def _async_get_json(self, url: str) -> dict | list:
        def _fetch():
            req = urllib.request.Request(
                url, headers={"User-Agent": "PredictiveScanner/2.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        return await asyncio.to_thread(_fetch)

    async def fetch_active_usdt_pairs(self) -> list[str]:
        endpoint = f"{self.base_url}/api/v3/ticker/24hr"
        tickers = await self._async_get_json(endpoint)

        valid_symbols = []
        for t in tickers:
            symbol = t.get("symbol", "")
            quote_vol = float(t.get("quoteVolume", 0.0))

            if symbol.endswith("USDT") and quote_vol >= self.min_24h_volume_usdt:
                valid_symbols.append(symbol)

        return valid_symbols

    async def fetch_klines(self, symbol: str, interval: str = "1h") -> pd.DataFrame:
        params = urllib.parse.urlencode({
            "symbol": symbol,
            "interval": interval,
            "limit": self.kline_limit,
        })
        endpoint = f"{self.base_url}/api/v3/klines?{params}"
        raw_klines = await self._async_get_json(endpoint)

        df = pd.DataFrame(
            raw_klines,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
            ],
        )

        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df

    async def scan_symbol(self, symbol: str, interval: str = "1h") -> dict | None:
        try:
            df = await self.fetch_klines(symbol, interval=interval)
            filter_res = self.apply_technical_filters(df)
            if filter_res["passed"]:
                return {"symbol": symbol, **filter_res["metrics"]}
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
        return None

    async def run_scan(self, interval: str = "1h", max_concurrent: int = 10) -> list[dict]:
        symbols = await self.fetch_active_usdt_pairs()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_scan(sym):
            async with semaphore:
                return await self.scan_symbol(sym, interval=interval)

        tasks = [_bounded_scan(sym) for sym in symbols]
        results = await asyncio.gather(*tasks)

        actionable_signals = [res for res in results if res is not None]
        return sorted(actionable_signals, key=lambda x: x["adx"], reverse=True)


# ------------------------------------------------------------------
# 2. Streamlit Web Interface
# ------------------------------------------------------------------

st.title("📈 Predictive Crypto Market Scanner")
st.markdown(
    "Scans active Binance USDT trading pairs with **ADX Trend Filtering**, "
    "**Volume Spike Detection**, and **SMA Trend Alignment**."
)

# Sidebar Parameters
st.sidebar.header("Filter Settings")
interval = st.sidebar.selectbox("Candle Timeframe", ["15m", "1h", "4h", "1d"], index=1)
min_volume = st.sidebar.number_input("Min 24h Volume (USDT)", value=10_000_000, step=1_000_000)
adx_cutoff = st.sidebar.slider("Minimum ADX Threshold", 15.0, 50.0, 25.0, 1.0)
vol_multiplier = st.sidebar.slider("Volume Multiplier vs SMA", 1.0, 3.0, 1.5, 0.1)
max_concurrency = st.sidebar.slider("Max Concurrent Scans", 5, 30, 15)

if st.sidebar.button("Launch Scan", type="primary"):
    scanner = PredictiveScanner(
        min_24h_volume_usdt=float(min_volume),
        volume_gate_multiplier=float(vol_multiplier),
        adx_threshold=float(adx_cutoff),
    )

    with st.spinner("Analyzing Binance market data..."):
        # Safely run asyncio scanner inside Streamlit
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            signals = loop.run_until_complete(
                scanner.run_scan(interval=interval, max_concurrent=max_concurrency)
            )
            loop.close()
        except Exception as e:
            st.error(f"Scan failed due to an execution error: {e}")
            signals = []

    if signals:
        st.success(f"Found {len(signals)} matching trading setup(s)!")

        # Top Summary Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Matching Setups", len(signals))
        col2.metric("Strongest Trend", signals[0]["symbol"])
        col3.metric("Peak ADX Value", signals[0]["adx"])

        # Clean Table Output
        df_display = pd.DataFrame(signals)[
            ["symbol", "close", "adx", "current_volume", "volume_threshold", "sma_short", "sma_long"]
        ]
        df_display.columns = [
            "Symbol", "Price ($)", "ADX", "Volume", "Vol Target", "SMA (Short)", "SMA (Long)"
        ]

        st.dataframe(
            df_display.style.format({
                "Price ($)": "{:.4f}",
                "ADX": "{:.2f}",
                "Volume": "{:,.0f}",
                "Vol Target": "{:,.0f}",
                "SMA (Short)": "{:.4f}",
                "SMA (Long)": "{:.4f}",
            }),
            use_container_width=True,
        )
    else:
        st.warning("No trading pairs currently meet all your filter criteria.")
