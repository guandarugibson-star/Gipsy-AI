import asyncio
import json
import logging
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# CORE SMC SCANNER ENGINE (UPGRADED WITH SHORT LOGIC & SL/TP)
# =====================================================================

class HighAccuracySMCScanner:
    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        min_24h_volume_usdt: float = 20_000_000.0,
        adx_threshold: float = 25.0,
        htf_interval: str = "4h",
        ltf_interval: str = "1h",
        kline_limit: int = 200,
    ):
        self.base_url = base_url.rstrip("/")
        self.min_24h_volume_usdt = min_24h_volume_usdt
        self.adx_threshold = adx_threshold
        self.htf_interval = htf_interval
        self.ltf_interval = ltf_interval
        self.kline_limit = kline_limit

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        safe_atr = np.where(atr == 0, 1e-9, atr)

        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / safe_atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / safe_atr)

        di_diff = (plus_di - minus_di).abs()
        di_sum = np.where((plus_di + minus_di) == 0, 1e-9, plus_di + minus_di)
        dx = 100 * (di_diff / di_sum)

        return dx.ewm(alpha=1 / period, adjust=False).mean()

    def evaluate_htf_bias(self, htf_df: pd.DataFrame) -> dict:
        if len(htf_df) < 50:
            return {"bias": "NEUTRAL", "adx": 0.0}

        htf_df["sma_short"] = htf_df["close"].rolling(20).mean()
        htf_df["sma_long"] = htf_df["close"].rolling(50).mean()
        htf_df["adx"] = self.calculate_adx(htf_df)

        close = htf_df["close"].iloc[-1]
        sma_short = htf_df["sma_short"].iloc[-1]
        sma_long = htf_df["sma_long"].iloc[-1]
        adx = htf_df["adx"].iloc[-1]

        is_bullish = (close > sma_short) and (sma_short > sma_long) and (adx >= self.adx_threshold)
        is_bearish = (close < sma_short) and (sma_short < sma_long) and (adx >= self.adx_threshold)

        if is_bullish:
            return {"bias": "BULLISH", "adx": round(float(adx), 2)}
        elif is_bearish:
            return {"bias": "BEARISH", "adx": round(float(adx), 2)}
        return {"bias": "NEUTRAL", "adx": round(float(adx), 2)}

    def evaluate_ltf_entry(self, ltf_df: pd.DataFrame, htf_bias: str) -> dict:
        if len(ltf_df) < 50:
            return {"passed": False, "reason": "Insufficient LTF candles"}

        # Evaluates completed candle (iloc[-2]) to prevent live repainting
        ltf_df["atr"] = self.calculate_atr(ltf_df)
        ltf_df["atr_sma"] = ltf_df["atr"].rolling(20).mean()
        atr_expansion = ltf_df["atr"].iloc[-2] >= (ltf_df["atr_sma"].iloc[-2] * 1.15)
        atr_val = float(ltf_df["atr"].iloc[-2])

        entry_price = float(ltf_df["close"].iloc[-2])
        entry_triggered = False
        direction = "NONE"
        stop_loss, take_profit = 0.0, 0.0

        # Bullish Signals
        recent_lows = ltf_df["low"].iloc[-14:-3]
        bull_liquidity_swept = ltf_df["low"].iloc[-3] < (recent_lows.min() if len(recent_lows) > 0 else 0)
        bull_bos = ltf_df["close"].iloc[-2] > ltf_df["high"].iloc[-20:-3].max()
        bull_fvg = ltf_df["low"].iloc[-2] > ltf_df["high"].iloc[-4]

        # Bearish Signals
        recent_highs = ltf_df["high"].iloc[-14:-3]
        bear_liquidity_swept = ltf_df["high"].iloc[-3] > (recent_highs.max() if len(recent_highs) > 0 else 0)
        bear_bos = ltf_df["close"].iloc[-2] < ltf_df["low"].iloc[-20:-3].min()
        bear_fvg = ltf_df["high"].iloc[-2] < ltf_df["low"].iloc[-4]

        if htf_bias == "BULLISH":
            entry_triggered = bull_bos and (bull_liquidity_swept or bull_fvg) and atr_expansion
            if entry_triggered:
                direction = "LONG"
                sweep_low = float(ltf_df["low"].iloc[-5:-2].min())
                stop_loss = sweep_low - (0.2 * atr_val)
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 2.5) if risk > 0 else entry_price * 1.05

        elif htf_bias == "BEARISH":
            entry_triggered = bear_bos and (bear_liquidity_swept or bear_fvg) and atr_expansion
            if entry_triggered:
                direction = "SHORT"
                sweep_high = float(ltf_df["high"].iloc[-5:-2].max())
                stop_loss = sweep_high + (0.2 * atr_val)
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 2.5) if risk > 0 else entry_price * 0.95

        return {
            "passed": entry_triggered,
            "direction": direction,
            "metrics": {
                "bos_confirmed": bool(bull_bos if direction == "LONG" else bear_bos),
                "liquidity_swept": bool(bull_liquidity_swept if direction == "LONG" else bear_liquidity_swept),
                "fvg_present": bool(bull_fvg if direction == "LONG" else bear_fvg),
                "atr_expansion": bool(atr_expansion),
                "entry_price": round(entry_price, 4),
                "stop_loss": round(stop_loss, 4),
                "take_profit": round(take_profit, 4),
                "risk_reward": 2.5,
            },
        }

    async def _async_get_json(self, url: str) -> dict | list:
        def _fetch():
            req = urllib.request.Request(url, headers={"User-Agent": "StreamlitSMC/1.0"})
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

    async def fetch_klines(self, symbol: str, interval: str) -> pd.DataFrame:
        params = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": self.kline_limit})
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
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        return df

    async def scan_symbol(self, symbol: str) -> dict | None:
        try:
            htf_df = await self.fetch_klines(symbol, interval=self.htf_interval)
            htf_res = self.evaluate_htf_bias(htf_df)

            if htf_res["bias"] == "NEUTRAL":
                return None

            ltf_df = await self.fetch_klines(symbol, interval=self.ltf_interval)
            ltf_res = self.evaluate_ltf_entry(ltf_df, htf_bias=htf_res["bias"])

            if ltf_res["passed"]:
                return {
                    "symbol": symbol,
                    "direction": ltf_res["direction"],
                    "htf_bias": htf_res["bias"],
                    "htf_adx": htf_res["adx"],
                    **ltf_res["metrics"],
                }
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
        return None

    async def run_scan(self, max_concurrent: int = 15) -> list[dict]:
        symbols = await self.fetch_active_usdt_pairs()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_scan(sym):
            async with semaphore:
                return await self.scan_symbol(sym)

        tasks = [_bounded_scan(sym) for sym in symbols]
        results = await asyncio.gather(*tasks)
        signals = [res for res in results if res is not None]
        return sorted(signals, key=lambda x: x["htf_adx"], reverse=True)


# =====================================================================
# STREAMLIT UI LAYOUT
# =====================================================================

st.set_page_config(
    page_title="High-Accuracy SMC Scanner",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ High-Accuracy SMC Crypto Scanner")
st.markdown("Automated Multi-Timeframe Smart Money Concepts Scanner with **ADX Trend Alignment**, **Liquidity Sweeps**, and **ATR Volatility Expansion**.")

# Sidebar Filters
st.sidebar.header("⚙️ Scanner Settings")
min_volume = st.sidebar.number_input("Min 24h Volume (USDT)", value=20_000_000.0, step=5_000_000.0, format="%.0f")
adx_val = st.sidebar.slider("HTF ADX Threshold", min_value=15.0, max_value=40.0, value=25.0, step=1.0)

col_htf, col_ltf = st.sidebar.columns(2)
with col_htf:
    htf_select = st.selectbox("HTF Interval", ["4h", "1d"], index=0)
with col_ltf:
    ltf_select = st.selectbox("LTF Interval", ["1h", "15m", "5m"], index=0)

concurrent_tasks = st.sidebar.slider("Max Concurrent Scans", min_value=5, max_value=25, value=12)

# Main Scanner Trigger
if st.button("🚀 Run SMC Market Scan", use_container_width=True):
    with st.spinner("Scanning active Binance USDT pairs..."):
        scanner = HighAccuracySMCScanner(
            min_24h_volume_usdt=min_volume,
            adx_threshold=adx_val,
            htf_interval=htf_select,
            ltf_interval=ltf_select,
        )
        signals = asyncio.run(scanner.run_scan(max_concurrent=concurrent_tasks))

    if signals:
        st.success(f"🎯 Found {len(signals)} Active SMC Setup(s)!")
        
        # Format Dataframe for UI Display
        df_display = pd.DataFrame(signals)
        df_display = df_display[[
            "symbol", "direction", "htf_bias", "htf_adx", 
            "entry_price", "stop_loss", "take_profit", "risk_reward"
        ]]
        df_display.columns = [
            "Symbol", "Direction", "HTF Bias", "HTF ADX", 
            "Entry Price", "Stop Loss", "Take Profit", "Target R:R"
        ]

        st.dataframe(
            df_display.style.highlight_max(subset=["HTF ADX"], color="#2e7d32"),
            use_container_width=True,
        )
    else:
        st.warning("No pairs currently match all strict SMC entry criteria. Check back in the next candle close.")
        
