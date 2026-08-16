import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# CORE SMC SCANNER ENGINE (WITH INTEGRATED NTFY NOTIFICATIONS)
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
        enable_notifications: bool = False,
        ntfy_topic: str = "",
    ):
        # List of alternative Binance API mirrors for failover/IP bypass
        self.fallback_urls = [
            base_url.rstrip("/"),
            "https://api1.binance.com",
            "https://api2.binance.com",
            "https://api3.binance.com",
            "https://data-api.binance.vision",
        ]
        self.current_base_idx = 0
        self.min_24h_volume_usdt = min_24h_volume_usdt
        self.adx_threshold = adx_threshold
        self.htf_interval = htf_interval
        self.ltf_interval = ltf_interval
        self.kline_limit = kline_limit
        self.enable_notifications = enable_notifications
        self.ntfy_topic = ntfy_topic.strip()
        
        self.debug_stats = {
            "total_pairs": 0,
            "volume_passed": 0,
            "htf_bias_passed": 0,
            "ltf_entry_passed": 0,
            "notifications_sent": 0,
            "errors": 0,
        }

    @property
    def base_url(self) -> str:
        return self.fallback_urls[self.current_base_idx]

    def _rotate_domain(self):
        self.current_base_idx = (self.current_base_idx + 1) % len(self.fallback_urls)
        logger.warning(f"Rotating Binance API domain to: {self.base_url}")

    async def send_ntfy_alert(self, signal: dict):
        """Dispatches real-time push notification to mobile via ntfy.sh"""
        if not self.enable_notifications or not self.ntfy_topic:
            return

        direction_emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
        title = f"{direction_emoji} {signal['direction']} SIGNAL: {signal['symbol']}"

        # Detailed trade alert formatting
        message = (
            f"🎯 Pair: {signal['symbol']}\n"
            f"📊 Direction: {signal['direction']}\n"
            f"💰 Entry Price: {signal['entry_price']}\n"
            f"🛑 Stop Loss (SL): {signal['stop_loss']}\n"
            f"🎯 Take Profit (TP): {signal['take_profit']}\n"
            f"⚖️ Risk/Reward: {signal['risk_reward']}R\n"
            f"📈 HTF ADX Strength: {signal['htf_adx']}"
        )

        req = urllib.request.Request(
            f"https://ntfy.sh/{self.ntfy_topic}",
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": "high",
                "Tags": "chart_with_upwards_trend,warning" if signal["direction"] == "LONG" else "chart_with_downwards_trend,warning",
            },
            method="POST",
        )

        try:
            await asyncio.to_thread(urllib.request.urlopen, req, timeout=5)
            self.debug_stats["notifications_sent"] += 1
            logger.info(f"Notification dispatched for {signal['symbol']} to topic '{self.ntfy_topic}'")
        except Exception as e:
            logger.error(f"Failed to send ntfy notification for {signal['symbol']}: {e}")

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

        ltf_df["atr"] = self.calculate_atr(ltf_df)
        ltf_df["atr_sma"] = ltf_df["atr"].rolling(20).mean()

        atr_expansion = ltf_df["atr"].iloc[-2] >= (ltf_df["atr_sma"].iloc[-2] * 1.15)
        atr_val = float(ltf_df["atr"].iloc[-2])

        entry_price = float(ltf_df["close"].iloc[-2])
        entry_triggered = False
        direction = "NONE"
        stop_loss, take_profit = 0.0, 0.0

        recent_lows = ltf_df["low"].iloc[-14:-3]
        bull_liquidity_swept = ltf_df["low"].iloc[-3] < (recent_lows.min() if len(recent_lows) > 0 else 0)
        bull_bos = ltf_df["close"].iloc[-2] > ltf_df["high"].iloc[-20:-3].max()
        bull_fvg = ltf_df["low"].iloc[-2] > ltf_df["high"].iloc[-4]

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

    async def _async_get_json(self, path: str) -> dict | list:
        max_retries = 3
        for attempt in range(max_retries):
            url = f"{self.base_url}{path}"

            def _fetch():
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=6) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            try:
                return await asyncio.to_thread(_fetch)
            except urllib.error.HTTPError as e:
                logger.warning(f"HTTP {e.code} on {url} (Attempt {attempt + 1}/{max_retries})")
                self.debug_stats["errors"] += 1
                if e.code in (403, 451, 429):
                    self._rotate_domain()
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                logger.error(f"Network error on {url}: {e}")
                self.debug_stats["errors"] += 1
                await asyncio.sleep(0.5)

        return []

    async def fetch_active_usdt_pairs(self) -> list[str]:
        tickers = await self._async_get_json("/api/v3/ticker/24hr")
        if not isinstance(tickers, list):
            return []

        self.debug_stats["total_pairs"] = len(tickers)

        valid_symbols = []
        for t in tickers:
            symbol = t.get("symbol", "")
            quote_vol = float(t.get("quoteVolume", 0.0))
            if (
                symbol.endswith("USDT")
                and not any(x in symbol for x in ["UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT"])
                and quote_vol >= self.min_24h_volume_usdt
            ):
                valid_symbols.append(symbol)

        self.debug_stats["volume_passed"] = len(valid_symbols)
        return valid_symbols

    async def fetch_klines(self, symbol: str, interval: str) -> pd.DataFrame:
        params = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": self.kline_limit})
        raw_klines = await self._async_get_json(f"/api/v3/klines?{params}")

        if not raw_klines or not isinstance(raw_klines, list):
            return pd.DataFrame()

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
            if htf_df.empty:
                return None

            htf_res = self.evaluate_htf_bias(htf_df)
            if htf_res["bias"] == "NEUTRAL":
                return None

            self.debug_stats["htf_bias_passed"] += 1

            ltf_df = await self.fetch_klines(symbol, interval=self.ltf_interval)
            if ltf_df.empty:
                return None

            ltf_res = self.evaluate_ltf_entry(ltf_df, htf_bias=htf_res["bias"])

            if ltf_res["passed"]:
                self.debug_stats["ltf_entry_passed"] += 1
                signal_data = {
                    "symbol": symbol,
                    "direction": ltf_res["direction"],
                    "htf_bias": htf_res["bias"],
                    "htf_adx": htf_res["adx"],
                    **ltf_res["metrics"],
                }

                # Dispatch push notification if enabled
                await self.send_ntfy_alert(signal_data)

                return signal_data
        except Exception as e:
            self.debug_stats["errors"] += 1
            logger.error(f"Error scanning {symbol}: {e}")
        return None

    async def run_scan(self, max_concurrent: int = 10) -> tuple[list[dict], dict]:
        symbols = await self.fetch_active_usdt_pairs()
        if not symbols:
            return [], self.debug_stats

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_scan(sym):
            async with semaphore:
                return await self.scan_symbol(sym)

        tasks = [_bounded_scan(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals = [res for res in results if isinstance(res, dict) and res is not None]
        sorted_signals = sorted(signals, key=lambda x: x["htf_adx"], reverse=True)
        return sorted_signals, self.debug_stats

# =====================================================================
# STREAMLIT USER INTERFACE & EXECUTION
# =====================================================================

st.set_page_config(page_title="SMC Crypto Scanner", page_icon="📈", layout="wide")

st.title("⚡ Multi-Timeframe SMC Scanner")
st.markdown("Identifies **High-Probability Smart Money Concepts (SMC)** setups across Binance USDT pairs.")

# Sidebar Configuration
st.sidebar.header("Scanner Configuration")

min_vol = st.sidebar.number_input(
    "Min 24h Volume ($ USDT)", min_value=1_000_000, value=20_000_000, step=5_000_000
)
adx_thresh = st.sidebar.slider("HTF ADX Trend Threshold", 10, 50, 25)
htf_tf = st.sidebar.selectbox("High Timeframe (HTF)", ["1d", "4h", "2h"], index=1)
ltf_tf = st.sidebar.selectbox("Low Timeframe (LTF)", ["1h", "15m", "5m"], index=0)
concurrency = st.sidebar.slider("Max Concurrent Async Requests", 1, 20, 10)

st.sidebar.markdown("---")
st.sidebar.header("🔔 Mobile Push Alerts (ntfy.sh)")
enable_ntfy = st.sidebar.checkbox("Enable Push Notifications", value=True)
ntfy_topic_input = st.sidebar.text_input(
    "ntfy Topic Name", 
    value="Gipsy_AI_888", 
    help="Enter your unique ntfy topic name to subscribe in the ntfy app."
)

if st.button("🚀 Run Market Scan", use_container_width=True):
    scanner = HighAccuracySMCScanner(
        min_24h_volume_usdt=float(min_vol),
        adx_threshold=float(adx_thresh),
        htf_interval=htf_tf,
        ltf_interval=ltf_tf,
        enable_notifications=enable_ntfy,
        ntfy_topic=ntfy_topic_input,
    )

    with st.spinner("Scanning active USDT pairs on Binance..."):
        signals, stats = asyncio.run(scanner.run_scan(max_concurrent=concurrency))

    # Display Execution Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Pairs Checked", stats["total_pairs"])
    col2.metric("Passed Vol Filter", stats["volume_passed"])
    col3.metric("Passed HTF Bias", stats["htf_bias_passed"])
    col4.metric("Valid Signals Found", len(signals))
    col5.metric("Alerts Dispatched", stats["notifications_sent"])

    st.markdown("---")

    if signals:
        df_results = pd.DataFrame(signals)

        df_results["Signal"] = df_results["direction"].apply(
            lambda x: "🟢 LONG" if x == "LONG" else "🔴 SHORT"
        )

        df_display = df_results[[
            "symbol", "Signal", "htf_adx", "entry_price",
            "stop_loss", "take_profit", "risk_reward",
            "bos_confirmed", "liquidity_swept", "fvg_present"
        ]].copy()

        df_display.columns = [
            "Pair", "Direction", "HTF ADX", "Entry Price",
            "Stop Loss", "Take Profit", "R:R",
            "BOS Confirmed", "Liq Swept", "FVG Present"
        ]

        st.success(f"Found **{len(signals)}** active SMC trade opportunities!")
        if enable_ntfy and stats["notifications_sent"] > 0:
            st.info(f"📲 Successfully sent **{stats['notifications_sent']}** trade alerts to ntfy topic **'{ntfy_topic_input}'**.")

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No SMC setups found matching your current threshold criteria.")
        
