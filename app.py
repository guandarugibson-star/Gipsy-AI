import asyncio
import json
import logging
import signal
import sys
import aiohttp
from binance import AsyncClient, BinanceSocketManager

# Setup clean structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Configuration
SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "adausdt"]
NTFY_TOPIC = "Gipsy_AI_888"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

class AsyncMarketScanner:
    def __init__(self, symbols, ntfy_url):
        self.symbols = [s.lower() for s in symbols]
        self.ntfy_url = ntfy_url
        self.session = None
        self.binance_client = None
        self.running = True

    async def send_notification(self, title: str, message: str, priority: str = "default"):
        """Non-blocking notification dispatch via aiohttp."""
        headers = {
            "Title": title,
            "Priority": priority,
            "Tags": "chart_with_upwards_trend,warning"
        }
        try:
            async with self.session.post(self.ntfy_url, data=message, headers=headers, timeout=5) as resp:
                if resp.status == 200:
                    logging.info(f"Notification sent: {title}")
                else:
                    logging.warning(f"ntfy.sh responded with status {resp.status}")
        except Exception as e:
            logging.error(f"Failed to dispatch notification: {e}")

    def evaluate_kline(self, kline_data: dict):
        """Processes real-time kline closed bar data."""
        k = kline_data['k']
        if not k['x']:  # Only process on candle close
            return

        symbol = k['s']
        close_price = float(k['c'])
        open_price = float(k['o'])
        high_price = float(k['h'])
        low_price = float(k['l'])
        volume = float(k['v'])

        # Calculate candle change percentage
        price_change_pct = ((close_price - open_price) / open_price) * 100

        # Strategy Trigger: Candle move > 1.5% in a single 1m bar
        if abs(price_change_pct) >= 1.5:
            direction = "PUMP 🚀" if price_change_pct > 0 else "DUMP 🔻"
            title = f"{symbol} Volatility Alert: {direction}"
            message = (
                f"Price: {close_price}\n"
                f"Change: {price_change_pct:+.2f}%\n"
                f"High: {high_price} | Low: {low_price}\n"
                f"Volume: {volume:.2f}"
            )
            # Schedule dispatch without blocking the WebSocket listener loop
            asyncio.create_task(
                self.send_notification(title, message, priority="high" if abs(price_change_pct) > 3 else "default")
            )

    async def start_stream(self):
        """Listens to WebSocket multiplexed stream with automatic reconnection loop."""
        stream_names = [f"{s}@kline_1m" for s in self.symbols]
        
        while self.running:
            try:
                logging.info("Initializing Async Binance Client...")
                self.binance_client = await AsyncClient.create()
                bm = BinanceSocketManager(self.binance_client)
                
                # Multiplex socket handles multiple symbols over a single TCP connection
                ms = bm.multiplex_socket(stream_names)
                
                logging.info(f"Subscribed to WebSocket streams for {len(self.symbols)} symbols.")
                
                async with ms as stream:
                    while self.running:
                        res = await stream.recv()
                        if res and 'data' in res:
                            self.evaluate_kline(res['data'])

            except asyncio.CancelledError:
                logging.info("WebSocket loop cancelled. Shutting down...")
                break
            except Exception as e:
                logging.error(f"WebSocket network connection dropped: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            finally:
                if self.binance_client:
                    await self.binance_client.close_connection()

    async def run(self):
        """Main lifecycle manager."""
        self.session = aiohttp.ClientSession()
        try:
            await self.start_stream()
        finally:
            await self.session.close()

    def stop(self):
        self.running = False


async def main():
    scanner = AsyncMarketScanner(SYMBOLS, NTFY_URL)
    
    # Graceful shutdown handler for UNIX / Termux environments
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, scanner.stop)
        except NotImplementedError:
            pass  # Fallback for OS environments without full signal support

    await scanner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Application terminated cleanly.")
