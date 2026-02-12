import ccxt
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize KuCoin exchange
# Note: KuCoin requires 'password' (this is your API Passphrase)
exchange = ccxt.kucoin({
    "apiKey": os.getenv("KUCOIN_API_KEY"),
    "secret": os.getenv("KUCOIN_API_SECRET"),
    "password": os.getenv("KUCOIN_PASSWORD"), # Your API Passphrase
    "enableRateLimit": True,
})

def fetch_ohlcv(symbol, timeframe, limit=200):
    """
    Fetch historical candle data (Open, High, Low, Close, Volume).
    Symbol example: 'BTC/USDT'
    """
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

def place_order(symbol, side, amount):
    """
    Place a market order.
    Side can be 'buy' or 'sell'.
    """
    return exchange.create_market_order(symbol, side, amount)