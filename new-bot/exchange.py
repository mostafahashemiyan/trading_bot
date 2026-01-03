import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

exchange = ccxt.kraken({
    "apiKey": os.getenv("KRAKEN_API_KEY"),
    "secret": os.getenv("KRAKEN_API_SECRET"),
    "enableRateLimit": True,
})

def fetch_ohlcv(symbol, timeframe, limit=200):
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

def place_order(symbol, side, amount):
    return exchange.create_market_order(symbol, side, amount)
