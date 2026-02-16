import os
import ccxt
import time
from dotenv import load_dotenv

load_dotenv()

exchange = ccxt.kucoin(
    {
        "apiKey": os.getenv("KUCOIN_API_KEY"),
        "secret": os.getenv("KUCOIN_API_SECRET"),
        "password": os.getenv("KUCOIN_API_PASSPHRASE"),
        "enableRateLimit": True,
    }
)


def fetch_ohlcv(symbol, timeframe, limit=200):
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)


def get_balance(currency="USDT"):
    """Fetch free balance for the quote currency."""
    try:
        bal = exchange.fetch_balance()
        return bal["free"].get(currency, 0.0)

    except Exception as e:
        print(f"Error fetching balance: {e}")
        return 0.0


def execute_trade(
    symbol, side, entry_price, stop_loss, take_profit, risk_per_trade=0.01
):
    """
    1. Calculates Position Size based on Risk %.
    2. Places Market Entry.
    3. Places Stop Loss Order.
    4. Places Take Profit Order.
    """
    try:
        # 1. Calculate Balance & Size
        quote_currency = symbol.split("/")[1]
        balance = get_balance(quote_currency)

        if balance == 0:
            print("Insufficient balance.")
            return None

        # Risk Calculation: (Balance * Risk%) / (Entry - Stop)
        risk_amt = balance * risk_per_trade
        price_diff = abs(entry_price - stop_loss)

        if price_diff == 0:
            print("Invalid Stop Loss (Price Diff is 0).")
            return None

        amount = risk_amt / price_diff

        # Sanity check: Kraken min order size (approx)
        if amount * entry_price < 10:
            print("Position size too small (< $10).")
            return None

        print(f"EXECUTING {side} {symbol}")
        print(f"Balance: {balance:.2f} | Risk: ${risk_amt:.2f} | Size: {amount:.4f}")

        # 2. Execute Market Entry
        order = exchange.create_order(symbol, "market", side, amount)
        print(f"Entry Filled: {order['id']}")

        # 3. Place Stop Loss (Wait a moment to ensure entry is processed)
        time.sleep(2)

        # Determine SL/TP side (Opposite of entry)
        exit_side = "sell" if side == "buy" else "buy"

        # Kraken Stop Loss
        sl_order = exchange.create_order(
            symbol, "stop-loss", exit_side, amount, params={"stopPrice": stop_loss}
        )  # `stop-loss` may need to be changed into `limit`

        print(f"SL Placed at {stop_loss}: {sl_order['id']}")

        # 4. Place Take Profit
        tp_order = exchange.create_order(
            symbol,
            "limit",  # `take-profit` if not `limit` didn't work
            exit_side,
            amount,
            price=take_profit,
            params={"reduceOnly": True},
        )

        print(f"TP Placed at {take_profit}: {tp_order['id']}")

        return {
            "entry_id": order["id"],
            "sl_id": sl_order["id"],
            "tp_id": tp_order["id"],
            "amount": amount,
            "entry_price": entry_price,
        }

    except Exception as e:
        print(f"Execution Failed: {e}")
        return None
