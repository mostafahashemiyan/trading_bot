# report.py
import os
import ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

exchange = ccxt.kucoin({
    "apiKey": os.getenv("KUCOIN_API_KEY"),
    "secret": os.getenv("KUCOIN_API_SECRET"),
    "password": os.getenv("KUCOIN_API_PASSPHRASE"),
})


def fetch_closed_trades():
    # Kucoin might require a symbol for fetch_closed_orders in some cases
    orders = exchange.fetch_closed_orders(limit=50)

    trades = []
    for o in orders:
        if o["status"] == "closed" and o["filled"] > 0:
            trades.append(
                {
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "amount": o["amount"],
                    "price": o["average"],  # Average fill price
                    "cost": o["cost"],
                    "timestamp": o["timestamp"],
                }
            )

    return pd.DataFrame(trades)


if __name__ == "__main__":
    print("Fetching account history...")
    df = fetch_closed_trades()

    if not df.empty:
        print("\n--- RECENT COMPLETED ORDERS ---")
        print(df[["symbol", "side", "price", "cost"]])

        # Basic PnL logic would require matching Entry orders with Exit orders
        # For now, this shows raw execution history

    else:
        print("No closed trades found recently.")
