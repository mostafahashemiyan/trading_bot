# report.py
import os
import ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

exchange = ccxt.kraken(
    {
        "apiKey": os.getenv("KRAKEN_API_KEY"),
        "secret": os.getenv("KRAKEN_API_SECRET"),
    }
)


def fetch_closed_trades():
    # Kraken 'closed' orders includes cancelled ones --> we filter by status='closed' and filled > 0
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
