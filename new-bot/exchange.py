import os
import ccxt
import time
from dotenv import load_dotenv

load_dotenv()

exchange = ccxt.kucoin({
    "apiKey": os.getenv("KUCOIN_API_KEY"),
    "secret": os.getenv("KUCOIN_API_SECRET"),
    "password": os.getenv("KUCOIN_API_PASSPHRASE"),
    "enableRateLimit": True,
})

def cancel_orphaned_orders(symbol, current_order_id, related_order_id):
    """
    Checks if one part of a trade pair (SL or TP) is filled.
    If it is, it cancels the other one.
    """
    try:
        order = exchange.fetch_order(current_order_id, symbol)
        if order['status'] == 'closed':
            print(f"🧹 Trade closed. Cancelling related order: {related_order_id}")
            exchange.cancel_order(related_order_id, symbol)
            return True
        return False
    except Exception as e:
        print(f"Cleanup Error: {e}")
        return False

def execute_trade(symbol, side, entry_price, stop_loss, take_profit, risk_per_trade=0.01):
    try:
        exchange.load_markets()
        balance = get_balance("USDT")
        
        # ... (keep your existing Risk and Market Entry logic) ...
        
        # 1. Market Entry
        funds = (balance * risk_per_trade / abs(entry_price - stop_loss)) * entry_price
        if side.lower() == "buy":
            order = exchange.create_market_buy_order(symbol, funds)
        else:
            order = exchange.create_market_sell_order(symbol, amount) # amount defined in your sizing logic
            
        print(f"✅ Entry Filled: {order['id']}")
        time.sleep(1.5)
        
        exit_side = "sell" if side.lower() == "buy" else "buy"
        qty = exchange.amount_to_precision(symbol, amount)

        # 2. Place SL and TP
        sl_order = exchange.create_order(
            symbol=symbol, type='market', side=exit_side, amount=qty,
            params={'stop': 'loss', 'stopPrice': exchange.price_to_precision(symbol, stop_loss), 'reduceOnly': True}
        )
        
        tp_order = exchange.create_order(
            symbol=symbol, type='limit', side=exit_side, amount=qty,
            price=exchange.price_to_precision(symbol, take_profit),
            params={'reduceOnly': True}
        )

        # Return all IDs so the bot can track them
        return {
            "entry_id": order["id"],
            "sl_id": sl_order["id"],
            "tp_id": tp_order["id"],
            "symbol": symbol
        }

    except Exception as e:
        print(f"❌ Execution Failed: {e}")
        return None