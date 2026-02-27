import os
import ccxt
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize Kucoin with Spot Trading settings
exchange = ccxt.kucoin({
    "apiKey": os.getenv("KUCOIN_API_KEY"),
    "secret": os.getenv("KUCOIN_API_SECRET"),
    "password": os.getenv("KUCOIN_API_PASSPHRASE"),
    "enableRateLimit": True,
})

def fetch_ohlcv(symbol, timeframe, limit=200):
    """Fetches historical data for indicator calculation."""
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

def get_balance(currency="USDT"):
    """Fetches available balance from the TRADE account."""
    try:
        # Kucoin requires specifying the 'trade' account type for spot trading
        bal = exchange.fetch_balance({'type': 'trade'})
        return float(bal["free"].get(currency, 0.0))
    except Exception as e:
        print(f"❌ Balance Error: {e}")
        return 0.0

def cancel_orphaned_orders(symbol, current_order_id, related_order_id):
    """
    Checks if one part of a trade pair (SL or TP) is filled.
    If it is, it cancels the other one to prevent "ghost" orders.
    """
    try:
        order = exchange.fetch_order(current_order_id, symbol)
        # Status 'closed' means the order was filled
        if order['status'] == 'closed':
            print(f"🧹 Trade closed via {current_order_id}. Cancelling related order: {related_order_id}")
            exchange.cancel_order(related_order_id, symbol)
            return True
        return False
    except Exception as e:
        # Silently fail if the order was already cancelled manually
        return False

def execute_trade(symbol, side, entry_price, stop_loss, take_profit, risk_per_trade=0.01):
    """
    Handles complete order lifecycle for Kucoin:
    Position sizing, Market Entry (funds-based), SL, and TP.
    """
    try:
        exchange.load_markets()
        balance = get_balance("USDT")
        
        if balance < 10:
            print(f"⚠️ Insufficient balance: ${balance}")
            return None

        # 1. Calculate Risk and Position Size
        risk_amt = balance * risk_per_trade
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return None

        amount = risk_amt / price_diff
        funds = amount * entry_price # Total USDT for market buy

        print(f"🚀 EXECUTING {side.upper()} on {symbol}")
        
        # 2. Market Entry
        if side.lower() == "buy":
            # Kucoin Market Buy uses funds (USDT)
            order = exchange.create_market_buy_order(symbol, funds)
        else:
            # Market Sell uses amount (Coin quantity)
            order = exchange.create_market_sell_order(symbol, amount)
            
        print(f"✅ Entry Filled: {order['id']}")
        time.sleep(2) # Brief pause for exchange synchronization
        
        # 3. Setup Exit Orders
        exit_side = "sell" if side.lower() == "buy" else "buy"
        qty = exchange.amount_to_precision(symbol, amount)
        sl_price = exchange.price_to_precision(symbol, stop_loss)
        tp_price = exchange.price_to_precision(symbol, take_profit)

        # 4. Place Stop Loss (Trigger Market Order)
        sl_order = exchange.create_order(
            symbol=symbol, 
            type='market', 
            side=exit_side, 
            amount=qty,
            params={
                'stop': 'loss', 
                'stopPrice': sl_price, 
                'reduceOnly': True
            }
        )
        print(f"🛡️ SL Set at {sl_price}")

        # 5. Place Take Profit (Limit Order)
        tp_order = exchange.create_order(
            symbol=symbol, 
            type='limit', 
            side=exit_side, 
            amount=qty,
            price=tp_price,
            params={'reduceOnly': True}
        )
        print(f"🎯 TP Set at {tp_price}")

        # Return IDs to bot.py for the safety tracker
        return {
            "entry_id": order["id"],
            "sl_id": sl_order["id"],
            "tp_id": tp_order["id"],
            "amount": qty,
            "symbol": symbol
        }

    except Exception as e:
        print(f"❌ Execution Failed: {e}")
        return None