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
    "options": {'code': 'USDT'} 
})

def fetch_ohlcv(symbol, timeframe, limit=200):
    """Fetches historical data for indicator calculation."""
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

def get_balance(currency="USDT"):
    """Fetches available balance from the TRADE account."""
    try:
        # Kucoin has Main, Trade, and Margin accounts. We need 'trade'.
        bal = exchange.fetch_balance({'type': 'trade'})
        return float(bal["free"].get(currency, 0.0))
    except Exception as e:
        print(f"❌ Balance Error: {e}")
        return 0.0

def execute_trade(symbol, side, entry_price, stop_loss, take_profit, risk_per_trade=0.01):
    """
    Handles complete order lifecycle:
    1. Position Sizing
    2. Market Entry (Funds-based for Buy)
    3. Trigger Stop Loss
    4. Limit Take Profit
    """
    try:
        # 1. Load Markets to handle decimal precision automatically
        exchange.load_markets()
        
        balance = get_balance("USDT")
        if balance < 10:
            print(f"⚠️ Insufficient balance: ${balance}")
            return None

        # 2. Risk Calculation
        risk_amt = balance * risk_per_trade
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return None

        amount = risk_amt / price_diff
        
        # Kucoin Market Buy requires 'funds' (USDT cost), not 'amount' (coin qty)
        funds = amount * entry_price 

        if funds < 5: # Kucoin minimum is usually $5-$10
            print(f"⚠️ Position too small: ${funds:.2f}")
            return None

        print(f"🚀 EXECUTING {side.upper()} on {symbol}")
        print(f"Risk: ${risk_amt:.2f} | Est. Size: {amount:.4f}")

        # 3. Market Entry
        if side.lower() == "buy":
            # For 'buy', we send the 'funds' (USDT)
            order = exchange.create_market_buy_order(symbol, funds)
        else:
            # For 'sell' (Short), we send the 'amount' (Qty)
            order = exchange.create_market_sell_order(symbol, amount)

        print(f"✅ Entry Filled: {order['id']}")
        
        # Wait for Kucoin to register the filled balance
        time.sleep(1.5)
        
        # 4. Exit Management
        exit_side = "sell" if side.lower() == "buy" else "buy"
        
        # Format prices to exchange-specific precision
        sl_price = exchange.price_to_precision(symbol, stop_loss)
        tp_price = exchange.price_to_precision(symbol, take_profit)
        # Re-fetch actual amount if it's a sell to ensure we close the whole position
        qty = exchange.amount_to_precision(symbol, amount)

        # 5. Place Stop Loss (Market Trigger)
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

        # 6. Place Take Profit (Limit Order)
        tp_order = exchange.create_order(
            symbol=symbol,
            type='limit',
            side=exit_side,
            amount=qty,
            price=tp_price,
            params={'reduceOnly': True}
        )
        print(f"🎯 TP Set at {tp_price}")

        return {
            "entry_id": order["id"],
            "sl_id": sl_order["id"],
            "tp_id": tp_order["id"],
            "amount": qty
        }

    except Exception as e:
        print(f"❌ Execution Failed: {str(e)}")
        return None