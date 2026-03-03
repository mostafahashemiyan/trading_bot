# bot.py
"""
Main live trading bot loop.
Uses strategy → gatekeeper → execute (or dry run)
"""

import asyncio
import time
from datetime import datetime
import ccxt.async_support as ccxt
from dotenv import load_dotenv
import os

# Local imports
import config
from indicators import prepare_df
from strategy import trend_pullback_signal
from risk import position_size, sl_tp_from_atr
from llm_gatekeeper import llm_decide

load_dotenv()

# ────────────────────────────────────────────────
# Exchange setup
# ────────────────────────────────────────────────
exchange = ccxt.kucoinfutures({
    'apiKey': os.getenv('KUCOIN_API_KEY'),
    'secret': os.getenv('KUCOIN_API_SECRET'),
    'password': os.getenv('KUCOIN_PASSPHRASE'),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

# ────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────
async def fetch_ohlcv(symbol, timeframe, limit=200):
    """Fetch OHLCV data asynchronously"""
    return await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)


async def get_balance():
    """Get USDT free balance"""
    bal = await exchange.fetch_balance()
    return float(bal['USDT']['free'])


async def execute_trade(symbol, side, amount, sl, tp):
    """Place market entry + SL + TP orders (dry run supported)"""
    if config.DRY_RUN:
        print(f"[DRY RUN] Would place {side} order for {amount:.4f} {symbol} | SL:{sl} TP:{tp}")
        return {"status": "dry_run", "entry_price": "simulated"}

    try:
        # Market entry
        order = await exchange.create_market_order(
            symbol=symbol,
            side=side.lower(),
            amount=amount
        )

        # Stop Loss
        sl_side = 'sell' if side == 'buy' else 'buy'
        sl_order = await exchange.create_order(
            symbol=symbol,
            type='stop_market',
            side=sl_side,
            amount=amount,
            params={'stopPrice': sl, 'reduceOnly': True}
        )

        # Take Profit
        tp_order = await exchange.create_limit_order(
            symbol=symbol,
            side=sl_side,
            amount=amount,
            price=tp,
            params={'reduceOnly': True}
        )

        return {
            "entry": order,
            "sl": sl_order,
            "tp": tp_order
        }

    except Exception as e:
        print(f"Execution failed: {e}")
        return None


# ────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────
async def main_loop():
    print(f"Bot started – Dry run: {config.DRY_RUN}")

    while True:
        for symbol in config.SYMBOLS:
            try:
                print(f"\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] Scanning {symbol}")

                # Fetch data
                ohlcv_high   = await fetch_ohlcv(symbol, config.HIGH_TF,   300)
                ohlcv_medium = await fetch_ohlcv(symbol, config.MEDIUM_TF, 300)
                ohlcv_low    = await fetch_ohlcv(symbol, config.LOW_TF,    300)

                df_high   = prepare_df(ohlcv_high)
                df_medium = prepare_df(ohlcv_medium)
                df_low    = prepare_df(ohlcv_low)

                if df_high.empty or df_medium.empty or df_low.empty:
                    print(f"Insufficient data for {symbol}")
                    continue

                # Generate raw signal
                signal = trend_pullback_signal(df_high, df_medium, df_low)

                if not signal["setup"]:
                    print(f"No setup → {signal['reasons']}")
                    continue

                print(f"Raw signal detected: {signal['side']} @ {signal['entry']}")

                # LLM Gatekeeper
                current_price = df_low['close'].iloc[-1]
                decision = llm_decide(signal, symbol, current_price)

                print(f"LLM decision: {decision['decision']} | {decision.get('reason')} | Confidence: {decision.get('confidence')}")

                if decision["decision"] != "APPROVE":
                    continue

                # Position sizing
                balance = await get_balance()
                size = position_size(balance, signal["entry"], signal["sl"])

                if size <= 0 or size * signal["entry"] < config.MIN_NOTIONAL_VALUE:
                    print("Position size too small")
                    continue

                # Execute
                side_exec = "buy" if signal["side"] == "LONG" else "sell"
                result = await execute_trade(symbol, side_exec, size, signal["sl"], signal["tp"])

                if result:
                    print(f"Trade executed: {result}")
                else:
                    print("Execution failed")

            except Exception as e:
                print(f"Error on {symbol}: {e}")

        await asyncio.sleep(300)  # هر ۵ دقیقه اسکن


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        asyncio.run(exchange.close())