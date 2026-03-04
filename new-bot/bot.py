# bot.py
"""
Main live trading bot loop.
Uses strategy -> gatekeeper -> execute (or dry run)
"""

import asyncio
import time
from datetime import datetime, timezone
import ccxt.async_support as ccxt
from ccxt.base.errors import ExchangeNotAvailable, NetworkError
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
    'timeout': 20000,
    'options': {'defaultType': 'swap'},
})

# ────────────────────────────────────────────────
# Error formatting
# ────────────────────────────────────────────────
def format_ccxt_error(e: Exception) -> str:
    parts = [f"{type(e).__name__}: {e}"]
    for attr in ("url", "status", "status_code"):
        val = getattr(e, attr, None)
        if val:
            parts.append(f"{attr}={val}")
    body = getattr(e, "body", None)
    if body:
        body_str = str(body)
        if len(body_str) > 500:
            body_str = body_str[:500] + "...(truncated)"
        parts.append(f"body={body_str}")
    return " | ".join(parts)

def is_dns_failure(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "could not contact dns servers" in msg
        or "name or service not known" in msg
        or "temporary failure in name resolution" in msg
        or "nodename nor servname provided" in msg
        or "getaddrinfo failed" in msg
        or "clientconnectordnserror" in msg.lower()
    )

async def retry(label: str, fn, *, attempts: int = 6, base_delay_s: float = 2.0, max_delay_s: float = 60.0):
    """
    Retry an async callable with exponential backoff.
    `fn` must be a zero-arg async callable (lambda: coro).
    """
    delay = base_delay_s
    last_exc: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return await fn()
        except (ExchangeNotAvailable, NetworkError, OSError) as e:
            last_exc = e
            dns_hint = ""
            if is_dns_failure(e):
                dns_hint = (
                    " (DNS unreachable: check VPN/proxy/firewall; try DNS 1.1.1.1 or 8.8.8.8)"
                )
            print(
                f"[Network] {label} failed (attempt {i}/{attempts}): {format_ccxt_error(e)}{dns_hint}",
                flush=True,
            )
            if i == attempts:
                raise
            await asyncio.sleep(delay)
            delay = min(max_delay_s, delay * 2)
        except Exception as e:
            # Non-network errors should surface immediately.
            raise e
    # Safety net (should be unreachable)
    raise last_exc if last_exc else RuntimeError(f"{label} failed")

# ────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────
async def fetch_ohlcv(symbol, timeframe, limit=200):
    """Fetch OHLCV data asynchronously"""
    return await retry(
        f"fetch_ohlcv {symbol} {timeframe}",
        lambda: exchange.fetch_ohlcv(symbol, timeframe, limit=limit),
        attempts=5,
    )


async def get_balance():
    """Get USDT free balance"""
    bal = await retry("fetch_balance", lambda: exchange.fetch_balance(), attempts=5)
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
    print(f"Bot started - Dry run: {config.DRY_RUN}", flush=True)
    # Load markets once (avoids repeated market bootstrap calls)
    print("Loading markets...", flush=True)
    await retry("load_markets", lambda: exchange.load_markets(), attempts=10, base_delay_s=2.0, max_delay_s=60.0)
    print("Markets loaded.", flush=True)

    while True:
        for symbol in config.SYMBOLS:
            try:
                now_utc = datetime.now(timezone.utc)
                print(f"\n[{now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}] Scanning {symbol}", flush=True)

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
                    print(f"No setup -> {signal['reasons']}", flush=True)
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
                print(f"Error on {symbol}: {format_ccxt_error(e)}", flush=True)

        await asyncio.sleep(300)  # هر ۵ دقیقه اسکن


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        asyncio.run(exchange.close())