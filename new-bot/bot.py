"""
LIVE TRADING BOT (Professional 2026)
------------------------------------
Supports:
- Multi-symbol scanning
- MTF strategy
- LLM gatekeeper
- KuCoin Futures execution
- DRY RUN mode for safety
"""

import asyncio
from datetime import datetime, timezone

from dotenv import load_dotenv

from indicators import prepare_df
from strategy import trend_pullback_signal
from llm_gatekeeper import llm_decide
from risk import position_size
from exchange import ExchangeClient
from tracker import TradeTracker
from logger import log, log_system
import config

load_dotenv()

# ───────────────────────────────────────────────
# Initialize Exchange + Tracker
# ───────────────────────────────────────────────
exchange = ExchangeClient()
tracker = TradeTracker()


# ───────────────────────────────────────────────
# Helper: Safe OHLCV fetch with retries
# ───────────────────────────────────────────────
async def fetch_ohlcv(symbol, timeframe, limit=200):
    for attempt in range(config.RETRY_ATTEMPTS):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception:
            await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
    return []


# ───────────────────────────────────────────────
# MAIN SCAN LOOP
# ───────────────────────────────────────────────
async def scan_symbol(symbol: str):
    """
    Handle scanning, signal building, LLM verification, and order execution for one symbol.
    """

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Scanning {symbol}...")

    # Fetch data for all 3 TFs
    h_tf = await fetch_ohlcv(symbol, config.HIGH_TF, 250)
    m_tf = await fetch_ohlcv(symbol, config.MEDIUM_TF, 250)
    l_tf = await fetch_ohlcv(symbol, config.LOW_TF, 250)

    df_h = prepare_df(h_tf)
    df_m = prepare_df(m_tf)
    df_l = prepare_df(l_tf)

    if df_h.empty or df_m.empty or df_l.empty:
        log("error", {"msg": "Insufficient data"}, symbol)

        tracker.save_decision_report(
            symbol=symbol,
            strategy_signal={
                "setup": False,
                "side": None,
                "entry": None,
                "sl": None,
                "tp": None,
                "confidence": 0,
                "reasons": ["Insufficient data"],
            },
            decision={
                "decision": "NO_TRADE",
                "side": None,
                "confidence": 0,
                "reason": "Insufficient data",
            },
            trade_result=None,
        )
        return

    # ------------------------------------------------------------
    # Generate raw signal from strategy
    # ------------------------------------------------------------
    signal = trend_pullback_signal(df_h, df_m, df_l)

    if not signal["setup"]:
        log("signal", {"msg": "No setup", "reasons": signal["reasons"]}, symbol)

        tracker.save_decision_report(
            symbol=symbol,
            strategy_signal=signal,
            decision={
                "decision": "NO_TRADE",
                "side": None,
                "confidence": 0,
                "reason": "Strategy conditions not met",
            },
            trade_result=None,
        )
        return

    entry = signal["entry"]
    side = signal["side"]

    log("signal", signal, symbol)

    # ------------------------------------------------------------
    # LLM Gatekeeper
    # ------------------------------------------------------------
    llm_res = llm_decide(signal, symbol)
    log("llm_decision", llm_res, symbol)

    if llm_res["decision"] != "APPROVE":
        tracker.save_decision_report(
            symbol=symbol,
            strategy_signal=signal,
            decision={
                "decision": "NO_TRADE",
                "side": side,
                "confidence": llm_res.get("confidence", 0),
                "reason": llm_res.get("reason", "LLM rejected trade"),
            },
            trade_result=None,
        )
        return

    # ------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------
    balance = exchange.get_balance_usdt()
    size = position_size(balance, entry, signal["sl"])

    if size <= 0 or size * entry < config.MIN_NOTIONAL_VALUE:
        log("error", {"msg": "Position too small"}, symbol)

        tracker.save_decision_report(
            symbol=symbol,
            strategy_signal=signal,
            decision={
                "decision": "NO_TRADE",
                "side": side,
                "confidence": llm_res.get("confidence", signal.get("confidence", 0)),
                "reason": "Position too small",
            },
            trade_result={
                "balance": balance,
                "size": size,
                "min_notional": config.MIN_NOTIONAL_VALUE,
            },
        )
        return

    # ------------------------------------------------------------
    # Determine buy/sell string for exchange
    # ------------------------------------------------------------
    side_exec = "buy" if side == "LONG" else "sell"

    # ------------------------------------------------------------
    # Execute trade (LIVE or DRY RUN)
    # ------------------------------------------------------------
    trade_res = exchange.execute_trade(
        symbol=symbol,
        side=side_exec,
        base_amount=size,
        sl=signal["sl"],
        tp=signal["tp"]
    )

    log("trade_open", {
        "side": side,
        "entry": entry,
        "sl": signal["sl"],
        "tp": signal["tp"],
        "size": size,
        "exchange_res": trade_res
    }, symbol)

    decision_status = "TRADE_OPENED"
    decision_reason = llm_res.get("reason", "Approved and executed")

    if isinstance(trade_res, dict) and trade_res.get("error"):
        decision_status = "TRADE_FAILED"
        decision_reason = trade_res.get("error", "Trade execution failed")

    tracker.save_decision_report(
        symbol=symbol,
        strategy_signal=signal,
        decision={
            "decision": decision_status,
            "side": side,
            "confidence": llm_res.get("confidence", signal.get("confidence", 0)),
            "reason": decision_reason,
        },
        trade_result=trade_res,
    )

    if isinstance(trade_res, dict) and trade_res.get("error"):
        return

    # Track open trades
    tracker.open_trade(
        symbol=symbol,
        side=side,
        entry=entry,
        sl=signal["sl"],
        tp=signal["tp"],
        size=size,
        entry_ts=datetime.now().timestamp(),
    )


# ───────────────────────────────────────────────
# GLOBAL LOOP
# ───────────────────────────────────────────────
async def main_loop():

    log_system(f"Bot started (DRY_RUN={config.DRY_RUN})")

    while True:

        if not config.SYMBOLS:
            print("⚠ No symbols set in config.SYMBOLS")
            await asyncio.sleep(5)
            continue

        tasks = [scan_symbol(sym) for sym in config.SYMBOLS]
        await asyncio.gather(*tasks)

        await asyncio.sleep(config.SCAN_DELAY_SECONDS)


# ───────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        log_system(f"CRASH: {e}")
        raise