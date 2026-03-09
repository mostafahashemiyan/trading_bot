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
from risk import position_size, validate_llm_levels
from exchange import ExchangeClient
from tracker import TradeTracker
from logger import log, log_system
import config
from datetime import timedelta
load_dotenv()

# ───────────────────────────────────────────────
# Initialize Exchange + Tracker
# ───────────────────────────────────────────────

exchange = ExchangeClient()
tracker = TradeTracker()

# NEW: cooldown tracker
last_trade_time = {}

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

async def reconcile_open_trades():
    """
    Reconcile locally tracked open trades with actual exchange positions.
    If a tracked trade no longer has an exchange position, mark it closed.
    """
    open_trades = tracker.get_open()

    for trade in open_trades:
        symbol = trade.get("symbol")
        if not symbol:
            continue

        pos = exchange.get_position(symbol)

        # Position still exists on exchange -> keep trade open
        if pos is not None:
            continue

        # No live position found -> trade was closed externally / by TP / by SL
        last_price = exchange.get_last_price(symbol)

        entry = trade.get("entry")
        sl = trade.get("sl")
        tp = trade.get("tp")
        side = trade.get("side")

        outcome = "CLOSED"
        pnl = None

        try:
            if (
                last_price is not None and
                entry is not None and
                sl is not None and
                tp is not None and
                side in {"LONG", "SHORT"}
            ):
                entry = float(entry)
                sl = float(sl)
                tp = float(tp)
                last_price = float(last_price)
                size = float(trade.get("size", 0) or 0)

                if side == "LONG":
                    if last_price >= tp:
                        outcome = "TP"
                    elif last_price <= sl:
                        outcome = "SL"
                    else:
                        outcome = "CLOSED"

                    pnl = (last_price - entry) * size if size > 0 else None

                elif side == "SHORT":
                    if last_price <= tp:
                        outcome = "TP"
                    elif last_price >= sl:
                        outcome = "SL"
                    else:
                        outcome = "CLOSED"

                    pnl = (entry - last_price) * size if size > 0 else None
        except Exception:
            outcome = "CLOSED"
            pnl = None

        closed = tracker.close_trade(
            symbol=symbol,
            outcome=outcome,
            exit_price=last_price,
            pnl=pnl,
        )

        if closed:
            log("trade_closed", {
                "symbol": symbol,
                "outcome": outcome,
                "exit_price": last_price,
                "pnl": pnl
            }, symbol)


def timeframe_to_seconds(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    return 0


# ───────────────────────────────────────────────
# MAIN SCAN LOOP
# ───────────────────────────────────────────────
async def scan_symbol(symbol: str):
    """
    Handle scanning, signal building, LLM verification, and order execution for one symbol.
    """
    log("scan_start", {"msg": "Scanning symbol"}, symbol)

    # ------------------------------------------------------------
    # Skip if tracker already has an open trade for this symbol
    # ------------------------------------------------------------
    tracked_open = tracker.get_open()

    if any(t.get("symbol") == symbol and t.get("status") == "OPEN" for t in tracked_open):
        log("skip_open_trade", {
            "msg": "Symbol already has an open tracked trade"
        }, symbol)
        return

    # ------------------------------------------------------------
    # Fetch market data
    # ------------------------------------------------------------
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

    # ───────────────────────────────────────────────
    # COOLDOWN CHECK
    # ───────────────────────────────────────────────

    if symbol in last_trade_time:

        tf_seconds = timeframe_to_seconds(config.HIGH_TF)
        cooldown_seconds = config.COOLDOWN_CANDLES * tf_seconds

        time_since = datetime.now(timezone.utc).timestamp() - last_trade_time[symbol]

        if time_since < cooldown_seconds:

            log("cooldown", {
                "msg": "Symbol in cooldown",
                "seconds_remaining": int(cooldown_seconds - time_since)
            }, symbol)

            tracker.save_decision_report(
                symbol=symbol,
                strategy_signal=signal,
                decision={
                    "decision": "NO_TRADE",
                    "side": None,
                    "confidence": 0,
                    "reason": "Cooldown active",
                },
                trade_result=None,
            )

            return


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

    if llm_res.get("latency_ms", 0) > int(config.LLM_TIMEOUT_SECONDS * 1000):
        log("llm_timeout", {
            "msg": "LLM response exceeded timeout threshold",
            "latency_ms": llm_res.get("latency_ms", 0),
            "timeout_seconds": config.LLM_TIMEOUT_SECONDS
        }, symbol)

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
            trade_result={
                "latency_ms": llm_res.get("latency_ms", 0)
            },
        )
        return

        # ------------------------------------------------------------
    # Safe LLM SL/TP override validation
    # ------------------------------------------------------------
    llm_sl = llm_res.get("suggested_sl")
    llm_tp = llm_res.get("suggested_tp")

    if llm_sl is not None and llm_tp is not None:
        validation = validate_llm_levels(
            side=side,
            entry=entry,
            sl=llm_sl,
            tp=llm_tp
        )

        if validation["valid"]:
            signal["sl"] = float(llm_sl)
            signal["tp"] = float(llm_tp)

            log("llm_levels_applied", {
                "entry": entry,
                "side": side,
                "new_sl": signal["sl"],
                "new_tp": signal["tp"],
                "rr": validation["rr"],
                "reason": validation["reason"]
            }, symbol)
        else:
            log("llm_levels_rejected", {
                "entry": entry,
                "side": side,
                "suggested_sl": llm_sl,
                "suggested_tp": llm_tp,
                "reason": validation["reason"]
            }, symbol)


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
    # ------------------------------------------------------------
    # Spread filter
    # ------------------------------------------------------------
    spread_info = exchange.get_spread_info(symbol)
    spread_pct = spread_info.get("spread_pct")

    if spread_pct is None:
        log("error", {"msg": "Spread unavailable"}, symbol)

        tracker.save_decision_report(
            symbol=symbol,
            strategy_signal=signal,
            decision={
                "decision": "NO_TRADE",
                "side": side,
                "confidence": llm_res.get("confidence", signal.get("confidence", 0)),
                "reason": "Spread unavailable",
            },
            trade_result=spread_info,
        )
        return

    if spread_pct > config.MAX_SPREAD_PCT:
        log("spread_rejected", {
            "bid": spread_info.get("bid"),
            "ask": spread_info.get("ask"),
            "spread": spread_info.get("spread"),
            "spread_pct": spread_pct,
            "max_spread_pct": config.MAX_SPREAD_PCT
        }, symbol)

        tracker.save_decision_report(
            symbol=symbol,
            strategy_signal=signal,
            decision={
                "decision": "NO_TRADE",
                "side": side,
                "confidence": llm_res.get("confidence", signal.get("confidence", 0)),
                "reason": f"Spread too wide ({spread_pct:.5f})",
            },
            trade_result=spread_info,
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
    last_trade_time[symbol] = datetime.now(timezone.utc).timestamp()


# ───────────────────────────────────────────────
# GLOBAL LOOP
# ───────────────────────────────────────────────
async def main_loop():

    log_system(f"Bot started (DRY_RUN={config.DRY_RUN})")

    while True:

        await reconcile_open_trades()

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