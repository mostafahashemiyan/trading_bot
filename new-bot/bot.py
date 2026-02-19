import asyncio
from datetime import datetime

from exchange import fetch_ohlcv, execute_trade
from indicators import prepare_df
from strategy import safe_short_signal
from llm_gatekeeper import llm_decide
from tracker import save_trade, generate_report
from logger import log
from config import SYMBOLS, DRY_RUN, RISK_PER_TRADE, HTF_TIMEFRAME


# Keep per-symbol state in-memory (for min-gap between trades)
LAST_TRADE_BAR: dict[str, int] = {}


def _map_side_to_exchange(side: str | None) -> str | None:
    """Map strategy side to exchange order side."""
    if side is None:
        return None
    s = side.upper()
    if s == "LONG":
        return "buy"
    if s == "SHORT":
        return "sell"
    return None


async def analyze_symbol(symbol: str) -> dict:
    # --------------------------------------------------
    # Fetch market data
    # --------------------------------------------------
    ohlcv_1h = fetch_ohlcv(symbol, "1h")
    ohlcv_htf = fetch_ohlcv(symbol, HTF_TIMEFRAME)
    ohlcv_5m = fetch_ohlcv(symbol, "5m")

    df_1h = prepare_df(ohlcv_1h)
    df_htf = prepare_df(ohlcv_htf)
    df_5m = prepare_df(ohlcv_5m)

    # --------------------------------------------------
    # Strategy (NEW)
    # --------------------------------------------------
    last_trade_bar = LAST_TRADE_BAR.get(symbol)
    signal = safe_short_signal(df_htf, df_5m, last_trade_bar=last_trade_bar)

    # --------------------------------------------------
    # No setup → log and return
    # --------------------------------------------------
    if not signal.get("setup"):
        result = {
            "symbol": symbol,
            "strategy_signal": signal,
            "decision": {
                "decision": "NO_TRADE",
                "side": None,
                "confidence": 0,
                "reason": "Strategy conditions not met",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        log(symbol, result)
        return result

    # --------------------------------------------------
    # LLM features
    # --------------------------------------------------
    features = {
        "symbol": symbol,
        "setup_detected": signal["setup"],
        "strategy_reasons": signal["reasons"],
        "trend": signal["trend"],
        "side": signal.get("side"),
        "entry": signal["entry"],
        "stop": signal["stop"],
        "tp": signal["tp"],
        "rr": signal["rr"],
        "timeframes": {
            "1h": {
                "ema50": round(float(df_1h["ema50"].iloc[-1]), 6),
                "ema200": round(float(df_1h["ema200"].iloc[-1]), 6),
            },
            f"{HTF_TIMEFRAME}": {
                "ema50": round(float(df_htf["ema50"].iloc[-1]), 6),
                "ema200": round(float(df_htf["ema200"].iloc[-1]), 6),
                "rsi": round(float(df_htf["rsi"].iloc[-1]), 2),
            },
            "5m": {
                "close": round(float(df_5m["close"].iloc[-1]), 6),
                "ema20": round(float(df_5m["ema20"].iloc[-1]), 6),
                "ema50": round(float(df_5m["ema50"].iloc[-1]), 6),
                "atr14": round(float(df_5m["atr14"].iloc[-1]), 6),
            },
        },
    }

    # --------------------------------------------------
    # LLM decision
    # --------------------------------------------------
    decision = llm_decide(features)

    # Hard constraint: this strategy is SHORT-only
    if decision.get("decision") == "TRADE" and decision.get("side") not in ("SHORT", "short"):
        decision = {
            "decision": "NO_TRADE",
            "side": None,
            "confidence": 0,
            "reason": "Strategy is SHORT-only; LLM did not approve SHORT.",
        }

    result = {
        "symbol": symbol,
        "strategy_signal": signal,
        "decision": decision,
        "timestamp": datetime.utcnow().isoformat(),
    }

    log(symbol, result)

    # --------------------------------------------------
    # Execution (still gated)
    # --------------------------------------------------
    if decision.get("decision") == "TRADE":
        exch_side = _map_side_to_exchange(decision.get("side") or signal.get("side"))

        if not exch_side:
            exch_side = "sell"  # SHORT fallback

        if DRY_RUN:
            print(f"[DRY RUN] Would execute {exch_side.upper()} on {symbol}")
            print(f"Entry: {signal['entry']} | SL: {signal['stop']} | TP: {signal['tp']}")

            save_trade(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "side": exch_side,
                    "type": "PAPER",
                    "entry": signal["entry"],
                    "stop": signal["stop"],
                    "tp": signal["tp"],
                    "size": "N/A",
                }
            )

        else:
            print(f"[LIVE] Initiating Trade on {symbol}...")

            execution_data = execute_trade(
                symbol=symbol,
                side=exch_side,
                entry_price=signal["entry"],
                stop_loss=signal["stop"],
                take_profit=signal["tp"],
                risk_per_trade=RISK_PER_TRADE,
            )

            if execution_data:
                save_trade(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": symbol,
                        "side": exch_side,
                        "type": "LIVE",
                        "entry": signal["entry"],
                        "stop": signal["stop"],
                        "tp": signal["tp"],
                        "size": execution_data["amount"],
                        "order_ids": execution_data,
                    }
                )

        # Update min-gap state only if we actually took the trade (paper or live)
        if signal.get("bar_index") is not None:
            LAST_TRADE_BAR[symbol] = int(signal["bar_index"])

        return result

    return result


async def run_loop():
    print("🟢 Multi-symbol bot started (60s interval)")

    # Startup report
    generate_report()

    while True:
        start = datetime.utcnow().isoformat()
        print(f"\n⏱ Scan started at {start}")

        tasks = [analyze_symbol(symbol) for symbol in SYMBOLS]
        results = await asyncio.gather(*tasks)

        for r in results:
            dec = r.get("decision", {}).get("decision", "N/A")
            print(f"{r['symbol']} → {dec}")

        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(run_loop())
