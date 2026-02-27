# bot.py

import asyncio
from exchange import fetch_ohlcv, execute_trade, cancel_orphaned_orders
from indicators import prepare_df
from strategy import trend_pullback_signal
from llm_gatekeeper import llm_decide
from tracker import save_trade, generate_report
from logger import log
from config import SYMBOLS, DRY_RUN, RISK_PER_TRADE
from datetime import datetime

# Global list to track active trade IDs for the Safety Check
active_trades = [] 

async def analyze_symbol(symbol: str) -> dict:
    # --------------------------------------------------
    # Fetch market data
    # --------------------------------------------------
    ohlcv_1h = fetch_ohlcv(symbol, "1h")
    ohlcv_15m = fetch_ohlcv(symbol, "15m")
    ohlcv_5m = fetch_ohlcv(symbol, "5m")

    df_1h = prepare_df(ohlcv_1h)
    df_15m = prepare_df(ohlcv_15m)
    df_5m = prepare_df(ohlcv_5m)

    # --------------------------------------------------
    # Strategy
    # --------------------------------------------------
    signal = trend_pullback_signal(df_1h, df_15m, df_5m)

    # --------------------------------------------------
    # No setup → log and return
    # --------------------------------------------------
    if not signal["setup"]:
        result = {
            "symbol": symbol,
            "strategy_signal": signal,
            "decision": {
                "decision": "NO_TRADE",
                "side": None,
                "confidence": 0,
                "reason": "Strategy conditions not met"
            },
            "timestamp": datetime.utcnow().isoformat()
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
        "entry": signal["entry"],
        "stop": signal["stop"],
        "tp": signal["tp"],
        "rr": signal["rr"],
        "timeframes": {
            "1h": {
                "ema50": round(df_1h["ema50"].iloc[-1], 2),
                "ema200": round(df_1h["ema200"].iloc[-1], 2),
            },
            "15m": {
                "rsi": round(df_15m["rsi"].iloc[-1], 2)
            },
            "5m": {
                "close": round(df_5m["close"].iloc[-1], 2),
                "ema20": round(df_5m["ema20"].iloc[-1], 2)
            }
        }
    }

    # --------------------------------------------------
    # LLM decision
    # --------------------------------------------------
    decision = llm_decide(features)

    result = {
        "symbol": symbol,
        "strategy_signal": signal,
        "decision": decision,
        "timestamp": datetime.utcnow().isoformat()
    }
    log(symbol, result)

    # --------------------------------------------------
    # Execution (Gated by LLM)
    # --------------------------------------------------
    decision_status = decision.get("decision", "NO_TRADE")

    if decision_status == "TRADE":
        llm_side = decision.get("side", "").upper()
        side = "buy" if llm_side == "LONG" else "sell" if llm_side == "SHORT" else None
        
        if not side:
            print(f"Skipping: Invalid side {llm_side}")
            return result

        if DRY_RUN:
            print(f"[DRY RUN] {side.upper()} {symbol} at {signal['entry']}")
        else:
            # LIVE EXECUTION
            execution_data = execute_trade(
                symbol=symbol,
                side=side,
                entry_price=signal["entry"],
                stop_loss=signal["stop"],
                take_profit=signal["tp"],
                risk_per_trade=RISK_PER_TRADE
            )

            if execution_data:
                # 1. Add to Safety Tracker for OCO cleanup
                execution_data['symbol'] = symbol
                active_trades.append(execution_data)

                # 2. Log success to tracker
                save_trade({
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "type": "LIVE",
                    "entry": signal["entry"],
                    "stop": signal["stop"],
                    "tp": signal["tp"],
                    "size": execution_data.get("amount", "N/A"),
                    "order_ids": execution_data
                })

    return result

async def run_loop():
    print("🟢 Multi-symbol bot started (60s interval)")
    generate_report()

    while True:
        # 🛡️ SAFETY CHECK: Cleanup orphaned orders (SL/TP)
        if active_trades:
            print(f"🔍 Checking {len(active_trades)} active trades for cleanup...")
            for trade in active_trades[:]:
                # If SL hit -> Cancel TP
                if cancel_orphaned_orders(trade['symbol'], trade['sl_id'], trade['tp_id']):
                    active_trades.remove(trade)
                # If TP hit -> Cancel SL
                elif cancel_orphaned_orders(trade['symbol'], trade['tp_id'], trade['sl_id']):
                    active_trades.remove(trade)

        start = datetime.utcnow().isoformat()
        print(f"\n⏱ Scan started at {start}")

        tasks = [analyze_symbol(symbol) for symbol in SYMBOLS]
        results = await asyncio.gather(*tasks)

        for r in results:
            dec = r.get('decision', {}).get('decision', 'N/A')
            print(f"{r['symbol']} → {dec}")

        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")