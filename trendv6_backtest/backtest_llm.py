"""
trendv6_backtest_with_llm.py

Separate backtesting script for the v6 signal system WITH the LLM strategy-selection layer.

Important design choices:
- This file does NOT modify trendv6_fixed_signal_validator.py.
- It imports the same v6 trend analysis, strategy engines, and hard validation functions.
- It mirrors the real pipeline:
    historical candles -> trend analysis -> LLM strategy suggestion -> validation -> selected strategy signal -> hard validation -> trade simulation
- It can optionally run a no-LLM baseline in the same run so you can compare the effect of the LLM selector.
- It generates signals candle-by-candle using only candles available up to that candle.
- It tests the signal only on future candles.
- If TP and SL are both touched in the same OHLC candle, it assumes SL first as a conservative rule.

Keep this file in the same folder as:
    trendv6_fixed_signal_validator.py

Run:
    python trendv6_backtest_with_llm.py

Dependencies:
    pip install ccxt pandas ta python-dotenv openai-agents

Environment:
    OPENAI_API_KEY must be set for true LLM backtesting.
    If the key/SDK is missing or the LLM response fails, this script can safely fall back to the rule-based selector and counts those cases.
"""

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import trendv6 as engine
except ImportError as exc:
    raise ImportError(
        "Could not import trendv6_fixed_signal_validator.py. "
        "Put this backtest file in the same folder as trendv6_fixed_signal_validator.py."
    ) from exc


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "backtest_outputs"
DEFAULT_MAX_HOLD_CANDLES = 48
DEFAULT_TARGET_MODE = "tp1"
DEFAULT_ENTRY_MODE = "signal_entry"
PREVENT_OVERLAPPING_TRADES = True
MIN_HISTORY_CANDLES = 240
INTRABAR_AMBIGUITY_MODE = "conservative_sl_first"

# LLM config. Keep the same model family as the real v6 pipeline unless you intentionally change it.
LLM_MODEL = os.getenv("BACKTEST_LLM_MODEL", "gpt-4o-mini")
LLM_SLEEP_SECONDS = float(os.getenv("BACKTEST_LLM_SLEEP_SECONDS", "0"))

# Exact mode means each historical signal point asks the LLM separately.
# This is the closest to your real pipeline, but it can be slower and uses API credits.
# You can enable cache from the prompt when you only want a faster approximation.


# ============================================================
# BASIC HELPERS
# ============================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_symbol_for_filename(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_executable_signal(signal: Dict) -> bool:
    if not signal:
        return False
    if signal.get("signal_status") != "valid_signal":
        return False
    if signal.get("trade_side") not in {"long", "short"}:
        return False
    required_fields = ["entry_price", "stop_loss", "take_profit_1", "take_profit_2"]
    return all(is_number(signal.get(field)) for field in required_fields)


def get_target_price(signal: Dict, target_mode: str) -> Optional[float]:
    if target_mode == "tp1":
        return signal.get("take_profit_1")
    if target_mode == "tp2":
        return signal.get("take_profit_2")
    raise ValueError("target_mode must be 'tp1' or 'tp2'")


def calculate_trade_r(entry: float, stop_loss: float, exit_price: float, side: str) -> float:
    if side == "long":
        risk = entry - stop_loss
        if risk <= 0:
            return 0.0
        return (exit_price - entry) / risk
    if side == "short":
        risk = stop_loss - entry
        if risk <= 0:
            return 0.0
        return (entry - exit_price) / risk
    return 0.0


def calculate_trade_return_pct(entry: float, exit_price: float, side: str) -> float:
    if entry <= 0:
        return 0.0
    if side == "long":
        return (exit_price - entry) / entry * 100
    if side == "short":
        return (entry - exit_price) / entry * 100
    return 0.0


def parse_json_from_text(raw_text: str) -> Dict:
    """Parse JSON robustly from an LLM response."""
    if raw_text is None:
        raise ValueError("LLM returned None")

    text = str(raw_text).strip()
    if not text:
        raise ValueError("LLM returned an empty response")

    # Remove markdown fences if the model ignores instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Extract the first JSON object from extra text.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def normalize_strategy_suggestion(obj: Dict, source: str) -> Dict:
    strategy = obj.get("strategy_style")
    condition = obj.get("market_condition", "unclear")
    confidence = obj.get("confidence", 0.0)
    reason = obj.get("reason", "")

    if strategy not in engine.ALLOWED_STRATEGIES:
        raise ValueError(f"LLM returned unsupported strategy_style: {strategy}")

    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    return {
        "strategy_style": strategy,
        "market_condition": str(condition),
        "confidence": confidence,
        "reason": str(reason),
        "source": source,
    }


# ============================================================
# LLM STRATEGY SELECTOR FOR HISTORICAL BACKTEST
# ============================================================

_STRATEGY_AGENT = None


def get_strategy_agent():
    global _STRATEGY_AGENT
    if _STRATEGY_AGENT is not None:
        return _STRATEGY_AGENT

    if getattr(engine, "Agent", None) is None or getattr(engine, "Runner", None) is None:
        return None

    _STRATEGY_AGENT = engine.Agent(
        name="Historical Backtest Strategy Selector Agent",
        model=LLM_MODEL,
        instructions="""
You are a crypto trading strategy classifier used inside a historical backtest.

Your task:
Suggest the best-fit trading strategy style from a completed trend analysis.

Rules:
- Use only the provided trend analysis output.
- Do not create a new signal.
- Do not approve or reject a trade.
- Do not say BUY, SELL, LONG, SHORT, ENTER, EXIT, NO_TRADE, approved, or rejected.
- Do not recalculate indicators.
- Do not add news, fundamentals, predictions, or extra market data.
- Keep reasoning strictly based on the provided trend analysis.
- If the trend is unclear, weak, or sideways, avoid trend-following strategies.
- In unclear or sideways conditions, prefer range-bound strategy or mixed/confirmation strategy.

Allowed strategy styles:
- pullback continuation strategy
- breakout-and-retest strategy
- range-bound strategy
- momentum continuation strategy
- reversal-confirmation strategy
- mixed/confirmation strategy

Return ONLY valid JSON.

JSON schema:
{
  "strategy_style": "<one allowed strategy style>",
  "market_condition": "<strong_trending | weak_trending | ranging | unclear>",
  "confidence": <float between 0 and 1>,
  "reason": "<short concise explanation based only on provided trend analysis>"
}
""",
    )
    return _STRATEGY_AGENT


def fallback_strategy_suggestion(trend_data: Dict, reason: str) -> Dict:
    fallback = engine.get_rule_based_strategy_suggestion(trend_data)
    fallback["source"] = "rule_based_fallback_after_llm_failure"
    fallback["llm_failure_reason"] = reason
    return fallback


def get_llm_strategy_suggestion_for_backtest(
    trend_output: str,
    trend_data: Dict,
    llm_stats: Dict,
    use_cache: bool = False,
    cache: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """Return an LLM strategy suggestion, with safe fallback and optional cache."""
    if cache is None:
        cache = {}

    # A cache key based on the full textual trend output is conservative.
    # If exact mode is requested, use_cache=False and every signal point calls the LLM.
    cache_key = trend_output
    if use_cache and cache_key in cache:
        llm_stats["cache_hits"] += 1
        return dict(cache[cache_key])

    if getattr(engine, "Agent", None) is None or getattr(engine, "Runner", None) is None:
        llm_stats["fallbacks"] += 1
        suggestion = fallback_strategy_suggestion(trend_data, "OpenAI Agents SDK is not installed")
        if use_cache:
            cache[cache_key] = dict(suggestion)
        return suggestion

    if not os.getenv("OPENAI_API_KEY"):
        llm_stats["fallbacks"] += 1
        suggestion = fallback_strategy_suggestion(trend_data, "OPENAI_API_KEY is not set")
        if use_cache:
            cache[cache_key] = dict(suggestion)
        return suggestion

    agent = get_strategy_agent()
    if agent is None:
        llm_stats["fallbacks"] += 1
        suggestion = fallback_strategy_suggestion(trend_data, "Strategy agent could not be created")
        if use_cache:
            cache[cache_key] = dict(suggestion)
        return suggestion

    agent_input = f"""
Trend output:
{trend_output}

Suggest the best-fit strategy style for this historical candle.
""".strip()

    try:
        if LLM_SLEEP_SECONDS > 0:
            time.sleep(LLM_SLEEP_SECONDS)
        llm_stats["calls_attempted"] += 1
        result = engine.Runner.run_sync(agent, agent_input, max_turns=1)
        raw = getattr(result, "final_output", "")
        parsed = parse_json_from_text(raw)
        suggestion = normalize_strategy_suggestion(parsed, source="llm_backtest")
        llm_stats["calls_successful"] += 1
        if use_cache:
            cache[cache_key] = dict(suggestion)
        return suggestion
    except Exception as exc:
        llm_stats["fallbacks"] += 1
        suggestion = fallback_strategy_suggestion(trend_data, f"LLM strategy suggestion failed: {exc}")
        if use_cache:
            cache[cache_key] = dict(suggestion)
        return suggestion


# ============================================================
# SIGNAL GENERATION PIPELINES
# ============================================================

def select_strategy_without_llm(trend_data: Dict) -> Dict:
    suggestion = engine.get_rule_based_strategy_suggestion(trend_data)
    suggestion["source"] = "rule_based_backtest_no_llm"
    return suggestion


def generate_selected_signal_from_suggestion(
    window_df: pd.DataFrame,
    trend_data: Dict,
    strategy_suggestion: Dict,
) -> Tuple[Dict, Dict]:
    validation = engine.validate_strategy_suggestion(
        trend=trend_data["trend"],
        strength=trend_data["signal_strength"],
        strategy_json=strategy_suggestion,
    )

    selected_strategy = validation.get("final_strategy_style") or strategy_suggestion.get("strategy_style")
    signal = engine.generate_signal(window_df, trend_data, selected_strategy)
    hard_validation = engine.validate_final_signal(signal, trend_data, selected_strategy)

    signal = dict(signal)
    if not hard_validation.get("is_hard_valid"):
        signal["backtest_executable"] = False
        signal["backtest_rejection_reason"] = hard_validation.get("validator_note")
    else:
        signal["backtest_executable"] = True
        signal["backtest_rejection_reason"] = None

    return validation, signal


def generate_selected_signal_with_llm(
    window_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    exchange_name: str,
    llm_stats: Dict,
    use_cache: bool,
    cache: Dict[str, Dict],
) -> Tuple[Dict, Dict, Dict, Dict]:
    trend_data = engine.analyze_trend(window_df)
    trend_output = engine.build_trend_output(symbol, timeframe, exchange_name, trend_data)
    strategy_suggestion = get_llm_strategy_suggestion_for_backtest(
        trend_output=trend_output,
        trend_data=trend_data,
        llm_stats=llm_stats,
        use_cache=use_cache,
        cache=cache,
    )
    validation, signal = generate_selected_signal_from_suggestion(window_df, trend_data, strategy_suggestion)
    return trend_data, strategy_suggestion, validation, signal


def generate_selected_signal_no_llm(window_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict]:
    trend_data = engine.analyze_trend(window_df)
    strategy_suggestion = select_strategy_without_llm(trend_data)
    validation, signal = generate_selected_signal_from_suggestion(window_df, trend_data, strategy_suggestion)
    return trend_data, strategy_suggestion, validation, signal


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_future_trade(
    df: pd.DataFrame,
    signal_index: int,
    signal: Dict,
    target_mode: str = DEFAULT_TARGET_MODE,
    entry_mode: str = DEFAULT_ENTRY_MODE,
    max_hold_candles: int = DEFAULT_MAX_HOLD_CANDLES,
) -> Optional[Dict]:
    if not is_executable_signal(signal):
        return None
    if signal_index + 1 >= len(df):
        return None

    side = signal["trade_side"]
    signal_entry = float(signal["entry_price"])
    stop_loss = float(signal["stop_loss"])
    target_price = get_target_price(signal, target_mode)
    if target_price is None:
        return None
    target_price = float(target_price)

    next_candle = df.iloc[signal_index + 1]

    if entry_mode == "signal_entry":
        entry_price = signal_entry
        entry_time = df.iloc[signal_index]["datetime"] if "datetime" in df.columns else None
        entry_index = signal_index
    elif entry_mode == "next_open":
        entry_price = float(next_candle["open"])
        entry_time = next_candle["datetime"] if "datetime" in df.columns else None
        entry_index = signal_index + 1
    else:
        raise ValueError("entry_mode must be 'signal_entry' or 'next_open'")

    if side == "long":
        if not (stop_loss < entry_price < target_price):
            return None
    elif side == "short":
        if not (target_price < entry_price < stop_loss):
            return None
    else:
        return None

    max_exit_index = min(len(df) - 1, signal_index + max_hold_candles)
    exit_price = None
    exit_reason = None
    exit_index = None

    for j in range(signal_index + 1, max_exit_index + 1):
        candle = df.iloc[j]
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        if side == "long":
            stop_hit = low <= stop_loss
            target_hit = high >= target_price
            if stop_hit and target_hit:
                exit_price = stop_loss
                exit_reason = "sl_hit_intrabar_ambiguous"
                exit_index = j
                break
            if stop_hit:
                exit_price = stop_loss
                exit_reason = "sl_hit"
                exit_index = j
                break
            if target_hit:
                exit_price = target_price
                exit_reason = f"{target_mode}_hit"
                exit_index = j
                break
        elif side == "short":
            stop_hit = high >= stop_loss
            target_hit = low <= target_price
            if stop_hit and target_hit:
                exit_price = stop_loss
                exit_reason = "sl_hit_intrabar_ambiguous"
                exit_index = j
                break
            if stop_hit:
                exit_price = stop_loss
                exit_reason = "sl_hit"
                exit_index = j
                break
            if target_hit:
                exit_price = target_price
                exit_reason = f"{target_mode}_hit"
                exit_index = j
                break

        if j == max_exit_index:
            exit_price = close
            exit_reason = "time_exit"
            exit_index = j
            break

    if exit_price is None or exit_index is None:
        return None

    result_r = calculate_trade_r(entry_price, stop_loss, exit_price, side)
    return_pct = calculate_trade_return_pct(entry_price, exit_price, side)

    signal_time = df.iloc[signal_index]["datetime"] if "datetime" in df.columns else None
    exit_time = df.iloc[exit_index]["datetime"] if "datetime" in df.columns else None

    return {
        "signal_index": signal_index,
        "entry_index": entry_index,
        "exit_index": exit_index,
        "signal_time": str(signal_time),
        "entry_time": str(entry_time),
        "exit_time": str(exit_time),
        "strategy_name": signal.get("strategy_name"),
        "signal_status": signal.get("signal_status"),
        "direction": signal.get("direction"),
        "trade_side": side,
        "entry_mode": entry_mode,
        "target_mode": target_mode,
        "entry_price": round(float(entry_price), 6),
        "signal_entry_price": round(float(signal_entry), 6),
        "stop_loss": round(float(stop_loss), 6),
        "target_price": round(float(target_price), 6),
        "take_profit_1": signal.get("take_profit_1"),
        "take_profit_2": signal.get("take_profit_2"),
        "exit_price": round(float(exit_price), 6),
        "exit_reason": exit_reason,
        "result_r": round(float(result_r), 4),
        "return_pct": round(float(return_pct), 4),
        "risk_reward_1": signal.get("risk_reward_1"),
        "risk_reward_2": signal.get("risk_reward_2"),
        "setup_quality": signal.get("setup_quality"),
        "confidence_score": signal.get("confidence_score"),
        "risk_level": signal.get("risk_level"),
        "reason": signal.get("reason"),
    }


# ============================================================
# BACKTEST ENGINES
# ============================================================

def backtest_selected_strategy_with_llm(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    exchange_name: str,
    target_mode: str = DEFAULT_TARGET_MODE,
    entry_mode: str = DEFAULT_ENTRY_MODE,
    max_hold_candles: int = DEFAULT_MAX_HOLD_CANDLES,
    use_cache: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict]:
    trades = []
    skipped_signals = []
    next_allowed_signal_index = MIN_HISTORY_CANDLES
    cache: Dict[str, Dict] = {}
    llm_stats = {
        "calls_attempted": 0,
        "calls_successful": 0,
        "fallbacks": 0,
        "cache_hits": 0,
    }

    for i in range(MIN_HISTORY_CANDLES, len(df) - 1):
        if PREVENT_OVERLAPPING_TRADES and i < next_allowed_signal_index:
            continue

        window_df = df.iloc[: i + 1].copy().reset_index(drop=True)

        try:
            trend_data, strategy_suggestion, validation, signal = generate_selected_signal_with_llm(
                window_df=window_df,
                symbol=symbol,
                timeframe=timeframe,
                exchange_name=exchange_name,
                llm_stats=llm_stats,
                use_cache=use_cache,
                cache=cache,
            )
        except Exception as exc:
            skipped_signals.append({"signal_index": i, "reason": f"signal_generation_error: {exc}"})
            continue

        selected_strategy = validation.get("final_strategy_style") or strategy_suggestion.get("strategy_style")

        if not signal.get("backtest_executable") or not is_executable_signal(signal):
            skipped_signals.append({
                "signal_index": i,
                "signal_time": str(df.iloc[i]["datetime"]) if "datetime" in df.columns else None,
                "trend": trend_data.get("trend"),
                "strength": trend_data.get("signal_strength"),
                "strategy_suggestion": strategy_suggestion.get("strategy_style"),
                "strategy_source": strategy_suggestion.get("source"),
                "selected_strategy": selected_strategy,
                "signal_status": signal.get("signal_status"),
                "trade_side": signal.get("trade_side"),
                "reason": signal.get("reason"),
                "backtest_rejection_reason": signal.get("backtest_rejection_reason"),
            })
            continue

        trade = simulate_future_trade(
            df=df,
            signal_index=i,
            signal=signal,
            target_mode=target_mode,
            entry_mode=entry_mode,
            max_hold_candles=max_hold_candles,
        )

        if trade is None:
            skipped_signals.append({
                "signal_index": i,
                "signal_time": str(df.iloc[i]["datetime"]) if "datetime" in df.columns else None,
                "trend": trend_data.get("trend"),
                "strength": trend_data.get("signal_strength"),
                "strategy_suggestion": strategy_suggestion.get("strategy_style"),
                "strategy_source": strategy_suggestion.get("source"),
                "selected_strategy": selected_strategy,
                "signal_status": signal.get("signal_status"),
                "trade_side": signal.get("trade_side"),
                "reason": "signal was executable but could not be simulated, likely due to entry/SL/TP geometry after entry mode.",
            })
            continue

        trade["trend"] = trend_data.get("trend")
        trade["trend_strength"] = trend_data.get("signal_strength")
        trade["strategy_suggestion"] = strategy_suggestion.get("strategy_style")
        trade["selected_strategy"] = selected_strategy
        trade["strategy_source"] = strategy_suggestion.get("source")
        trade["llm_confidence"] = strategy_suggestion.get("confidence")
        trades.append(trade)

        if PREVENT_OVERLAPPING_TRADES:
            next_allowed_signal_index = trade["exit_index"] + 1

    return trades, skipped_signals, llm_stats


def backtest_selected_strategy_no_llm(
    df: pd.DataFrame,
    target_mode: str = DEFAULT_TARGET_MODE,
    entry_mode: str = DEFAULT_ENTRY_MODE,
    max_hold_candles: int = DEFAULT_MAX_HOLD_CANDLES,
) -> Tuple[List[Dict], List[Dict]]:
    trades = []
    skipped_signals = []
    next_allowed_signal_index = MIN_HISTORY_CANDLES

    for i in range(MIN_HISTORY_CANDLES, len(df) - 1):
        if PREVENT_OVERLAPPING_TRADES and i < next_allowed_signal_index:
            continue

        window_df = df.iloc[: i + 1].copy().reset_index(drop=True)

        try:
            trend_data, strategy_suggestion, validation, signal = generate_selected_signal_no_llm(window_df)
        except Exception as exc:
            skipped_signals.append({"signal_index": i, "reason": f"signal_generation_error: {exc}"})
            continue

        selected_strategy = validation.get("final_strategy_style") or strategy_suggestion.get("strategy_style")

        if not signal.get("backtest_executable") or not is_executable_signal(signal):
            skipped_signals.append({
                "signal_index": i,
                "signal_time": str(df.iloc[i]["datetime"]) if "datetime" in df.columns else None,
                "trend": trend_data.get("trend"),
                "strength": trend_data.get("signal_strength"),
                "strategy_suggestion": strategy_suggestion.get("strategy_style"),
                "strategy_source": strategy_suggestion.get("source"),
                "selected_strategy": selected_strategy,
                "signal_status": signal.get("signal_status"),
                "trade_side": signal.get("trade_side"),
                "reason": signal.get("reason"),
                "backtest_rejection_reason": signal.get("backtest_rejection_reason"),
            })
            continue

        trade = simulate_future_trade(
            df=df,
            signal_index=i,
            signal=signal,
            target_mode=target_mode,
            entry_mode=entry_mode,
            max_hold_candles=max_hold_candles,
        )

        if trade is None:
            skipped_signals.append({
                "signal_index": i,
                "signal_time": str(df.iloc[i]["datetime"]) if "datetime" in df.columns else None,
                "trend": trend_data.get("trend"),
                "strength": trend_data.get("signal_strength"),
                "strategy_suggestion": strategy_suggestion.get("strategy_style"),
                "strategy_source": strategy_suggestion.get("source"),
                "selected_strategy": selected_strategy,
                "signal_status": signal.get("signal_status"),
                "trade_side": signal.get("trade_side"),
                "reason": "signal was executable but could not be simulated, likely due to entry/SL/TP geometry after entry mode.",
            })
            continue

        trade["trend"] = trend_data.get("trend")
        trade["trend_strength"] = trend_data.get("signal_strength")
        trade["strategy_suggestion"] = strategy_suggestion.get("strategy_style")
        trade["selected_strategy"] = selected_strategy
        trade["strategy_source"] = strategy_suggestion.get("source")
        trades.append(trade)

        if PREVENT_OVERLAPPING_TRADES:
            next_allowed_signal_index = trade["exit_index"] + 1

    return trades, skipped_signals


def backtest_each_strategy_independently(
    df: pd.DataFrame,
    target_mode: str = DEFAULT_TARGET_MODE,
    entry_mode: str = DEFAULT_ENTRY_MODE,
    max_hold_candles: int = DEFAULT_MAX_HOLD_CANDLES,
) -> Dict[str, List[Dict]]:
    trades_by_strategy = {strategy: [] for strategy in engine.ALLOWED_STRATEGIES}
    next_allowed_by_strategy = {strategy: MIN_HISTORY_CANDLES for strategy in engine.ALLOWED_STRATEGIES}

    for i in range(MIN_HISTORY_CANDLES, len(df) - 1):
        window_df = df.iloc[: i + 1].copy().reset_index(drop=True)

        try:
            trend_data = engine.analyze_trend(window_df)
            all_signals = engine.generate_all_strategy_signals(window_df, trend_data)
        except Exception:
            continue

        for strategy_name, signal in all_signals.items():
            if PREVENT_OVERLAPPING_TRADES and i < next_allowed_by_strategy[strategy_name]:
                continue
            validation = engine.validate_final_signal(signal, trend_data, strategy_name)
            if not validation.get("is_hard_valid") or not is_executable_signal(signal):
                continue
            trade = simulate_future_trade(
                df=df,
                signal_index=i,
                signal=signal,
                target_mode=target_mode,
                entry_mode=entry_mode,
                max_hold_candles=max_hold_candles,
            )
            if trade is None:
                continue
            trade["trend"] = trend_data.get("trend")
            trade["trend_strength"] = trend_data.get("signal_strength")
            trade["selected_strategy"] = strategy_name
            trade["strategy_source"] = "independent_strategy_diagnostic"
            trades_by_strategy[strategy_name].append(trade)
            if PREVENT_OVERLAPPING_TRADES:
                next_allowed_by_strategy[strategy_name] = trade["exit_index"] + 1

    return trades_by_strategy


# ============================================================
# METRICS
# ============================================================

def max_drawdown_from_r(trades: List[Dict]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for trade in trades:
        equity += float(trade.get("result_r", 0.0))
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return round(max_dd, 4)


def summarize_trades(trades: List[Dict]) -> Dict:
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate_pct": 0.0,
            "net_r": 0.0,
            "average_r": 0.0,
            "median_r": 0.0,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor": None,
            "max_drawdown_r": 0.0,
            "average_return_pct": 0.0,
        }

    results = [float(t.get("result_r", 0.0)) for t in trades]
    returns_pct = [float(t.get("return_pct", 0.0)) for t in trades]
    wins = [r for r in results if r > 0]
    losses = [r for r in results if r < 0]
    breakeven = [r for r in results if r == 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 0 else None

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate_pct": round(len(wins) / total * 100, 2),
        "net_r": round(sum(results), 4),
        "average_r": round(sum(results) / total, 4),
        "median_r": round(float(pd.Series(results).median()), 4),
        "gross_profit_r": round(gross_profit, 4),
        "gross_loss_r": round(gross_loss, 4),
        "profit_factor": profit_factor,
        "max_drawdown_r": max_drawdown_from_r(trades),
        "average_return_pct": round(sum(returns_pct) / total, 4),
    }


def summarize_by_strategy(trades: List[Dict]) -> Dict[str, Dict]:
    grouped = {}
    for trade in trades:
        strategy = trade.get("strategy_name") or trade.get("selected_strategy") or "unknown"
        grouped.setdefault(strategy, []).append(trade)
    return {strategy: summarize_trades(group_trades) for strategy, group_trades in grouped.items()}


def summarize_by_side(trades: List[Dict]) -> Dict[str, Dict]:
    grouped = {}
    for trade in trades:
        side = trade.get("trade_side", "unknown")
        grouped.setdefault(side, []).append(trade)
    return {side: summarize_trades(group_trades) for side, group_trades in grouped.items()}


def compare_metrics(llm_metrics: Dict, baseline_metrics: Optional[Dict]) -> Optional[Dict]:
    if not baseline_metrics:
        return None
    keys = ["total_trades", "win_rate_pct", "net_r", "average_r", "profit_factor", "max_drawdown_r", "average_return_pct"]
    comparison = {}
    for key in keys:
        a = llm_metrics.get(key)
        b = baseline_metrics.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            comparison[key] = {
                "llm": a,
                "no_llm": b,
                "difference_llm_minus_no_llm": round(a - b, 4),
            }
        else:
            comparison[key] = {"llm": a, "no_llm": b, "difference_llm_minus_no_llm": None}
    return comparison


# ============================================================
# OUTPUT
# ============================================================

def save_backtest_outputs(symbol: str, timeframe: str, report: Dict, trades: List[Dict], skipped: List[Dict], prefix: str) -> Tuple[str, str, str]:
    ensure_output_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_symbol = clean_symbol_for_filename(symbol)
    json_path = os.path.join(OUTPUT_DIR, f"{prefix}_report_{safe_symbol}_{timeframe}_{stamp}.json")
    trades_csv_path = os.path.join(OUTPUT_DIR, f"{prefix}_trades_{safe_symbol}_{timeframe}_{stamp}.csv")
    skipped_csv_path = os.path.join(OUTPUT_DIR, f"{prefix}_skipped_{safe_symbol}_{timeframe}_{stamp}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(engine.make_json_safe(report), f, indent=2)
    pd.DataFrame(trades).to_csv(trades_csv_path, index=False)
    pd.DataFrame(skipped).to_csv(skipped_csv_path, index=False)
    return json_path, trades_csv_path, skipped_csv_path


def print_summary(report: Dict, json_path: str, trades_csv_path: str, skipped_csv_path: str) -> None:
    print("\n====================")
    print("LLM BACKTEST SUMMARY")
    print("====================")
    print(f"Symbol: {report['symbol']}")
    print(f"Timeframe: {report['timeframe']}")
    print(f"Exchange: {report['exchange']}")
    print(f"Execution Mode: {report['execution_mode']}")
    print(f"Target Mode: {report['target_mode']}")
    print(f"Entry Mode: {report['entry_mode']}")
    print(f"Max Hold Candles: {report['max_hold_candles']}")
    print(f"LLM Model: {report['llm_config']['model']}")
    print(f"LLM Cache Enabled: {report['llm_config']['use_cache']}")
    print(f"LLM Stats: {report['llm_strategy_backtest']['llm_stats']}")
    print("--------------------")
    metrics = report["llm_strategy_backtest"]["metrics"]
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("--------------------")
    print("By Strategy:")
    for strategy, metrics in report["llm_strategy_backtest"].get("by_strategy", {}).items():
        print(f"\n{strategy}")
        print(f"  trades: {metrics['total_trades']}")
        print(f"  win_rate_pct: {metrics['win_rate_pct']}")
        print(f"  net_r: {metrics['net_r']}")
        print(f"  profit_factor: {metrics['profit_factor']}")
        print(f"  max_drawdown_r: {metrics['max_drawdown_r']}")
    if report.get("comparison_llm_vs_no_llm"):
        print("--------------------")
        print("LLM vs No-LLM difference:")
        for key, value in report["comparison_llm_vs_no_llm"].items():
            print(f"{key}: {value}")
    print("--------------------")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved trades CSV: {trades_csv_path}")
    print(f"Saved skipped signals CSV: {skipped_csv_path}")
    print("====================")


# ============================================================
# MAIN
# ============================================================

def main():
    raw_symbol = input("Symbol (e.g. BTC/USDT, ETH, ETHUSDT): ").strip().upper()
    timeframe = input("Timeframe (e.g. 15m,1h,4h,1d): ").strip()

    raw_limit = input("Candle limit for backtest (default 1000): ").strip()
    limit = int(raw_limit) if raw_limit else 1000

    target_mode = input("Target mode tp1/tp2 (default tp1): ").strip().lower() or DEFAULT_TARGET_MODE
    if target_mode not in {"tp1", "tp2"}:
        raise ValueError("Target mode must be tp1 or tp2")

    entry_mode = input("Entry mode signal_entry/next_open (default signal_entry): ").strip().lower() or DEFAULT_ENTRY_MODE
    if entry_mode not in {"signal_entry", "next_open"}:
        raise ValueError("Entry mode must be signal_entry or next_open")

    raw_max_hold = input(f"Max hold candles (default {DEFAULT_MAX_HOLD_CANDLES}): ").strip()
    max_hold_candles = int(raw_max_hold) if raw_max_hold else DEFAULT_MAX_HOLD_CANDLES

    use_cache = input("Use LLM cache? exact comparison = n, faster approximate = y (default n): ").strip().lower() == "y"
    run_no_llm_baseline = input("Also run no-LLM baseline comparison? (y/n, default y): ").strip().lower()
    run_no_llm_baseline = False if run_no_llm_baseline == "n" else True
    run_independent = input("Also run independent all-strategy diagnostic? (y/n, default n): ").strip().lower() == "y"

    symbol = engine.normalize_symbol(raw_symbol)
    raw_df, exchange_name = engine.fetch_ohlcv_dataframe(symbol=symbol, timeframe=timeframe, limit=limit)
    df = engine.add_indicators(raw_df)

    if len(df) <= MIN_HISTORY_CANDLES + 10:
        raise RuntimeError(
            f"Not enough candles after indicator warmup. Have {len(df)}, need more than {MIN_HISTORY_CANDLES + 10}. "
            "Increase candle limit."
        )

    llm_trades, llm_skipped, llm_stats = backtest_selected_strategy_with_llm(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        exchange_name=exchange_name,
        target_mode=target_mode,
        entry_mode=entry_mode,
        max_hold_candles=max_hold_candles,
        use_cache=use_cache,
    )

    baseline_report = None
    baseline_metrics = None
    if run_no_llm_baseline:
        baseline_trades, baseline_skipped = backtest_selected_strategy_no_llm(
            df=df,
            target_mode=target_mode,
            entry_mode=entry_mode,
            max_hold_candles=max_hold_candles,
        )
        baseline_metrics = summarize_trades(baseline_trades)
        baseline_report = {
            "metrics": baseline_metrics,
            "by_strategy": summarize_by_strategy(baseline_trades),
            "by_side": summarize_by_side(baseline_trades),
            "completed_trades_count": len(baseline_trades),
            "skipped_signals_count": len(baseline_skipped),
        }

    independent_report = None
    if run_independent:
        trades_by_strategy = backtest_each_strategy_independently(
            df=df,
            target_mode=target_mode,
            entry_mode=entry_mode,
            max_hold_candles=max_hold_candles,
        )
        independent_report = {strategy: summarize_trades(strategy_trades) for strategy, strategy_trades in trades_by_strategy.items()}

    llm_metrics = summarize_trades(llm_trades)
    report = {
        "created_at_utc": now_utc_iso(),
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": exchange_name,
        "execution_mode": "selected_strategy_with_llm_selector",
        "target_mode": target_mode,
        "entry_mode": entry_mode,
        "max_hold_candles": max_hold_candles,
        "prevent_overlapping_trades": PREVENT_OVERLAPPING_TRADES,
        "minimum_history_candles": MIN_HISTORY_CANDLES,
        "candles_after_indicator_warmup": len(df),
        "llm_config": {
            "model": LLM_MODEL,
            "use_cache": use_cache,
            "sleep_seconds_between_calls": LLM_SLEEP_SECONDS,
            "exact_mode_note": "If use_cache is false, every historical signal point asks the LLM separately.",
        },
        "important_note": (
            "This backtest uses the same v6 trend and signal functions, with the LLM strategy selector added. "
            "Signals are generated candle-by-candle using only candles available up to that point. "
            "If LLM is unavailable or returns invalid output, the script falls back to the rule-based selector and counts it in llm_stats. "
            "Intrabar TP/SL ambiguity is handled conservatively with SL first."
        ),
        "llm_strategy_backtest": {
            "metrics": llm_metrics,
            "by_strategy": summarize_by_strategy(llm_trades),
            "by_side": summarize_by_side(llm_trades),
            "completed_trades_count": len(llm_trades),
            "skipped_signals_count": len(llm_skipped),
            "llm_stats": llm_stats,
        },
        "no_llm_baseline_backtest": baseline_report,
        "comparison_llm_vs_no_llm": compare_metrics(llm_metrics, baseline_metrics),
        "independent_all_strategy_diagnostic": independent_report,
        "disclaimer": "Educational backtest only. Not financial advice. Past performance does not guarantee future results.",
    }

    json_path, trades_csv_path, skipped_csv_path = save_backtest_outputs(
        symbol=symbol,
        timeframe=timeframe,
        report=report,
        trades=llm_trades,
        skipped=llm_skipped,
        prefix="llm_backtest",
    )
    print_summary(report, json_path, trades_csv_path, skipped_csv_path)


if __name__ == "__main__":
    main()
