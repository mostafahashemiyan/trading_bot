"""
WHAT IS NEW IN v7
------------------
1. Multi-timeframe orchestration:
   - Higher timeframe (HTF) decides bias + strategy.
   - Lower timeframe (LTF) decides entry trigger.
   - Single-timeframe still supported (LTF == HTF).

2. Strategy tuning based on documented backtest results:
   - Pullback continuation (the winner): candle confirmation, volume filter,
     "too close to S/R" guard, slightly higher minimum quality.
   - Momentum continuation (weak as execution): stricter (5/5 quality required,
     extra ATR slope and RSI-not-extreme checks). By default DEMOTED to
     confirmation-only; toggleable.
   - Range-bound: hard-gated to Sideways/Range only.
   - Breakout-and-retest (was producing zero trades): rules relaxed so it can
     actually fire — wider retest tolerance, lookback window for the breakout,
     volume expansion check.
   - Reversal-confirmation: stricter (4/5), only allowed when trend is not
     strongly extended.

3. Position sizing module:
   - account balance, risk per trade %, max daily loss in R, max open trades.
   - Computes USD risk, units, and position size before any "execution".

4. Unified backtester (no-LLM and with-LLM) in this same file:
   - Fees and slippage are modelled.
   - Walk-forward style: signals are generated using only candles available up
     to that point, and tested against future candles.
   - Sharpe-like, Sortino-like, Calmar, expectancy, max drawdown.

5. argparse CLI with three subcommands:
   - analyze   — single-shot live analysis
   - backtest  — historical simulation (no-LLM or with-LLM)
   - compare   — runs both and reports the difference
"""

from __future__ import annotations

import os
import re
import sys
import time
import math
import json
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import ta
import ccxt
import pandas as pd
from dotenv import load_dotenv

try:
    from agents import Agent, Runner
except ImportError:
    Agent = None
    Runner = None


load_dotenv()


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "outputs"
BACKTEST_DIR = "backtest_outputs"

# Default candle limits.
DEFAULT_LIVE_LIMIT = 1000
DEFAULT_BACKTEST_LIMIT = 1500

# Execution modes:
# - "selected_only": only the selected strategy is used for the final signal.
# - "confirmation_mode": selected strategy is used for the final signal,
#   other valid strategies only increase confidence.
# - "multi_strategy_mode": report all valid strategies, but still do not auto-execute.
EXECUTION_MODE = "confirmation_mode"

# Strategy gating based on documented backtest results.
# Momentum continuation was net-negative in the documented runs, so by default
# it is NOT allowed to be the selected execution strategy. Flip this to True if
# you have re-validated it on a wider dataset.
ALLOW_MOMENTUM_AS_EXECUTION = False

# Hard signal floor — every executable signal must clear this risk/reward.
MINIMUM_RR = 1.50  # was 1.20 in v6; stricter floor improves expectancy

MAX_CONFIDENCE_AFTER_CONFIRMATION = 0.95
ENABLE_LLM_SIGNAL_REVIEW = True

# LLM configuration.
LLM_MODEL = os.getenv("TRENDV7_LLM_MODEL", "gpt-4o-mini")
LLM_SLEEP_SECONDS_BACKTEST = float(os.getenv("TRENDV7_LLM_SLEEP", "0"))

# Backtest defaults.
DEFAULT_MAX_HOLD_CANDLES = 48
DEFAULT_TARGET_MODE = "tp1"  # "tp1" or "tp2"
DEFAULT_ENTRY_MODE = "next_open"  # "signal_entry" or "next_open"
INTRABAR_AMBIGUITY_MODE = "conservative_sl_first"
PREVENT_OVERLAPPING_TRADES = True
MIN_HISTORY_CANDLES = 240

# Fees and slippage applied per round-trip trade in backtests.
DEFAULT_FEE_PCT_PER_SIDE = 0.05  # 0.05% per side -> 0.1% round trip
DEFAULT_SLIPPAGE_PCT = 0.02  # 0.02% one-time slippage on entry

# Position sizing defaults (used by the live analyze command).
DEFAULT_ACCOUNT_BALANCE_USD = 10_000.0
DEFAULT_RISK_PER_TRADE_PCT = 1.0
DEFAULT_MAX_DAILY_LOSS_R = -3.0
DEFAULT_MAX_OPEN_TRADES = 1


# ============================================================
# BASIC HELPERS
# ============================================================


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_symbol_for_filename(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def normalize_symbol(value: str) -> str:
    """Accept common user inputs and convert them to a ccxt spot symbol."""
    value = value.replace("-", "/").strip().upper()
    if "/" in value:
        return value
    if value.endswith("USDT"):
        return f"{value[:-4]}/USDT"
    return f"{value}/USDT"


def make_json_safe(obj: Any) -> Any:
    """Convert pandas/numpy scalar values to plain Python for json.dumps."""
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(item) for item in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def safe_quality_score(points: float, max_points: float) -> float:
    if max_points == 0:
        return 0.0
    return round(points / max_points, 2)


def infer_direction_from_trend(trend: str) -> str:
    if "Bullish" in trend:
        return "bullish"
    if "Bearish" in trend:
        return "bearish"
    return "neutral"


def trade_side_from_direction(direction: str) -> str:
    if direction == "bullish":
        return "long"
    if direction == "bearish":
        return "short"
    return "none"


def round_price(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def parse_timeframe_minutes(timeframe: str) -> int:
    """Return the size of one candle in minutes — used for HTF/LTF comparison."""
    timeframe = timeframe.strip().lower()
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 60 * 24
    if unit == "w":
        return value * 60 * 24 * 7
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def timeframe_to_milliseconds(timeframe: str) -> int:
    return parse_timeframe_minutes(timeframe) * 60 * 1000


# ============================================================
# DATA FUNCTIONS
# ============================================================


def fetch_ohlcv_dataframe(
    symbol: str, timeframe: str, limit: int = 1000
) -> Tuple[pd.DataFrame, str]:
    """
    Fetch OHLCV candles with pagination.

    Tries Binance first, KuCoin as fallback. Returns (df, exchange_id).
    """
    exchanges = [
        ccxt.binance({"enableRateLimit": True}),
        ccxt.kucoin({"enableRateLimit": True}),
    ]
    last_error: Optional[Exception] = None

    for exchange in exchanges:
        try:
            exchange.load_markets()
            tf_ms = timeframe_to_milliseconds(timeframe)
            now_ms = exchange.milliseconds()

            buffer_candles = 5
            since = now_ms - ((limit + buffer_candles) * tf_ms)

            all_ohlcv: List[List[float]] = []
            batch_limit = 1000

            while len(all_ohlcv) < limit:
                remaining = limit - len(all_ohlcv)
                current_limit = min(batch_limit, remaining)

                batch = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=current_limit,
                )
                if not batch:
                    break

                if all_ohlcv:
                    last_ts = all_ohlcv[-1][0]
                    batch = [row for row in batch if row[0] > last_ts]
                if not batch:
                    break

                all_ohlcv.extend(batch)
                since = batch[-1][0] + tf_ms

                if len(batch) < current_limit:
                    break

            if not all_ohlcv:
                raise RuntimeError(f"No candles returned for {symbol}")

            all_ohlcv = all_ohlcv[-limit:]

            df = pd.DataFrame(
                all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df = (
                df.drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            return df, exchange.id

        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not fetch candles for {symbol}: {last_error}")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA20/50/200, RSI14, MACD, ATR14, and volume MA20."""
    df = df.copy()

    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["atr"] = ta.volatility.average_true_range(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )

    # Volume context — needed for the new pullback/breakout volume filters.
    df["volume_ma20"] = df["volume"].rolling(window=20, min_periods=5).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]

    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        raise RuntimeError("Not enough candles after indicator warmup.")

    return df


# ============================================================
# MARKET STRUCTURE
# ============================================================


def get_recent_structure(df: pd.DataFrame, lookback: int = 8) -> Dict[str, Any]:
    highs = df["high"].tail(lookback).values
    lows = df["low"].tail(lookback).values

    hh = sum(highs[i] > highs[i - 1] for i in range(1, len(highs)))
    hl = sum(lows[i] > lows[i - 1] for i in range(1, len(lows)))
    lh = sum(highs[i] < highs[i - 1] for i in range(1, len(highs)))
    ll = sum(lows[i] < lows[i - 1] for i in range(1, len(lows)))
    span = lookback - 1

    return {
        "higher_high_count": hh,
        "higher_low_count": hl,
        "lower_high_count": lh,
        "lower_low_count": ll,
        "structure_lookback": span,
        "bullish_structure": hh >= 4 and hl >= 4,
        "bearish_structure": lh >= 4 and ll >= 4,
    }


def find_swing_points(
    df: pd.DataFrame, left: int = 3, right: int = 3, lookback: int = 60
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Identify pivot highs and lows using a (left, right) fractal rule.
    A candle at index i is a swing high if its high is the maximum of the
    window [i-left, i+right]. Same idea for swing lows.

    Returns:
        {"swing_highs": [(index, price), ...], "swing_lows": [(index, price), ...]}
    """
    sub = df.tail(lookback + left + right).reset_index(drop=True)
    highs = sub["high"].values
    lows = sub["low"].values
    n = len(sub)

    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []

    for i in range(left, n - right):
        window_high = highs[i - left : i + right + 1]
        window_low = lows[i - left : i + right + 1]
        if highs[i] == max(window_high) and highs[i] > 0:
            swing_highs.append((i, float(highs[i])))
        if lows[i] == min(window_low) and lows[i] > 0:
            swing_lows.append((i, float(lows[i])))

    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def get_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    """
    Support/resistance with a swing-aware preference:
    use the most recent valid swing high/low when available, otherwise fall
    back to the rolling min/max of the last `lookback` candles.
    """
    recent = df.tail(lookback)
    swings = find_swing_points(df, left=3, right=3, lookback=lookback)

    if swings["swing_highs"]:
        resistance = max(price for _, price in swings["swing_highs"])
    else:
        resistance = float(recent["high"].max())

    if swings["swing_lows"]:
        support = min(price for _, price in swings["swing_lows"])
    else:
        support = float(recent["low"].min())

    midpoint = (support + resistance) / 2
    last_close = float(df.iloc[-1]["close"])
    range_pct = (resistance - support) / last_close * 100 if last_close > 0 else 0.0

    return {
        "support": support,
        "resistance": resistance,
        "midpoint": midpoint,
        "range_percent": range_pct,
    }


def recent_swing_low(df: pd.DataFrame, lookback: int = 12) -> float:
    return float(df["low"].tail(lookback).min())


def recent_swing_high(df: pd.DataFrame, lookback: int = 12) -> float:
    return float(df["high"].tail(lookback).max())


def too_close_to_opposite_level(
    direction: str,
    price: float,
    atr: float,
    sr: Dict[str, float],
    min_atr_distance: float = 1.5,
) -> bool:
    """
    Pullback/momentum guard: avoid going LONG right under resistance,
    or SHORT right above support. Distance is measured in ATRs.
    """
    if atr <= 0:
        return False
    if direction == "bullish":
        distance = (sr["resistance"] - price) / atr
        return distance < min_atr_distance
    if direction == "bearish":
        distance = (price - sr["support"]) / atr
        return distance < min_atr_distance
    return False


# ============================================================
# TREND ANALYSIS
# ============================================================


def analyze_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """Weighted-vote trend analyzer based on EMA stack, MACD, RSI, slope, structure."""
    latest = df.iloc[-1]

    price = float(latest["close"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])
    ema200 = float(latest["ema200"])
    rsi = float(latest["rsi"])
    macd_value = float(latest["macd"])
    macd_signal_value = float(latest["macd_signal"])
    atr = float(latest["atr"])

    recent_highs = df["high"].tail(8).values
    recent_lows = df["low"].tail(8).values
    recent_closes = df["close"].tail(8).values

    hh = sum(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
    hl = sum(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))
    lh = sum(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))
    ll = sum(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))
    span = len(recent_highs) - 1

    higher_highs = hh >= 4
    higher_lows = hl >= 4
    lower_highs = lh >= 4
    lower_lows = ll >= 4

    close_change_atr = (recent_closes[-1] - recent_closes[0]) / atr if atr > 0 else 0
    ema_distance = abs(ema50 - ema200) / price * 100 if price > 0 else 0
    ema20_50_distance = abs(ema20 - ema50) / price * 100 if price > 0 else 0
    atr_percent = atr / price * 100 if price > 0 else 0
    recent_range = (
        (df["high"].tail(20).max() - df["low"].tail(20).min()) / price * 100
        if price > 0
        else 0
    )

    bull_score = 0
    bear_score = 0
    notes: List[str] = []

    def vote(condition: bool, bull_note: str, bear_note: str) -> None:
        nonlocal bull_score, bear_score
        if condition:
            bull_score += 1
            notes.append(bull_note)
        else:
            bear_score += 1
            notes.append(bear_note)

    vote(price > ema20, "price is above EMA20", "price is below EMA20")
    vote(ema20 > ema50, "EMA20 is above EMA50", "EMA20 is below EMA50")
    vote(ema50 > ema200, "EMA50 is above EMA200", "EMA50 is below EMA200")
    vote(macd_value > macd_signal_value, "MACD is above signal", "MACD is below signal")

    if rsi > 53:
        bull_score += 1
        notes.append(f"RSI {rsi:.1f} has bullish bias")
    elif rsi < 47:
        bear_score += 1
        notes.append(f"RSI {rsi:.1f} has bearish bias")
    else:
        notes.append(f"RSI {rsi:.1f} is neutral")

    if close_change_atr > 0.35:
        bull_score += 1
        notes.append(f"recent closes rose {close_change_atr:.2f} ATR")
    elif close_change_atr < -0.35:
        bear_score += 1
        notes.append(f"recent closes fell {close_change_atr:.2f} ATR")
    else:
        notes.append(f"recent close slope is flat ({close_change_atr:.2f} ATR)")

    if higher_highs and higher_lows:
        bull_score += 1
        notes.append(f"market structure leans higher ({hh}/{span} HH, {hl}/{span} HL)")
    elif lower_highs and lower_lows:
        bear_score += 1
        notes.append(f"market structure leans lower ({lh}/{span} LH, {ll}/{span} LL)")
    else:
        notes.append(
            f"market structure is mixed ({hh}/{span} HH, {hl}/{span} HL, {lh}/{span} LH, {ll}/{span} LL)"
        )

    is_sideways = (
        ema20_50_distance < 0.25
        and ema_distance < 1
        and 47 <= rsi <= 53
        and abs(close_change_atr) < 0.35
        and recent_range < max(2.5, atr_percent * 4)
    )

    trend = "Unclear"
    if is_sideways:
        trend = "Sideways/Range"
    elif bull_score >= 5 and bull_score >= bear_score + 2:
        trend = "Bullish"
    elif bear_score >= 5 and bear_score >= bull_score + 2:
        trend = "Bearish"
    elif bull_score > bear_score:
        trend = "Weak Bullish / Not confirmed"
    elif bear_score > bull_score:
        trend = "Weak Bearish / Not confirmed"

    leading = max(bull_score, bear_score)
    if leading >= 6:
        signal_strength = "Strong"
    elif leading >= 5:
        signal_strength = "Moderate"
    else:
        signal_strength = "Weak"

    if trend == "Bullish" and ema50 < ema200:
        signal_strength = "Moderate"
    if trend == "Bearish" and ema20 > ema50:
        signal_strength = "Moderate"

    sr = get_support_resistance(df)

    return {
        "trend": trend,
        "signal_strength": signal_strength,
        "direction": infer_direction_from_trend(trend),
        "bull_score": bull_score,
        "bear_score": bear_score,
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "macd_value": macd_value,
        "macd_signal": macd_signal_value,
        "atr": atr,
        "atr_percent": atr_percent,
        "recent_range": recent_range,
        "close_change_atr": close_change_atr,
        "higher_high_count": hh,
        "higher_low_count": hl,
        "lower_high_count": lh,
        "lower_low_count": ll,
        "structure_lookback": span,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "range_midpoint": sr["midpoint"],
        "notes": notes,
    }


def build_trend_output(
    symbol: str, timeframe: str, exchange_name: str, td: Dict[str, Any]
) -> str:
    notes = td["notes"]
    return f"""
Symbol: {symbol}
Timeframe: {timeframe}
Exchange: {exchange_name}
Trend: {td["trend"]}
Strength: {td["signal_strength"]}
Score: {td["bull_score"]} bullish / {td["bear_score"]} bearish
Price: {td["price"]:.6f}
EMA20 / EMA50 / EMA200: {td["ema20"]:.6f} / {td["ema50"]:.6f} / {td["ema200"]:.6f}
RSI: {td["rsi"]:.2f}
MACD / Signal: {td["macd_value"]:.6f} / {td["macd_signal"]:.6f}
ATR: {td["atr"]:.6f} ({td["atr_percent"]:.2f}% of price)
Recent range: {td["recent_range"]:.2f}%
Support / Resistance: {td["support"]:.6f} / {td["resistance"]:.6f}
Reasons:
{chr(10).join(f"- {n}" for n in notes)}
""".strip()


def print_trend_report(
    symbol: str, timeframe: str, exchange_name: str, td: Dict[str, Any]
) -> None:
    print("\n====================")
    print("TREND REPORT")
    print("====================")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Exchange: {exchange_name}")
    print(f"Trend: {td['trend']}")
    print(f"Strength: {td['signal_strength']}")
    print(f"Score: {td['bull_score']} bullish / {td['bear_score']} bearish")
    print("--------------------")
    print(f"Price: {td['price']:.6f}")
    print(
        f"EMA20 / EMA50 / EMA200: {td['ema20']:.6f} / {td['ema50']:.6f} / {td['ema200']:.6f}"
    )
    print(f"RSI: {td['rsi']:.2f}")
    print(f"MACD / Signal: {td['macd_value']:.6f} / {td['macd_signal']:.6f}")
    print(f"ATR: {td['atr']:.6f} ({td['atr_percent']:.2f}% of price)")
    print(f"Recent range: {td['recent_range']:.2f}%")
    print(f"Support / Resistance: {td['support']:.6f} / {td['resistance']:.6f}")
    print("--------------------")
    print("Reasons:")
    for n in td["notes"]:
        print(f"- {n}")


# ============================================================
# STRATEGY SELECTION
# ============================================================

ALLOWED_STRATEGIES = {
    "pullback continuation strategy",
    "breakout-and-retest strategy",
    "range-bound strategy",
    "momentum continuation strategy",
    "reversal-confirmation strategy",
    "mixed/confirmation strategy",
}


def get_rule_based_strategy_suggestion(trend_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback strategy selector when the LLM is not used.

    v7 tuning, based on documented backtest results:
    - Sideways/Range -> range-bound
    - Weak/Unclear   -> mixed/confirmation (do not force a trend trade)
    - Strong trend   -> pullback continuation by default (the documented winner).
                        Momentum is only used if explicitly enabled.
    - Moderate trend -> pullback continuation
    """
    trend = trend_data["trend"]
    strength = trend_data["signal_strength"]

    if "Sideways" in trend:
        return {
            "strategy_style": "range-bound strategy",
            "market_condition": "ranging",
            "confidence": 0.72,
            "reason": "Market is classified as sideways/ranging.",
            "source": "rule_based_fallback",
        }

    if "Weak" in trend or "Unclear" in trend or strength == "Weak":
        return {
            "strategy_style": "mixed/confirmation strategy",
            "market_condition": "weak_trending" if "Weak" in trend else "unclear",
            "confidence": 0.60,
            "reason": "Trend is weak or unclear, so confirmation is required before any directional setup.",
            "source": "rule_based_fallback",
        }

    if strength == "Strong" and ALLOW_MOMENTUM_AS_EXECUTION:
        return {
            "strategy_style": "momentum continuation strategy",
            "market_condition": "strong_trending",
            "confidence": 0.72,
            "reason": "Strong trend; momentum continuation is allowed by current config.",
            "source": "rule_based_fallback",
        }

    return {
        "strategy_style": "pullback continuation strategy",
        "market_condition": (
            "strong_trending" if strength == "Strong" else "weak_trending"
        ),
        "confidence": 0.70,
        "reason": "Directional trend without forcing momentum; pullback continuation is preferred.",
        "source": "rule_based_fallback",
    }


def get_llm_strategy_suggestion(
    trend_output: str,
    trend_data: Dict[str, Any],
    *,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """LLM strategy classifier with rule-based fallback."""
    if (
        not use_llm
        or Agent is None
        or Runner is None
        or not os.getenv("OPENAI_API_KEY")
    ):
        return get_rule_based_strategy_suggestion(trend_data)

    agent = Agent(
        name="Trend Strategy Suggestion Agent",
        model=LLM_MODEL,
        instructions="""
You are a crypto trading strategy classifier.

Task:
Suggest the best-fit trading strategy style from a completed trend analysis.

Rules:
- Use only the provided trend analysis output.
- Do not create a new signal.
- Do not approve or reject a trade.
- Do not say BUY, SELL, LONG, SHORT, ENTER, EXIT, NO_TRADE, approved, or rejected.
- Do not recalculate indicators.
- Do not add news, fundamentals, predictions, or extra market data.
- If trend is weak, unclear, or sideways, prefer range-bound or mixed/confirmation.
- Prefer pullback continuation over momentum continuation in trending conditions.

Allowed strategy styles:
- pullback continuation strategy
- breakout-and-retest strategy
- range-bound strategy
- momentum continuation strategy
- reversal-confirmation strategy
- mixed/confirmation strategy

Return ONLY valid JSON, no markdown fences.

JSON schema:
{
  "strategy_style": "<one allowed strategy style>",
  "market_condition": "<strong_trending | weak_trending | ranging | unclear>",
  "confidence": <float between 0 and 1>,
  "reason": "<short concise explanation based only on provided trend analysis>"
}
""",
    )

    agent_input = f"""
Trend output:
{trend_output}

Suggest the best-fit strategy style for this trend.
""".strip()

    try:
        result = Runner.run_sync(agent, agent_input, max_turns=1)
        raw = getattr(result, "final_output", "")
        parsed = extract_json_object(raw)
        # Validate the strategy name; fall back if the model invented one.
        if parsed.get("strategy_style") not in ALLOWED_STRATEGIES:
            return get_rule_based_strategy_suggestion(trend_data)
        parsed["source"] = "llm"
        return parsed
    except Exception:
        return get_rule_based_strategy_suggestion(trend_data)


def validate_strategy_suggestion(
    trend: str,
    strength: str,
    strategy_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Rule-based strategy validator. Forces consistency with the trend regime.

    v7 additions:
    - Range-bound is only allowed when trend is Sideways/Range.
    - Momentum continuation is redirected to pullback continuation when
      ALLOW_MOMENTUM_AS_EXECUTION is False.
    """
    strategy = strategy_json.get("strategy_style")
    confidence = strategy_json.get("confidence", 0)

    warnings: List[str] = []
    result = {
        "is_valid": True,
        "warnings": warnings,
        "final_strategy_style": strategy,
        "validator_note": "",
    }

    if strategy not in ALLOWED_STRATEGIES:
        result["is_valid"] = False
        warnings.append("Strategy style is not in the allowed list.")
        result["final_strategy_style"] = "mixed/confirmation strategy"

    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        result["is_valid"] = False
        warnings.append("Confidence must be a number between 0 and 1.")

    weak_or_unclear = (
        "Weak" in trend
        or "Unclear" in trend
        or "Sideways" in trend
        or strength == "Weak"
    )

    trend_following = {
        "pullback continuation strategy",
        "breakout-and-retest strategy",
        "momentum continuation strategy",
    }

    if weak_or_unclear and strategy in trend_following:
        result["is_valid"] = False
        warnings.append(
            "Weak/unclear/sideways trend should not use a trend-following strategy."
        )
        result["final_strategy_style"] = "mixed/confirmation strategy"

    if "Sideways" in trend:
        result["final_strategy_style"] = "range-bound strategy"
        if strategy != "range-bound strategy":
            warnings.append(
                "Sideways condition was redirected to range-bound strategy."
            )

    if strategy == "range-bound strategy" and "Sideways" not in trend:
        warnings.append(
            "Range-bound was redirected — only valid in Sideways/Range markets."
        )
        result["final_strategy_style"] = (
            "pullback continuation strategy"
            if not weak_or_unclear
            else "mixed/confirmation strategy"
        )

    if (
        result["final_strategy_style"] == "momentum continuation strategy"
        and not ALLOW_MOMENTUM_AS_EXECUTION
    ):
        warnings.append(
            "Momentum continuation was redirected to pullback continuation "
            "(documented backtest performance was net-negative; toggle ALLOW_MOMENTUM_AS_EXECUTION to re-enable)."
        )
        result["final_strategy_style"] = "pullback continuation strategy"

    if result["is_valid"]:
        result["validator_note"] = (
            "Strategy suggestion is consistent with trend output."
        )
    else:
        result["validator_note"] = (
            "Strategy suggestion conflicts with rule-based validation."
        )

    return result


# ============================================================
# STRATEGY TEMPLATES (informational)
# ============================================================

STRATEGY_TEMPLATES = {
    "pullback continuation strategy": {
        "type": "trend_following",
        "best_for": ["strong_trending", "moderate_trending"],
        "confirmation_needed": [
            "trend remains aligned",
            "price pulls back near EMA20 or EMA50",
            "candle confirms pullback completion",
            "momentum improves again",
            "price is not too close to opposite S/R level",
        ],
        "risk_note": "Avoid if structure is mixed or trend strength is weak.",
    },
    "breakout-and-retest strategy": {
        "type": "trend_following",
        "best_for": ["strong_trending", "expansion"],
        "confirmation_needed": [
            "price breaks a recent range/high/low",
            "retest holds within ATR tolerance",
            "volume expansion on breakout",
            "trend alignment",
        ],
        "risk_note": "Avoid during low range or unclear structure.",
    },
    "range-bound strategy": {
        "type": "mean_reversion",
        "best_for": ["ranging"],
        "confirmation_needed": [
            "trend classified as Sideways/Range",
            "price near range support or resistance",
            "RSI is neutral",
            "EMA distances are compressed",
        ],
        "risk_note": "Avoid if breakout pressure increases.",
    },
    "momentum continuation strategy": {
        "type": "momentum",
        "best_for": ["strong_trending"],
        "confirmation_needed": [
            "strong trend score",
            "MACD confirms momentum",
            "market structure is clean",
            "RSI not extreme",
            "ATR slope is positive",
        ],
        "risk_note": "Documented to be net-negative as execution. Use only as confirmation by default.",
    },
    "reversal-confirmation strategy": {
        "type": "reversal",
        "best_for": ["exhaustion", "failed continuation"],
        "confirmation_needed": [
            "trend weakens but not strongly extended",
            "RSI extremes reversing",
            "MACD turning",
            "price reclaiming EMA20",
            "structure starts improving",
        ],
        "risk_note": "Requires strong confirmation; reversal trades are higher risk.",
    },
    "mixed/confirmation strategy": {
        "type": "confirmation_waiting",
        "best_for": ["weak_trending", "unclear"],
        "confirmation_needed": [
            "wait for cleaner market structure",
            "wait for EMA alignment or rejection",
            "wait for MACD confirmation",
        ],
        "risk_note": "Used when signals conflict and the trend is not confirmed.",
    },
}


def get_strategy_template(strategy_name: str) -> Dict[str, Any]:
    template = STRATEGY_TEMPLATES.get(strategy_name)
    if template is None:
        return {
            "template_found": False,
            "error": "No template found for this strategy style.",
        }
    return {
        "template_found": True,
        "strategy_style": strategy_name,
        "template": template,
    }


# ============================================================
# SIGNAL / RISK HELPERS
# ============================================================


def calculate_risk_reward(
    direction: str,
    entry: Optional[float],
    stop_loss: Optional[float],
    take_profit: Optional[float],
) -> Optional[float]:
    if entry is None or stop_loss is None or take_profit is None:
        return None
    if direction == "bullish":
        risk = entry - stop_loss
        reward = take_profit - entry
    elif direction == "bearish":
        risk = stop_loss - entry
        reward = entry - take_profit
    else:
        return None
    if risk <= 0:
        return None
    return round(reward / risk, 2)


def build_signal_result(
    strategy_name: str,
    signal_status: str,
    direction: str = "neutral",
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit_1: Optional[float] = None,
    take_profit_2: Optional[float] = None,
    setup_quality: float = 0.0,
    confidence_score: float = 0.0,
    risk_level: str = "unknown",
    reason: str = "",
    notes: Optional[List[str]] = None,
    entry_zone_low: Optional[float] = None,
    entry_zone_high: Optional[float] = None,
) -> Dict[str, Any]:
    rr1 = calculate_risk_reward(direction, entry_price, stop_loss, take_profit_1)
    rr2 = calculate_risk_reward(direction, entry_price, stop_loss, take_profit_2)
    trade_side = trade_side_from_direction(direction)

    return {
        "strategy_name": strategy_name,
        "signal_status": signal_status,
        "direction": direction,
        "trade_side": trade_side,
        "entry_price": round_price(entry_price),
        "entry_zone_low": round_price(entry_zone_low),
        "entry_zone_high": round_price(entry_zone_high),
        "stop_loss": round_price(stop_loss),
        "take_profit_1": round_price(take_profit_1),
        "take_profit_2": round_price(take_profit_2),
        "risk_reward_1": rr1,
        "risk_reward_2": rr2,
        "setup_quality": round(float(setup_quality), 2),
        "confidence_score": round(float(confidence_score), 2),
        "risk_level": risk_level,
        "invalidation_level": round_price(stop_loss),
        "reason": reason,
        "notes": notes or [],
    }


def no_signal(
    strategy_name: str, reason: str, notes: Optional[List[str]] = None
) -> Dict[str, Any]:
    return build_signal_result(
        strategy_name=strategy_name,
        signal_status="no_valid_signal",
        direction="neutral",
        reason=reason,
        notes=notes or [],
    )


def make_r_multiple_targets(
    direction: str, entry: float, stop_loss: float, r1: float = 1.5, r2: float = 2.5
) -> Tuple[Optional[float], Optional[float]]:
    if direction == "bullish":
        risk = entry - stop_loss
        return entry + risk * r1, entry + risk * r2
    if direction == "bearish":
        risk = stop_loss - entry
        return entry - risk * r1, entry - risk * r2
    return None, None


def valid_minimum_rr(signal: Dict[str, Any], minimum_rr: float = MINIMUM_RR) -> bool:
    rr1 = signal.get("risk_reward_1")
    rr2 = signal.get("risk_reward_2")
    if rr1 is None or rr2 is None:
        return False
    return rr1 >= minimum_rr or rr2 >= minimum_rr


def is_valid_trade_signal(signal: Dict[str, Any]) -> bool:
    return signal.get("signal_status") == "valid_signal"


def get_strategy_min_quality(strategy_name: str) -> float:
    """Minimum setup_quality required to take an executable trade per strategy."""
    return {
        "pullback continuation strategy": 0.70,  # was 0.65; lift the bar slightly
        "breakout-and-retest strategy": 0.70,  # was 0.75; relaxed so it can fire
        "range-bound strategy": 0.70,
        "momentum continuation strategy": 1.00,  # only fires if 5/5 — virtually demoted
        "reversal-confirmation strategy": 0.75,  # was 0.65; stricter
        "mixed/confirmation strategy": 1.00,
    }.get(strategy_name, 0.70)


# ============================================================
# CANDLE-LEVEL CONFIRMATION (new in v7)
# ============================================================


def is_bullish_confirmation_candle(df: pd.DataFrame) -> bool:
    """
    A pullback in an uptrend is considered "complete" when the latest closed
    candle prints a bullish reaction:
    - close above open (a green bar), AND
    - close above the previous bar's high, OR a bullish engulfing.
    """
    if len(df) < 3:
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_bar = float(last["close"]) > float(last["open"])
    if not bullish_bar:
        return False

    breaks_prev_high = float(last["close"]) > float(prev["high"])
    bullish_engulfing = (
        float(prev["close"]) < float(prev["open"])
        and float(last["close"]) > float(prev["open"])
        and float(last["open"]) < float(prev["close"])
    )
    return breaks_prev_high or bullish_engulfing


def is_bearish_confirmation_candle(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]

    bearish_bar = float(last["close"]) < float(last["open"])
    if not bearish_bar:
        return False

    breaks_prev_low = float(last["close"]) < float(prev["low"])
    bearish_engulfing = (
        float(prev["close"]) > float(prev["open"])
        and float(last["close"]) < float(prev["open"])
        and float(last["open"]) > float(prev["close"])
    )
    return breaks_prev_low or bearish_engulfing


def has_acceptable_volume(df: pd.DataFrame, min_ratio: float = 0.8) -> bool:
    """Volume of the latest candle must be at least min_ratio * 20-bar average."""
    if "volume_ratio" not in df.columns:
        return True
    last = df.iloc[-1]
    ratio = last.get("volume_ratio")
    if ratio is None or not is_number(ratio):
        return True
    return float(ratio) >= min_ratio


# ============================================================
# STRATEGY SIGNAL ENGINES (v7 — tuned)
# ============================================================


def generate_pullback_continuation_signal(
    df: pd.DataFrame, trend_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PULLBACK CONTINUATION — the documented best-performing strategy.

    v7 enhancements:
    - Candle confirmation: latest bar must print a bullish/bearish reaction.
    - "Too close to opposite S/R" guard: no longs right under resistance,
      no shorts right above support.
    - Volume filter: latest bar must show acceptable participation.
    - Slightly higher minimum quality.
    """
    strategy = "pullback continuation strategy"
    latest = df.iloc[-1]
    direction = trend_data["direction"]

    if direction not in ("bullish", "bearish"):
        return no_signal(
            strategy, "Trend direction is not clear enough for pullback continuation."
        )

    price = float(latest["close"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])
    ema200 = float(latest["ema200"])
    rsi = float(latest["rsi"])
    macd_val = float(latest["macd"])
    macd_sig = float(latest["macd_signal"])
    atr = float(latest["atr"])
    structure = get_recent_structure(df)
    sr = get_support_resistance(df)

    dist_ema20_atr = abs(price - ema20) / atr if atr > 0 else 999
    dist_ema50_atr = abs(price - ema50) / atr if atr > 0 else 999
    nearest_ema_atr = min(dist_ema20_atr, dist_ema50_atr)

    near_ema20 = abs(price - ema20) / price < 0.006 and dist_ema20_atr <= 0.75
    near_ema50 = abs(price - ema50) / price < 0.010 and dist_ema50_atr <= 0.90
    pullback_area = near_ema20 or near_ema50
    entry_zone_low = min(ema20, ema50)
    entry_zone_high = max(ema20, ema50)

    if direction == "bullish":
        trend_alignment = ema20 > ema50
        ema200_supportive = price > ema200
        momentum_ok = rsi > 50 and macd_val > macd_sig
        structure_ok = structure["bullish_structure"]
        candle_confirm = is_bullish_confirmation_candle(df)
        swing_low = recent_swing_low(df)
        stop_candidates = [
            swing_low - atr * 0.10,
            price - atr * 1.50,
            ema50 - atr * 0.50,
        ]
        valid_stops = [x for x in stop_candidates if x < price]
        stop_loss = max(valid_stops) if valid_stops else price - atr * 1.5
    else:
        trend_alignment = ema20 < ema50 or price < ema20
        ema200_supportive = price < ema200
        momentum_ok = rsi < 50 and macd_val < macd_sig
        structure_ok = structure["bearish_structure"]
        candle_confirm = is_bearish_confirmation_candle(df)
        swing_high = recent_swing_high(df)
        stop_candidates = [
            swing_high + atr * 0.10,
            price + atr * 1.50,
            ema50 + atr * 0.50,
        ]
        valid_stops = [x for x in stop_candidates if x > price]
        stop_loss = min(valid_stops) if valid_stops else price + atr * 1.5

    volume_ok = has_acceptable_volume(df, min_ratio=0.8)
    s_r_safe = not too_close_to_opposite_level(
        direction, price, atr, sr, min_atr_distance=1.0
    )

    # 6-point quality: trend, ema200, pullback location, momentum, structure, candle confirm.
    # Volume and S/R safe are pass/fail guards, not weighted points.
    points = 0
    points += 1 if trend_alignment else 0
    points += 1 if ema200_supportive else 0
    points += 1 if pullback_area else 0
    points += 1 if momentum_ok else 0
    points += 1 if structure_ok else 0
    points += 1 if candle_confirm else 0
    setup_quality = safe_quality_score(points, 6)

    common_notes = [
        f"near_ema20={near_ema20}",
        f"near_ema50={near_ema50}",
        f"distance_to_ema20_atr={dist_ema20_atr:.2f}",
        f"distance_to_ema50_atr={dist_ema50_atr:.2f}",
        f"trend_alignment={trend_alignment}",
        f"ema200_supportive={ema200_supportive}",
        f"momentum_ok={momentum_ok}",
        f"structure_ok={structure_ok}",
        f"candle_confirm={candle_confirm}",
        f"volume_ok={volume_ok}",
        f"s_r_safe={s_r_safe}",
        f"setup_quality={setup_quality}",
    ]

    if not pullback_area:
        # Waiting state — useful for live alerts.
        return build_signal_result(
            strategy_name=strategy,
            signal_status="waiting_for_pullback",
            direction=direction,
            setup_quality=setup_quality,
            confidence_score=min(0.65, setup_quality * 0.65),
            risk_level="medium" if setup_quality >= 0.75 else "high",
            reason="Trend is directional, but price is not close enough to the EMA pullback entry zone.",
            entry_zone_low=entry_zone_low,
            entry_zone_high=entry_zone_high,
            notes=common_notes,
        )

    if not volume_ok:
        return no_signal(
            strategy,
            "Pullback exists but volume is too thin for a confident entry.",
            common_notes,
        )
    if not s_r_safe:
        return no_signal(
            strategy,
            "Pullback exists but price is too close to the opposite S/R level.",
            common_notes,
        )
    if not candle_confirm:
        return no_signal(
            strategy,
            "Pullback exists but the latest candle has not confirmed direction yet.",
            common_notes,
        )

    if setup_quality < get_strategy_min_quality(strategy):
        return no_signal(
            strategy,
            "Pullback exists, but confirmation quality is not strong enough.",
            common_notes,
        )

    entry = price
    tp1, tp2 = make_r_multiple_targets(direction, entry, stop_loss, r1=1.5, r2=2.5)
    risk_level = "medium" if ema200_supportive and setup_quality >= 0.80 else "high"
    confidence = min(0.90, setup_quality * 0.85 + 0.10)

    return build_signal_result(
        strategy_name=strategy,
        signal_status="valid_signal",
        direction=direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        setup_quality=setup_quality,
        confidence_score=confidence,
        risk_level=risk_level,
        reason="Directional trend with price near EMA pullback area, candle confirmation, and sufficient confluence.",
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        notes=common_notes,
    )


def generate_range_bound_signal(
    df: pd.DataFrame, trend_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    RANGE-BOUND — v7: hard-gated to Sideways/Range trends only.
    """
    strategy = "range-bound strategy"

    if "Sideways" not in trend_data.get("trend", ""):
        return no_signal(
            strategy, "Range-bound strategy only runs in Sideways/Range markets."
        )

    latest = df.iloc[-1]
    price = float(latest["close"])
    rsi = float(latest["rsi"])
    atr = float(latest["atr"])

    sr = get_support_resistance(df, lookback=50)
    support, resistance, midpoint = sr["support"], sr["resistance"], sr["midpoint"]

    near_support = abs(price - support) <= atr * 1.0
    near_resistance = abs(price - resistance) <= atr * 1.0
    inside_range = support <= price <= resistance
    rsi_neutral = 40 <= rsi <= 60

    if not inside_range:
        return no_signal(strategy, "Price is not inside the detected range.")
    if not rsi_neutral:
        return no_signal(strategy, "RSI is not neutral enough for a range-bound setup.")

    if near_support:
        direction = "bullish"
        entry = price
        stop_loss = support - atr * 0.50
        tp1 = midpoint
        tp2 = resistance
        reason = "Price is near range support with neutral RSI."
    elif near_resistance:
        direction = "bearish"
        entry = price
        stop_loss = resistance + atr * 0.50
        tp1 = midpoint
        tp2 = support
        reason = "Price is near range resistance with neutral RSI."
    else:
        return no_signal(
            strategy,
            "Price is inside the range but not close to support or resistance.",
            [
                f"support={support:.6f}",
                f"resistance={resistance:.6f}",
                f"price={price:.6f}",
            ],
        )

    points = 0
    points += 1 if inside_range else 0
    points += 1 if rsi_neutral else 0
    points += 1 if near_support or near_resistance else 0
    points += 1 if sr["range_percent"] < max(4.0, trend_data["atr_percent"] * 6) else 0
    setup_quality = safe_quality_score(points, 4)

    if setup_quality < get_strategy_min_quality(strategy):
        return no_signal(strategy, "Range setup quality is not strong enough.")

    confidence = min(0.85, setup_quality * 0.80 + 0.10)

    return build_signal_result(
        strategy_name=strategy,
        signal_status="valid_signal",
        direction=direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        setup_quality=setup_quality,
        confidence_score=confidence,
        risk_level="medium",
        reason=reason,
        notes=[
            f"support={support:.6f}",
            f"resistance={resistance:.6f}",
            f"midpoint={midpoint:.6f}",
            f"near_support={near_support}",
            f"near_resistance={near_resistance}",
            f"rsi_neutral={rsi_neutral}",
        ],
    )


def generate_breakout_retest_signal(
    df: pd.DataFrame, trend_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    BREAKOUT-AND-RETEST — v7 relaxed so it can actually generate trades.

    Changes vs v6:
    - Looks for a breakout anywhere within the last `breakout_lookback` bars,
      not only the previous bar.
    - Retest tolerance widened to 1.0 ATR (was 0.75).
    - Volume expansion check on the breakout bar.
    """
    strategy = "breakout-and-retest strategy"
    direction = trend_data["direction"]
    if direction not in ("bullish", "bearish"):
        return no_signal(
            strategy, "Trend direction is not clear enough for breakout-and-retest."
        )

    latest = df.iloc[-1]
    price = float(latest["close"])
    atr = float(latest["atr"])
    rsi = float(latest["rsi"])
    macd_val = float(latest["macd"])
    macd_sig = float(latest["macd_signal"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])

    # Look at the prior 20 bars (excluding the most recent two) to set the breakout level,
    # then scan the last 5 bars to see if any of them broke that level.
    history = df.iloc[-25:-2] if len(df) > 25 else df.iloc[:-2]
    breakout_window = df.iloc[-5:-1]  # 4 bars before the current one

    if len(history) < 5 or len(breakout_window) < 1:
        return no_signal(strategy, "Not enough history to detect a breakout.")

    recent_resistance = float(history["high"].max())
    recent_support = float(history["low"].min())

    if direction == "bullish":
        broke_level = bool((breakout_window["close"] > recent_resistance).any())
        retest_near_level = abs(price - recent_resistance) <= atr * 1.0
        trend_ok = ema20 > ema50
        momentum_ok = rsi > 50 and macd_val > macd_sig
        # Volume expansion on the breakout candle.
        if broke_level:
            breakout_idx = breakout_window[
                breakout_window["close"] > recent_resistance
            ].index.max()
            breakout_volume_ratio = (
                float(df.loc[breakout_idx, "volume_ratio"])
                if "volume_ratio" in df.columns
                else 1.0
            )
        else:
            breakout_volume_ratio = 0.0
        volume_expansion = breakout_volume_ratio >= 1.1
        entry = price
        stop_loss = recent_resistance - atr * 0.75
        tp1, tp2 = make_r_multiple_targets(direction, entry, stop_loss, r1=1.5, r2=2.5)
        level_name = "resistance_retest_as_support"
    else:
        broke_level = bool((breakout_window["close"] < recent_support).any())
        retest_near_level = abs(price - recent_support) <= atr * 1.0
        trend_ok = ema20 < ema50 or price < ema20
        momentum_ok = rsi < 50 and macd_val < macd_sig
        if broke_level:
            breakout_idx = breakout_window[
                breakout_window["close"] < recent_support
            ].index.max()
            breakout_volume_ratio = (
                float(df.loc[breakout_idx, "volume_ratio"])
                if "volume_ratio" in df.columns
                else 1.0
            )
        else:
            breakout_volume_ratio = 0.0
        volume_expansion = breakout_volume_ratio >= 1.1
        entry = price
        stop_loss = recent_support + atr * 0.75
        tp1, tp2 = make_r_multiple_targets(direction, entry, stop_loss, r1=1.5, r2=2.5)
        level_name = "support_retest_as_resistance"

    points = 0
    points += 1 if broke_level else 0
    points += 1 if retest_near_level else 0
    points += 1 if trend_ok else 0
    points += 1 if momentum_ok else 0
    points += 1 if volume_expansion else 0
    setup_quality = safe_quality_score(points, 5)

    common_notes = [
        f"level_name={level_name}",
        f"recent_resistance={recent_resistance:.6f}",
        f"recent_support={recent_support:.6f}",
        f"broke_level={broke_level}",
        f"retest_near_level={retest_near_level}",
        f"trend_ok={trend_ok}",
        f"momentum_ok={momentum_ok}",
        f"volume_expansion={volume_expansion}",
        f"breakout_volume_ratio={breakout_volume_ratio:.2f}",
        f"setup_quality={setup_quality}",
    ]

    if not broke_level or not retest_near_level:
        return no_signal(
            strategy,
            "Breakout and retest conditions are not both present.",
            common_notes,
        )

    if setup_quality < get_strategy_min_quality(strategy):
        return no_signal(
            strategy,
            "Breakout/retest is present but confirmation is weak.",
            common_notes,
        )

    confidence = min(0.88, setup_quality * 0.80 + 0.10)
    return build_signal_result(
        strategy_name=strategy,
        signal_status="valid_signal",
        direction=direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        setup_quality=setup_quality,
        confidence_score=confidence,
        risk_level="medium",
        reason="Price broke a recent level and is retesting it with trend/momentum/volume confirmation.",
        notes=common_notes,
    )


def generate_momentum_continuation_signal(
    df: pd.DataFrame, trend_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    MOMENTUM CONTINUATION — v7: stricter rules + min_quality 1.0 (5/5)
    so it almost never produces a trade unless conditions are pristine.

    Also: by default ALLOW_MOMENTUM_AS_EXECUTION is False, so the strategy
    validator will redirect it. This engine remains available for the
    "supporting strategies" confirmation pool.
    """
    strategy = "momentum continuation strategy"
    direction = trend_data["direction"]
    if direction not in ("bullish", "bearish"):
        return no_signal(
            strategy, "Trend direction is not clear enough for momentum continuation."
        )

    latest = df.iloc[-1]
    price = float(latest["close"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])
    ema200 = float(latest["ema200"])
    rsi = float(latest["rsi"])
    macd_val = float(latest["macd"])
    macd_sig = float(latest["macd_signal"])
    atr = float(latest["atr"])
    structure = get_recent_structure(df)

    recent_closes = df["close"].tail(8).values
    close_change_atr = (recent_closes[-1] - recent_closes[0]) / atr if atr > 0 else 0

    if direction == "bullish":
        trend_fully_aligned = price > ema20 > ema50 > ema200
        momentum_ok = macd_val > macd_sig and 55 <= rsi <= 72  # exclude extremes
        rsi_not_extreme = rsi < 72
        slope_strong = close_change_atr > 1.0  # was 0.7
        structure_ok = structure["bullish_structure"]
        entry = price
        stop_loss = min(ema20 - atr * 0.50, price - atr * 1.25)
        tp1, tp2 = make_r_multiple_targets(direction, entry, stop_loss, r1=1.5, r2=3.0)
    else:
        trend_fully_aligned = price < ema20 < ema50 < ema200
        momentum_ok = macd_val < macd_sig and 28 <= rsi <= 45
        rsi_not_extreme = rsi > 28
        slope_strong = close_change_atr < -1.0
        structure_ok = structure["bearish_structure"]
        entry = price
        stop_loss = max(ema20 + atr * 0.50, price + atr * 1.25)
        tp1, tp2 = make_r_multiple_targets(direction, entry, stop_loss, r1=1.5, r2=3.0)

    points = 0
    points += 1 if trend_fully_aligned else 0
    points += 1 if momentum_ok else 0
    points += 1 if rsi_not_extreme else 0
    points += 1 if slope_strong else 0
    points += 1 if structure_ok else 0
    setup_quality = safe_quality_score(points, 5)

    common_notes = [
        f"trend_fully_aligned={trend_fully_aligned}",
        f"momentum_ok={momentum_ok}",
        f"rsi_not_extreme={rsi_not_extreme}",
        f"slope_strong={slope_strong}",
        f"structure_ok={structure_ok}",
        f"close_change_atr={close_change_atr:.2f}",
        f"setup_quality={setup_quality}",
    ]

    if setup_quality < get_strategy_min_quality(strategy):
        return no_signal(
            strategy,
            "Momentum continuation quality is not strong enough.",
            common_notes,
        )

    confidence = min(0.90, setup_quality * 0.85 + 0.05)
    return build_signal_result(
        strategy_name=strategy,
        signal_status="valid_signal",
        direction=direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        setup_quality=setup_quality,
        confidence_score=confidence,
        risk_level="medium" if rsi_not_extreme else "high",
        reason="Strong directional alignment, healthy momentum (no RSI extremes), and clear structure.",
        notes=common_notes,
    )


def generate_reversal_confirmation_signal(
    df: pd.DataFrame, trend_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    REVERSAL-CONFIRMATION — v7: stricter (4/5 quality minimum, candle confirmation,
    avoid trading against very strong trends).
    """
    strategy = "reversal-confirmation strategy"
    if trend_data.get("signal_strength") == "Strong":
        # Avoid reversal trades against an unambiguously strong trend.
        return no_signal(
            strategy, "Trend is strong; reversal is too risky to initiate."
        )

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(latest["close"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])
    ema200 = float(latest["ema200"])
    rsi = float(latest["rsi"])
    macd_val = float(latest["macd"])
    macd_sig = float(latest["macd_signal"])
    atr = float(latest["atr"])
    prev_rsi = float(previous["rsi"])
    prev_macd = float(previous["macd"])
    prev_macd_sig = float(previous["macd_signal"])
    structure = get_recent_structure(df)

    bullish_conditions = {
        "trend_was_weak": ema20 < ema50 or price < ema200,
        "rsi_recovering": prev_rsi < 45 and rsi > prev_rsi,
        "macd_turning": prev_macd < prev_macd_sig and macd_val > macd_sig,
        "price_reclaiming_ema20": float(previous["close"]) < float(previous["ema20"])
        and price > ema20,
        "structure_improving": structure["higher_low_count"] >= 4,
        "candle_confirms": is_bullish_confirmation_candle(df),
    }
    bearish_conditions = {
        "trend_was_weak": ema20 > ema50 or price > ema200,
        "rsi_recovering": prev_rsi > 55 and rsi < prev_rsi,
        "macd_turning": prev_macd > prev_macd_sig and macd_val < macd_sig,
        "price_reclaiming_ema20": float(previous["close"]) > float(previous["ema20"])
        and price < ema20,
        "structure_improving": structure["lower_high_count"] >= 4,
        "candle_confirms": is_bearish_confirmation_candle(df),
    }

    bullish_points = sum(1 for v in bullish_conditions.values() if v)
    bearish_points = sum(1 for v in bearish_conditions.values() if v)

    if bullish_points > bearish_points:
        direction = "bullish"
        conditions = bullish_conditions
        setup_quality = safe_quality_score(bullish_points, 6)
        entry = price
        stop_loss = recent_swing_low(df) - atr * 0.25
        tp1 = max(ema50, entry + (entry - stop_loss) * 1.0)
        tp2 = max(ema200, entry + (entry - stop_loss) * 2.0)
    elif bearish_points > bullish_points:
        direction = "bearish"
        conditions = bearish_conditions
        setup_quality = safe_quality_score(bearish_points, 6)
        entry = price
        stop_loss = recent_swing_high(df) + atr * 0.25
        tp1 = min(ema50, entry - (stop_loss - entry) * 1.0)
        tp2 = min(ema200, entry - (stop_loss - entry) * 2.0)
    else:
        return no_signal(strategy, "No clear bullish or bearish reversal direction.")

    if setup_quality < get_strategy_min_quality(strategy):
        return no_signal(
            strategy,
            "Reversal confirmation is not strong enough.",
            [f"direction_candidate={direction}", f"setup_quality={setup_quality}"]
            + [f"{k}={v}" for k, v in conditions.items()],
        )

    confidence = min(0.80, setup_quality * 0.75 + 0.05)
    return build_signal_result(
        strategy_name=strategy,
        signal_status="valid_signal",
        direction=direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        setup_quality=setup_quality,
        confidence_score=confidence,
        risk_level="high",
        reason="Multiple reversal confirmation signs present; treat as higher-risk.",
        notes=[f"{k}={v}" for k, v in conditions.items()],
    )


def generate_mixed_confirmation_signal(
    df: pd.DataFrame, trend_data: Dict[str, Any]
) -> Dict[str, Any]:
    return no_signal(
        "mixed/confirmation strategy",
        "Market signals are mixed or weak. Waiting for clearer confirmation is preferred.",
        trend_data["notes"],
    )


SIGNAL_GENERATORS = {
    "pullback continuation strategy": generate_pullback_continuation_signal,
    "breakout-and-retest strategy": generate_breakout_retest_signal,
    "range-bound strategy": generate_range_bound_signal,
    "momentum continuation strategy": generate_momentum_continuation_signal,
    "reversal-confirmation strategy": generate_reversal_confirmation_signal,
    "mixed/confirmation strategy": generate_mixed_confirmation_signal,
}


def generate_signal(
    df: pd.DataFrame, trend_data: Dict[str, Any], strategy_name: str
) -> Dict[str, Any]:
    generator = SIGNAL_GENERATORS.get(strategy_name)
    if generator is None:
        return no_signal(strategy_name, "No signal generator exists for this strategy.")
    signal = generator(df, trend_data)

    if signal["signal_status"] == "valid_signal" and not valid_minimum_rr(
        signal, minimum_rr=MINIMUM_RR
    ):
        signal["signal_status"] = "no_valid_signal"
        signal["reason"] = (
            "Setup detected, but risk/reward is below the minimum acceptable threshold."
        )
        signal["risk_level"] = "high"

    return signal


def generate_all_strategy_signals(
    df: pd.DataFrame, trend_data: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name in [
        "pullback continuation strategy",
        "breakout-and-retest strategy",
        "range-bound strategy",
        "momentum continuation strategy",
        "reversal-confirmation strategy",
        "mixed/confirmation strategy",
    ]:
        results[name] = generate_signal(
            df=df, trend_data=trend_data, strategy_name=name
        )
    return results


def print_all_strategy_signals(all_signals: Dict[str, Dict[str, Any]]) -> None:
    print("\n====================")
    print("ALL STRATEGY SIGNALS")
    print("====================")
    for name, sig in all_signals.items():
        print(f"\nStrategy: {name}")
        print(f"Signal Status: {sig['signal_status']}")
        print(f"Direction: {sig['direction']}")
        print(f"Trade Side: {sig.get('trade_side')}")
        print(f"Entry Price: {sig['entry_price']}")
        print(f"Stop Loss: {sig['stop_loss']}")
        print(f"TP1 / TP2: {sig['take_profit_1']} / {sig['take_profit_2']}")
        print(f"RR1 / RR2: {sig['risk_reward_1']} / {sig['risk_reward_2']}")
        print(f"Setup Quality: {sig['setup_quality']}")
        print(f"Confidence: {sig['confidence_score']}")
        print(f"Risk Level: {sig['risk_level']}")
        print(f"Reason: {sig['reason']}")


# ============================================================
# MULTI-TIMEFRAME ORCHESTRATION (new in v7)
# ============================================================


def validate_timeframe_pair(htf: str, ltf: Optional[str]) -> None:
    """LTF must be strictly smaller than HTF."""
    if ltf is None:
        return
    if parse_timeframe_minutes(ltf) >= parse_timeframe_minutes(htf):
        raise ValueError(
            f"Lower timeframe '{ltf}' must be strictly smaller than higher timeframe '{htf}'."
        )


def ltf_trigger_passes(
    ltf_df: pd.DataFrame,
    ltf_trend: Dict[str, Any],
    htf_direction: str,
    htf_strategy: str,
) -> Tuple[bool, List[str]]:
    """
    Cheap, conservative LTF entry trigger.

    Logic:
    - LTF direction should align with HTF direction (or be neutral, which is
      acceptable while a pullback is forming on the HTF).
    - A confirmation candle on the LTF in the HTF direction is required.
    - For breakout/momentum strategies, the LTF should also show a closing
      cross of EMA20 in the HTF direction in the last few bars.
    """
    notes: List[str] = []
    if htf_direction not in ("bullish", "bearish"):
        notes.append("HTF direction is not bullish/bearish; LTF trigger skipped.")
        return False, notes

    ltf_dir = ltf_trend["direction"]
    aligned_or_neutral = (ltf_dir == htf_direction) or (ltf_dir == "neutral")
    notes.append(
        f"ltf_dir={ltf_dir}, htf_dir={htf_direction}, aligned_or_neutral={aligned_or_neutral}"
    )
    if not aligned_or_neutral:
        return False, notes

    if htf_direction == "bullish":
        candle_ok = is_bullish_confirmation_candle(ltf_df)
    else:
        candle_ok = is_bearish_confirmation_candle(ltf_df)
    notes.append(f"ltf_candle_confirm={candle_ok}")

    extra_cross_required = htf_strategy in {
        "breakout-and-retest strategy",
        "momentum continuation strategy",
    }
    cross_ok = True
    if extra_cross_required:
        recent = ltf_df.tail(5)
        if htf_direction == "bullish":
            cross_ok = bool((recent["close"] > recent["ema20"]).iloc[-1]) and bool(
                (ltf_df["close"].iloc[-2] <= ltf_df["ema20"].iloc[-2])
                or (recent["close"] > recent["ema20"]).any()
            )
        else:
            cross_ok = bool((recent["close"] < recent["ema20"]).iloc[-1]) and bool(
                (ltf_df["close"].iloc[-2] >= ltf_df["ema20"].iloc[-2])
                or (recent["close"] < recent["ema20"]).any()
            )
        notes.append(f"ltf_ema20_cross_ok={cross_ok}")

    return candle_ok and cross_ok, notes


# ============================================================
# POSITION SIZING (new in v7)
# ============================================================


@dataclass
class PositionSizingConfig:
    account_balance_usd: float = DEFAULT_ACCOUNT_BALANCE_USD
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT
    max_daily_loss_r: float = DEFAULT_MAX_DAILY_LOSS_R
    max_open_trades: int = DEFAULT_MAX_OPEN_TRADES


def compute_position_sizing(
    signal: Dict[str, Any],
    config: PositionSizingConfig,
) -> Dict[str, Any]:
    """
    Compute the dollar risk, position size in base units, and notional exposure
    for an executable trade signal.

    The function does not execute anything; it only documents what an account
    with the given configuration would risk on this signal.
    """
    if signal.get("signal_status") != "valid_signal":
        return {
            "sizing_status": "not_applicable",
            "note": "Signal is not executable; position sizing is not computed.",
            "config": asdict(config),
        }

    entry = signal.get("entry_price")
    stop = signal.get("stop_loss")
    side = signal.get("trade_side")
    if not (is_number(entry) and is_number(stop)) or side not in ("long", "short"):
        return {
            "sizing_status": "missing_inputs",
            "note": "Signal is missing entry/stop or has an invalid side.",
            "config": asdict(config),
        }

    risk_per_unit = abs(float(entry) - float(stop))
    if risk_per_unit <= 0:
        return {
            "sizing_status": "invalid_geometry",
            "note": "Entry and stop_loss imply zero or negative risk.",
            "config": asdict(config),
        }

    risk_amount_usd = round(
        config.account_balance_usd * (config.risk_per_trade_pct / 100.0), 2
    )
    units = risk_amount_usd / risk_per_unit
    notional_usd = units * float(entry)

    return {
        "sizing_status": "ok",
        "config": asdict(config),
        "risk_per_unit_quote": round(risk_per_unit, 8),
        "risk_amount_usd": risk_amount_usd,
        "position_size_units": round(units, 8),
        "position_notional_usd": round(notional_usd, 2),
        "leverage_implied": round(notional_usd / config.account_balance_usd, 2),
        "max_daily_loss_usd": round(
            config.account_balance_usd
            * abs(config.max_daily_loss_r)
            * (config.risk_per_trade_pct / 100.0),
            2,
        ),
        "max_open_trades": config.max_open_trades,
        "note": "Position size assumes one trade; you must reserve capacity for max_open_trades.",
    }


# ============================================================
# FINAL SIGNAL VALIDATION
# ============================================================


def validate_final_signal(
    signal: Dict[str, Any], trend_data: Dict[str, Any], selected_strategy: str
) -> Dict[str, Any]:
    """Hard rule-based validator. This is the authority, not the LLM."""
    errors: List[str] = []
    warnings: List[str] = []

    status = signal.get("signal_status")
    direction = signal.get("direction")
    trade_side = signal.get("trade_side")
    expected_trade_side = trade_side_from_direction(direction)

    result = {
        "hard_validation_status": "not_trade_signal",
        "is_hard_valid": False,
        "errors": errors,
        "warnings": warnings,
        "validator_note": "",
    }

    if status != "valid_signal":
        result["validator_note"] = "No executable trade signal was produced."
        if status == "waiting_for_pullback":
            result["hard_validation_status"] = "waiting"
        return result

    if trade_side != expected_trade_side:
        errors.append(f"trade_side={trade_side} does not match direction={direction}.")
    if trade_side not in ("long", "short"):
        errors.append("Executable signal must have trade_side long or short.")

    entry = signal.get("entry_price")
    stop = signal.get("stop_loss")
    tp1 = signal.get("take_profit_1")
    tp2 = signal.get("take_profit_2")

    if entry is None or stop is None or tp1 is None or tp2 is None:
        errors.append(
            "Executable signal must include entry_price, stop_loss, take_profit_1, and take_profit_2."
        )
    else:
        entry_f = float(entry)
        stop_f = float(stop)
        tp1_f = float(tp1)
        tp2_f = float(tp2)
        if trade_side == "long":
            if not (stop_f < entry_f < tp1_f <= tp2_f):
                errors.append(
                    "LONG price logic failed: expected stop_loss < entry_price < TP1 <= TP2."
                )
        elif trade_side == "short":
            if not (tp2_f <= tp1_f < entry_f < stop_f):
                errors.append(
                    "SHORT price logic failed: expected TP2 <= TP1 < entry_price < stop_loss."
                )

        atr = float(trend_data.get("atr", 0) or 0)
        if atr > 0:
            stop_distance_atr = abs(entry_f - stop_f) / atr
            if stop_distance_atr > 4:
                warnings.append(
                    f"Stop loss is far from entry: {stop_distance_atr:.2f} ATR."
                )
            if stop_distance_atr < 0.25:
                warnings.append(
                    f"Stop loss is very tight: {stop_distance_atr:.2f} ATR."
                )

    rr1 = signal.get("risk_reward_1")
    rr2 = signal.get("risk_reward_2")
    if rr1 is None or rr2 is None:
        errors.append("Risk/reward values are missing.")
    elif max(float(rr1), float(rr2)) < MINIMUM_RR:
        errors.append(f"Risk/reward is below minimum threshold {MINIMUM_RR}.")

    setup_quality = float(signal.get("setup_quality", 0) or 0)
    min_quality = get_strategy_min_quality(selected_strategy)
    if setup_quality < min_quality:
        errors.append(
            f"Setup quality {setup_quality:.2f} is below required minimum {min_quality:.2f} for {selected_strategy}."
        )

    confidence = float(signal.get("confidence_score", 0) or 0)
    if confidence > 0.92:
        warnings.append(
            "Confidence is very high; check that scoring is not over-optimistic."
        )

    trend_direction = trend_data.get("direction")
    trend_following = {
        "pullback continuation strategy",
        "breakout-and-retest strategy",
        "momentum continuation strategy",
    }
    if (
        selected_strategy in trend_following
        and trend_direction in ("bullish", "bearish")
        and direction != trend_direction
    ):
        errors.append(
            "Trend-following signal direction does not match detected trend direction."
        )

    if selected_strategy == "mixed/confirmation strategy":
        errors.append(
            "Mixed/confirmation strategy is a waiting mode and should not produce executable trades."
        )

    if errors:
        result["hard_validation_status"] = "rejected"
        result["validator_note"] = "Signal failed hard rule-based validation."
        return result

    result["hard_validation_status"] = "approved"
    result["is_hard_valid"] = True
    result["validator_note"] = "Signal passed hard rule-based validation."
    return result


def find_supporting_strategies(
    selected_strategy: str,
    final_signal: Dict[str, Any],
    all_strategy_signals: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not is_valid_trade_signal(final_signal):
        return []
    selected_side = final_signal.get("trade_side")
    supporting: List[Dict[str, Any]] = []
    for name, sig in all_strategy_signals.items():
        if name == selected_strategy:
            continue
        if not is_valid_trade_signal(sig):
            continue
        if sig.get("trade_side") != selected_side:
            continue
        supporting.append(
            {
                "strategy_name": name,
                "trade_side": sig.get("trade_side"),
                "setup_quality": sig.get("setup_quality"),
                "confidence_score": sig.get("confidence_score"),
                "reason": sig.get("reason"),
            }
        )
    return supporting


def apply_supporting_confirmation(
    final_signal: Dict[str, Any], supporting_strategies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    adjusted = dict(final_signal)
    base = float(adjusted.get("confidence_score", 0) or 0)
    if adjusted.get("signal_status") != "valid_signal" or not supporting_strategies:
        adjusted["confirmation_boost"] = 0.0
        adjusted["supporting_strategy_count"] = len(supporting_strategies)
        return adjusted
    boost = min(0.08, 0.03 * len(supporting_strategies))
    adjusted["confidence_score_before_confirmation"] = round(base, 2)
    adjusted["confirmation_boost"] = round(boost, 2)
    adjusted["confidence_score"] = round(
        min(MAX_CONFIDENCE_AFTER_CONFIRMATION, base + boost), 2
    )
    adjusted["supporting_strategy_count"] = len(supporting_strategies)
    return adjusted


# ============================================================
# LLM JSON HELPERS / REVIEW
# ============================================================


def extract_json_object(text: Optional[str]) -> Dict[str, Any]:
    """Best-effort JSON extraction for LLM outputs (handles fences, leading text)."""
    if text is None:
        raise ValueError("LLM returned None instead of JSON.")
    cleaned = str(text).strip()
    if not cleaned:
        raise ValueError("LLM returned an empty response instead of JSON.")

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM response: {cleaned[:200]!r}")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        ch = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : index + 1])

    raise ValueError(
        f"Could not find a complete JSON object in LLM response: {cleaned[:200]!r}"
    )


def normalize_llm_review(
    review_data: Any, hard_validation: Dict[str, Any], signal_result: Dict[str, Any]
) -> Dict[str, Any]:
    if not isinstance(review_data, dict):
        raise ValueError("LLM review JSON must be an object/dict.")

    allowed = {
        "approved",
        "approved_with_warning",
        "rejected_by_hard_validator",
        "not_executable",
    }
    status = review_data.get("agent_review_status")
    if status not in allowed:
        if signal_result.get("signal_status") != "valid_signal":
            status = "not_executable"
        elif not hard_validation.get("is_hard_valid"):
            status = "rejected_by_hard_validator"
        else:
            status = "approved_with_warning"

    try:
        adj = float(review_data.get("confidence_adjustment", 0.0))
    except (TypeError, ValueError):
        adj = 0.0
    adj = max(-0.20, min(0.05, adj))

    warnings = review_data.get("warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings]
    if not isinstance(warnings, list):
        warnings = []
    warnings = [str(item) for item in warnings if item is not None]

    summary = (
        review_data.get("reviewer_summary")
        or review_data.get("reviewer_note")
        or "LLM review completed."
    )

    if signal_result.get("signal_status") != "valid_signal":
        status = "not_executable"
        adj = min(adj, 0.0)
    elif not hard_validation.get("is_hard_valid"):
        status = "rejected_by_hard_validator"
        adj = min(adj, 0.0)

    return {
        "agent_review_status": status,
        "confidence_adjustment": round(adj, 2),
        "warnings": warnings,
        "reviewer_summary": str(summary),
        "source": "llm",
    }


def rule_based_signal_review(
    signal_result: Dict[str, Any],
    hard_validation: Dict[str, Any],
    supporting_strategies: List[Dict[str, Any]],
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    status_signal = signal_result.get("signal_status")
    strategy = signal_result.get("strategy_name")
    setup_quality = float(signal_result.get("setup_quality", 0) or 0)
    confidence = float(signal_result.get("confidence_score", 0) or 0)

    if status_signal != "valid_signal":
        status = "not_executable"
        summary = f"No executable trade signal was produced; current status is {status_signal}."
    elif not hard_validation.get("is_hard_valid"):
        status = "rejected_by_hard_validator"
        summary = "The signal failed hard Python validation, so it is not approved."
    else:
        status = (
            "approved"
            if not hard_validation.get("warnings")
            else "approved_with_warning"
        )
        summary = "The signal passed hard Python validation."

    if setup_quality >= 0.95:
        warnings.append(
            "Setup quality is very high; check that scoring is not over-optimistic."
        )
    if confidence >= 0.90:
        warnings.append(
            "Confidence is high; keep validating with backtesting/paper tracking."
        )
    if strategy == "pullback continuation strategy" and status_signal == "valid_signal":
        warnings.append(
            "Pullback strategy: confirm price is close enough to the EMA pullback zone before execution."
        )
    if supporting_strategies:
        summary += f" {len(supporting_strategies)} supporting strategy/strategies agree with the same trade side."
    if fallback_reason:
        warnings.append(f"LLM review fallback used: {fallback_reason}")

    if warnings and status == "approved":
        status = "approved_with_warning"

    return {
        "agent_review_status": status,
        "confidence_adjustment": 0.0,
        "warnings": warnings,
        "reviewer_summary": summary,
        "source": "rule_based_fallback",
    }


def review_signal_with_llm(
    trend_output: str,
    strategy_json: Dict[str, Any],
    validation: Dict[str, Any],
    signal_result: Dict[str, Any],
    hard_validation: Dict[str, Any],
    supporting_strategies: List[Dict[str, Any]],
    *,
    enabled: bool = ENABLE_LLM_SIGNAL_REVIEW,
) -> Dict[str, Any]:
    if not enabled:
        fb = rule_based_signal_review(
            signal_result, hard_validation, supporting_strategies
        )
        fb["review_status"] = "disabled"
        fb["reviewer_note"] = (
            "LLM signal review is disabled in config. Rule-based fallback review was used."
        )
        return fb

    if Agent is None or Runner is None or not os.getenv("OPENAI_API_KEY"):
        fb = rule_based_signal_review(
            signal_result, hard_validation, supporting_strategies
        )
        fb["review_status"] = "not_available"
        fb["reviewer_note"] = (
            "OpenAI Agents SDK or OPENAI_API_KEY missing. Rule-based fallback used."
        )
        return fb

    agent = Agent(
        name="Final Signal Review Agent",
        model=LLM_MODEL,
        instructions="""
You are a conservative trading-signal consistency reviewer.

Rules:
- You are NOT allowed to override the hard Python validator.
- Do not give financial advice.
- Do not approve a signal whose hard validation failed.
- Review only consistency, contradictions, and risk warnings.
- Use only the provided trend, strategy, signal, validation, and supporting-strategy data.

Return ONLY valid JSON. No markdown fences. No text before or after.

Schema:
{
  "agent_review_status": "approved | approved_with_warning | rejected_by_hard_validator | not_executable",
  "confidence_adjustment": <float between -0.20 and 0.05>,
  "warnings": ["short warning 1", "short warning 2"],
  "reviewer_summary": "short explanation"
}
""",
    )

    agent_input = json.dumps(
        make_json_safe(
            {
                "trend_output": trend_output,
                "strategy_suggestion": strategy_json,
                "strategy_validation": validation,
                "final_signal": signal_result,
                "hard_validation": hard_validation,
                "supporting_strategies": supporting_strategies,
                "instruction_reminder": "Return only a single valid JSON object matching the schema.",
            }
        ),
        indent=2,
    )

    try:
        result = Runner.run_sync(agent, agent_input, max_turns=1)
        raw = getattr(result, "final_output", None)
        parsed = extract_json_object(raw)
        return normalize_llm_review(parsed, hard_validation, signal_result)
    except Exception as exc:
        return rule_based_signal_review(
            signal_result=signal_result,
            hard_validation=hard_validation,
            supporting_strategies=supporting_strategies,
            fallback_reason=str(exc),
        )


def build_final_decision(
    final_signal: Dict[str, Any],
    hard_validation: Dict[str, Any],
    llm_signal_review: Dict[str, Any],
    supporting_strategies: List[Dict[str, Any]],
    ltf_trigger: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if final_signal.get("signal_status") != "valid_signal":
        decision_status = final_signal.get("signal_status")
    elif hard_validation.get("is_hard_valid"):
        if ltf_trigger is not None and not ltf_trigger.get("passed", True):
            decision_status = "waiting_for_ltf_trigger"
        else:
            decision_status = "approved_for_paper_tracking"
    else:
        decision_status = "rejected"

    return {
        "execution_mode": EXECUTION_MODE,
        "decision_status": decision_status,
        "trade_side": final_signal.get("trade_side"),
        "selected_strategy": final_signal.get("strategy_name"),
        "supporting_strategies": supporting_strategies,
        "hard_validation_status": hard_validation.get("hard_validation_status"),
        "llm_review_status": (
            llm_signal_review.get("agent_review_status")
            or llm_signal_review.get("review_status")
        ),
        "ltf_trigger": ltf_trigger,
        "note": "Educational decision-support only. Use for paper tracking/backtesting before any real trading.",
    }


# ============================================================
# REPORTING (LIVE)
# ============================================================


def print_signal_report(
    symbol: str,
    timeframe: str,
    exchange_name: str,
    trend_data: Dict[str, Any],
    strategy_json: Dict[str, Any],
    validation: Dict[str, Any],
    template_result: Dict[str, Any],
    signal_result: Dict[str, Any],
    hard_validation: Optional[Dict[str, Any]] = None,
    llm_signal_review: Optional[Dict[str, Any]] = None,
    final_decision: Optional[Dict[str, Any]] = None,
    position_sizing: Optional[Dict[str, Any]] = None,
) -> None:
    print("\n====================")
    print("FINAL SIGNAL REPORT")
    print("====================")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Exchange: {exchange_name}")
    print("--------------------")
    print(f"Trend: {trend_data['trend']}")
    print(f"Strength: {trend_data['signal_strength']}")
    print(f"Suggested Strategy: {strategy_json.get('strategy_style')}")
    print(f"Strategy Source: {strategy_json.get('source')}")
    print(f"Final Strategy: {validation.get('final_strategy_style')}")
    print(f"Validation Valid: {validation.get('is_valid')}")
    if validation.get("warnings"):
        print("Validation Warnings:")
        for w in validation["warnings"]:
            print(f"- {w}")
    print("--------------------")
    print(f"Signal Status: {signal_result['signal_status']}")
    print(f"Direction: {signal_result['direction']}")
    print(f"Trade Side: {signal_result.get('trade_side')}")
    print(f"Entry Price: {signal_result['entry_price']}")
    print(
        f"Entry Zone: {signal_result.get('entry_zone_low')} - {signal_result.get('entry_zone_high')}"
    )
    print(f"Stop Loss: {signal_result['stop_loss']}")
    print(f"Take Profit 1: {signal_result['take_profit_1']}")
    print(f"Take Profit 2: {signal_result['take_profit_2']}")
    print(f"Risk/Reward 1: {signal_result['risk_reward_1']}")
    print(f"Risk/Reward 2: {signal_result['risk_reward_2']}")
    print(f"Setup Quality: {signal_result['setup_quality']}")
    print(f"Confidence: {signal_result['confidence_score']}")
    print(f"Risk Level: {signal_result['risk_level']}")
    print(f"Invalidation Level: {signal_result['invalidation_level']}")
    if hard_validation:
        print("--------------------")
        print(f"Hard Validation: {hard_validation.get('hard_validation_status')}")
        for e in hard_validation.get("errors", []):
            print(f"  ERROR: {e}")
        for w in hard_validation.get("warnings", []):
            print(f"  WARNING: {w}")
    if llm_signal_review:
        print("--------------------")
        print(
            "LLM Signal Review:",
            llm_signal_review.get("agent_review_status")
            or llm_signal_review.get("review_status"),
        )
        for w in llm_signal_review.get("warnings", []):
            print(f"- {w}")
        if llm_signal_review.get("reviewer_summary"):
            print(f"Reviewer Summary: {llm_signal_review.get('reviewer_summary')}")
        elif llm_signal_review.get("reviewer_note"):
            print(f"Reviewer Note: {llm_signal_review.get('reviewer_note')}")
    if final_decision:
        print("--------------------")
        print(f"Final Decision: {final_decision.get('decision_status')}")
        print(f"Execution Mode: {final_decision.get('execution_mode')}")
        print(
            f"Supporting Strategies: {len(final_decision.get('supporting_strategies', []))}"
        )
        ltf = final_decision.get("ltf_trigger")
        if ltf is not None:
            print(f"LTF Trigger Passed: {ltf.get('passed')}")
            for n in ltf.get("notes", []):
                print(f"  - {n}")
    if position_sizing:
        print("--------------------")
        print(f"Position Sizing Status: {position_sizing.get('sizing_status')}")
        if position_sizing.get("sizing_status") == "ok":
            print(f"  Risk Amount USD:      {position_sizing.get('risk_amount_usd')}")
            print(
                f"  Position Size (units):{position_sizing.get('position_size_units')}"
            )
            print(
                f"  Notional USD:         {position_sizing.get('position_notional_usd')}"
            )
            print(f"  Implied Leverage:     {position_sizing.get('leverage_implied')}x")
            print(
                f"  Max Daily Loss USD:   {position_sizing.get('max_daily_loss_usd')}"
            )
    print("--------------------")
    print(f"Reason: {signal_result['reason']}")
    if signal_result.get("notes"):
        print("Notes:")
        for n in signal_result["notes"]:
            print(f"- {n}")
    print("====================")


def save_report_json(
    symbol: str, timeframe: str, report: Dict[str, Any], folder: str = OUTPUT_DIR
) -> str:
    ensure_dir(folder)
    safe_symbol = clean_symbol_for_filename(symbol)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"signal_{safe_symbol}_{timeframe}_{stamp}.json"
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2)
    return path


# ============================================================
# BACKTEST: TRADE SIMULATION
# ============================================================


def is_executable_signal(signal: Dict[str, Any]) -> bool:
    if not signal:
        return False
    if signal.get("signal_status") != "valid_signal":
        return False
    if signal.get("trade_side") not in ("long", "short"):
        return False
    required = ("entry_price", "stop_loss", "take_profit_1", "take_profit_2")
    return all(is_number(signal.get(f)) for f in required)


def get_target_price(signal: Dict[str, Any], target_mode: str) -> Optional[float]:
    if target_mode == "tp1":
        return signal.get("take_profit_1")
    if target_mode == "tp2":
        return signal.get("take_profit_2")
    raise ValueError("target_mode must be 'tp1' or 'tp2'")


def calculate_trade_r(
    entry: float, stop_loss: float, exit_price: float, side: str
) -> float:
    if side == "long":
        risk = entry - stop_loss
        return (exit_price - entry) / risk if risk > 0 else 0.0
    if side == "short":
        risk = stop_loss - entry
        return (entry - exit_price) / risk if risk > 0 else 0.0
    return 0.0


def calculate_trade_return_pct(entry: float, exit_price: float, side: str) -> float:
    if entry <= 0:
        return 0.0
    if side == "long":
        return (exit_price - entry) / entry * 100
    if side == "short":
        return (entry - exit_price) / entry * 100
    return 0.0


def apply_fees_and_slippage(
    side: str,
    entry: float,
    exit_price: float,
    stop_loss: float,
    fee_pct_per_side: float,
    slippage_pct: float,
) -> Tuple[float, float, float]:
    """
    Apply trading frictions to the simulated trade.

    Returns:
        (adj_entry, adj_exit, result_r_after_costs)

    Slippage: entry fills slightly worse than the signal price.
    Fees: charged on both entry and exit notionals.
    """
    slip = entry * (slippage_pct / 100.0)
    fee_in = entry * (fee_pct_per_side / 100.0)
    fee_out = exit_price * (fee_pct_per_side / 100.0)

    if side == "long":
        adj_entry = entry + slip
    else:
        adj_entry = entry - slip

    if side == "long":
        # gross P/L on adjusted entry
        pl_per_unit = exit_price - adj_entry - fee_in - fee_out
        risk = adj_entry - stop_loss
    else:
        pl_per_unit = adj_entry - exit_price - fee_in - fee_out
        risk = stop_loss - adj_entry

    if risk <= 0:
        return adj_entry, exit_price, 0.0
    return adj_entry, exit_price, pl_per_unit / risk


def simulate_future_trade(
    df: pd.DataFrame,
    signal_index: int,
    signal: Dict[str, Any],
    *,
    target_mode: str = DEFAULT_TARGET_MODE,
    entry_mode: str = DEFAULT_ENTRY_MODE,
    max_hold_candles: int = DEFAULT_MAX_HOLD_CANDLES,
    fee_pct_per_side: float = DEFAULT_FEE_PCT_PER_SIDE,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
) -> Optional[Dict[str, Any]]:
    """
    Simulate one future trade after a valid signal.

    signal_index is the candle index where the signal was generated.
    The trade is evaluated starting from signal_index + 1.
    """
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
        entry_time = (
            df.iloc[signal_index]["datetime"] if "datetime" in df.columns else None
        )
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
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_index: Optional[int] = None

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

    result_r_gross = calculate_trade_r(entry_price, stop_loss, exit_price, side)
    return_pct = calculate_trade_return_pct(entry_price, exit_price, side)
    adj_entry, _, result_r_net = apply_fees_and_slippage(
        side=side,
        entry=entry_price,
        exit_price=exit_price,
        stop_loss=stop_loss,
        fee_pct_per_side=fee_pct_per_side,
        slippage_pct=slippage_pct,
    )

    signal_time = (
        df.iloc[signal_index]["datetime"] if "datetime" in df.columns else None
    )
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
        "entry_price_after_slippage": round(float(adj_entry), 6),
        "signal_entry_price": round(float(signal_entry), 6),
        "stop_loss": round(float(stop_loss), 6),
        "target_price": round(float(target_price), 6),
        "take_profit_1": signal.get("take_profit_1"),
        "take_profit_2": signal.get("take_profit_2"),
        "exit_price": round(float(exit_price), 6),
        "exit_reason": exit_reason,
        "result_r_gross": round(float(result_r_gross), 4),
        "result_r": round(float(result_r_net), 4),  # net R after fees + slippage
        "return_pct": round(float(return_pct), 4),
        "risk_reward_1": signal.get("risk_reward_1"),
        "risk_reward_2": signal.get("risk_reward_2"),
        "setup_quality": signal.get("setup_quality"),
        "confidence_score": signal.get("confidence_score"),
        "risk_level": signal.get("risk_level"),
        "reason": signal.get("reason"),
    }


# ============================================================
# BACKTEST: SIGNAL GENERATION PIPELINES
# ============================================================


def _generate_selected_signal_with_suggestion(
    window_df: pd.DataFrame,
    trend_data: Dict[str, Any],
    strategy_suggestion: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    validation = validate_strategy_suggestion(
        trend=trend_data["trend"],
        strength=trend_data["signal_strength"],
        strategy_json=strategy_suggestion,
    )
    selected = validation.get("final_strategy_style") or strategy_suggestion.get(
        "strategy_style"
    )
    signal = generate_signal(window_df, trend_data, selected)
    hard_validation = validate_final_signal(signal, trend_data, selected)

    signal = dict(signal)
    if not hard_validation.get("is_hard_valid"):
        signal["backtest_executable"] = False
        signal["backtest_rejection_reason"] = hard_validation.get("validator_note")
    else:
        signal["backtest_executable"] = True
        signal["backtest_rejection_reason"] = None
    return validation, signal


def generate_selected_signal_no_llm(
    window_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    trend_data = analyze_trend(window_df)
    strategy_suggestion = get_rule_based_strategy_suggestion(trend_data)
    strategy_suggestion["source"] = "rule_based_backtest_no_llm"
    validation, signal = _generate_selected_signal_with_suggestion(
        window_df, trend_data, strategy_suggestion
    )
    return trend_data, strategy_suggestion, validation, signal


# LLM strategy agent — cached at module level so we don't recreate it per call.
_BACKTEST_STRATEGY_AGENT = None


def _get_backtest_strategy_agent():
    global _BACKTEST_STRATEGY_AGENT
    if _BACKTEST_STRATEGY_AGENT is not None:
        return _BACKTEST_STRATEGY_AGENT
    if Agent is None or Runner is None:
        return None
    _BACKTEST_STRATEGY_AGENT = Agent(
        name="Historical Backtest Strategy Selector",
        model=LLM_MODEL,
        instructions="""
You are a crypto trading strategy classifier used inside a historical backtest.

Suggest the best-fit trading strategy style from a completed trend analysis.

Rules:
- Use only the provided trend analysis output.
- Do not create a new signal.
- Do not approve or reject a trade.
- Do not say BUY, SELL, LONG, SHORT, ENTER, EXIT, NO_TRADE, approved, or rejected.
- Do not recalculate indicators.
- Do not add news, fundamentals, predictions, or extra market data.
- If the trend is weak/unclear/sideways, prefer range-bound or mixed/confirmation.
- Prefer pullback continuation over momentum continuation in trending markets.

Allowed strategy styles:
- pullback continuation strategy
- breakout-and-retest strategy
- range-bound strategy
- momentum continuation strategy
- reversal-confirmation strategy
- mixed/confirmation strategy

Return ONLY valid JSON. No markdown fences.

JSON schema:
{
  "strategy_style": "<one allowed strategy style>",
  "market_condition": "<strong_trending | weak_trending | ranging | unclear>",
  "confidence": <float between 0 and 1>,
  "reason": "<short concise explanation based only on provided trend analysis>"
}
""",
    )
    return _BACKTEST_STRATEGY_AGENT


def _normalize_strategy_suggestion(obj: Dict[str, Any], source: str) -> Dict[str, Any]:
    strategy = obj.get("strategy_style")
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f"LLM returned unsupported strategy_style: {strategy}")
    confidence = obj.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))
    return {
        "strategy_style": strategy,
        "market_condition": str(obj.get("market_condition", "unclear")),
        "confidence": confidence,
        "reason": str(obj.get("reason", "")),
        "source": source,
    }


def _llm_strategy_suggestion_for_backtest(
    trend_output: str,
    trend_data: Dict[str, Any],
    llm_stats: Dict[str, int],
    *,
    use_cache: bool,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if use_cache and trend_output in cache:
        llm_stats["cache_hits"] += 1
        return dict(cache[trend_output])

    if Agent is None or Runner is None or not os.getenv("OPENAI_API_KEY"):
        llm_stats["fallbacks"] += 1
        suggestion = get_rule_based_strategy_suggestion(trend_data)
        suggestion["source"] = "rule_based_fallback_after_llm_unavailable"
        if use_cache:
            cache[trend_output] = dict(suggestion)
        return suggestion

    agent = _get_backtest_strategy_agent()
    if agent is None:
        llm_stats["fallbacks"] += 1
        suggestion = get_rule_based_strategy_suggestion(trend_data)
        suggestion["source"] = "rule_based_fallback_after_llm_unavailable"
        if use_cache:
            cache[trend_output] = dict(suggestion)
        return suggestion

    agent_input = f"""
Trend output:
{trend_output}

Suggest the best-fit strategy style for this historical candle.
""".strip()

    try:
        if LLM_SLEEP_SECONDS_BACKTEST > 0:
            time.sleep(LLM_SLEEP_SECONDS_BACKTEST)
        llm_stats["calls_attempted"] += 1
        result = Runner.run_sync(agent, agent_input, max_turns=1)
        parsed = extract_json_object(getattr(result, "final_output", ""))
        suggestion = _normalize_strategy_suggestion(parsed, source="llm_backtest")
        llm_stats["calls_successful"] += 1
        if use_cache:
            cache[trend_output] = dict(suggestion)
        return suggestion
    except Exception as exc:
        llm_stats["fallbacks"] += 1
        suggestion = get_rule_based_strategy_suggestion(trend_data)
        suggestion["source"] = "rule_based_fallback_after_llm_failure"
        suggestion["llm_failure_reason"] = str(exc)
        if use_cache:
            cache[trend_output] = dict(suggestion)
        return suggestion


def generate_selected_signal_with_llm(
    window_df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    exchange_name: str,
    llm_stats: Dict[str, int],
    use_cache: bool,
    cache: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    trend_data = analyze_trend(window_df)
    trend_output = build_trend_output(symbol, timeframe, exchange_name, trend_data)
    strategy_suggestion = _llm_strategy_suggestion_for_backtest(
        trend_output=trend_output,
        trend_data=trend_data,
        llm_stats=llm_stats,
        use_cache=use_cache,
        cache=cache,
    )
    validation, signal = _generate_selected_signal_with_suggestion(
        window_df, trend_data, strategy_suggestion
    )
    return trend_data, strategy_suggestion, validation, signal


# ============================================================
# BACKTEST: ENGINES
# ============================================================


def backtest_selected_strategy_no_llm(
    df: pd.DataFrame,
    *,
    target_mode: str = DEFAULT_TARGET_MODE,
    entry_mode: str = DEFAULT_ENTRY_MODE,
    max_hold_candles: int = DEFAULT_MAX_HOLD_CANDLES,
    fee_pct_per_side: float = DEFAULT_FEE_PCT_PER_SIDE,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trades: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    next_allowed = MIN_HISTORY_CANDLES

    for i in range(MIN_HISTORY_CANDLES, len(df) - 1):
        if PREVENT_OVERLAPPING_TRADES and i < next_allowed:
            continue
        window = df.iloc[: i + 1].copy().reset_index(drop=True)
        try:
            trend_data, suggestion, validation, signal = (
                generate_selected_signal_no_llm(window)
            )
        except Exception as exc:
            skipped.append(
                {"signal_index": i, "reason": f"signal_generation_error: {exc}"}
            )
            continue

        selected = validation.get("final_strategy_style") or suggestion.get(
            "strategy_style"
        )
        if not signal.get("backtest_executable") or not is_executable_signal(signal):
            skipped.append(
                {
                    "signal_index": i,
                    "signal_time": (
                        str(df.iloc[i]["datetime"])
                        if "datetime" in df.columns
                        else None
                    ),
                    "trend": trend_data.get("trend"),
                    "strength": trend_data.get("signal_strength"),
                    "selected_strategy": selected,
                    "signal_status": signal.get("signal_status"),
                    "trade_side": signal.get("trade_side"),
                    "reason": signal.get("reason"),
                    "backtest_rejection_reason": signal.get(
                        "backtest_rejection_reason"
                    ),
                }
            )
            continue

        trade = simulate_future_trade(
            df=df,
            signal_index=i,
            signal=signal,
            target_mode=target_mode,
            entry_mode=entry_mode,
            max_hold_candles=max_hold_candles,
            fee_pct_per_side=fee_pct_per_side,
            slippage_pct=slippage_pct,
        )
        if trade is None:
            skipped.append(
                {
                    "signal_index": i,
                    "selected_strategy": selected,
                    "reason": "signal was executable but could not be simulated.",
                }
            )
            continue

        trade["trend"] = trend_data.get("trend")
        trade["trend_strength"] = trend_data.get("signal_strength")
        trade["selected_strategy"] = selected
        trade["strategy_source"] = suggestion.get("source")
        trades.append(trade)

        if PREVENT_OVERLAPPING_TRADES:
            next_allowed = trade["exit_index"] + 1

    return trades, skipped


def backtest_selected_strategy_with_llm(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    exchange_name: str,
    target_mode: str = DEFAULT_TARGET_MODE,
    entry_mode: str = DEFAULT_ENTRY_MODE,
    max_hold_candles: int = DEFAULT_MAX_HOLD_CANDLES,
    fee_pct_per_side: float = DEFAULT_FEE_PCT_PER_SIDE,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    use_cache: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    trades: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    next_allowed = MIN_HISTORY_CANDLES
    cache: Dict[str, Dict[str, Any]] = {}
    llm_stats = {
        "calls_attempted": 0,
        "calls_successful": 0,
        "fallbacks": 0,
        "cache_hits": 0,
    }

    for i in range(MIN_HISTORY_CANDLES, len(df) - 1):
        if PREVENT_OVERLAPPING_TRADES and i < next_allowed:
            continue
        window = df.iloc[: i + 1].copy().reset_index(drop=True)
        try:
            trend_data, suggestion, validation, signal = (
                generate_selected_signal_with_llm(
                    window,
                    symbol=symbol,
                    timeframe=timeframe,
                    exchange_name=exchange_name,
                    llm_stats=llm_stats,
                    use_cache=use_cache,
                    cache=cache,
                )
            )
        except Exception as exc:
            skipped.append(
                {"signal_index": i, "reason": f"signal_generation_error: {exc}"}
            )
            continue

        selected = validation.get("final_strategy_style") or suggestion.get(
            "strategy_style"
        )
        if not signal.get("backtest_executable") or not is_executable_signal(signal):
            skipped.append(
                {
                    "signal_index": i,
                    "signal_time": (
                        str(df.iloc[i]["datetime"])
                        if "datetime" in df.columns
                        else None
                    ),
                    "trend": trend_data.get("trend"),
                    "strength": trend_data.get("signal_strength"),
                    "strategy_suggestion": suggestion.get("strategy_style"),
                    "strategy_source": suggestion.get("source"),
                    "selected_strategy": selected,
                    "signal_status": signal.get("signal_status"),
                    "trade_side": signal.get("trade_side"),
                    "reason": signal.get("reason"),
                    "backtest_rejection_reason": signal.get(
                        "backtest_rejection_reason"
                    ),
                }
            )
            continue

        trade = simulate_future_trade(
            df=df,
            signal_index=i,
            signal=signal,
            target_mode=target_mode,
            entry_mode=entry_mode,
            max_hold_candles=max_hold_candles,
            fee_pct_per_side=fee_pct_per_side,
            slippage_pct=slippage_pct,
        )
        if trade is None:
            skipped.append(
                {
                    "signal_index": i,
                    "selected_strategy": selected,
                    "reason": "signal was executable but could not be simulated.",
                }
            )
            continue

        trade["trend"] = trend_data.get("trend")
        trade["trend_strength"] = trend_data.get("signal_strength")
        trade["strategy_suggestion"] = suggestion.get("strategy_style")
        trade["selected_strategy"] = selected
        trade["strategy_source"] = suggestion.get("source")
        trade["llm_confidence"] = suggestion.get("confidence")
        trades.append(trade)

        if PREVENT_OVERLAPPING_TRADES:
            next_allowed = trade["exit_index"] + 1

    return trades, skipped, llm_stats


def backtest_each_strategy_independently(
    df: pd.DataFrame,
    *,
    target_mode: str = DEFAULT_TARGET_MODE,
    entry_mode: str = DEFAULT_ENTRY_MODE,
    max_hold_candles: int = DEFAULT_MAX_HOLD_CANDLES,
    fee_pct_per_side: float = DEFAULT_FEE_PCT_PER_SIDE,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
) -> Dict[str, List[Dict[str, Any]]]:
    """Diagnostic: test every strategy engine in isolation."""
    trades_by_strategy: Dict[str, List[Dict[str, Any]]] = {
        s: [] for s in ALLOWED_STRATEGIES
    }
    next_allowed_by: Dict[str, int] = {
        s: MIN_HISTORY_CANDLES for s in ALLOWED_STRATEGIES
    }

    for i in range(MIN_HISTORY_CANDLES, len(df) - 1):
        window = df.iloc[: i + 1].copy().reset_index(drop=True)
        try:
            trend_data = analyze_trend(window)
            all_signals = generate_all_strategy_signals(window, trend_data)
        except Exception:
            continue
        for name, signal in all_signals.items():
            if PREVENT_OVERLAPPING_TRADES and i < next_allowed_by[name]:
                continue
            validation = validate_final_signal(signal, trend_data, name)
            if not validation.get("is_hard_valid") or not is_executable_signal(signal):
                continue
            trade = simulate_future_trade(
                df=df,
                signal_index=i,
                signal=signal,
                target_mode=target_mode,
                entry_mode=entry_mode,
                max_hold_candles=max_hold_candles,
                fee_pct_per_side=fee_pct_per_side,
                slippage_pct=slippage_pct,
            )
            if trade is None:
                continue
            trade["trend"] = trend_data.get("trend")
            trade["trend_strength"] = trend_data.get("signal_strength")
            trade["selected_strategy"] = name
            trade["strategy_source"] = "independent_strategy_diagnostic"
            trades_by_strategy[name].append(trade)
            if PREVENT_OVERLAPPING_TRADES:
                next_allowed_by[name] = trade["exit_index"] + 1

    return trades_by_strategy


# ============================================================
# BACKTEST: METRICS
# ============================================================


def max_drawdown_from_r(trades: List[Dict[str, Any]]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += float(t.get("result_r", 0.0))
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return round(max_dd, 4)


def sharpe_like_ratio(results: List[float]) -> Optional[float]:
    """
    Sharpe-like ratio per-trade (mean R / stdev R).
    This is not annualized because trade frequency varies.
    Returns None if fewer than 2 trades or stdev is zero.
    """
    if len(results) < 2:
        return None
    series = pd.Series(results)
    stdev = float(series.std(ddof=1))
    if stdev == 0:
        return None
    return round(float(series.mean()) / stdev, 4)


def sortino_like_ratio(results: List[float]) -> Optional[float]:
    """Sortino-like: mean / downside stdev (only negative results)."""
    if len(results) < 2:
        return None
    series = pd.Series(results)
    downside = series[series < 0]
    if len(downside) < 2:
        return None
    dstd = float(downside.std(ddof=1))
    if dstd == 0:
        return None
    return round(float(series.mean()) / dstd, 4)


def expectancy(results: List[float]) -> float:
    """Average R per trade."""
    if not results:
        return 0.0
    return round(sum(results) / len(results), 4)


def calmar_like_ratio(net_r: float, max_dd_r: float) -> Optional[float]:
    if max_dd_r <= 0:
        return None
    return round(net_r / max_dd_r, 4)


def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate_pct": 0.0,
            "net_r": 0.0,
            "net_r_gross": 0.0,
            "average_r": 0.0,
            "expectancy_r": 0.0,
            "median_r": 0.0,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor": None,
            "max_drawdown_r": 0.0,
            "sharpe_like": None,
            "sortino_like": None,
            "calmar_like": None,
            "average_return_pct": 0.0,
        }

    results = [float(t.get("result_r", 0.0)) for t in trades]
    results_gross = [
        float(t.get("result_r_gross", t.get("result_r", 0.0))) for t in trades
    ]
    returns_pct = [float(t.get("return_pct", 0.0)) for t in trades]

    wins = [r for r in results if r > 0]
    losses = [r for r in results if r < 0]
    breakeven = [r for r in results if r == 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 0 else None

    net_r = round(sum(results), 4)
    max_dd = max_drawdown_from_r(trades)

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate_pct": round(len(wins) / total * 100, 2),
        "net_r": net_r,
        "net_r_gross": round(sum(results_gross), 4),
        "average_r": round(sum(results) / total, 4),
        "expectancy_r": expectancy(results),
        "median_r": round(float(pd.Series(results).median()), 4),
        "gross_profit_r": round(gross_profit, 4),
        "gross_loss_r": round(gross_loss, 4),
        "profit_factor": profit_factor,
        "max_drawdown_r": max_dd,
        "sharpe_like": sharpe_like_ratio(results),
        "sortino_like": sortino_like_ratio(results),
        "calmar_like": calmar_like_ratio(net_r, max_dd),
        "average_return_pct": round(sum(returns_pct) / total, 4),
    }


def summarize_by_strategy(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        key = t.get("strategy_name") or t.get("selected_strategy") or "unknown"
        grouped.setdefault(key, []).append(t)
    return {k: summarize_trades(v) for k, v in grouped.items()}


def summarize_by_side(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        grouped.setdefault(t.get("trade_side", "unknown"), []).append(t)
    return {k: summarize_trades(v) for k, v in grouped.items()}


def compare_metrics(
    llm_metrics: Dict[str, Any], baseline_metrics: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not baseline_metrics:
        return None
    keys = [
        "total_trades",
        "win_rate_pct",
        "net_r",
        "average_r",
        "profit_factor",
        "max_drawdown_r",
        "sharpe_like",
        "sortino_like",
        "calmar_like",
        "average_return_pct",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        a = llm_metrics.get(k)
        b = baseline_metrics.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = {
                "llm": a,
                "no_llm": b,
                "difference_llm_minus_no_llm": round(a - b, 4),
            }
        else:
            out[k] = {"llm": a, "no_llm": b, "difference_llm_minus_no_llm": None}
    return out


# ============================================================
# BACKTEST: OUTPUT
# ============================================================


def save_backtest_outputs(
    symbol: str,
    timeframe: str,
    report: Dict[str, Any],
    trades: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
    prefix: str,
) -> Tuple[str, str, str]:
    ensure_dir(BACKTEST_DIR)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = clean_symbol_for_filename(symbol)
    json_path = os.path.join(
        BACKTEST_DIR, f"{prefix}_report_{safe}_{timeframe}_{stamp}.json"
    )
    trades_csv = os.path.join(
        BACKTEST_DIR, f"{prefix}_trades_{safe}_{timeframe}_{stamp}.csv"
    )
    skipped_csv = os.path.join(
        BACKTEST_DIR, f"{prefix}_skipped_{safe}_{timeframe}_{stamp}.csv"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2)
    pd.DataFrame(trades).to_csv(trades_csv, index=False)
    pd.DataFrame(skipped).to_csv(skipped_csv, index=False)
    return json_path, trades_csv, skipped_csv


def print_backtest_summary(
    report: Dict[str, Any], section_key: str, json_path: str
) -> None:
    metrics = report[section_key]["metrics"]
    print("\n====================")
    print(f"BACKTEST SUMMARY ({section_key})")
    print("====================")
    print(f"Symbol: {report['symbol']}")
    print(f"Timeframe: {report['timeframe']}")
    print(f"Exchange: {report['exchange']}")
    print(f"Execution Mode: {report['execution_mode']}")
    print(f"Target Mode: {report['target_mode']}")
    print(f"Entry Mode: {report['entry_mode']}")
    print(f"Max Hold Candles: {report['max_hold_candles']}")
    print(f"Fee % per side: {report['fee_pct_per_side']}")
    print(f"Slippage %: {report['slippage_pct']}")
    print("--------------------")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    by_strat = report[section_key].get("by_strategy", {})
    if by_strat:
        print("--------------------")
        print("By Strategy:")
        for name, m in by_strat.items():
            print(f"\n{name}")
            print(f"  trades:        {m['total_trades']}")
            print(f"  win_rate_pct:  {m['win_rate_pct']}")
            print(f"  net_r:         {m['net_r']}")
            print(f"  profit_factor: {m['profit_factor']}")
            print(f"  expectancy_r:  {m['expectancy_r']}")
            print(f"  max_dd_r:      {m['max_drawdown_r']}")
            print(f"  sharpe_like:   {m.get('sharpe_like')}")
    if report.get("comparison_llm_vs_no_llm"):
        print("--------------------")
        print("LLM vs No-LLM:")
        for k, v in report["comparison_llm_vs_no_llm"].items():
            print(f"  {k}: {v}")
    print("--------------------")
    print(f"Saved JSON report: {json_path}")
    print("====================")


# ============================================================
# COMMAND: ANALYZE (live signal generation)
# ============================================================


def cmd_analyze(args: argparse.Namespace) -> int:
    symbol = normalize_symbol(args.symbol)
    htf = args.timeframe
    ltf = args.ltf if args.ltf else None
    if ltf:
        validate_timeframe_pair(htf, ltf)

    # HTF data
    raw_df, exchange_name = fetch_ohlcv_dataframe(
        symbol=symbol, timeframe=htf, limit=args.limit
    )
    print(f"HTF candles fetched: {len(raw_df)} ({exchange_name})")
    df = add_indicators(raw_df)
    print(f"HTF candles after warmup: {len(df)}")

    trend_data = analyze_trend(df)
    trend_output = build_trend_output(symbol, htf, exchange_name, trend_data)
    print_trend_report(symbol, htf, exchange_name, trend_data)

    strategy_json = get_llm_strategy_suggestion(
        trend_output, trend_data, use_llm=not args.no_llm
    )
    print("\nSuggested Strategy:")
    print(json.dumps(make_json_safe(strategy_json), indent=2))

    validation = validate_strategy_suggestion(
        trend=trend_data["trend"],
        strength=trend_data["signal_strength"],
        strategy_json=strategy_json,
    )
    print("\nValidation:")
    print(json.dumps(make_json_safe(validation), indent=2))

    final_strategy = validation["final_strategy_style"]
    template_result = get_strategy_template(final_strategy)

    # All strategy signals (internal — for confirmation context).
    all_signals = generate_all_strategy_signals(df=df, trend_data=trend_data)
    signal_result = all_signals.get(final_strategy) or generate_signal(
        df=df, trend_data=trend_data, strategy_name=final_strategy
    )

    supporting_strategies: List[Dict[str, Any]] = []
    if EXECUTION_MODE in ("confirmation_mode", "multi_strategy_mode"):
        supporting_strategies = find_supporting_strategies(
            selected_strategy=final_strategy,
            final_signal=signal_result,
            all_strategy_signals=all_signals,
        )
        signal_result = apply_supporting_confirmation(
            signal_result, supporting_strategies
        )

    hard_validation = validate_final_signal(
        signal=signal_result, trend_data=trend_data, selected_strategy=final_strategy
    )

    # Multi-timeframe LTF trigger (optional).
    ltf_trigger: Optional[Dict[str, Any]] = None
    if ltf and signal_result.get("signal_status") == "valid_signal":
        try:
            ltf_raw, _ = fetch_ohlcv_dataframe(symbol=symbol, timeframe=ltf, limit=300)
            ltf_df = add_indicators(ltf_raw)
            ltf_trend = analyze_trend(ltf_df)
            passed, notes = ltf_trigger_passes(
                ltf_df=ltf_df,
                ltf_trend=ltf_trend,
                htf_direction=trend_data["direction"],
                htf_strategy=final_strategy,
            )
            ltf_trigger = {
                "ltf_timeframe": ltf,
                "passed": passed,
                "ltf_trend": ltf_trend["trend"],
                "ltf_direction": ltf_trend["direction"],
                "notes": notes,
            }
        except Exception as exc:
            ltf_trigger = {"ltf_timeframe": ltf, "passed": False, "error": str(exc)}

    llm_signal_review = review_signal_with_llm(
        trend_output=trend_output,
        strategy_json=strategy_json,
        validation=validation,
        signal_result=signal_result,
        hard_validation=hard_validation,
        supporting_strategies=supporting_strategies,
        enabled=ENABLE_LLM_SIGNAL_REVIEW and not args.no_llm,
    )

    # Position sizing (only if signal is valid).
    sizing_config = PositionSizingConfig(
        account_balance_usd=args.balance,
        risk_per_trade_pct=args.risk_pct,
        max_daily_loss_r=args.max_daily_loss_r,
        max_open_trades=args.max_open_trades,
    )
    position_sizing = compute_position_sizing(signal_result, sizing_config)

    final_decision = build_final_decision(
        final_signal=signal_result,
        hard_validation=hard_validation,
        llm_signal_review=llm_signal_review,
        supporting_strategies=supporting_strategies,
        ltf_trigger=ltf_trigger,
    )

    if args.run_all:
        print_all_strategy_signals(all_signals)

    print_signal_report(
        symbol=symbol,
        timeframe=htf,
        exchange_name=exchange_name,
        trend_data=trend_data,
        strategy_json=strategy_json,
        validation=validation,
        template_result=template_result,
        signal_result=signal_result,
        hard_validation=hard_validation,
        llm_signal_review=llm_signal_review,
        final_decision=final_decision,
        position_sizing=position_sizing,
    )

    report = {
        "created_at_utc": now_utc_iso(),
        "version": "trendv7",
        "symbol": symbol,
        "htf_timeframe": htf,
        "ltf_timeframe": ltf,
        "exchange": exchange_name,
        "execution_mode": EXECUTION_MODE,
        "allow_momentum_as_execution": ALLOW_MOMENTUM_AS_EXECUTION,
        "trend_data": trend_data,
        "strategy_suggestion": strategy_json,
        "validation": validation,
        "strategy_template": template_result,
        "signal": signal_result,
        "hard_signal_validation": hard_validation,
        "llm_signal_review": llm_signal_review,
        "supporting_strategies": supporting_strategies,
        "ltf_trigger": ltf_trigger,
        "position_sizing": position_sizing,
        "final_decision": final_decision,
        "all_strategy_signals": all_signals if args.run_all else None,
        "disclaimer": "Educational decision-support output only. Not financial advice.",
    }
    if args.save:
        saved = save_report_json(symbol, htf, report)
        print(f"\nSaved JSON report to: {saved}")

    return 0


# ============================================================
# COMMAND: BACKTEST
# ============================================================


def cmd_backtest(args: argparse.Namespace) -> int:
    symbol = normalize_symbol(args.symbol)
    timeframe = args.timeframe
    use_llm = args.use_llm
    use_cache = args.cache

    raw_df, exchange_name = fetch_ohlcv_dataframe(
        symbol=symbol, timeframe=timeframe, limit=args.limit
    )
    print(f"Raw candles: {len(raw_df)} ({exchange_name})")
    df = add_indicators(raw_df)
    print(f"Candles after warmup: {len(df)}")

    if len(df) <= MIN_HISTORY_CANDLES + 10:
        raise RuntimeError(
            f"Not enough candles after indicator warmup. Have {len(df)}, "
            f"need more than {MIN_HISTORY_CANDLES + 10}. Increase --limit."
        )

    fee_pct = args.fee_pct
    slippage_pct = args.slippage_pct

    if use_llm:
        trades, skipped, llm_stats = backtest_selected_strategy_with_llm(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            exchange_name=exchange_name,
            target_mode=args.target_mode,
            entry_mode=args.entry_mode,
            max_hold_candles=args.max_hold_candles,
            fee_pct_per_side=fee_pct,
            slippage_pct=slippage_pct,
            use_cache=use_cache,
        )
        prefix = "llm_backtest"
        section_key = "selected_strategy_backtest"
    else:
        trades, skipped = backtest_selected_strategy_no_llm(
            df=df,
            target_mode=args.target_mode,
            entry_mode=args.entry_mode,
            max_hold_candles=args.max_hold_candles,
            fee_pct_per_side=fee_pct,
            slippage_pct=slippage_pct,
        )
        llm_stats = None
        prefix = "no_llm_backtest"
        section_key = "selected_strategy_backtest"

    independent = None
    if args.diagnostic:
        ind_trades = backtest_each_strategy_independently(
            df=df,
            target_mode=args.target_mode,
            entry_mode=args.entry_mode,
            max_hold_candles=args.max_hold_candles,
            fee_pct_per_side=fee_pct,
            slippage_pct=slippage_pct,
        )
        independent = {name: summarize_trades(t) for name, t in ind_trades.items()}

    report = {
        "created_at_utc": now_utc_iso(),
        "version": "trendv7",
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": exchange_name,
        "execution_mode": (
            "selected_strategy_with_llm_selector"
            if use_llm
            else "selected_strategy_no_llm_rule_based_selector"
        ),
        "allow_momentum_as_execution": ALLOW_MOMENTUM_AS_EXECUTION,
        "target_mode": args.target_mode,
        "entry_mode": args.entry_mode,
        "max_hold_candles": args.max_hold_candles,
        "fee_pct_per_side": fee_pct,
        "slippage_pct": slippage_pct,
        "prevent_overlapping_trades": PREVENT_OVERLAPPING_TRADES,
        "minimum_history_candles": MIN_HISTORY_CANDLES,
        "candles_after_indicator_warmup": len(df),
        "llm_config": {"model": LLM_MODEL, "use_cache": use_cache} if use_llm else None,
        "important_note": (
            "Walk-forward backtest. Signals are generated using only candles "
            "available up to that point. Fees and slippage are modelled. "
            "Intrabar TP/SL ambiguity is handled conservatively with SL first."
        ),
        section_key: {
            "metrics": summarize_trades(trades),
            "by_strategy": summarize_by_strategy(trades),
            "by_side": summarize_by_side(trades),
            "completed_trades_count": len(trades),
            "skipped_signals_count": len(skipped),
            "llm_stats": llm_stats,
        },
        "independent_all_strategy_diagnostic": independent,
        "disclaimer": "Educational backtest only. Not financial advice.",
    }

    json_path, trades_csv, skipped_csv = save_backtest_outputs(
        symbol=symbol,
        timeframe=timeframe,
        report=report,
        trades=trades,
        skipped=skipped,
        prefix=prefix,
    )
    print_backtest_summary(report, section_key, json_path)
    print(f"Saved trades CSV: {trades_csv}")
    print(f"Saved skipped CSV: {skipped_csv}")
    return 0


# ============================================================
# COMMAND: COMPARE (run both backtests and report the delta)
# ============================================================


def cmd_compare(args: argparse.Namespace) -> int:
    symbol = normalize_symbol(args.symbol)
    timeframe = args.timeframe

    raw_df, exchange_name = fetch_ohlcv_dataframe(
        symbol=symbol, timeframe=timeframe, limit=args.limit
    )
    print(f"Raw candles: {len(raw_df)} ({exchange_name})")
    df = add_indicators(raw_df)
    print(f"Candles after warmup: {len(df)}")
    if len(df) <= MIN_HISTORY_CANDLES + 10:
        raise RuntimeError(
            f"Not enough candles after indicator warmup. Have {len(df)}, "
            f"need more than {MIN_HISTORY_CANDLES + 10}. Increase --limit."
        )

    fee_pct = args.fee_pct
    slippage_pct = args.slippage_pct

    print("\nRunning LLM backtest...")
    llm_trades, llm_skipped, llm_stats = backtest_selected_strategy_with_llm(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        exchange_name=exchange_name,
        target_mode=args.target_mode,
        entry_mode=args.entry_mode,
        max_hold_candles=args.max_hold_candles,
        fee_pct_per_side=fee_pct,
        slippage_pct=slippage_pct,
        use_cache=args.cache,
    )

    print("\nRunning no-LLM baseline...")
    base_trades, base_skipped = backtest_selected_strategy_no_llm(
        df=df,
        target_mode=args.target_mode,
        entry_mode=args.entry_mode,
        max_hold_candles=args.max_hold_candles,
        fee_pct_per_side=fee_pct,
        slippage_pct=slippage_pct,
    )

    llm_metrics = summarize_trades(llm_trades)
    base_metrics = summarize_trades(base_trades)

    independent = None
    if args.diagnostic:
        ind_trades = backtest_each_strategy_independently(
            df=df,
            target_mode=args.target_mode,
            entry_mode=args.entry_mode,
            max_hold_candles=args.max_hold_candles,
            fee_pct_per_side=fee_pct,
            slippage_pct=slippage_pct,
        )
        independent = {name: summarize_trades(t) for name, t in ind_trades.items()}

    report = {
        "created_at_utc": now_utc_iso(),
        "version": "trendv7",
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": exchange_name,
        "execution_mode": "compare_llm_vs_no_llm",
        "target_mode": args.target_mode,
        "entry_mode": args.entry_mode,
        "max_hold_candles": args.max_hold_candles,
        "fee_pct_per_side": fee_pct,
        "slippage_pct": slippage_pct,
        "llm_config": {"model": LLM_MODEL, "use_cache": args.cache},
        "llm_backtest": {
            "metrics": llm_metrics,
            "by_strategy": summarize_by_strategy(llm_trades),
            "by_side": summarize_by_side(llm_trades),
            "completed_trades_count": len(llm_trades),
            "skipped_signals_count": len(llm_skipped),
            "llm_stats": llm_stats,
        },
        "no_llm_baseline_backtest": {
            "metrics": base_metrics,
            "by_strategy": summarize_by_strategy(base_trades),
            "by_side": summarize_by_side(base_trades),
            "completed_trades_count": len(base_trades),
            "skipped_signals_count": len(base_skipped),
        },
        "comparison_llm_vs_no_llm": compare_metrics(llm_metrics, base_metrics),
        "independent_all_strategy_diagnostic": independent,
        "disclaimer": "Educational backtest only. Not financial advice.",
    }

    json_path, trades_csv, skipped_csv = save_backtest_outputs(
        symbol=symbol,
        timeframe=timeframe,
        report=report,
        trades=llm_trades,
        skipped=llm_skipped,
        prefix="compare",
    )
    print_backtest_summary(report, "llm_backtest", json_path)

    print("\n====================")
    print("NO-LLM BASELINE METRICS")
    print("====================")
    for k, v in base_metrics.items():
        print(f"{k}: {v}")

    print("\n====================")
    print("LLM - NO_LLM DELTA")
    print("====================")
    for k, v in report["comparison_llm_vs_no_llm"].items():
        print(f"  {k}: {v}")

    print(f"\nSaved trades CSV: {trades_csv}")
    print(f"Saved skipped CSV: {skipped_csv}")
    return 0


# ============================================================
# CLI
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trendv7",
        description="trendv7 — consolidated crypto trend-analysis, signal, and backtesting tool.",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # analyze
    p_a = subparsers.add_parser("analyze", help="Run a single live analysis cycle.")
    p_a.add_argument("--symbol", required=True, help="e.g. BTC/USDT, ETH, ETHUSDT")
    p_a.add_argument("--timeframe", required=True, help="HTF, e.g. 15m, 1h, 4h, 1d")
    p_a.add_argument(
        "--ltf", default=None, help="Optional lower timeframe for entry trigger."
    )
    p_a.add_argument("--limit", type=int, default=DEFAULT_LIVE_LIMIT)
    p_a.add_argument(
        "--no-llm",
        action="store_true",
        help="Use only the rule-based selector and skip LLM review.",
    )
    p_a.add_argument(
        "--run-all", action="store_true", help="Print every strategy engine output too."
    )
    p_a.add_argument(
        "--save", action="store_true", help="Save the full JSON report to disk."
    )
    p_a.add_argument("--balance", type=float, default=DEFAULT_ACCOUNT_BALANCE_USD)
    p_a.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PER_TRADE_PCT)
    p_a.add_argument("--max-daily-loss-r", type=float, default=DEFAULT_MAX_DAILY_LOSS_R)
    p_a.add_argument("--max-open-trades", type=int, default=DEFAULT_MAX_OPEN_TRADES)
    p_a.set_defaults(func=cmd_analyze)

    # backtest
    p_b = subparsers.add_parser("backtest", help="Run a historical backtest.")
    p_b.add_argument("--symbol", required=True)
    p_b.add_argument("--timeframe", required=True)
    p_b.add_argument("--limit", type=int, default=DEFAULT_BACKTEST_LIMIT)
    p_b.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the LLM strategy selector (otherwise rule-based).",
    )
    p_b.add_argument(
        "--target-mode", choices=("tp1", "tp2"), default=DEFAULT_TARGET_MODE
    )
    p_b.add_argument(
        "--entry-mode",
        choices=("signal_entry", "next_open"),
        default=DEFAULT_ENTRY_MODE,
    )
    p_b.add_argument("--max-hold-candles", type=int, default=DEFAULT_MAX_HOLD_CANDLES)
    p_b.add_argument(
        "--fee-pct",
        type=float,
        default=DEFAULT_FEE_PCT_PER_SIDE,
        help="Fee percent per side, e.g. 0.05 for 0.05%%.",
    )
    p_b.add_argument("--slippage-pct", type=float, default=DEFAULT_SLIPPAGE_PCT)
    p_b.add_argument(
        "--cache",
        action="store_true",
        help="Cache LLM responses by trend output (faster, approximate).",
    )
    p_b.add_argument(
        "--diagnostic",
        action="store_true",
        help="Also run independent per-strategy diagnostic.",
    )
    p_b.set_defaults(func=cmd_backtest)

    # compare
    p_c = subparsers.add_parser(
        "compare", help="Run LLM and no-LLM backtests on the same window and diff them."
    )
    p_c.add_argument("--symbol", required=True)
    p_c.add_argument("--timeframe", required=True)
    p_c.add_argument("--limit", type=int, default=DEFAULT_BACKTEST_LIMIT)
    p_c.add_argument(
        "--target-mode", choices=("tp1", "tp2"), default=DEFAULT_TARGET_MODE
    )
    p_c.add_argument(
        "--entry-mode",
        choices=("signal_entry", "next_open"),
        default=DEFAULT_ENTRY_MODE,
    )
    p_c.add_argument("--max-hold-candles", type=int, default=DEFAULT_MAX_HOLD_CANDLES)
    p_c.add_argument("--fee-pct", type=float, default=DEFAULT_FEE_PCT_PER_SIDE)
    p_c.add_argument("--slippage-pct", type=float, default=DEFAULT_SLIPPAGE_PCT)
    p_c.add_argument("--cache", action="store_true")
    p_c.add_argument("--diagnostic", action="store_true")
    p_c.set_defaults(func=cmd_compare)

    return parser


def interactive_main() -> int:
    """Fallback prompt-based UX when no subcommand is given."""
    print("trendv7 — interactive mode")
    print("Commands: analyze | backtest | compare")
    command = input("Command [analyze]: ").strip().lower() or "analyze"

    raw_symbol = input("Symbol (e.g. BTC/USDT, ETH, ETHUSDT): ").strip().upper()
    timeframe = input("Timeframe (e.g. 15m, 1h, 4h, 1d): ").strip()

    parser = build_arg_parser()

    if command == "analyze":
        ltf = input("Optional LTF for entry trigger (blank = none): ").strip() or None
        raw_limit = input(f"Candle limit (default {DEFAULT_LIVE_LIMIT}): ").strip()
        limit = int(raw_limit) if raw_limit else DEFAULT_LIVE_LIMIT
        use_llm = input("Use LLM? (y/n, default y): ").strip().lower() != "n"
        run_all = (
            input("Print every strategy engine output? (y/n, default n): ")
            .strip()
            .lower()
            == "y"
        )
        save = input("Save JSON report? (y/n, default y): ").strip().lower() != "n"
        args = parser.parse_args(
            [
                "analyze",
                "--symbol",
                raw_symbol,
                "--timeframe",
                timeframe,
                *(["--ltf", ltf] if ltf else []),
                "--limit",
                str(limit),
                *(["--no-llm"] if not use_llm else []),
                *(["--run-all"] if run_all else []),
                *(["--save"] if save else []),
            ]
        )
        return cmd_analyze(args)

    if command in ("backtest", "compare"):
        raw_limit = input(f"Candle limit (default {DEFAULT_BACKTEST_LIMIT}): ").strip()
        limit = int(raw_limit) if raw_limit else DEFAULT_BACKTEST_LIMIT
        target = (
            input(f"Target mode tp1/tp2 (default {DEFAULT_TARGET_MODE}): ")
            .strip()
            .lower()
            or DEFAULT_TARGET_MODE
        )
        entry = (
            input(f"Entry mode signal_entry/next_open (default {DEFAULT_ENTRY_MODE}): ")
            .strip()
            .lower()
            or DEFAULT_ENTRY_MODE
        )
        max_hold_raw = input(
            f"Max hold candles (default {DEFAULT_MAX_HOLD_CANDLES}): "
        ).strip()
        max_hold = int(max_hold_raw) if max_hold_raw else DEFAULT_MAX_HOLD_CANDLES

        if command == "backtest":
            use_llm = (
                input("Use LLM selector? (y/n, default n): ").strip().lower() == "y"
            )
            diagnostic = (
                input("Also run per-strategy diagnostic? (y/n, default n): ")
                .strip()
                .lower()
                == "y"
            )
            args_list = [
                "backtest",
                "--symbol",
                raw_symbol,
                "--timeframe",
                timeframe,
                "--limit",
                str(limit),
                "--target-mode",
                target,
                "--entry-mode",
                entry,
                "--max-hold-candles",
                str(max_hold),
            ]
            if use_llm:
                args_list.append("--use-llm")
            if diagnostic:
                args_list.append("--diagnostic")
            args = parser.parse_args(args_list)
            return cmd_backtest(args)

        # compare
        diagnostic = (
            input("Also run per-strategy diagnostic? (y/n, default n): ")
            .strip()
            .lower()
            == "y"
        )
        args_list = [
            "compare",
            "--symbol",
            raw_symbol,
            "--timeframe",
            timeframe,
            "--limit",
            str(limit),
            "--target-mode",
            target,
            "--entry-mode",
            entry,
            "--max-hold-candles",
            str(max_hold),
        ]
        if diagnostic:
            args_list.append("--diagnostic")
        args = parser.parse_args(args_list)
        return cmd_compare(args)

    print(f"Unknown command: {command}")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return interactive_main()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
