import os
import json
from datetime import datetime, timezone

import ccxt
import pandas as pd
import ta
from dotenv import load_dotenv

try:
    from agents import Agent, Runner
except ImportError:
    Agent = None
    Runner = None


load_dotenv()

# ==========================================
# CONFIG
# ==========================================

OHLCV_LIMIT = 1000
OUTPUT_DIR = "outputs"


# ==========================================
# BASIC HELPERS
# ==========================================

def normalize_symbol(value):
    """Accept common user inputs and convert them to a ccxt spot symbol."""
    value = value.replace("-", "/").strip().upper()

    if "/" in value:
        return value

    if value.endswith("USDT"):
        return f"{value[:-4]}/USDT"

    return f"{value}/USDT"


def make_json_safe(obj):
    """Convert pandas/numpy scalar values to normal Python values for json.dumps."""
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]

    if hasattr(obj, "item"):
        return obj.item()

    return obj


def safe_quality_score(points, max_points):
    if max_points == 0:
        return 0.0
    return round(points / max_points, 2)


def infer_direction_from_trend(trend):
    if "Bullish" in trend:
        return "bullish"
    if "Bearish" in trend:
        return "bearish"
    return "neutral"


def round_price(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


# ==========================================
# DATA FUNCTIONS
# ==========================================

def fetch_ohlcv_dataframe(symbol, timeframe, limit=OHLCV_LIMIT):
    exchanges = [
        ccxt.binance({"enableRateLimit": True}),
        ccxt.kucoin({"enableRateLimit": True}),
    ]

    last_error = None

    for exchange in exchanges:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df, exchange.id
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not fetch candles for {symbol}: {last_error}")


def add_indicators(df):
    df = df.copy()

    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["atr"] = ta.volatility.average_true_range(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )

    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        raise RuntimeError("Not enough candles after indicator warmup.")

    return df


# ==========================================
# MARKET STRUCTURE
# ==========================================

def get_recent_structure(df, lookback=8):
    recent_highs = df["high"].tail(lookback).values
    recent_lows = df["low"].tail(lookback).values

    higher_high_count = sum(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
    higher_low_count = sum(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))
    lower_high_count = sum(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))
    lower_low_count = sum(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))

    structure_lookback = lookback - 1

    return {
        "higher_high_count": higher_high_count,
        "higher_low_count": higher_low_count,
        "lower_high_count": lower_high_count,
        "lower_low_count": lower_low_count,
        "structure_lookback": structure_lookback,
        "bullish_structure": higher_high_count >= 4 and higher_low_count >= 4,
        "bearish_structure": lower_high_count >= 4 and lower_low_count >= 4,
    }


def get_support_resistance(df, lookback=40):
    recent = df.tail(lookback)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    midpoint = (support + resistance) / 2
    return {
        "support": support,
        "resistance": resistance,
        "midpoint": midpoint,
        "range_percent": (resistance - support) / float(df.iloc[-1]["close"]) * 100,
    }


def recent_swing_low(df, lookback=12):
    return float(df["low"].tail(lookback).min())


def recent_swing_high(df, lookback=12):
    return float(df["high"].tail(lookback).max())


# ==========================================
# TREND ANALYSIS
# ==========================================

def analyze_trend(df):
    latest = df.iloc[-1]

    price = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    ema200 = latest["ema200"]
    rsi = latest["rsi"]
    macd_value = latest["macd"]
    macd_signal = latest["macd_signal"]
    atr = latest["atr"]

    recent_highs = df["high"].tail(8).values
    recent_lows = df["low"].tail(8).values
    recent_closes = df["close"].tail(8).values

    higher_high_count = sum(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
    higher_low_count = sum(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))
    lower_high_count = sum(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))
    lower_low_count = sum(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))

    structure_lookback = len(recent_highs) - 1
    higher_highs = higher_high_count >= 4
    higher_lows = higher_low_count >= 4
    lower_highs = lower_high_count >= 4
    lower_lows = lower_low_count >= 4

    close_change_atr = (recent_closes[-1] - recent_closes[0]) / atr if atr > 0 else 0

    ema_distance = abs(ema50 - ema200) / price * 100
    ema20_ema50_distance = abs(ema20 - ema50) / price * 100
    atr_percent = atr / price * 100

    recent_range = (df["high"].tail(20).max() - df["low"].tail(20).min()) / price * 100

    trend = "Unclear"
    signal_strength = "Weak"
    bull_score = 0
    bear_score = 0
    notes = []

    def add_vote(condition, bull_note, bear_note):
        nonlocal bull_score, bear_score
        if condition:
            bull_score += 1
            notes.append(bull_note)
        else:
            bear_score += 1
            notes.append(bear_note)

    add_vote(price > ema20, "price is above EMA20", "price is below EMA20")
    add_vote(ema20 > ema50, "EMA20 is above EMA50", "EMA20 is below EMA50")
    add_vote(ema50 > ema200, "EMA50 is above EMA200", "EMA50 is below EMA200")
    add_vote(macd_value > macd_signal, "MACD is above signal", "MACD is below signal")

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
        notes.append(
            f"market structure leans higher ({higher_high_count}/{structure_lookback} HH, "
            f"{higher_low_count}/{structure_lookback} HL)"
        )
    elif lower_highs and lower_lows:
        bear_score += 1
        notes.append(
            f"market structure leans lower ({lower_high_count}/{structure_lookback} LH, "
            f"{lower_low_count}/{structure_lookback} LL)"
        )
    else:
        notes.append(
            f"market structure is mixed ({higher_high_count}/{structure_lookback} HH, "
            f"{higher_low_count}/{structure_lookback} HL, "
            f"{lower_high_count}/{structure_lookback} LH, {lower_low_count}/{structure_lookback} LL)"
        )

    is_sideways = (
        ema20_ema50_distance < 0.25
        and ema_distance < 1
        and 47 <= rsi <= 53
        and abs(close_change_atr) < 0.35
        and recent_range < max(2.5, atr_percent * 4)
    )

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

    leading_score = max(bull_score, bear_score)

    if leading_score >= 6:
        signal_strength = "Strong"
    elif leading_score >= 5:
        signal_strength = "Moderate"

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
        "macd_signal": macd_signal,
        "atr": atr,
        "atr_percent": atr_percent,
        "recent_range": recent_range,
        "close_change_atr": close_change_atr,
        "higher_high_count": higher_high_count,
        "higher_low_count": higher_low_count,
        "lower_high_count": lower_high_count,
        "lower_low_count": lower_low_count,
        "structure_lookback": structure_lookback,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "range_midpoint": sr["midpoint"],
        "notes": notes,
    }


def build_trend_output(symbol, timeframe, exchange_name, trend_data):
    notes = trend_data["notes"]

    return f"""
Symbol: {symbol}
Timeframe: {timeframe}
Exchange: {exchange_name}
Trend: {trend_data["trend"]}
Strength: {trend_data["signal_strength"]}
Score: {trend_data["bull_score"]} bullish / {trend_data["bear_score"]} bearish
Price: {trend_data["price"]:.4f}
EMA20 / EMA50 / EMA200: {trend_data["ema20"]:.4f} / {trend_data["ema50"]:.4f} / {trend_data["ema200"]:.4f}
RSI: {trend_data["rsi"]:.2f}
MACD / Signal: {trend_data["macd_value"]:.4f} / {trend_data["macd_signal"]:.4f}
ATR: {trend_data["atr"]:.4f} ({trend_data["atr_percent"]:.2f}% of price)
Recent range: {trend_data["recent_range"]:.2f}%
Support / Resistance: {trend_data["support"]:.4f} / {trend_data["resistance"]:.4f}
Reasons:
{chr(10).join(f"- {note}" for note in notes)}
""".strip()


def print_trend_report(symbol, timeframe, exchange_name, trend_data):
    print("\n====================")
    print("TREND REPORT")
    print("====================")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Exchange: {exchange_name}")
    print(f"Trend: {trend_data['trend']}")
    print(f"Strength: {trend_data['signal_strength']}")
    print(f"Score: {trend_data['bull_score']} bullish / {trend_data['bear_score']} bearish")
    print("--------------------")
    print(f"Price: {trend_data['price']:.4f}")
    print(
        f"EMA20 / EMA50 / EMA200: "
        f"{trend_data['ema20']:.4f} / {trend_data['ema50']:.4f} / {trend_data['ema200']:.4f}"
    )
    print(f"RSI: {trend_data['rsi']:.2f}")
    print(f"MACD / Signal: {trend_data['macd_value']:.4f} / {trend_data['macd_signal']:.4f}")
    print(f"ATR: {trend_data['atr']:.4f} ({trend_data['atr_percent']:.2f}% of price)")
    print(f"Recent range: {trend_data['recent_range']:.2f}%")
    print(f"Support / Resistance: {trend_data['support']:.4f} / {trend_data['resistance']:.4f}")
    print("--------------------")
    print("Reasons:")
    for note in trend_data["notes"]:
        print(f"- {note}")


# ==========================================
# STRATEGY SELECTION
# ==========================================

ALLOWED_STRATEGIES = {
    "pullback continuation strategy",
    "breakout-and-retest strategy",
    "range-bound strategy",
    "momentum continuation strategy",
    "reversal-confirmation strategy",
    "mixed/confirmation strategy",
}


def get_rule_based_strategy_suggestion(trend_data):
    """Fallback selector when the LLM is unavailable."""
    trend = trend_data["trend"]
    strength = trend_data["signal_strength"]

    if "Sideways" in trend:
        strategy = "range-bound strategy"
        condition = "ranging"
        confidence = 0.72
        reason = "Trend analysis classified the market as sideways/ranging."
    elif "Weak" in trend or "Unclear" in trend or strength == "Weak":
        strategy = "mixed/confirmation strategy"
        condition = "weak_trending" if "Weak" in trend else "unclear"
        confidence = 0.60
        reason = "Trend is weak or unclear, so confirmation is needed before choosing a directional setup."
    elif strength == "Strong":
        strategy = "momentum continuation strategy"
        condition = "strong_trending"
        confidence = 0.75
        reason = "Trend strength is strong, so momentum continuation is the default fit."
    else:
        strategy = "pullback continuation strategy"
        condition = "weak_trending" if strength == "Weak" else "strong_trending"
        confidence = 0.68
        reason = "Trend is directional but not extreme, so pullback continuation is preferred."

    return {
        "strategy_style": strategy,
        "market_condition": condition,
        "confidence": confidence,
        "reason": reason,
        "source": "rule_based_fallback",
    }


def get_llm_strategy_suggestion(trend_output, trend_data):
    """Use LLM if available; otherwise use rule-based fallback so the script still works."""
    if Agent is None or Runner is None or not os.getenv("OPENAI_API_KEY"):
        return get_rule_based_strategy_suggestion(trend_data)

    strategy_agent = Agent(
        name="Trend Strategy Suggestion Agent",
        model="gpt-4o-mini",
        instructions="""
You are a crypto trading strategy classifier.

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

    agent_input = f"""
Trend output:
{trend_output}

Suggest the best-fit strategy style for this trend.
""".strip()

    strategy_result = Runner.run_sync(strategy_agent, agent_input, max_turns=1)

    raw_output = (strategy_result.final_output or "").strip()

    if not raw_output:
        print("LLM returned empty output — falling back to rule-based strategy.")
        fallback = get_rule_based_strategy_suggestion(trend_data)
        fallback["llm_raw_output"] = raw_output
        fallback["source"] = "llm_empty_fallback"
        return fallback

    try:
        result = json.loads(raw_output)
    except Exception:
        # Try to extract a JSON object from noisy LLM output
        import re

        m = re.search(r"(\{.*\})", raw_output, re.S)
        if m:
            candidate = m.group(1)
            try:
                result = json.loads(candidate)
            except Exception as e:
                print("Failed to parse JSON from LLM output:", e)
                print("LLM raw output:", raw_output)
                fallback = get_rule_based_strategy_suggestion(trend_data)
                fallback["llm_raw_output"] = raw_output
                fallback["source"] = "llm_invalid_json_fallback"
                return fallback
        else:
            print("LLM output did not contain JSON. Raw output:", raw_output)
            fallback = get_rule_based_strategy_suggestion(trend_data)
            fallback["llm_raw_output"] = raw_output
            fallback["source"] = "llm_no_json_fallback"
            return fallback

    result["source"] = "llm"
    return result


def validate_strategy_suggestion(trend, strength, strategy_json):
    strategy = strategy_json.get("strategy_style")
    confidence = strategy_json.get("confidence", 0)

    result = {
        "is_valid": True,
        "warnings": [],
        "final_strategy_style": strategy,
        "validator_note": "",
    }

    if strategy not in ALLOWED_STRATEGIES:
        result["is_valid"] = False
        result["warnings"].append("Strategy style is not in the allowed list.")

    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        result["is_valid"] = False
        result["warnings"].append("Confidence must be a number between 0 and 1.")

    weak_or_unclear = "Weak" in trend or "Unclear" in trend or "Sideways" in trend or strength == "Weak"

    trend_following_strategies = {
        "pullback continuation strategy",
        "breakout-and-retest strategy",
        "momentum continuation strategy",
    }

    if weak_or_unclear and strategy in trend_following_strategies:
        result["is_valid"] = False
        result["warnings"].append(
            "Weak/unclear/sideways trend should not use a trend-following strategy."
        )
        result["final_strategy_style"] = "mixed/confirmation strategy"

    if "Sideways" in trend:
        result["final_strategy_style"] = "range-bound strategy"
        if strategy != "range-bound strategy":
            result["warnings"].append("Sideways condition was redirected to range-bound strategy.")

    if result["is_valid"]:
        result["validator_note"] = "Strategy suggestion is consistent with trend output."
    else:
        result["validator_note"] = "Strategy suggestion conflicts with rule-based validation."

    return result


# ==========================================
# STRATEGY TEMPLATES
# ==========================================

STRATEGY_TEMPLATES = {
    "pullback continuation strategy": {
        "type": "trend_following",
        "best_for": ["strong_trending", "moderate_trending"],
        "confirmation_needed": [
            "trend remains aligned",
            "price pulls back near EMA20 or EMA50",
            "momentum improves again",
        ],
        "risk_note": "Avoid if structure is mixed or trend strength is weak.",
    },
    "breakout-and-retest strategy": {
        "type": "trend_following",
        "best_for": ["strong_trending", "expansion"],
        "confirmation_needed": [
            "price breaks a recent range/high/low",
            "retest holds",
            "volume or momentum confirms",
        ],
        "risk_note": "Avoid during low range or unclear structure.",
    },
    "range-bound strategy": {
        "type": "mean_reversion",
        "best_for": ["ranging"],
        "confirmation_needed": [
            "price stays inside recent range",
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
        ],
        "risk_note": "Avoid if RSI is extreme or price is too extended.",
    },
    "reversal-confirmation strategy": {
        "type": "reversal",
        "best_for": ["exhaustion", "failed continuation"],
        "confirmation_needed": [
            "trend weakens",
            "momentum turns",
            "structure starts reversing",
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


def get_strategy_template(strategy_name):
    template = STRATEGY_TEMPLATES.get(strategy_name)
    if template is None:
        return {"template_found": False, "error": "No template found for this strategy style."}
    return {"template_found": True, "strategy_style": strategy_name, "template": template}


# ==========================================
# SIGNAL / RISK HELPERS
# ==========================================

def calculate_risk_reward(direction, entry, stop_loss, take_profit):
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
    strategy_name,
    signal_status,
    direction="neutral",
    entry_price=None,
    stop_loss=None,
    take_profit_1=None,
    take_profit_2=None,
    setup_quality=0.0,
    confidence_score=0.0,
    risk_level="unknown",
    reason="",
    notes=None,
):
    rr1 = calculate_risk_reward(direction, entry_price, stop_loss, take_profit_1)
    rr2 = calculate_risk_reward(direction, entry_price, stop_loss, take_profit_2)

    return {
        "strategy_name": strategy_name,
        "signal_status": signal_status,
        "direction": direction,
        "entry_price": round_price(entry_price),
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


def no_signal(strategy_name, reason, notes=None):
    return build_signal_result(
        strategy_name=strategy_name,
        signal_status="no_valid_signal",
        direction="neutral",
        reason=reason,
        notes=notes or [],
    )


def make_r_multiple_targets(direction, entry, stop_loss, r1=1.0, r2=2.0):
    if direction == "bullish":
        risk = entry - stop_loss
        return entry + risk * r1, entry + risk * r2
    if direction == "bearish":
        risk = stop_loss - entry
        return entry - risk * r1, entry - risk * r2
    return None, None


def valid_minimum_rr(signal, minimum_rr=1.2):
    rr1 = signal.get("risk_reward_1")
    rr2 = signal.get("risk_reward_2")
    if rr1 is None or rr2 is None:
        return False
    return rr1 >= minimum_rr or rr2 >= minimum_rr


# ==========================================
# STRATEGY SIGNAL ENGINES
# ==========================================

def generate_pullback_continuation_signal(df, trend_data):
    strategy = "pullback continuation strategy"
    latest = df.iloc[-1]
    direction = trend_data["direction"]

    if direction not in ["bullish", "bearish"]:
        return no_signal(strategy, "Trend direction is not clear enough for pullback continuation.")

    price = float(latest["close"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])
    ema200 = float(latest["ema200"])
    rsi = float(latest["rsi"])
    macd = float(latest["macd"])
    macd_signal = float(latest["macd_signal"])
    atr = float(latest["atr"])
    structure = get_recent_structure(df)

    near_ema20 = abs(price - ema20) / price < 0.006
    near_ema50 = abs(price - ema50) / price < 0.010
    pullback_area = near_ema20 or near_ema50

    if direction == "bullish":
        trend_alignment = ema20 > ema50
        ema200_supportive = price > ema200
        momentum_ok = rsi > 50 and macd > macd_signal
        structure_ok = structure["bullish_structure"]
        swing_low = recent_swing_low(df)
        stop_candidates = [swing_low - atr * 0.10, price - atr * 1.50, ema50 - atr * 0.50]
        valid_stops = [x for x in stop_candidates if x < price]
        stop_loss = max(valid_stops) if valid_stops else price - atr * 1.5
    else:
        trend_alignment = ema20 < ema50 or price < ema20
        ema200_supportive = price < ema200
        momentum_ok = rsi < 50 and macd < macd_signal
        structure_ok = structure["bearish_structure"]
        swing_high = recent_swing_high(df)
        stop_candidates = [swing_high + atr * 0.10, price + atr * 1.50, ema50 + atr * 0.50]
        valid_stops = [x for x in stop_candidates if x > price]
        stop_loss = min(valid_stops) if valid_stops else price + atr * 1.5

    points = 0
    points += 1 if trend_alignment else 0
    points += 1 if ema200_supportive else 0
    points += 1 if pullback_area else 0
    points += 1 if momentum_ok else 0
    points += 1 if structure_ok else 0
    setup_quality = safe_quality_score(points, 5)

    if not pullback_area:
        return no_signal(
            strategy,
            "Price is not close enough to EMA20 or EMA50 pullback area.",
            [
                f"near_ema20={near_ema20}",
                f"near_ema50={near_ema50}",
                f"setup_quality={setup_quality}",
            ],
        )

    if setup_quality < 0.65:
        return no_signal(
            strategy,
            "Pullback exists, but confirmation quality is not strong enough.",
            [
                f"trend_alignment={trend_alignment}",
                f"ema200_supportive={ema200_supportive}",
                f"momentum_ok={momentum_ok}",
                f"structure_ok={structure_ok}",
                f"setup_quality={setup_quality}",
            ],
        )

    entry = price
    tp1, tp2 = make_r_multiple_targets(direction, entry, stop_loss, r1=1.5, r2=2.5)

    risk_level = "medium" if ema200_supportive and setup_quality >= 0.75 else "high"
    confidence = min(0.90, setup_quality * 0.85 + 0.10)

    signal = build_signal_result(
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
        reason="Directional trend with price near EMA pullback area and sufficient confirmation.",
        notes=[
            f"trend_alignment={trend_alignment}",
            f"ema200_supportive={ema200_supportive}",
            f"pullback_area={pullback_area}",
            f"momentum_ok={momentum_ok}",
            f"structure_ok={structure_ok}",
        ],
    )

    return signal


def generate_range_bound_signal(df, trend_data):
    strategy = "range-bound strategy"
    latest = df.iloc[-1]
    price = float(latest["close"])
    rsi = float(latest["rsi"])
    atr = float(latest["atr"])

    sr = get_support_resistance(df, lookback=40)
    support = sr["support"]
    resistance = sr["resistance"]
    midpoint = sr["midpoint"]

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
            [f"support={support:.4f}", f"resistance={resistance:.4f}", f"price={price:.4f}"],
        )

    points = 0
    points += 1 if inside_range else 0
    points += 1 if rsi_neutral else 0
    points += 1 if near_support or near_resistance else 0
    points += 1 if sr["range_percent"] < max(4.0, trend_data["atr_percent"] * 6) else 0
    setup_quality = safe_quality_score(points, 4)
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
            f"support={support:.4f}",
            f"resistance={resistance:.4f}",
            f"midpoint={midpoint:.4f}",
            f"near_support={near_support}",
            f"near_resistance={near_resistance}",
            f"rsi_neutral={rsi_neutral}",
        ],
    )


def generate_breakout_retest_signal(df, trend_data):
    strategy = "breakout-and-retest strategy"
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    direction = trend_data["direction"]

    if direction not in ["bullish", "bearish"]:
        return no_signal(strategy, "Trend direction is not clear enough for breakout-and-retest.")

    price = float(latest["close"])
    atr = float(latest["atr"])
    rsi = float(latest["rsi"])
    macd = float(latest["macd"])
    macd_signal = float(latest["macd_signal"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])

    recent_resistance = float(df["high"].tail(20).iloc[:-1].max())
    recent_support = float(df["low"].tail(20).iloc[:-1].min())

    if direction == "bullish":
        broke_level = float(previous["close"]) > recent_resistance
        retest_near_level = abs(price - recent_resistance) <= atr * 0.75
        trend_ok = ema20 > ema50
        momentum_ok = rsi > 50 and macd > macd_signal
        entry = price
        stop_loss = recent_resistance - atr * 0.75
        tp1, tp2 = make_r_multiple_targets(direction, entry, stop_loss, r1=1.5, r2=2.5)
        level_name = "resistance_retest_as_support"
    else:
        broke_level = float(previous["close"]) < recent_support
        retest_near_level = abs(price - recent_support) <= atr * 0.75
        trend_ok = ema20 < ema50 or price < ema20
        momentum_ok = rsi < 50 and macd < macd_signal
        entry = price
        stop_loss = recent_support + atr * 0.75
        tp1, tp2 = make_r_multiple_targets(direction, entry, stop_loss, r1=1.5, r2=2.5)
        level_name = "support_retest_as_resistance"

    points = 0
    points += 1 if broke_level else 0
    points += 1 if retest_near_level else 0
    points += 1 if trend_ok else 0
    points += 1 if momentum_ok else 0
    setup_quality = safe_quality_score(points, 4)

    if not broke_level or not retest_near_level:
        return no_signal(
            strategy,
            "Breakout and retest conditions are not both present.",
            [
                f"level_name={level_name}",
                f"recent_resistance={recent_resistance:.4f}",
                f"recent_support={recent_support:.4f}",
                f"broke_level={broke_level}",
                f"retest_near_level={retest_near_level}",
                f"setup_quality={setup_quality}",
            ],
        )

    if setup_quality < 0.75:
        return no_signal(strategy, "Breakout/retest is present but confirmation is weak.")

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
        reason="Price broke a recent level and is retesting it with trend/momentum confirmation.",
        notes=[
            f"level_name={level_name}",
            f"broke_level={broke_level}",
            f"retest_near_level={retest_near_level}",
            f"trend_ok={trend_ok}",
            f"momentum_ok={momentum_ok}",
        ],
    )


def generate_momentum_continuation_signal(df, trend_data):
    strategy = "momentum continuation strategy"
    latest = df.iloc[-1]
    direction = trend_data["direction"]

    if direction not in ["bullish", "bearish"]:
        return no_signal(strategy, "Trend direction is not clear enough for momentum continuation.")

    price = float(latest["close"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])
    ema200 = float(latest["ema200"])
    rsi = float(latest["rsi"])
    macd = float(latest["macd"])
    macd_signal = float(latest["macd_signal"])
    atr = float(latest["atr"])
    structure = get_recent_structure(df)

    recent_closes = df["close"].tail(8).values
    close_change_atr = (recent_closes[-1] - recent_closes[0]) / atr if atr > 0 else 0

    if direction == "bullish":
        trend_fully_aligned = price > ema20 > ema50 > ema200
        momentum_ok = macd > macd_signal and rsi > 55
        rsi_not_extreme = rsi < 75
        slope_strong = close_change_atr > 0.7
        structure_ok = structure["bullish_structure"]
        entry = price
        stop_loss = min(ema20 - atr * 0.50, price - atr * 1.25)
        tp1, tp2 = make_r_multiple_targets(direction, entry, stop_loss, r1=1.5, r2=3.0)
    else:
        trend_fully_aligned = price < ema20 < ema50 < ema200
        momentum_ok = macd < macd_signal and rsi < 45
        rsi_not_extreme = rsi > 25
        slope_strong = close_change_atr < -0.7
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

    if setup_quality < 0.75:
        return no_signal(
            strategy,
            "Momentum continuation quality is not strong enough.",
            [
                f"trend_fully_aligned={trend_fully_aligned}",
                f"momentum_ok={momentum_ok}",
                f"rsi_not_extreme={rsi_not_extreme}",
                f"slope_strong={slope_strong}",
                f"structure_ok={structure_ok}",
                f"setup_quality={setup_quality}",
            ],
        )

    confidence = min(0.90, setup_quality * 0.85 + 0.05)
    risk_level = "medium" if rsi_not_extreme else "high"

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
        reason="Strong directional alignment and momentum support continuation.",
        notes=[
            f"trend_fully_aligned={trend_fully_aligned}",
            f"momentum_ok={momentum_ok}",
            f"rsi_not_extreme={rsi_not_extreme}",
            f"slope_strong={slope_strong}",
            f"structure_ok={structure_ok}",
            f"close_change_atr={close_change_atr:.2f}",
        ],
    )


def generate_reversal_confirmation_signal(df, trend_data):
    strategy = "reversal-confirmation strategy"
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(latest["close"])
    ema20 = float(latest["ema20"])
    ema50 = float(latest["ema50"])
    ema200 = float(latest["ema200"])
    rsi = float(latest["rsi"])
    macd = float(latest["macd"])
    macd_signal = float(latest["macd_signal"])
    atr = float(latest["atr"])
    previous_rsi = float(previous["rsi"])
    previous_macd = float(previous["macd"])
    previous_macd_signal = float(previous["macd_signal"])
    structure = get_recent_structure(df)

    bullish_conditions = {
        "trend_was_weak": ema20 < ema50 or price < ema200,
        "rsi_recovering": previous_rsi < 47 and rsi > previous_rsi,
        "macd_turning": previous_macd < previous_macd_signal and macd > macd_signal,
        "price_reclaiming_ema20": float(previous["close"]) < float(previous["ema20"]) and price > ema20,
        "structure_improving": structure["higher_low_count"] >= 4,
    }

    bearish_conditions = {
        "trend_was_weak": ema20 > ema50 or price > ema200,
        "rsi_recovering": previous_rsi > 53 and rsi < previous_rsi,
        "macd_turning": previous_macd > previous_macd_signal and macd < macd_signal,
        "price_reclaiming_ema20": float(previous["close"]) > float(previous["ema20"]) and price < ema20,
        "structure_improving": structure["lower_high_count"] >= 4,
    }

    bullish_points = sum(1 for value in bullish_conditions.values() if value)
    bearish_points = sum(1 for value in bearish_conditions.values() if value)

    if bullish_points > bearish_points:
        direction = "bullish"
        conditions = bullish_conditions
        setup_quality = safe_quality_score(bullish_points, 5)
        entry = price
        stop_loss = recent_swing_low(df) - atr * 0.25
        tp1 = max(ema50, entry + (entry - stop_loss) * 1.0)
        tp2 = max(ema200, entry + (entry - stop_loss) * 2.0)
    elif bearish_points > bullish_points:
        direction = "bearish"
        conditions = bearish_conditions
        setup_quality = safe_quality_score(bearish_points, 5)
        entry = price
        stop_loss = recent_swing_high(df) + atr * 0.25
        tp1 = min(ema50, entry - (stop_loss - entry) * 1.0)
        tp2 = min(ema200, entry - (stop_loss - entry) * 2.0)
    else:
        return no_signal(strategy, "No clear bullish or bearish reversal direction.")

    if setup_quality < 0.65:
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
        reason="Potential reversal has multiple confirmation signs, but reversal setups remain high risk.",
        notes=[f"{k}={v}" for k, v in conditions.items()],
    )


def generate_mixed_confirmation_signal(df, trend_data):
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


def generate_signal(df, trend_data, strategy_name):
    generator = SIGNAL_GENERATORS.get(strategy_name)
    if generator is None:
        return no_signal(strategy_name, "No signal generator exists for this strategy.")

    signal = generator(df, trend_data)

    if signal["signal_status"] == "valid_signal" and not valid_minimum_rr(signal, minimum_rr=1.2):
        signal["signal_status"] = "no_valid_signal"
        signal["reason"] = "Setup detected, but risk/reward is below the minimum acceptable threshold."
        signal["risk_level"] = "high"

    return signal


# ==========================================
# REPORTING
# ==========================================

def print_signal_report(symbol, timeframe, exchange_name, trend_data, strategy_json, validation, template_result, signal_result):
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
        for warning in validation["warnings"]:
            print(f"- {warning}")
    print("--------------------")
    print(f"Signal Status: {signal_result['signal_status']}")
    print(f"Direction: {signal_result['direction']}")
    print(f"Entry Price: {signal_result['entry_price']}")
    print(f"Stop Loss: {signal_result['stop_loss']}")
    print(f"Take Profit 1: {signal_result['take_profit_1']}")
    print(f"Take Profit 2: {signal_result['take_profit_2']}")
    print(f"Risk/Reward 1: {signal_result['risk_reward_1']}")
    print(f"Risk/Reward 2: {signal_result['risk_reward_2']}")
    print(f"Setup Quality: {signal_result['setup_quality']}")
    print(f"Confidence Score: {signal_result['confidence_score']}")
    print(f"Risk Level: {signal_result['risk_level']}")
    print(f"Invalidation Level: {signal_result['invalidation_level']}")
    print("--------------------")
    print(f"Reason: {signal_result['reason']}")
    if signal_result.get("notes"):
        print("Notes:")
        for note in signal_result["notes"]:
            print(f"- {note}")
    print("====================")


def save_report_json(symbol, timeframe, report):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_symbol = symbol.replace("/", "_")
    filename = f"signal_{safe_symbol}_{timeframe}.json"
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2)

    return path


# ==========================================
# MAIN
# ==========================================

def main():
    raw_symbol = input("Symbol (e.g. BTC/USDT, BNB, BNBUSDT): ").strip().upper()
    timeframe = input("Timeframe (e.g. 1h,4h,1d): ").strip()

    symbol = normalize_symbol(raw_symbol)

    raw_df, exchange_name = fetch_ohlcv_dataframe(symbol=symbol, timeframe=timeframe, limit=OHLCV_LIMIT)
    df = add_indicators(raw_df)

    trend_data = analyze_trend(df)
    trend_output = build_trend_output(symbol, timeframe, exchange_name, trend_data)

    print_trend_report(symbol, timeframe, exchange_name, trend_data)

    strategy_json = get_llm_strategy_suggestion(trend_output, trend_data)

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

    print("\nStrategy Template:")
    print(json.dumps(make_json_safe(template_result), indent=2))

    signal_result = generate_signal(df=df, trend_data=trend_data, strategy_name=final_strategy)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": exchange_name,
        "trend_data": trend_data,
        "strategy_suggestion": strategy_json,
        "validation": validation,
        "strategy_template": template_result,
        "signal": signal_result,
        "disclaimer": "Educational decision-support output only. Not financial advice.",
    }

    print_signal_report(
        symbol=symbol,
        timeframe=timeframe,
        exchange_name=exchange_name,
        trend_data=trend_data,
        strategy_json=strategy_json,
        validation=validation,
        template_result=template_result,
        signal_result=signal_result,
    )

    saved_path = save_report_json(symbol, timeframe, report)
    print(f"\nSaved JSON report to: {saved_path}")


if __name__ == "__main__":
    main()
