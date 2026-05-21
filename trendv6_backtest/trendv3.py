import os
import json

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


# ==========================================
# HELPERS
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
    elif "Bearish" in trend:
        return "bearish"
    else:
        return "neutral"


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
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            return df, exchange.id

        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not fetch candles for {symbol}: {last_error}")


def add_indicators(df):
    df = df.copy()

    df["ema50"] = ta.trend.ema_indicator(
        df["close"],
        window=50,
    )

    df["ema200"] = ta.trend.ema_indicator(
        df["close"],
        window=200,
    )

    df["ema20"] = ta.trend.ema_indicator(
        df["close"],
        window=20,
    )

    df["rsi"] = ta.momentum.rsi(
        df["close"],
        window=14,
    )

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

    higher_high_count = sum(
        recent_highs[i] > recent_highs[i - 1]
        for i in range(1, len(recent_highs))
    )

    higher_low_count = sum(
        recent_lows[i] > recent_lows[i - 1]
        for i in range(1, len(recent_lows))
    )

    lower_high_count = sum(
        recent_highs[i] < recent_highs[i - 1]
        for i in range(1, len(recent_highs))
    )

    lower_low_count = sum(
        recent_lows[i] < recent_lows[i - 1]
        for i in range(1, len(recent_lows))
    )

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


# ==========================================
# TREND ANALYSIS FUNCTION
# This is used both for the current candle and inside historical backtest.
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

    higher_high_count = sum(
        recent_highs[i] > recent_highs[i - 1]
        for i in range(1, len(recent_highs))
    )

    higher_low_count = sum(
        recent_lows[i] > recent_lows[i - 1]
        for i in range(1, len(recent_lows))
    )

    lower_high_count = sum(
        recent_highs[i] < recent_highs[i - 1]
        for i in range(1, len(recent_highs))
    )

    lower_low_count = sum(
        recent_lows[i] < recent_lows[i - 1]
        for i in range(1, len(recent_lows))
    )

    structure_lookback = len(recent_highs) - 1
    higher_highs = higher_high_count >= 4
    higher_lows = higher_low_count >= 4
    lower_highs = lower_high_count >= 4
    lower_lows = lower_low_count >= 4

    close_change_atr = (
        (recent_closes[-1] - recent_closes[0]) / atr
        if atr > 0 else 0
    )

    ema_distance = abs(ema50 - ema200) / price * 100
    ema20_ema50_distance = abs(ema20 - ema50) / price * 100
    atr_percent = atr / price * 100

    recent_range = (
        (df["high"].tail(20).max() - df["low"].tail(20).min())
        / price
        * 100
    )

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

    add_vote(
        price > ema20,
        "price is above EMA20",
        "price is below EMA20",
    )

    add_vote(
        ema20 > ema50,
        "EMA20 is above EMA50",
        "EMA20 is below EMA50",
    )

    add_vote(
        ema50 > ema200,
        "EMA50 is above EMA200",
        "EMA50 is below EMA200",
    )

    add_vote(
        macd_value > macd_signal,
        "MACD is above signal",
        "MACD is below signal",
    )

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

    # Prevent overconfident "Strong" when EMA structure is not fully aligned.
    if trend == "Bullish" and ema50 < ema200:
        signal_strength = "Moderate"

    if trend == "Bearish" and ema20 > ema50:
        signal_strength = "Moderate"

    return {
        "trend": trend,
        "signal_strength": signal_strength,
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
Reasons:
{chr(10).join(f"- {note}" for note in notes)}
""".strip()


def print_trend_report(symbol, timeframe, exchange_name, trend_data):
    print("\n====================")
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
    print("--------------------")
    print("Reasons:")
    for note in trend_data["notes"]:
        print(f"- {note}")
    print("====================")


# ==========================================
# STRATEGY AGENT
# ==========================================

def get_llm_strategy_suggestion(trend_output):
    if Agent is None or Runner is None:
        raise RuntimeError(
            "OpenAI Agents SDK is not installed. Run: pip install openai-agents"
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env file")

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
- Your reason must not contradict the provided Reasons section.
- If the Reasons section says market structure leans higher, do not call it mixed.
- If the Reasons section says market structure leans lower, do not call it mixed.
- Only say market structure is mixed if the Reasons section says it is mixed.
- If the trend is unclear, weak, or sideways, avoid trend-following strategies.
- In unclear or sideways conditions, prefer:
  - range-bound strategy
  - mixed/confirmation strategy

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

Constraints:
- strategy_style must exactly match one allowed strategy style.
- confidence must be numeric between 0 and 1.
- reason must be concise.
- No markdown.
- No extra text outside JSON.
"""
    )

    agent_input = f"""
Trend output:
{trend_output}

Suggest the best-fit strategy style for this trend.
""".strip()

    strategy_result = Runner.run_sync(
        strategy_agent,
        agent_input,
        max_turns=1,
    )

    return json.loads(strategy_result.final_output.strip())


# ==========================================
# RULE-BASED VALIDATOR
# ==========================================

ALLOWED_STRATEGIES = {
    "pullback continuation strategy",
    "breakout-and-retest strategy",
    "range-bound strategy",
    "momentum continuation strategy",
    "reversal-confirmation strategy",
    "mixed/confirmation strategy",
}


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

    weak_or_unclear = (
        "Weak" in trend
        or "Unclear" in trend
        or "Sideways" in trend
        or strength == "Weak"
    )

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

    if "Sideways" in trend and strategy != "range-bound strategy":
        result["warnings"].append(
            "Sideways condition usually fits range-bound strategy better."
        )

    if "Weak" in trend and strategy != "mixed/confirmation strategy":
        result["warnings"].append(
            "Weak trend usually fits mixed/confirmation strategy better."
        )

    if result["is_valid"]:
        result["validator_note"] = "LLM strategy suggestion is consistent with trend output."
    else:
        result["validator_note"] = "LLM strategy suggestion conflicts with rule-based validation."

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
        "risk_note": "Avoid if RSI is high but structure is mixed.",
    },

    "reversal-confirmation strategy": {
        "type": "reversal",
        "best_for": ["exhaustion", "failed continuation"],
        "confirmation_needed": [
            "trend weakens",
            "momentum divergence appears",
            "structure starts reversing",
        ],
        "risk_note": "Requires strong confirmation; do not use only because price is extended.",
    },

    "mixed/confirmation strategy": {
        "type": "confirmation_waiting",
        "best_for": ["weak_trending", "unclear"],
        "confirmation_needed": [
            "wait for cleaner market structure",
            "wait for EMA200 alignment or rejection",
            "wait for MACD confirmation",
        ],
        "risk_note": "Used when signals conflict and the trend is not confirmed.",
    },
}


def get_strategy_template(strategy_json):
    strategy = strategy_json.get("strategy_style")

    template = STRATEGY_TEMPLATES.get(strategy)

    if template is None:
        return {
            "template_found": False,
            "error": "No template found for this strategy style.",
        }

    return {
        "template_found": True,
        "strategy_style": strategy,
        "template": template,
    }






# ==========================================
# STRATEGY CHECKERS
# ==========================================

def pullback_continuation_check(df, direction):
    latest = df.iloc[-1]

    price = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    ema200 = latest["ema200"]
    rsi = latest["rsi"]
    macd = latest["macd"]
    macd_signal = latest["macd_signal"]

    structure = get_recent_structure(df)

    near_ema20 = abs(price - ema20) / price < 0.006
    near_ema50 = abs(price - ema50) / price < 0.01
    pullback_area = near_ema20 or near_ema50

    if direction == "bullish":
        trend_alignment = ema20 > ema50
        ema200_supportive = price > ema200
        momentum_ok = rsi > 50 and macd > macd_signal
        structure_ok = structure["bullish_structure"]

    elif direction == "bearish":
        trend_alignment = ema20 < ema50 or price < ema20
        ema200_supportive = price < ema200
        momentum_ok = rsi < 50 and macd < macd_signal
        structure_ok = structure["bearish_structure"]

    else:
        trend_alignment = False
        ema200_supportive = False
        momentum_ok = False
        structure_ok = False

    points = 0
    points += 1 if trend_alignment else 0
    points += 1 if ema200_supportive else 0
    points += 1 if pullback_area else 0
    points += 1 if momentum_ok else 0
    points += 1 if structure_ok else 0

    setup_quality = safe_quality_score(points, 5)

    setup_detected = (
        direction in ["bullish", "bearish"]
        and setup_quality >= 0.65
        and pullback_area
    )

    if direction == "neutral":
        confirmation_status = "not_applicable"
    elif not pullback_area:
        confirmation_status = "needs_pullback"
    elif setup_detected:
        confirmation_status = "confirmed"
    else:
        confirmation_status = "needs_confirmation"

    return {
        "setup_detected": setup_detected,
        "confirmation_status": confirmation_status,
        "setup_quality": setup_quality,
        "risk_level": "medium" if ema200_supportive else "high",
        "notes": [
            f"direction={direction}",
            f"trend_alignment={trend_alignment}",
            f"ema200_supportive={ema200_supportive}",
            f"pullback_area={pullback_area}",
            f"momentum_ok={momentum_ok}",
            f"structure_ok={structure_ok}",
        ],
    }


def breakout_retest_check(df, direction):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    price = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    ema200 = latest["ema200"]
    rsi = latest["rsi"]
    macd = latest["macd"]
    macd_signal = latest["macd_signal"]
    atr = latest["atr"]

    recent_resistance = df["high"].tail(20).iloc[:-1].max()
    recent_support = df["low"].tail(20).iloc[:-1].min()

    if direction == "bullish":
        broke_level = previous["close"] > recent_resistance
        retest_near_level = abs(price - recent_resistance) <= atr * 0.5
        trend_ok = ema20 > ema50
        ema200_supportive = price > ema200
        momentum_ok = rsi > 50 and macd > macd_signal
        level_name = "resistance"

    elif direction == "bearish":
        broke_level = previous["close"] < recent_support
        retest_near_level = abs(price - recent_support) <= atr * 0.5
        trend_ok = ema20 < ema50 or price < ema20
        ema200_supportive = price < ema200
        momentum_ok = rsi < 50 and macd < macd_signal
        level_name = "support"

    else:
        broke_level = False
        retest_near_level = False
        trend_ok = False
        ema200_supportive = False
        momentum_ok = False
        level_name = "none"

    points = 0
    points += 1 if broke_level else 0
    points += 1 if retest_near_level else 0
    points += 1 if trend_ok else 0
    points += 1 if ema200_supportive else 0
    points += 1 if momentum_ok else 0

    setup_quality = safe_quality_score(points, 5)
    setup_detected = setup_quality >= 0.65 and broke_level and retest_near_level

    return {
        "setup_detected": setup_detected,
        "confirmation_status": "confirmed" if setup_detected and momentum_ok else "needs_confirmation",
        "setup_quality": setup_quality,
        "risk_level": "medium" if ema200_supportive else "high",
        "notes": [
            f"direction={direction}",
            f"level_name={level_name}",
            f"recent_resistance={recent_resistance:.4f}",
            f"recent_support={recent_support:.4f}",
            f"broke_level={broke_level}",
            f"retest_near_level={retest_near_level}",
            f"trend_ok={trend_ok}",
            f"ema200_supportive={ema200_supportive}",
            f"momentum_ok={momentum_ok}",
        ],
    }


def range_bound_check(df, direction=None):
    latest = df.iloc[-1]

    price = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    ema200 = latest["ema200"]
    rsi = latest["rsi"]
    atr = latest["atr"]

    atr_percent = atr / price * 100

    recent_high = df["high"].tail(20).max()
    recent_low = df["low"].tail(20).min()
    recent_range_percent = (recent_high - recent_low) / price * 100

    ema20_ema50_distance = abs(ema20 - ema50) / price * 100
    ema50_ema200_distance = abs(ema50 - ema200) / price * 100

    ema_compressed = ema20_ema50_distance < 0.25 and ema50_ema200_distance < 1
    rsi_neutral = 45 <= rsi <= 55
    range_not_too_wide = recent_range_percent < max(2.5, atr_percent * 4)

    near_range_low = abs(price - recent_low) <= atr
    near_range_high = abs(price - recent_high) <= atr
    inside_range = recent_low <= price <= recent_high

    points = 0
    points += 1 if ema_compressed else 0
    points += 1 if rsi_neutral else 0
    points += 1 if range_not_too_wide else 0
    points += 1 if inside_range else 0
    points += 1 if near_range_low or near_range_high else 0

    setup_quality = safe_quality_score(points, 5)
    setup_detected = setup_quality >= 0.65

    return {
        "setup_detected": setup_detected,
        "confirmation_status": "confirmed" if setup_detected and rsi_neutral else "needs_confirmation",
        "setup_quality": setup_quality,
        "risk_level": "low" if setup_quality >= 0.8 else "medium",
        "notes": [
            f"direction={direction}",
            f"recent_high={recent_high:.4f}",
            f"recent_low={recent_low:.4f}",
            f"recent_range_percent={recent_range_percent:.2f}",
            f"ema_compressed={ema_compressed}",
            f"rsi_neutral={rsi_neutral}",
            f"inside_range={inside_range}",
            f"near_range_low={near_range_low}",
            f"near_range_high={near_range_high}",
        ],
    }


def momentum_continuation_check(df, direction):
    latest = df.iloc[-1]

    price = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    ema200 = latest["ema200"]
    rsi = latest["rsi"]
    macd = latest["macd"]
    macd_signal = latest["macd_signal"]
    atr = latest["atr"]

    structure = get_recent_structure(df)

    recent_closes = df["close"].tail(8).values
    close_change_atr = (
        (recent_closes[-1] - recent_closes[0]) / atr
        if atr > 0 else 0
    )

    if direction == "bullish":
        trend_fully_aligned = price > ema20 > ema50 > ema200
        momentum_ok = macd > macd_signal and rsi > 55
        rsi_not_extreme = rsi < 75
        slope_strong = close_change_atr > 0.7
        structure_ok = structure["bullish_structure"]

    elif direction == "bearish":
        trend_fully_aligned = price < ema20 < ema50 < ema200
        momentum_ok = macd < macd_signal and rsi < 45
        rsi_not_extreme = rsi > 25
        slope_strong = close_change_atr < -0.7
        structure_ok = structure["bearish_structure"]

    else:
        trend_fully_aligned = False
        momentum_ok = False
        rsi_not_extreme = False
        slope_strong = False
        structure_ok = False

    points = 0
    points += 1 if trend_fully_aligned else 0
    points += 1 if momentum_ok else 0
    points += 1 if rsi_not_extreme else 0
    points += 1 if slope_strong else 0
    points += 1 if structure_ok else 0

    setup_quality = safe_quality_score(points, 5)
    setup_detected = setup_quality >= 0.75

    return {
        "setup_detected": setup_detected,
        "confirmation_status": "confirmed" if setup_detected else "needs_confirmation",
        "setup_quality": setup_quality,
        "risk_level": "medium" if rsi_not_extreme else "high",
        "notes": [
            f"direction={direction}",
            f"trend_fully_aligned={trend_fully_aligned}",
            f"momentum_ok={momentum_ok}",
            f"rsi_not_extreme={rsi_not_extreme}",
            f"slope_strong={slope_strong}",
            f"close_change_atr={close_change_atr:.2f}",
            f"structure_ok={structure_ok}",
        ],
    }


def reversal_confirmation_check(df, direction):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    price = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    ema200 = latest["ema200"]
    rsi = latest["rsi"]
    macd = latest["macd"]
    macd_signal = latest["macd_signal"]

    structure = get_recent_structure(df)

    previous_rsi = previous["rsi"]
    previous_macd = previous["macd"]
    previous_macd_signal = previous["macd_signal"]

    if direction == "bullish":
        trend_was_weak = ema20 < ema50 or price < ema200
        rsi_recovering = previous_rsi < 47 and rsi > previous_rsi
        macd_turning = previous_macd < previous_macd_signal and macd > macd_signal
        price_reclaiming_ema20 = previous["close"] < previous["ema20"] and price > ema20
        structure_improving = structure["higher_low_count"] >= 4

    elif direction == "bearish":
        trend_was_weak = ema20 > ema50 or price > ema200
        rsi_recovering = previous_rsi > 53 and rsi < previous_rsi
        macd_turning = previous_macd > previous_macd_signal and macd < macd_signal
        price_reclaiming_ema20 = previous["close"] > previous["ema20"] and price < ema20
        structure_improving = structure["lower_high_count"] >= 4

    else:
        trend_was_weak = False
        rsi_recovering = False
        macd_turning = False
        price_reclaiming_ema20 = False
        structure_improving = False

    points = 0
    points += 1 if trend_was_weak else 0
    points += 1 if rsi_recovering else 0
    points += 1 if macd_turning else 0
    points += 1 if price_reclaiming_ema20 else 0
    points += 1 if structure_improving else 0

    setup_quality = safe_quality_score(points, 5)
    setup_detected = setup_quality >= 0.65

    return {
        "setup_detected": setup_detected,
        "confirmation_status": "confirmed" if setup_detected and macd_turning else "needs_confirmation",
        "setup_quality": setup_quality,
        "risk_level": "high",
        "notes": [
            f"direction={direction}",
            f"trend_was_weak={trend_was_weak}",
            f"rsi_recovering={rsi_recovering}",
            f"macd_turning={macd_turning}",
            f"price_reclaiming_ema20={price_reclaiming_ema20}",
            f"structure_improving={structure_improving}",
        ],
    }


def mixed_confirmation_check(df, direction):
    latest = df.iloc[-1]

    price = latest["close"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    ema200 = latest["ema200"]
    rsi = latest["rsi"]
    macd = latest["macd"]
    macd_signal = latest["macd_signal"]

    structure = get_recent_structure(df)

    short_term_bullish = price > ema20 and rsi > 53 and macd > macd_signal
    short_term_bearish = price < ema20 and rsi < 47 and macd < macd_signal

    bullish_ema_alignment = ema20 > ema50 > ema200
    bearish_ema_alignment = ema20 < ema50 < ema200

    bullish_structure = structure["bullish_structure"]
    bearish_structure = structure["bearish_structure"]
    mixed_structure = not bullish_structure and not bearish_structure

    price_near_ema200 = abs(price - ema200) / price < 0.015

    bullish_conflict = short_term_bullish and not bullish_ema_alignment
    bearish_conflict = short_term_bearish and not bearish_ema_alignment

    neutral_momentum = 47 <= rsi <= 53
    macd_flat = abs(macd - macd_signal) / price < 0.0005

    conflicting_signals = (
        bullish_conflict
        or bearish_conflict
        or price_near_ema200
        or mixed_structure
        or neutral_momentum
        or macd_flat
    )

    points = 0
    points += 1 if bullish_conflict or bearish_conflict else 0
    points += 1 if price_near_ema200 else 0
    points += 1 if mixed_structure else 0
    points += 1 if neutral_momentum else 0
    points += 1 if macd_flat else 0

    setup_quality = safe_quality_score(points, 5)
    setup_detected = setup_quality >= 0.5

    return {
        "setup_detected": setup_detected,
        "confirmation_status": "needs_confirmation",
        "setup_quality": setup_quality,
        "risk_level": "medium" if setup_quality >= 0.7 else "high",
        "notes": [
            f"direction={direction}",
            f"short_term_bullish={short_term_bullish}",
            f"short_term_bearish={short_term_bearish}",
            f"bullish_ema_alignment={bullish_ema_alignment}",
            f"bearish_ema_alignment={bearish_ema_alignment}",
            f"bullish_conflict={bullish_conflict}",
            f"bearish_conflict={bearish_conflict}",
            f"price_near_ema200={price_near_ema200}",
            f"mixed_structure={mixed_structure}",
            f"neutral_momentum={neutral_momentum}",
            f"macd_flat={macd_flat}",
            f"conflicting_signals={conflicting_signals}",
        ],
    }


STRATEGY_CHECKERS = {
    "pullback continuation strategy": pullback_continuation_check,
    "breakout-and-retest strategy": breakout_retest_check,
    "range-bound strategy": range_bound_check,
    "momentum continuation strategy": momentum_continuation_check,
    "reversal-confirmation strategy": reversal_confirmation_check,
    "mixed/confirmation strategy": mixed_confirmation_check,
}


def run_strategy_checker(df, strategy_name, trend):
    checker = STRATEGY_CHECKERS.get(strategy_name)
    direction = infer_direction_from_trend(trend)

    if checker is None:
        return {
            "setup_detected": False,
            "confirmation_status": "not_available",
            "setup_quality": 0.0,
            "risk_level": "unknown",
            "notes": [
                f"No checker found for selected strategy: {strategy_name}",
            ],
        }

    return checker(df, direction)


## Backtest functionality removed to restore original trend-only workflow.


# ==========================================
# MAIN
# ==========================================

def main():
    raw_symbol = input("Symbol (e.g. BTC/USDT, BNB, BNBUSDT): ").strip().upper()
    timeframe = input("Timeframe (e.g. 1h,4h,1d): ").strip()

    symbol = normalize_symbol(raw_symbol)

    raw_df, exchange_name = fetch_ohlcv_dataframe(
        symbol=symbol,
        timeframe=timeframe,
        limit=OHLCV_LIMIT,
    )

    df = add_indicators(raw_df)

    current_trend_data = analyze_trend(df)

    trend = current_trend_data["trend"]
    signal_strength = current_trend_data["signal_strength"]

    trend_output = build_trend_output(
        symbol=symbol,
        timeframe=timeframe,
        exchange_name=exchange_name,
        trend_data=current_trend_data,
    )

    print_trend_report(
        symbol=symbol,
        timeframe=timeframe,
        exchange_name=exchange_name,
        trend_data=current_trend_data,
    )

    strategy_json = get_llm_strategy_suggestion(trend_output)

    print("Suggested Strategy:")
    print(json.dumps(make_json_safe(strategy_json), indent=2))

    validation = validate_strategy_suggestion(
        trend=trend,
        strength=signal_strength,
        strategy_json=strategy_json,
    )

    print("\nValidation:")
    print(json.dumps(make_json_safe(validation), indent=2))

    template_result = get_strategy_template(strategy_json)

    print("\nStrategy Template:")
    print(json.dumps(make_json_safe(template_result), indent=2))

    selected_strategy = strategy_json["strategy_style"]

    setup_result = run_strategy_checker(
        df=df,
        strategy_name=selected_strategy,
        trend=trend,
    )

    print("\nStrategy Setup Check:")
    print(json.dumps(make_json_safe(setup_result), indent=2))
    # Backtest functionality removed; end of workflow.



if __name__ == "__main__":
    main()
