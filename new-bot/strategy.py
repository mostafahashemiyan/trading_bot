"""
Trading signal generator.
Conservative multi-timeframe logic.
"""

import pandas as pd
import config


def trend_pullback_signal(df_high: pd.DataFrame, df_medium: pd.DataFrame, df_low: pd.DataFrame) -> dict:
    """
    Main signal function.
    Returns dict with setup info or rejection reason.
    """
    signal = {
        "trend": "neutral",
        "setup": False,
        "side": None,
        "entry": None,
        "reasons": []
    }

    # 1. Trend direction (higher TF)
    curr_high = df_high.iloc[-1]
    if curr_high["ema_fast"] > curr_high["ema_slow"]:
        signal["trend"] = "bullish"
        signal["side"] = "LONG"
    elif curr_high["ema_fast"] < curr_high["ema_slow"]:
        signal["trend"] = "bearish"
        signal["side"] = "SHORT"
    else:
        signal["reasons"].append("No clear EMA crossover on high TF")
        return signal

    # 2. Strong trend filter (ADX)
    adx_high = df_high["adx"].iloc[-1]
    if adx_high < config.ADX_THRESHOLD:
        signal["reasons"].append(f"ADX too weak ({adx_high:.1f} < {config.ADX_THRESHOLD})")
        return signal

    # 3. Volume filter (momentum confirmation)
    recent_vol = df_low["volume"].rolling(20).mean().iloc[-1]
    curr_vol = df_low["volume"].iloc[-1]
    if curr_vol < recent_vol * config.MIN_VOLUME_MULTIPLIER:
        signal["reasons"].append("Insufficient volume")
        return signal

    # 4. Low TF momentum
    last = df_low.iloc[-1]
    body = abs(last["close"] - last["open"])
    rng = last["high"] - last["low"]

    if signal["side"] == "LONG" and last["close"] > last["open"] and body > 0.6 * rng:
        signal["setup"] = True
    elif signal["side"] == "SHORT" and last["close"] < last["open"] and body > 0.6 * rng:
        signal["setup"] = True
    else:
        signal["reasons"].append("No momentum/wick confirmation on low TF")

    if signal["setup"]:
        signal["entry"] = last["close"]

    return signal