"""
STRATEGY MODULE (Professional redesign 2026)
--------------------------------------------
Multi-timeframe trend + pullback + momentum system.
Produces extremely clean signals for the LLM gatekeeper.
"""

import pandas as pd
import config
from risk import sl_tp_from_atr


def trend_pullback_signal(df_high: pd.DataFrame,
                          df_med: pd.DataFrame,
                          df_low: pd.DataFrame) -> dict:
    """
    Multi-timeframe trend strategy.

    Returns:
        {
          setup: True/False,
          side: "LONG" / "SHORT",
          entry: float,
          sl: float,
          tp: float,
          confidence: 0-100,
          reasons: [...]
        }
    """

    signal = {
        "setup": False,
        "side": None,
        "entry": None,
        "sl": None,
        "tp": None,
        "confidence": 0,
        "reasons": []
    }

    # Validate data
    if df_high.empty or df_med.empty or df_low.empty:
        signal["reasons"].append("Insufficient data")
        return signal

    # Last candles
    h = df_high.iloc[-1]
    m = df_med.iloc[-1]
    l = df_low.iloc[-1]

    # ───────────────────────────────────────────────
    # 1) Trend direction (Higher Timeframe)
    # ───────────────────────────────────────────────
    if h["ema_fast"] > h["ema_slow"]:
        trend = "bullish"
        signal["side"] = "LONG"
    elif h["ema_fast"] < h["ema_slow"]:
        trend = "bearish"
        signal["side"] = "SHORT"
    else:
        signal["reasons"].append("No clear trend on HTF")
        return signal

    # ───────────────────────────────────────────────
    # 2) ADX filter (trend strength)
    # ───────────────────────────────────────────────
    if h["adx"] < config.ADX_THRESHOLD:
        signal["reasons"].append(f"ADX too weak ({h['adx']:.1f})")
        return signal

    # ───────────────────────────────────────────────
    # 3) Volume filter on LTF
    # ───────────────────────────────────────────────
    vol_ma = df_low["volume"].rolling(20).mean().iloc[-1]
    if l["volume"] < vol_ma * config.MIN_VOLUME_MULTIPLIER:
        signal["reasons"].append("Low volume, momentum not confirmed")
        return signal

    # ───────────────────────────────────────────────
    # 4) Candle momentum (LTF confirmation)
    # ───────────────────────────────────────────────
    body = abs(l["close"] - l["open"])
    rng = l["high"] - l["low"]

    has_momentum = body > 0.55 * rng and rng > 0
    direction_ok = (
        (signal["side"] == "LONG" and l["close"] > l["open"]) or
        (signal["side"] == "SHORT" and l["close"] < l["open"])
    )

    if not (has_momentum and direction_ok):
        signal["reasons"].append("No LTF momentum confirmation")
        return signal

    # ───────────────────────────────────────────────
    # 5) Everything passes → We have a setup
    # ───────────────────────────────────────────────
    signal["setup"] = True
    signal["entry"] = float(l["close"])

    # Build SL/TP using ATR from Medium TF
    atr = float(m["atr"])
    sl, tp = sl_tp_from_atr(signal["side"], signal["entry"], atr)
    signal["sl"] = sl
    signal["tp"] = tp

    # Confidence score (simple heuristic)
    conf = 50
    conf += min(20, h["adx"])          # ADX strength adds confidence
    conf += 10 if has_momentum else 0
    conf += 10 if l["volume"] > vol_ma else 0
    conf = min(conf, 100)

    signal["confidence"] = conf

    return signal