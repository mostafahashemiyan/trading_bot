import pandas as pd
import config
from risk import sl_tp_from_atr


def _ema_slope(series: pd.Series, lookback: int) -> float:
    """
    Simple slope approximation using lookback bars.
    Positive = rising, negative = falling.
    """
    if len(series) <= lookback:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-1 - lookback])


def trend_pullback_signal(
    df_high: pd.DataFrame,
    df_med: pd.DataFrame,
    df_low: pd.DataFrame
) -> dict:
    """
    Multi-timeframe trend strategy.

    Returns:
        {
          "setup": True/False,
          "side": "LONG" / "SHORT",
          "entry": float,
          "sl": float,
          "tp": float,
          "confidence": 0-100,
          "reasons": [...],
          "market_context": {...}
        }
    """

    signal = {
        "setup": False,
        "side": None,
        "entry": None,
        "sl": None,
        "tp": None,
        "confidence": 0,
        "reasons": [],
        "market_context": {
            "adx": None,
            "ema_fast": None,
            "ema_slow": None,
            "ema_distance": None,
            "ema_fast_slope": None,
            "htf_close": None,
            "atr": None,
            "volume_current": None,
            "volume_ma20": None,
            "volume_ratio": None,
            "candle_body_ratio": None,
            "trend_quality_pass": None
        }
    }

    # Validate data
    if df_high.empty or df_med.empty or df_low.empty:
        signal["reasons"].append("Insufficient data")
        return signal

    # Last candles
    h = df_high.iloc[-1]
    m = df_med.iloc[-1]
    l = df_low.iloc[-1]

    # Pre-calculate common values for context
    ema_fast = float(h["ema_fast"])
    ema_slow = float(h["ema_slow"])
    atr_h = float(h["atr"]) if pd.notna(h["atr"]) else 0.0
    adx_h = float(h["adx"]) if pd.notna(h["adx"]) else 0.0
    close_h = float(h["close"])

    ema_distance = abs(ema_fast - ema_slow)
    min_distance = atr_h * config.MIN_EMA_DISTANCE_ATR if atr_h > 0 else 0.0
    fast_slope = _ema_slope(df_high["ema_fast"], config.EMA_SLOPE_LOOKBACK)

    # LTF values
    body = abs(float(l["close"]) - float(l["open"]))
    rng = float(l["high"]) - float(l["low"])
    body_ratio = float(body / rng) if rng > 0 else 0.0

    vol_ma = df_low["volume"].rolling(20).mean().iloc[-1]
    vol_ma = float(vol_ma) if pd.notna(vol_ma) else 0.0
    volume_current = float(l["volume"])
    volume_ratio = float(volume_current / vol_ma) if vol_ma > 0 else 0.0

    atr_m = float(m["atr"]) if pd.notna(m["atr"]) else 0.0

    # Fill market context as early as possible
    signal["market_context"]["adx"] = adx_h
    signal["market_context"]["ema_fast"] = ema_fast
    signal["market_context"]["ema_slow"] = ema_slow
    signal["market_context"]["ema_distance"] = float(ema_distance)
    signal["market_context"]["ema_fast_slope"] = float(fast_slope)
    signal["market_context"]["htf_close"] = close_h
    signal["market_context"]["atr"] = atr_m
    signal["market_context"]["volume_current"] = volume_current
    signal["market_context"]["volume_ma20"] = vol_ma
    signal["market_context"]["volume_ratio"] = volume_ratio
    signal["market_context"]["candle_body_ratio"] = body_ratio
    signal["market_context"]["trend_quality_pass"] = False

    # ───────────────────────────────────────────────
    # 1) Strong trend detection (Higher Timeframe)
    # ───────────────────────────────────────────────
    if ema_fast > ema_slow:
        if ema_distance < min_distance:
            signal["reasons"].append(
                f"Weak bullish trend: EMA distance too small ({ema_distance:.4f})"
            )
            return signal

        if fast_slope <= 0:
            signal["reasons"].append(
                f"Weak bullish trend: EMA fast slope not rising ({fast_slope:.4f})"
            )
            return signal

        if close_h <= ema_fast:
            signal["reasons"].append("Weak bullish trend: price not above fast EMA")
            return signal

        signal["side"] = "LONG"

    elif ema_fast < ema_slow:
        if ema_distance < min_distance:
            signal["reasons"].append(
                f"Weak bearish trend: EMA distance too small ({ema_distance:.4f})"
            )
            return signal

        if fast_slope >= 0:
            signal["reasons"].append(
                f"Weak bearish trend: EMA fast slope not falling ({fast_slope:.4f})"
            )
            return signal

        if close_h >= ema_fast:
            signal["reasons"].append("Weak bearish trend: price not below fast EMA")
            return signal

        signal["side"] = "SHORT"

    else:
        signal["reasons"].append("No clear trend on HTF")
        return signal

    signal["market_context"]["trend_quality_pass"] = True

    # ───────────────────────────────────────────────
    # 2) ADX filter (trend strength)
    # ───────────────────────────────────────────────
    if adx_h < config.ADX_THRESHOLD:
        signal["reasons"].append(f"ADX too weak ({adx_h:.1f})")
        return signal

    # ───────────────────────────────────────────────
    # 3) Volume filter on LTF
    # ───────────────────────────────────────────────
    if volume_current < vol_ma * config.MIN_VOLUME_MULTIPLIER:
        signal["reasons"].append("Low volume, momentum not confirmed")
        return signal

    # ───────────────────────────────────────────────
    # 4) Candle momentum (LTF confirmation)
    # ───────────────────────────────────────────────
    has_momentum = body_ratio > 0.55 and rng > 0
    direction_ok = (
        (signal["side"] == "LONG" and float(l["close"]) > float(l["open"])) or
        (signal["side"] == "SHORT" and float(l["close"]) < float(l["open"]))
    )

    if not (has_momentum and direction_ok):
        signal["reasons"].append("No LTF momentum confirmation")
        return signal

    # ───────────────────────────────────────────────
    # 5) Setup confirmed
    # ───────────────────────────────────────────────
    signal["setup"] = True
    signal["entry"] = float(l["close"])

    if atr_m <= 0:
        signal["setup"] = False
        signal["reasons"].append("Invalid ATR on medium timeframe")
        return signal

    # Build SL/TP using ATR from Medium TF
    sl, tp = sl_tp_from_atr(signal["side"], signal["entry"], atr_m)
    signal["sl"] = float(sl)
    signal["tp"] = float(tp)

    # Confidence score
    conf = 50
    conf += min(20, int(adx_h))
    conf += 10 if has_momentum else 0
    conf += 10 if volume_ratio > config.MIN_VOLUME_MULTIPLIER else 0
    conf += 5 if ema_distance >= min_distance else 0
    conf += 5 if (
        (signal["side"] == "LONG" and fast_slope > 0) or
        (signal["side"] == "SHORT" and fast_slope < 0)
    ) else 0

    signal["confidence"] = min(conf, 100)

    return signal