import numpy as np
import pandas as pd

from indicators import ema
from config import (
    RR_RATIO,
    PIVOT_LEN,
    EMA_FAST,
    EMA_SLOW,
    MIN_GAP_BARS,
    MIN_RISK_ATR_MULT,
    ATR_LEN,
)


def _pivot_high_confirmed(high: pd.Series, left: int, right: int) -> pd.Series:
    """Mimic Pine ta.pivothigh(high, left, right).

    Returns a series where the value is the pivot HIGH price, but it becomes
    available only on the *confirmation* bar (pivot bar + right).

    Example: if a pivot is at index i, then out[i+right] = high[i].
    """
    h = high.astype(float).to_numpy()
    n = len(h)
    out = np.full(n, np.nan, dtype=float)

    for i in range(left, n - right):
        window = h[i - left : i + right + 1]
        if np.isnan(window).any():
            continue
        center = h[i]
        others = np.concatenate([window[:left], window[left + 1:]])
        if center > np.max(others):
            out[i + right] = center

    return pd.Series(out, index=high.index)


def safe_short_signal(df_htf: pd.DataFrame, df_ltf: pd.DataFrame, last_trade_bar: int | None = None) -> dict:
    """Python port of the user's Pine strategy:
    5M Trend + Structure + Pullback (Safe Short)

    - HTF trend: EMA50 < EMA200 on df_htf
    - LTF trend: EMA(EMA_FAST) < EMA(EMA_SLOW) on df_ltf
    - Structure: Lower High using confirmed pivot highs
    - Pullback: high >= EMA(EMA_FAST)
    - Confirmation: close < low[1]
    - Gap control: min bars between trades
    - SL: last pivot high
    - TP: RR_RATIO
    - Reject tiny/unrealistic risk: risk > ATR(ATR_LEN) * MIN_RISK_ATR_MULT
    """

    signal = {
        "trend": "neutral",
        "setup": False,
        "side": None,
        "entry": None,
        "stop": None,
        "tp": None,
        "rr": None,
        "reasons": [],
        # meta
        "bar_index": int(len(df_ltf) - 1) if len(df_ltf) else None,
    }

    if df_htf is None or df_ltf is None or df_htf.empty or df_ltf.empty:
        signal["reasons"].append("Missing dataframe(s)")
        return signal

    # ----------------------
    # HTF TREND (15m)
    # ----------------------
    htf_ema50 = float(df_htf["ema50"].iloc[-1]) if "ema50" in df_htf else float(ema(df_htf["close"], 50).iloc[-1])
    htf_ema200 = float(df_htf["ema200"].iloc[-1]) if "ema200" in df_htf else float(ema(df_htf["close"], 200).iloc[-1])
    htf_bear = htf_ema50 < htf_ema200
    if not htf_bear:
        signal["reasons"].append("HTF not bearish (EMA50 >= EMA200)")
        return signal

    # ----------------------
    # LTF TREND (5m)
    # ----------------------
    if f"ema{EMA_FAST}" in df_ltf:
        ltf_fast = df_ltf[f"ema{EMA_FAST}"].astype(float)
    else:
        ltf_fast = ema(df_ltf["close"], EMA_FAST)

    if f"ema{EMA_SLOW}" in df_ltf:
        ltf_slow = df_ltf[f"ema{EMA_SLOW}"].astype(float)
    else:
        ltf_slow = ema(df_ltf["close"], EMA_SLOW)

    trend_bear = float(ltf_fast.iloc[-1]) < float(ltf_slow.iloc[-1])
    if not trend_bear:
        signal["reasons"].append("LTF not bearish (EMA fast >= EMA slow)")
        return signal

    signal["trend"] = "bearish"

    # ----------------------
    # STRUCTURE: LOWER HIGH
    # ----------------------
    ph_confirmed = _pivot_high_confirmed(df_ltf["high"], PIVOT_LEN, PIVOT_LEN)
    pivots = ph_confirmed.dropna().astype(float)
    if len(pivots) < 2:
        signal["reasons"].append("Not enough confirmed pivot highs")
        return signal

    prev_high = float(pivots.iloc[-2])
    last_high = float(pivots.iloc[-1])
    lower_high = last_high < prev_high
    if not lower_high:
        signal["reasons"].append("No lower-high structure")
        return signal

    # ----------------------
    # PULLBACK
    # ----------------------
    last = df_ltf.iloc[-1]
    pullback = float(last["high"]) >= float(ltf_fast.iloc[-1])
    if not pullback:
        signal["reasons"].append("No pullback to fast EMA")
        return signal

    # ----------------------
    # CONFIRMATION BREAK
    # ----------------------
    if len(df_ltf) < 2:
        signal["reasons"].append("Not enough bars for confirmation")
        return signal

    confirm_break = float(last["close"]) < float(df_ltf["low"].iloc[-2])
    if not confirm_break:
        signal["reasons"].append("No confirmation break (close >= prev low)")
        return signal

    # ----------------------
    # GAP CONTROL
    # ----------------------
    bar_index = int(signal["bar_index"]) if signal["bar_index"] is not None else None
    can_trade = (last_trade_bar is None) or (bar_index is not None and (bar_index - int(last_trade_bar) > MIN_GAP_BARS))
    if not can_trade:
        signal["reasons"].append("Min-gap cooldown active")
        return signal

    # ----------------------
    # LEVELS + RISK FILTER
    # ----------------------
    entry = float(last["close"])
    stop_loss = last_high

    if not (stop_loss > entry):
        signal["reasons"].append("Stop not above entry")
        return signal

    risk = stop_loss - entry

    # ATR
    if "atr14" in df_ltf:
        atr_val = float(df_ltf["atr14"].iloc[-1])
    else:
        # fallback (shouldn't happen if prepare_df used)
        from indicators import atr
        atr_val = float(atr(df_ltf, ATR_LEN).iloc[-1])

    min_allowed_risk = atr_val * float(MIN_RISK_ATR_MULT)
    if not (risk > min_allowed_risk):
        signal["reasons"].append("Risk too small vs ATR filter")
        return signal

    take_profit = entry - (risk * float(RR_RATIO))
    rr = float(RR_RATIO)

    signal.update(
        {
            "setup": True,
            "side": "SHORT",
            "entry": entry,
            "stop": stop_loss,
            "tp": take_profit,
            "rr": rr,
            "reasons": [
                "HTF bearish",
                "LTF bearish",
                "Lower High confirmed",
                "Pullback to fast EMA",
                "Confirmation break",
                f"ATR filter ok (risk {risk:.4f} > {min_allowed_risk:.4f})",
            ],
        }
    )
    print(signal)
    return signal
    
# --- Old strategy kept for reference / fallback (not used by bot anymore unless you wire it back) ---

def trend_pullback_signal(df_1h, df_15m, df_5m):
    signal = {
        "trend": "neutral",
        "setup": False,
        "entry": None,
        "stop": None,
        "tp": None,
        "rr": None,
        "reasons": [],
    }

    # 1️⃣ Higher timeframe trend (1H)
    if df_1h["ema50"].iloc[-1] > df_1h["ema200"].iloc[-1]:
        signal["trend"] = "bullish"
    else:
        signal["reasons"].append("1H trend not bullish")
        return signal

    # 2️⃣ Pullback condition (15M)
    rsi_15 = df_15m["rsi"].iloc[-1]
    if not (40 <= rsi_15 <= 60):
        signal["reasons"].append("15M RSI not in pullback zone")
        return signal

    # 3️⃣ Entry logic (5M) — MOMENTUM OR WICK
    last = df_5m.iloc[-1]

    body = abs(last["close"] - last["open"])
    range_ = last["high"] - last["low"]
    lower_wick = min(last["open"], last["close"]) - last["low"]

    bullish_momentum = (
        last["close"] > last["open"]
        and last["close"] > last["ema20"]
        and body > 0.6 * range_
    )

    wick_rejection = last["close"] > last["open"] and lower_wick > 2 * body

    if bullish_momentum or wick_rejection:
        signal["setup"] = True
    else:
        signal["reasons"].append("No momentum or wick rejection on 5M")

    # 4️⃣ Trade levels
    entry = last["close"]

    # Stop-loss: recent swing low (last 5 candles)
    recent_lows = df_5m["low"].iloc[-6:-1]
    stop = recent_lows.min()

    # Safety buffer (0.1%)
    stop *= 0.999

    risk = entry - stop
    if risk <= 0:
        signal["reasons"].append("Invalid stop placement")
        return signal

    # Take-profit: 2.2:1 reward-to-risk ratio
    tp = entry + (risk * 2.2)
    rr = (tp - entry) / risk

    # 5️⃣ Finalize
    signal.update({"setup": True, "entry": entry, "stop": stop, "tp": tp, "rr": round(rr, 2)})

    return signal
