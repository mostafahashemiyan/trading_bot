def trend_pullback_signal(df_1h, df_15m, df_5m):
    # Initialize the signal dictionary with default 'Neutral' states
    signal = {
        "trend": "neutral",
        "setup": False,
        "side": None,
        "entry": None,
        "stop": None,
        "tp": None,
        "rr": 0,
        "reasons": []
    }

    # 1. 1H Trend Analysis (Identify, don't exit early)
    curr_1h = df_1h.iloc[-1]
    if curr_1h["close"] > curr_1h["ema200"]:
        signal["trend"] = "bullish"
        signal["side"] = "LONG"
    elif curr_1h["close"] < curr_1h["ema200"]:
        signal["trend"] = "bearish"
        signal["side"] = "SHORT"
    else:
        signal["reasons"].append("Price oscillating on 1H EMA200")

    # 2. 15M Pullback Logic (Broadened for LLM context)
    rsi_15 = df_15m["rsi"].iloc[-1]
    if signal["side"] == "LONG" and rsi_15 > 65:
        signal["reasons"].append(f"15M RSI ({round(rsi_15, 2)}) high for a long pullback")
    elif signal["side"] == "SHORT" and rsi_15 < 35:
        signal["reasons"].append(f"15M RSI ({round(rsi_15, 2)}) low for a short pullback")

    # 3. 5M Entry Logic (Momentum & Wicks)
    last = df_5m.iloc[-1]
    body = abs(last["close"] - last["open"])
    range_ = last["high"] - last["low"]
    lower_wick = min(last["open"], last["close"]) - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])

    # Bullish and Bearish conditions
    bull_mom = last["close"] > last["open"] and last["close"] > last["ema20"] and body > 0.5 * range_
    bull_wick = last["close"] > last["open"] and lower_wick > 1.5 * body
    bear_mom = last["close"] < last["open"] and last["close"] < last["ema20"] and body > 0.5 * range_
    bear_wick = last["close"] < last["open"] and upper_wick > 1.5 * body

    if signal["side"] == "LONG" and (bull_mom or bull_wick):
        signal["setup"] = True
    elif signal["side"] == "SHORT" and (bear_mom or bear_wick):
        signal["setup"] = True
    else:
        signal["reasons"].append("No 5M momentum/wick confirmation")

    # 4. Trade Levels Calculation
    entry = last["close"]
    recent_5m = df_5m.iloc[-6:-1]
    
    if signal["side"] == "LONG":
        stop = recent_5m["low"].min() * 0.998 
        risk = entry - stop
        if risk > 0:
            tp = entry + (risk * 2.2)
            signal.update({"entry": entry, "stop": stop, "tp": tp, "rr": round((tp-entry)/risk, 2)})
    elif signal["side"] == "SHORT":
        stop = recent_5m["high"].max() * 1.002
        risk = stop - entry
        if risk > 0:
            tp = entry - (risk * 2.2)
            signal.update({"entry": entry, "stop": stop, "tp": tp, "rr": round((entry-tp)/risk, 2)})

    return signal