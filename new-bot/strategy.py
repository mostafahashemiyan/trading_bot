def trend_pullback_signal(df_1h, df_15m, df_5m):
    signal = {
        "trend": "neutral",
        "setup": False,
        "entry": None,
        "stop": None,
        "tp": None,
        "rr": None,
        "reasons": []
    }

    # --------------------------------------------------
    # 1️⃣ Higher timeframe trend (1H)
    # --------------------------------------------------
    if df_1h["ema50"].iloc[-1] > df_1h["ema200"].iloc[-1]:
        signal["trend"] = "bullish"
    else:
        signal["reasons"].append("1H trend not bullish")
        return signal

    # --------------------------------------------------
    # 2️⃣ Pullback condition (15M)
    # --------------------------------------------------
    rsi_15 = df_15m["rsi"].iloc[-1]
    if not (40 <= rsi_15 <= 60):
        signal["reasons"].append("15M RSI not in pullback zone")
        return signal

    # --------------------------------------------------
    # 3️⃣ Entry logic (5M) — MOMENTUM OR WICK
    # --------------------------------------------------
    last = df_5m.iloc[-1]

    body = abs(last["close"] - last["open"])
    range_ = last["high"] - last["low"]
    lower_wick = min(last["open"], last["close"]) - last["low"]

    bullish_momentum = (
        last["close"] > last["open"] and
        last["close"] > last["ema20"] and
        body > 0.6 * range_
    )

    wick_rejection = (
        last["close"] > last["open"] and
        lower_wick > 2 * body
    )

    if bullish_momentum or wick_rejection:
        signal["setup"] = True
    else:
        signal["reasons"].append("No momentum or wick rejection on 5M")

    # --------------------------------------------------
    # 4️⃣ Trade levels
    # --------------------------------------------------
    entry = last["close"]

    # --------------------------------------------------
    # Stop-loss: recent swing low (last 5 candles)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 5️⃣ Finalize
    # --------------------------------------------------
    signal.update({
        "setup": True,
        "entry": entry,
        "stop": stop,
        "tp": tp,
        "rr": round(rr, 2)
    })

    return signal
