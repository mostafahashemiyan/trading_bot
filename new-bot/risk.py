import config


def position_size(balance: float, entry: float, stop: float, risk_pct: float) -> float:
    """Position size in *base units* (e.g. ETH) for linear USDT-margined futures.

    balance: available USDT collateral
    entry/stop: prices
    risk_pct: fraction of balance to risk
    """
    risk_amount = float(balance) * float(risk_pct)
    stop_distance = abs(float(entry) - float(stop))
    if stop_distance <= 0:
        return 0.0
    return risk_amount / stop_distance


def sl_tp_from_atr(side: str, entry: float, atr: float):
    """Compute SL/TP using ATR multipliers from config.

    side: "LONG" or "SHORT"
    """
    entry = float(entry)
    atr = float(atr)
    sl_mult = float(getattr(config, "SL_ATR_MULT", 1.2))
    tp_mult = float(getattr(config, "TP_ATR_MULT", 2.0))

    if side == "LONG":
        sl = entry - atr * sl_mult
        tp = entry + atr * tp_mult
    else:
        sl = entry + atr * sl_mult
        tp = entry - atr * tp_mult

    return sl, tp
