"""
Position sizing and SL/TP calculation.
"""

import config


def position_size(balance: float, entry: float, stop: float) -> float:
    """
    Calculate position size in base asset (e.g. ETH).
    Applies leverage and strict safety cap.
    """
    if balance <= 0:
        return 0.0

    risk_amount = balance * config.RISK_PER_TRADE
    stop_distance = abs(entry - stop)

    if stop_distance <= 0:
        return 0.0

    size_base = risk_amount / stop_distance

    # Hard cap on notional value
    max_notional = balance * config.LEVERAGE * config.SAFETY_FACTOR
    max_size = max_notional / entry

    return min(size_base, max_size)


def sl_tp_from_atr(side: str, entry: float, atr: float) -> tuple:
    """
    Return (stop_loss, take_profit) using ATR multipliers.
    """
    entry = float(entry)
    atr = float(atr)

    if side.upper() == "LONG":
        sl = entry - atr * config.SL_ATR_MULT
        tp = entry + atr * config.TP_ATR_MULT
    else:
        sl = entry + atr * config.SL_ATR_MULT
        tp = entry - atr * config.TP_ATR_MULT

    return sl, tp