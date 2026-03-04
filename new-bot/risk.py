"""
RISK MODULE (Professional 2026)
--------------------------------
Handles:
- Position sizing for futures
- ATR-based Stop Loss and Take Profit
"""

import config


def position_size(balance: float, entry: float, stop: float) -> float:
    """
    Calculate SAFE position size (base currency amount).

    Ensures:
    - Risk is fixed percentage of balance
    - Leverage limit obeyed
    - Notional cap applied (Safety Factor)
    """

    if balance <= 0:
        return 0.0

    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return 0.0

    # Risk amount in USDT
    risk_amount = balance * config.RISK_PER_TRADE

    # Size according to risk
    size_base = risk_amount / stop_distance

    # Apply notional safety cap
    max_notional = balance * config.LEVERAGE * config.SAFETY_FACTOR
    max_size = max_notional / entry

    return max(min(size_base, max_size), 0.0)


def sl_tp_from_atr(side: str, entry: float, atr: float) -> tuple:
    """
    Build SL/TP using ATR and global multipliers.
    """

    entry = float(entry)
    atr = float(atr)

    if side.upper() == "LONG":
        sl = entry - atr * config.SL_ATR_MULT
        tp = entry + atr * config.TP_ATR_MULT
    else:
        sl = entry + atr * config.SL_ATR_MULT
        tp = entry - atr * config.TP_ATR_MULT

    return float(sl), float(tp)