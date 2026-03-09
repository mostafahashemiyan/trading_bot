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

def validate_llm_levels(side: str, entry: float, sl: float, tp: float):
    """
    Validate stop-loss / take-profit levels proposed by the LLM.

    Returns:
        {
            "valid": bool,
            "reason": str,
            "rr": float | None
        }

    Rules:
    - LONG: sl < entry < tp
    - SHORT: tp < entry < sl
    - stop distance must be > 0
    - risk/reward must satisfy config.LLM_MIN_RR
    """

    try:
        side = str(side).upper()
        entry = float(entry)
        sl = float(sl)
        tp = float(tp)
    except Exception:
        return {
            "valid": False,
            "reason": "Non-numeric LLM levels",
            "rr": None
        }

    if side == "LONG":
        if not (sl < entry < tp):
            return {
                "valid": False,
                "reason": "Invalid LONG structure: require sl < entry < tp",
                "rr": None
            }
        risk = entry - sl
        reward = tp - entry

    elif side == "SHORT":
        if not (tp < entry < sl):
            return {
                "valid": False,
                "reason": "Invalid SHORT structure: require tp < entry < sl",
                "rr": None
            }
        risk = sl - entry
        reward = entry - tp

    else:
        return {
            "valid": False,
            "reason": "Unknown side",
            "rr": None
        }

    if risk <= 0:
        return {
            "valid": False,
            "reason": "Stop distance must be positive",
            "rr": None
        }

    rr = reward / risk if risk > 0 else None

    if rr is None or rr < config.LLM_MIN_RR:
        return {
            "valid": False,
            "reason": f"RR too low ({rr:.2f})" if rr is not None else "RR invalid",
            "rr": rr
        }

    return {
        "valid": True,
        "reason": "LLM levels validated",
        "rr": rr
    }