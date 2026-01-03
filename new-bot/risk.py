def position_size(balance, entry, stop, risk_pct):
    risk_amount = balance * risk_pct
    stop_distance = abs(entry - stop)
    return risk_amount / stop_distance
