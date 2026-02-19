import pandas as pd
import numpy as np


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA (a.k.a. smoothed moving average) — matches Pine ta.rma."""
    series = series.astype(float)
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average — matches Pine ta.ema (recursive form)."""
    series = series.astype(float)
    return series.ewm(span=int(period), adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI — matches Pine ta.rsi (Wilder smoothing)."""
    delta = series.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(0.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR — matches Pine ta.atr (TR with Wilder RMA)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return rma(tr, period)


def prepare_df(ohlcv) -> pd.DataFrame:
    df = pd.DataFrame(
        ohlcv,
        columns=["ts", "open", "high", "low", "close", "volume"],
    )

    # Common indicators (kept for LLM + old strategy compatibility)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)

    return df
