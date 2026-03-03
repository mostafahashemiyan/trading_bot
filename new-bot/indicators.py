"""
Technical indicators used in strategy.
All functions are pure pandas/numpy.
"""

import pandas as pd
import numpy as np

import config


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Manual ADX without ta-lib."""
    tr = calculate_atr(df, period=1)  # raw TR
    up = df['high'] - df['high'].shift()
    down = df['low'].shift() - df['low']

    pdm = np.where((up > down) & (up > 0), up, 0)
    mdm = np.where((down > up) & (down > 0), down, 0)

    pdm_s = pd.Series(pdm, index=df.index)
    mdm_s = pd.Series(mdm, index=df.index)

    atr = calculate_atr(df, period)
    pdi = 100 * pdm_s.ewm(span=period, adjust=False).mean() / atr
    mdi = 100 * mdm_s.ewm(span=period, adjust=False).mean() / atr

    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return dx.ewm(span=period, adjust=False).mean()


def prepare_df(ohlcv_list: list) -> pd.DataFrame:
    """
    Convert ccxt OHLCV list → DataFrame with indicators.
    """
    if not ohlcv_list or len(ohlcv_list) < 200:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv_list, columns=["ts", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df["ema_fast"] = ema(df["close"], config.FAST_EMA_PERIOD)
    df["ema_slow"] = ema(df["close"], config.SLOW_EMA_PERIOD)
    df["rsi"] = rsi(df["close"], config.RSI_PERIOD)
    df["atr"] = calculate_atr(df, config.ATR_PERIOD)
    df["adx"] = calculate_adx(df, config.ADX_PERIOD)

    return df