"""
INDICATORS MODULE
-----------------
A clean, reliable, high-performance indicator engine for both
live trading & backtesting. No TA-Lib required.
"""

import pandas as pd
import numpy as np
import config


# ─────────────────────────────────────────────────────────────
# Basic Indicators
# ─────────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()

    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────
# ATR (Average True Range)
# ─────────────────────────────────────────────────────────────
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


# ─────────────────────────────────────────────────────────────
# ADX (Accurate custom implementation without TA-Lib)
# ─────────────────────────────────────────────────────────────
def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = calculate_atr(df, period=1)

    up_move = df['high'].diff()
    down_move = df['low'].diff() * -1

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = calculate_atr(df, period)

    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx


# ─────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────
def prepare_df(ohlcv: list) -> pd.DataFrame:
    """
    Convert CCXT OHLCV list → DataFrame with indicators.
    Ensures stable columns and numeric types.
    """

    if not ohlcv or len(ohlcv) < 50:
        return pd.DataFrame()

    df = pd.DataFrame(
        ohlcv,
        columns=["ts", "open", "high", "low", "close", "volume"]
    )

    # Convert to numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    # Indicators
    df["ema_fast"] = ema(df["close"], config.FAST_EMA_PERIOD)
    df["ema_slow"] = ema(df["close"], config.SLOW_EMA_PERIOD)
    df["rsi"] = rsi(df["close"], config.RSI_PERIOD)
    df["atr"] = calculate_atr(df, config.ATR_PERIOD)
    df["adx"] = calculate_adx(df, config.ADX_PERIOD)

    return df