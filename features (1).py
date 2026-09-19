"""
Vectorized Intraday Feature Engineering for Tata Steel (NSE: TATASTEEL)
"""
import numpy as np
import pandas as pd
from config import EMA_FAST, EMA_SLOW, RSI_PERIOD, ATR_PERIOD, SUPERTREND_MULTIPLIER


def calculate_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # EMA 9 & 21
    data['EMA_Fast'] = data['Close'].ewm(span=EMA_FAST, adjust=False).mean()
    data['EMA_Slow'] = data['Close'].ewm(span=EMA_SLOW, adjust=False).mean()
    data['EMA_Spread'] = (data['EMA_Fast'] - data['EMA_Slow']) / data['EMA_Slow']

    # ATR 14
    high_low = data['High'] - data['Low']
    high_close_prev = (data['High'] - data['Close'].shift(1)).abs()
    low_close_prev = (data['Low'] - data['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=ATR_PERIOD).mean()

    # Supertrend (10, 3.0)
    hl2 = (data['High'] + data['Low']) / 2
    upper_basic = hl2 + (SUPERTREND_MULTIPLIER * data['ATR'])
    lower_basic = hl2 - (SUPERTREND_MULTIPLIER * data['ATR'])
    supertrend = [0.0] * len(data)
    direction = [1] * len(data)

    for i in range(1, len(data)):
        upper_band = upper_basic.iloc[i] if (upper_basic.iloc[i] < upper_basic.iloc[i - 1] or data['Close'].iloc[i - 1] > upper_basic.iloc[i - 1]) else upper_basic.iloc[i - 1]
        lower_band = lower_basic.iloc[i] if (lower_basic.iloc[i] > lower_basic.iloc[i - 1] or data['Close'].iloc[i - 1] < lower_basic.iloc[i - 1]) else lower_basic.iloc[i - 1]
        if data['Close'].iloc[i] > upper_band:
            direction[i] = 1
        elif data['Close'].iloc[i] < lower_band:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        supertrend[i] = lower_band if direction[i] == 1 else upper_band

    data['Supertrend'] = supertrend
    data['Supertrend_Direction'] = direction

    # VWAP
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (typical_price * data['Volume']).cumsum() / (data['Volume'].cumsum() + 1e-9)
    data['VWAP_Diff'] = (data['Close'] - data['VWAP']) / data['VWAP']

    # RSI
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(window=RSI_PERIOD).mean()
    data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    # Volume Ratio & Lags
    data['Volume_Ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1)
    data['Return_1'] = data['Close'].pct_change(1)
    data['Return_3'] = data['Close'].pct_change(3)
    data['Lag_Close_1'] = data['Close'].shift(1)

    # Target
    data['Target_Close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    return data
