import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    """Berechnet den RSI (Relative Strength Index)."""
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=window - 1, adjust=True, min_periods=window).mean()
    ma_down = down.ewm(com=window - 1, adjust=True, min_periods=window).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_indicators(df):
    """Fügt SMA, Bollinger, RSI, MACD, ATR und OBV hinzu."""
    if df is None or df.empty:
        return None

    df = df.copy()

    # --- Bestehende Indikatoren ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    std_dev = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['SMA_20'] + (2 * std_dev)
    df['Bollinger_Lower'] = df['SMA_20'] - (2 * std_dev)
    
    df['Daily_Return'] = df['Close'].pct_change()

    # --- NEU: MACD (Trend) ---
    # EMA 12 (schnell) - EMA 26 (langsam)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    # Signal Linie (9-Tage EMA des MACD)
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Histogramm (Differenz)
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # --- NEU: ATR (Volatilität) ---
    # True Range ist das Maximum aus 3 Werten
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    # ATR ist der gleitende Durchschnitt der True Range
    df['ATR'] = true_range.rolling(window=14).mean()

    # --- NEU: OBV (Volumen-Fluss) ---
    # Wenn Close > Vorheriges Close: Addiere Volumen
    # Wenn Close < Vorheriges Close: Subtrahiere Volumen
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Bereinigen
    df.dropna(inplace=True)
    return df