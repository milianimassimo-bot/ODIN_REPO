# strats/utils.py
import pandas as pd
from cache_ohlc import fetch_ohlc_cached

def ema(s: pd.Series, n:int)->pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n:int=14)->pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / down.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))

def bbands(s: pd.Series, n:int=20, k:float=2.0):
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    up = ma + k*sd
    lo = ma - k*sd
    return lo, ma, up

def get_df(symbol:str, timeframe:str)->pd.DataFrame:
    interval = "1day" if timeframe=="D1" else "4h"
    
    # --- PATCH COMPLETA: Più storico per D1 e H4 ---
    if interval == "4h":
        outputsize = 1200  # Circa 200 giorni di trading a 4h
        ttl = 3600       # Cache di 1 ora
    elif interval == "1day":
        outputsize = 2000  # Circa 8 anni di dati D1
        ttl = 4 * 3600   # Cache di 4 ore
    else: # Fallback per altri timeframe
        outputsize = 400
        ttl = 15 * 60

    out = fetch_ohlc_cached(symbol, interval, outputsize, max_age_seconds=ttl)
    # --- FINE PATCH ---

    out = out.set_index("datetime")
    return out