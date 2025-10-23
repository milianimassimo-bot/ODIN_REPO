# backtester_light.py — micro backtest per BB_MR / WEEKLY (ultimi N mesi)
import os, math, logging
from typing import Dict, Tuple
import pandas as pd
from cache_ohlc import fetch_ohlc_cached

log = logging.getLogger("bt_light")

def ema(s: pd.Series, n: int): 
    return s.ewm(span=n, adjust=False).mean()

def bbands(s: pd.Series, n=20, std=2.0):
    ma = s.rolling(n).mean(); sd = s.rolling(n).std(ddof=0)
    return ma - std*sd, ma, ma + std*sd

def atr_df(df: pd.DataFrame, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def metrics_from_R(Rs: list[float]) -> dict:
    """Calcola tr/pf/exp/dd da una lista di R-multipli (+1 win, -0.5 loss, ecc.)."""
    trades = len(Rs)
    wins   = [r for r in Rs if r > 0]
    losses = [-r for r in Rs if r <= 0]
    gross_up = sum(wins)
    gross_dn = sum(losses)
    if gross_dn > 0:
        pf = gross_up / gross_dn
    else:
        pf = float("inf") if gross_up > 0 else 0.0
    # equity cumulata per DD
    eq, peak, maxdd = 0.0, 0.0, 0.0
    for r in Rs:
        eq += r
        if eq > peak: peak = eq
        dd = peak - eq
        if dd > maxdd: maxdd = dd
    exp = (sum(Rs) / trades) if trades > 0 else 0.0
    return {"tr": trades, "pf": round(pf, 2), "exp": round(exp, 3), "dd": round(maxdd, 2)}


def bt_bbmr(symbol: str, months: int = 6) -> Dict[str, float]:
    d1 = fetch_ohlc_cached(symbol, "1day", 260)
    if d1 is None or d1.empty or len(d1) < 60:
        return {"tr": 0, "pf": 0.0, "exp": 0.0, "dd": 0.0}
    d1 = d1.sort_index().iloc[-max(22*months, 60):].copy()
    lo, ma, up = bbands(d1["close"], 20, 2.0)
    atr = atr_df(d1, 14)

    Rs = []
    for i in range(2, len(d1)):
        prev, last = d1.iloc[i-1], d1.iloc[i]
        prev_lo, last_lo = float(lo.iloc[i-1]), float(lo.iloc[i])
        prev_up, last_up = float(up.iloc[i-1]), float(up.iloc[i])
        prev_c,  last_c  = float(prev.close),  float(last.close)
        last_ma, last_atr = float(ma.iloc[i]), float(atr.iloc[i])
        if any(map(math.isnan, [prev_lo, last_lo, prev_up, last_up, last_ma, last_atr])):
            continue

        # LONG re-entry (rientro da sottobanda verso l'interno)
        if prev_c < prev_lo and last_c > last_lo:
            entry = last_c; tp = last_ma; sl = entry - 1.5*last_atr
            if tp > entry:
                R  = 1.5 * last_atr
                take = tp - entry
                rr = (take / R) if R > 0 else 0.0
                Rs.append(+1.0 if rr >= 1.0 else -0.5)

        # SHORT re-entry (rientro da sovrabanda verso l'interno)
        if prev_c > prev_up and last_c < last_up:
            entry = last_c; tp = last_ma; sl = entry + 1.5*last_atr
            if tp < entry:
                R  = 1.5 * last_atr
                take = entry - tp
                rr = (take / R) if R > 0 else 0.0
                Rs.append(+1.0 if rr >= 1.0 else -0.5)

    return metrics_from_R(Rs)


def bt_weekly(symbol: str, months: int = 6) -> Dict[str, float]:
    d1 = fetch_ohlc_cached(symbol, "1day", 260)
    if d1 is None or d1.empty or len(d1) < 200:
        return {"tr": 0, "pf": 0.0, "exp": 0.0, "dd": 0.0}
    d1 = d1.sort_index().iloc[-max(22*months, 200):].copy()
    ema50  = ema(d1["close"], 50); 
    ema200 = ema(d1["close"], 200)
    a14    = atr_df(d1, 14)

    Rs = []
    for i in range(200, len(d1)):
        c  = float(d1["close"].iloc[i])
        e50, e200, a = float(ema50.iloc[i]), float(ema200.iloc[i]), float(a14.iloc[i])
        if math.isnan(e50) or math.isnan(e200) or math.isnan(a): 
            continue
        dirL = (e50 > e200)
        # Proxy: RR > 1 ? win : half-loss
        R = 2.5*a
        take = 1.8*R
        rr = (take / R) if R > 0 else 0.0
        Rs.append(+1.0 if rr >= 1.0 else -0.5)

    return metrics_from_R(Rs)

