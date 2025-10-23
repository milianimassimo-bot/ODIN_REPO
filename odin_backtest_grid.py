from __future__ import annotations
import os, json, math, time, itertools, random
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple, Literal
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ====== .env ======
try:
    from dotenv import load_dotenv
    load_dotenv(os.getenv("ODIN_DOTENV", ".env"))
except Exception:
    pass

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_BASE_URL = "https://api.twelvedata.com/time_series"
OUT_DIR = "./backtests_grid"
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Universe ======
ASSET_PORTFOLIO = {
    "EUR/USD": ("EUR/USD", "forex"),
    "GBP/USD": ("GBP/USD", "forex"),
    "USD/JPY": ("USD/JPY", "forex"),
    "AUD/USD": ("AUD/USD", "forex"),
    "EUR/JPY": ("EUR/JPY", "forex"),
    "GBP/JPY": ("GBP/JPY", "forex"),
    "GOLD":    ("XAU/USD", "metal"),
    "SILVER":  ("XAG/USD", "metal"),
}

# ====== Fetch TwelveData ======
def fetch_td(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    if not TD_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY mancante (.env)")
    r = requests.get(
        TD_BASE_URL,
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "order": "ASC",
            "timezone": "UTC",
            "apikey": TD_API_KEY,
        },
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    vals = data.get("values")
    if not vals:
        raise RuntimeError(f"Nessun dato per {symbol} {interval}")
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df = df.sort_values("datetime").set_index("datetime")
    # escludi candela odierna aperta
    if len(df) > 1 and df.index[-1].date() == datetime.now(timezone.utc).date():
        df = df.iloc[:-1]
    return df

# ====== Indicatori ======
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr_d1(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def bbands(s: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return lower, ma, upper

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def adx_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(n).mean()
    pdi = 100 * (plus_dm.rolling(n).mean() / atr14)
    mdi = 100 * (minus_dm.rolling(n).mean() / atr14)
    dx = (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.rolling(n).mean()

def r2_trend_strength(series: pd.Series, lookback: int = 60) -> pd.Series:
    y = np.log(series)
    out = pd.Series(index=series.index, dtype=float)
    for i in range(lookback, len(series)):
        ys = y.iloc[i - lookback : i]
        x = np.arange(len(ys))
        slope, intercept = np.polyfit(x, ys.values, 1)
        y_hat = slope * x + intercept
        ss_res = float(((ys.values - y_hat) ** 2).sum())
        ss_tot = float(((ys.values - ys.values.mean()) ** 2).sum())
        out.iloc[i] = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return out

# ====== Regime (replica compatta Modulo 1) ======
def compute_regime_df(d1: pd.DataFrame, roll_win: int = 180) -> pd.DataFrame:
    out = d1.copy()
    out["ADX"] = adx_like(out, 14)
    lo, ma, up = bbands(out["close"], 20, 2.0)
    out["BBW"] = (up - lo) / ma
    out["ATR_PCT"] = (atr_d1(out, 14) / out["close"]) * 100.0
    out["R2_60"] = r2_trend_strength(out["close"], 60)
    minp = max(20, roll_win // 5)
    out["BBW_Q25"] = out["BBW"].rolling(roll_win, min_periods=minp).quantile(0.25)
    out["BBW_Q50"] = out["BBW"].rolling(roll_win, min_periods=minp).quantile(0.50)
    out["BBW_Q75"] = out["BBW"].rolling(roll_win, min_periods=minp).quantile(0.75)
    out["ATRQ50"] = out["ATR_PCT"].rolling(roll_win, min_periods=minp).quantile(0.50)
    out["ATRQ75"] = out["ATR_PCT"].rolling(roll_win, min_periods=minp).quantile(0.75)
    return out

def classify_row(row: pd.Series) -> str:
    adx, bbw, atrp, r2 = row["ADX"], row["BBW"], row["ATR_PCT"], row["R2_60"]
    if pd.notna(adx) and pd.notna(r2) and pd.notna(row["ATRQ50"]) and pd.notna(row["BBW_Q50"]):
        if adx >= 25 and r2 >= 0.28 and (atrp >= row["ATRQ50"] or bbw >= row["BBW_Q50"]):
            return "TRENDING"
    if pd.notna(row["ATRQ75"]) and pd.notna(row["BBW_Q75"]) and atrp >= row["ATRQ75"] and bbw >= row["BBW_Q75"]:
        return "VOLATILE"
    if pd.notna(adx) and pd.notna(row["BBW_Q25"]) and adx < 20 and bbw <= row["BBW_Q25"]:
        return "LATERALE (Compressione)"
    if pd.notna(adx) and adx < 20:
        return "LATERALE"
    return "INDEFINITO"

# ====== Dataclasses ======
@dataclass
class Signal:
    dt: pd.Timestamp
    strategy: str
    direction: Literal["LONG", "SHORT"]
    entry: float
    sl: float
    tp: float

@dataclass
class Trade:
    open_dt: pd.Timestamp
    close_dt: pd.Timestamp
    strategy: str
    direction: str
    entry: float
    sl: float
    tp: float
    result_r: float

# ====== Generatori segnali (WEEKLY raffinata + BB_MR confermata) ======
def gen_weekly_signal(d1: pd.DataFrame, reg_row: pd.Series, params: dict, dt: pd.Timestamp) -> Optional[Signal]:
    if reg_row["REGIME"] != "TRENDING":
        return None
    adx_min = params["adx_min"]
    r2_min = params["r2_min"]
    if not (reg_row["ADX"] >= adx_min and reg_row["R2_60"] >= r2_min):
        return None

    ema10 = ema(d1["close"], 10).loc[dt]
    ema50 = ema(d1["close"], 50).loc[dt]
    ema200 = ema(d1["close"], 200).loc[dt]
    atr14 = atr_d1(d1, 14).loc[dt]
    close_ = float(d1["close"].loc[dt])

    # anti-turbolenza
    atrp = (atr14 / close_) * 100.0 if close_ != 0 else 0.0
    lo, ma, up = bbands(d1["close"], 20, 2.0)
    bbw = float(((up.loc[dt] - lo.loc[dt]) / ma.loc[dt]) if pd.notna(ma.loc[dt]) and ma.loc[dt] != 0 else 0.0)
    if (pd.notna(reg_row["ATRQ75"]) and atrp >= reg_row["ATRQ75"]) or (pd.notna(reg_row["BBW_Q75"]) and bbw >= reg_row["BBW_Q75"]):
        return None

    # anti-inseguimento
    if abs(close_ - float(ema10)) > params["anti_chase_mult"] * float(atr14):
        return None

    direction = "LONG" if ema50 > ema200 else "SHORT"
    entry = float(ema10)
    sl_mult = params["sl_atr_mult"]
    rr = params["rr"]
    if direction == "LONG":
        sl = entry - sl_mult * float(atr14)
        r = entry - sl
        tp = entry + rr * r
    else:
        sl = entry + sl_mult * float(atr14)
        r = sl - entry
        tp = entry - rr * r
    return Signal(dt, "WEEKLY", direction, round(entry, 5), round(sl, 5), round(tp, 5))

def gen_bbmr_signal(d1: pd.DataFrame, reg_row: pd.Series, params: dict, dt: pd.Timestamp) -> Optional[Signal]:
    regime = reg_row["REGIME"]
    if regime not in ("LATERALE", "LATERALE (Compressione)"):
        return None
    if reg_row["ADX"] >= params["adx_max"]:
        return None
    if pd.notna(reg_row["BBW_Q50"]):
        # compressione richiesta
        lo, ma, up = bbands(d1["close"], 20, 2.0)
        bbw_now = float(((up.loc[dt] - lo.loc[dt]) / ma.loc[dt]) if pd.notna(ma.loc[dt]) and ma.loc[dt] != 0 else 0.0)
        if bbw_now > reg_row["BBW_Q50"] * params["bbw_relax"]:
            return None

    # conferma rientro
    lo, ma, up = bbands(d1["close"], params["bb_len"], params["bb_std"])
    rsi14 = rsi(d1["close"], params["rsi_len"])
    if d1.index.get_loc(dt) < 1:
        return None
    prev_dt = d1.index[d1.index.get_loc(dt) - 1]
    prev_close = float(d1["close"].loc[prev_dt])
    last_close = float(d1["close"].loc[dt])
    prev_low = float(lo.loc[prev_dt]); last_low = float(lo.loc[dt])
    prev_up  = float(up.loc[prev_dt]); last_up  = float(up.loc[dt])
    last_rsi = float(rsi14.loc[dt]) if pd.notna(rsi14.loc[dt]) else 50.0
    atr14 = float(atr_d1(d1, 14).loc[dt])
    ma20 = float(ema(d1["close"], 20).loc[dt])

    # LONG
    if prev_close < prev_low and last_close > last_low and last_rsi <= params["rsi_buy"]:
        entry = last_close
        sl = entry - params["atr_stop_mult"] * atr14
        tp = ma20 if params["tp_to_ma"] else last_up
        return Signal(dt, "BB_MR", "LONG", round(entry, 5), round(sl, 5), round(tp, 5))
    # SHORT
    if prev_close > prev_up and last_close < last_up and last_rsi >= params["rsi_sell"]:
        entry = last_close
        sl = entry + params["atr_stop_mult"] * atr14
        tp = ma20 if params["tp_to_ma"] else last_low
        return Signal(dt, "BB_MR", "SHORT", round(entry, 5), round(sl, 5), round(tp, 5))
    return None

# ====== Simulazione (D1, bar-by-bar) ======
def simulate_trade(d1: pd.DataFrame, start_idx: int, sig: Signal, time_stop: int) -> Trade:
    start_bar = start_idx + 1
    end_bar = min(len(d1) - 1, start_bar + time_stop)
    opened = d1.index[start_bar]
    for i in range(start_bar, end_bar + 1):
        hi = float(d1["high"].iloc[i]); lo = float(d1["low"].iloc[i])
        ts = d1.index[i]
        if sig.direction == "LONG":
            if lo <= sig.sl:
                return Trade(opened, ts, sig.strategy, sig.direction, sig.entry, sig.sl, sig.tp, -1.0)
            if hi >= sig.tp:
                return Trade(opened, ts, sig.strategy, sig.direction, sig.entry, sig.sl, sig.tp, +1.0)
        else:
            if hi >= sig.sl:
                return Trade(opened, ts, sig.strategy, sig.direction, sig.entry, sig.sl, sig.tp, -1.0)
            if lo <= sig.tp:
                return Trade(opened, ts, sig.strategy, sig.direction, sig.entry, sig.sl, sig.tp, +1.0)
    # time stop → chiudi a close
    px = float(d1["close"].iloc[end_bar])
    if sig.direction == "LONG":
        r = (px - sig.entry) / (sig.entry - sig.sl) if (sig.entry - sig.sl) != 0 else 0.0
    else:
        r = (sig.entry - px) / (sig.sl - sig.entry) if (sig.sl - sig.entry) != 0 else 0.0
    return Trade(d1.index[start_bar], d1.index[end_bar], sig.strategy, sig.direction, sig.entry, sig.sl, sig.tp, float(r))

# ====== Backtest per asset con parametri ======
def backtest_asset(label: str, symbol: str, start: Optional[str], end: Optional[str], mode: str, params: dict) -> Dict[str, Any]:
    d1 = fetch_td(symbol, "1day", 1200)
    h4 = fetch_td(symbol, "4h", 4000)  # (lo teniamo per eventuali DAILY, qui non serve)
    if start: d1 = d1[d1.index >= pd.Timestamp(start, tz="UTC")]
    if end:   d1 = d1[d1.index <= pd.Timestamp(end, tz="UTC")]

    reg = compute_regime_df(d1)
    reg["REGIME"] = reg.apply(classify_row, axis=1)

    trades: List[Trade] = []

    # warmup: dopo quantili
    start_i = max(200, 180)
    for i in range(start_i, len(d1) - 1):
        dt = d1.index[i]
        regrow = reg.iloc[i]
        sigs: List[Signal] = []
        if mode in ("weekly", "both"):
            s = gen_weekly_signal(d1, regrow, params["WEEKLY"], dt)
            if s: sigs.append(s)
        if mode in ("bbmr", "both"):
            s = gen_bbmr_signal(d1, regrow, params["BB_MR"], dt)
            if s: sigs.append(s)

        for s in sigs:
            tstop = params["WEEKLY"]["time_stop"] if s.strategy == "WEEKLY" else params["BB_MR"]["time_stop"]
            tr = simulate_trade(d1, i, s, tstop)
            trades.append(tr)

    if not trades:
        return {"asset": label, "trades": 0, "summary": {}, "files": {}}

    df_tr = pd.DataFrame([asdict(t) for t in trades])
    df_tr["cum_R"] = df_tr["result_r"].cumsum()
    wins = (df_tr["result_r"] > 0).sum()
    losses = (df_tr["result_r"] < 0).sum()
    pf = (df_tr.loc[df_tr["result_r"] > 0, "result_r"].sum() / abs(df_tr.loc[df_tr["result_r"] < 0, "result_r"].sum())) if losses > 0 else float("inf")
    max_dd = (df_tr["cum_R"].cummax() - df_tr["cum_R"]).max()
    summary = {
        "trades": int(len(df_tr)),
        "winrate_%": round(100 * wins / len(df_tr), 2),
        "expectancy_R": round(df_tr["result_r"].mean(), 3),
        "profit_factor": round(float(pf), 3) if math.isfinite(pf) else float("inf"),
        "max_drawdown_R": round(float(max_dd), 3),
        "sum_R": round(float(df_tr["result_r"].sum()), 3),
        "by_strategy": df_tr.groupby("strategy")["result_r"].agg(["count", "mean", "sum"]).round(3).to_dict(),
    }
    stamp = int(time.time())
    base = f"{label.replace('/','_')}_{mode}_{stamp}"
    f_trades = os.path.join(OUT_DIR, base + "_trades.csv")
    f_equity = os.path.join(OUT_DIR, base + "_equity.csv")
    df_tr.to_csv(f_trades, index=False)
    df_tr[["close_dt", "cum_R"]].to_csv(f_equity, index=False)
    return {"asset": label, "trades": int(len(df_tr)), "summary": summary, "files": {"trades_csv": f_trades, "equity_csv": f_equity}}

# ====== Grid dei parametri (PICCOLA, PRUDENTE) ======
WEEKLY_GRID = {
    "adx_min": [25, 30],
    "r2_min": [0.28, 0.34],
    "anti_chase_mult": [0.3, 0.5],
    "sl_atr_mult": [2.0, 2.5, 3.0],
    "rr": [1.6, 1.8],
    "time_stop": [30],
}
BBMR_GRID = {
    "bb_len": [20],
    "bb_std": [2.0],
    "rsi_len": [14],
    "rsi_buy": [30, 35, 40],
    "rsi_sell": [60, 65, 70],
    "atr_stop_mult": [1.2, 1.5, 1.8],
    "tp_to_ma": [True],
    "adx_max": [20],
    "bbw_relax": [1.0, 1.1],  # 1.0 = stretta su q50, 1.1 = leggera tolleranza
    "time_stop": [20],
}

def grid_dicts(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    out = []
    for prod in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, prod)})
    return out

# ====== Orchestratore Grid-Search ======
def run_grid_for_asset(label: str, symbol: str, start: Optional[str], end: Optional[str], mode: str, max_combos: int) -> Dict[str, Any]:
    weekly_list = grid_dicts(WEEKLY_GRID) if mode in ("weekly", "both") else [{}]
    bbmr_list   = grid_dicts(BBMR_GRID)   if mode in ("bbmr", "both")   else [{}]
    combos = []
    for w in weekly_list:
        for b in bbmr_list:
            combos.append({"WEEKLY": w if w else WEEKLY_GRID, "BB_MR": b if b else BBMR_GRID})
    # limita combinazioni
    if max_combos and len(combos) > max_combos:
        random.seed(42)
        combos = random.sample(combos, max_combos)

    best = None
    best_metrics = None
    all_runs = []
    for params in combos:
        res = backtest_asset(label, symbol, start, end, mode, params)
        if res["trades"] == 0:
            continue
        m = res["summary"]
        score = (
            (m.get("profit_factor", 0) if math.isfinite(m.get("profit_factor", 0)) else 0)
            - max(0.0, m.get("max_drawdown_R", 0)) * 0.01
            + m.get("expectancy_R", 0)
        )
        all_runs.append({"params": params, "metrics": m, "files": res["files"]})
        if best is None or score > best_metrics:
            best, best_metrics = {"params": params, "metrics": m, "files": res["files"]}, score

    return {"asset": label, "best": best, "runs": all_runs}

# ====== CLI ======
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ODIN Backtest Grid – WEEKLY + BB_MR")
    ap.add_argument("--assets", default=",".join(ASSET_PORTFOLIO.keys()), help="Lista separata da virgole (es. 'EUR/USD,USD/JPY,GOLD')")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--mode", default="both", choices=["weekly", "bbmr", "both"], help="Strategie da testare")
    ap.add_argument("--max-combos", type=int, default=120, help="Limite combinazioni per asset (sampling casuale se >)")
    args = ap.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    report = {"asof_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode, "assets": {}}

    for label in assets:
        symbol = ASSET_PORTFOLIO.get(label, (label, "forex"))[0]
        print(f"\n=== Grid {label} [{symbol}] ===")
        try:
            res = run_grid_for_asset(label, symbol, args.start, args.end, args.mode, args.max_combos)
            report["assets"][label] = res
            best = res["best"]
            if best:
                print("BEST:", json.dumps({"params": best["params"], "metrics": best["metrics"]}, indent=2))
            else:
                print("Nessun trade valido per le combinazioni provate.")
        except Exception as e:
            print(f"Errore su {label}: {e!r}")

    out_json = os.path.join(OUT_DIR, f"GRID_REPORT_{int(time.time())}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport salvato: {out_json}")
