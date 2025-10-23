# ==============================================================================
# ODIN Project - Module 1: Regime Filter
# Version: 2.2.1 (Telegram Hotfix)
# ==============================================================================

import os
import math
import json
import time
import random
import hashlib
import requests # Aggiunto per Telegram
from datetime import datetime, timezone

import pandas as pd
import pandas_ta as ta
from twelvedata import TDClient
from dotenv import load_dotenv
load_dotenv()

# ================== CONFIG ==================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(ROOT_DIR, "params")
UNIVERSE_FILE = os.getenv("ODIN_UNIVERSE_JSON", os.path.join(PARAMS_DIR, "universe.json"))
ALIASES_FILE = os.path.join(PARAMS_DIR, "symbol_aliases.json")

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

INTERVAL = "1day"
OUTPUTSIZE = 420
ROLL_WIN = 180
VERSION = "2.2.1"

WINSORIZE_FOR_QUANTILES = True
WINS_CLAMP_PCT = 0.01

# ================== TELEGRAM SENDER ==================
def tg_send(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
# ...dentro la funzione tg_send()
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "disable_web_page_preview": True}
        requests.post(url, json=payload, timeout=10).raise_for_status()
    except Exception as e:
# ...
        _log(f"⚠️ Telegram send failed: {e!r}")

def _load_aliases(path=ALIASES_FILE):
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return {}

ALIASES = _load_aliases()

def _infer_kind(asset_label: str) -> str:
    a = (asset_label or "").upper().replace(" ", "")
    if "/" in a: return "forex"
    if a in ("GOLD","XAUUSD","XAU/USD","SILVER","XAGUSD","XAG/USD"): return "metal"
    if any(k in a for k in ("OIL","WTI","CL","BRENT")): return "commodity"
    if any(k in a for k in ("SP500","SPX","US500","NAS","NDX","DAX","DJI","FTSE","EU50","STOXX", "UK100")): return "index"
    return "OTHER"

def _default_vendor_symbol(asset_label: str) -> str:
    a = (asset_label or "").upper().strip()
    if "/" in a and len(a) == 7: return a
    if a in ("GOLD","XAUUSD"): return "XAU/USD"
    if a in ("SILVER","XAGUSD"): return "XAG/USD"
    if a in ("OIL_WTI","WTI","USOIL","CL"): return "CL"
    if a in ("SP500","US500","SPX"): return "SPX"
    return asset_label

def load_universe_portfolio(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    except Exception: return {}
    labels = []
    for row in (data.get("picks") or []):
        if a := (row.get("asset") or "").strip(): labels.append((a, row.get("vendorSymbol")))
    if not labels:
        for row in (data.get("items") or []):
            if a := (row.get("asset") or "").strip(): labels.append((a, row.get("vendorSymbol")))
    if not labels and isinstance(data.get("allow"), dict):
        for a in data["allow"].keys():
            if (a or "").strip(): labels.append((a.strip(), None))
    out = {}
    for (a, vs) in labels:
        lab = a
        kind = _infer_kind(lab)
        sym  = vs or _default_vendor_symbol(lab)
        if ali := ALIASES.get(lab):
            if ali.get("twelvedata"):
                sym = ali["twelvedata"]
                kind = ali.get("kind", kind)
        out[lab] = (sym, kind)
    return out

RISK_BUCKETS = { "forex": "FX", "metal": "XAU_XAG", "commodity": "COMMOD", "index": "INDEX" }
def bucket_for(kind: str) -> str: return RISK_BUCKETS.get(kind, "OTHER")

def _log(msg: str): print(msg, flush=True)

def _retry_backoff(fn, tries=4, base=0.4, cap=5.0):
    last_exc = None
    for i in range(tries):
        try: return fn()
        except Exception as e:
            last_exc = e
            time.sleep(min(cap, base * (2 ** i)) + random.uniform(0, 0.25))
    raise last_exc

def _drop_last_if_open(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name != "datetime": df.index = pd.to_datetime(df.index, utc=True)
    if df.index[-1].date() == datetime.now(timezone.utc).date(): return df.iloc[:-1].copy()
    return df

def _nan_to_none(x):
    try: return None if (x != x) else float(x)
    except Exception: return x

def safe_get_series(td, symbol, interval, outputsize):
    from cache_ohlc import fetch_ohlc_twelvedata
    df = fetch_ohlc_twelvedata(symbol, interval, outputsize)
    if df is None or df.empty: raise ValueError(f"Serie vuota per {symbol}")
    df = df.set_index("datetime").sort_index()
    return df

def preflight_symbols(td, portfolio: dict, interval: str, outputsize: int):
    resolved, problems = {}, []
    _log("--- Avvio Preflight Check ---")
    for label, (symbol, kind) in portfolio.items():
        def _probe():
            ts = td.time_series(symbol=symbol, interval=interval, outputsize=min(10, outputsize), timezone="UTC")
            df = ts.as_pandas()
            if df is None or df.empty: raise ValueError("Serie vuota")
            if not {"open", "high", "low", "close"}.issubset({c.lower() for c in df.columns}):
                raise ValueError("Colonne OHLC mancanti")
            return True
        try:
            _retry_backoff(_probe, tries=2, base=0.2, cap=2.0)
            resolved[label] = (symbol, kind)
            _log(f"  ✅ {label} [{symbol}]")
        except Exception as e:
            problems.append((label, symbol, f"{type(e).__name__}: {e!r}"))
    if problems:
        _log("\n⚠️  Preflight: esclusi per problemi:")
        for label, sym, err in problems: _log(f"  ❌ {label} [{sym}] -> {err}")
    _log("--- Preflight completato ---\n")
    return resolved

def atr_manual(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    atr.name = f"ATRm_{length}"
    return atr

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    adx = df.ta.adx(length=14); bb = df.ta.bbands(length=20, std=2.0); atr = df.ta.atr(length=14)
    out = pd.concat([df, adx, bb, atr], axis=1).dropna(how="any")
    atr_cols = [c for c in out.columns if c.upper().startswith("ATR") and not c.upper().startswith("NATR")]
    if not atr_cols:
        out = pd.concat([out, atr_manual(out, length=14)], axis=1)
        atr_cols = ["ATRm_14"]
    atr_col = atr_cols[0]
    bbw_cols = [c for c in out.columns if c.upper().startswith("BBB_")]
    if not bbw_cols: raise ValueError("Colonna BBB_* non trovata.")
    bbw_col = bbw_cols[0]
    out["atr_pct"] = (out[atr_col] / out["close"]) * 100.0
    out = out.rename(columns={bbw_col: "BBW"})
    return out

def _winsorize_series(s: pd.Series, clamp_pct: float = 0.01) -> pd.Series:
    if clamp_pct <= 0.0: return s
    p_low, p_hi = s.quantile(clamp_pct), s.quantile(1.0 - clamp_pct)
    return s.clip(lower=p_low, upper=p_hi)

def rolling_quantiles(s: pd.Series, win: int = 180, winsorize: bool = False):
    minp = max(20, win // 5)
    src = _winsorize_series(s, clamp_pct=WINS_CLAMP_PCT) if winsorize else s
    q25 = src.rolling(win, min_periods=minp).quantile(0.25)
    q50 = src.rolling(win, min_periods=minp).quantile(0.50)
    q75 = src.rolling(win, min_periods=minp).quantile(0.75)
    return q25, q50, q75

def r2_trend_strength(series: pd.Series, lookback: int = 60) -> float:
    s = series.tail(lookback)
    if len(s) < max(20, lookback // 3): return math.nan
    import numpy as np
    y, x = np.log(s.values), np.arange(len(s))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res, ss_tot = float(((y - y_hat) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return (1 - ss_res / ss_tot) if ss_tot > 0 else math.nan

def classify_regime(row, ctx):
    adx, bbw, atrp, r2 = row["adx"], row["bbw"], row["atr_pct"], row["r2_60"]
    q = ctx
    if (adx is not None and r2 is not None and q.get("atrp_q50") is not None and q.get("bbw_q50") is not None
        and adx >= 25 and r2 >= 0.28 and (atrp >= q["atrp_q50"] or bbw >= q["bbw_q50"])): return "TRENDING"
    if (q.get("atrp_q75") is not None and q.get("bbw_q75") is not None and atrp >= q["atrp_q75"] and bbw >= q["bbw_q75"]): return "VOLATILE"
    if adx is not None and q.get("bbw_q25") is not None and adx < 20 and bbw <= q["bbw_q25"]: return "LATERALE (Compressione)"
    if adx is not None and adx < 20: return "LATERALE"
    return "INDEFINITO"

def _meta_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def main():
    _log(f"--- Avvio Filtro di Regime ODIN v{VERSION} ---")
    if not TD_API_KEY:
        _log("‼️ ERRORE CRITICO: TWELVEDATA_API_KEY non trovata (.env).")
        return
    td = TDClient(apikey=TD_API_KEY)
    universe_portfolio = load_universe_portfolio(UNIVERSE_FILE)
    if not universe_portfolio:
        _log(f"‼️ Universe vuoto o non leggibile: {UNIVERSE_FILE}")
        return
    ASSET_RESOLVED = preflight_symbols(td, universe_portfolio, INTERVAL, OUTPUTSIZE)
    if not ASSET_RESOLVED:
        _log("‼️ Nessun simbolo valido dopo il preflight. Stop.")
        return
    records = []
    asof_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, (label, (symbol, kind)) in enumerate(ASSET_RESOLVED.items(), 1):
        _log(f"Analizzo: {label}  (TD: {symbol}, tipo: {kind})")
        try:
            df = safe_get_series(td, symbol, INTERVAL, OUTPUTSIZE)
            df = compute_indicators(df)
            if (kind == "index") or any(tag in label.upper() for tag in ("SP", "SPX", "NAS", "NDX", "DAX", "DJI", "FTSE", "EU50", "STOXX", "UK100")):
                q_lo, q_hi = float(df["atr_pct"].quantile(0.01)), min(float(df["atr_pct"].quantile(0.99)), 4.5)
                df["atr_pct"] = df["atr_pct"].clip(lower=q_lo, upper=q_hi)
                _log(f"[ATR% clamp] {label}: clip 1–99% ({q_lo:.2f}–{q_hi:.2f})")
            adx_cols = [c for c in df.columns if c.upper().startswith("ADX_")]
            if not adx_cols: raise ValueError("Colonna ADX_* non trovata")
            adx_last = df[adx_cols[0]].dropna()
            if adx_last.empty: raise ValueError("ADX indisponibile")
            last_adx_val = float(adx_last.iloc[-1])
            bbw_q25, bbw_q50, bbw_q75 = rolling_quantiles(df["BBW"], win=ROLL_WIN, winsorize=WINSORIZE_FOR_QUANTILES)
            atrp_q25, atrp_q50, atrp_q75 = rolling_quantiles(df["atr_pct"], win=ROLL_WIN, winsorize=WINSORIZE_FOR_QUANTILES)
            r2, last = r2_trend_strength(df["close"], lookback=60), df.iloc[-1]
            ctx = {k: _nan_to_none(v.iloc[-1]) for k, v in locals().items() if k.startswith(("atrp_q", "bbw_q"))}
            metrics = {"adx": last_adx_val, "bbw": float(last["BBW"]), "atr_pct": float(last["atr_pct"]), "r2_60": _nan_to_none(r2)}
            regime = classify_regime(metrics, ctx)
            record = {"asset": label, "vendorSymbol": symbol, "kind": kind, "bucket": bucket_for(kind), "timeframe": "D1", "regime": regime, "metrics": metrics, "context": ctx}
            r2_log = metrics["r2_60"] if metrics["r2_60"] is not None else float('nan')
            _log(f"--> ADX={metrics['adx']:.2f} | BBW={metrics['bbw']:.3f} | ATR%={metrics['atr_pct']:.2f} | R2={r2_log:.2f} -> {regime}")
            records.append(record)
            if i % 8 == 0: time.sleep(0.5)
        except Exception as e:
            _log(f"‼️ ERRORE su {label} ({symbol}) [{type(e).__name__}]: {e!r}")
            records.append({"asset": label, "vendorSymbol": symbol, "kind": kind, "bucket": bucket_for(kind), "timeframe": "D1", "regime": "ERRORE_DATI", "metrics": None, "context": None})

    order = ["TRENDING", "VOLATILE", "LATERALE (Compressione)", "LATERALE", "INDEFINITO", "ERRORE_DATI"]
    df_out = pd.DataFrame(records)
    if not df_out.empty:
        df_out["regime"] = pd.Categorical(df_out["regime"], categories=order, ordered=True)
        df_out = df_out.sort_values(["regime", "asset"]).reset_index(drop=True)

    meta = {
        "version": VERSION, "vendor": "twelvedata", "interval_api": INTERVAL, "timeframe": "D1",
        "roll_window": ROLL_WIN, "universe": list(ASSET_RESOLVED.keys()), "assets_processed": len(ASSET_RESOLVED),
        "asof_utc": asof_utc,
    }
    meta.update({"universe_file": os.path.abspath(UNIVERSE_FILE), "symbols_td": {k: v[0] for k, v in ASSET_RESOLVED.items()}})
    
    payload = {"meta":  meta, "items": json.loads(df_out.to_json(orient="records"))}
    payload["meta"]["sha1"] = _meta_hash(payload)

    out_csv, out_json = "odin_regime_report_d1.csv", "odin_regime_report_d1.json"
    pd.json_normalize(payload["items"], sep="_").to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2, ensure_ascii=False)

    _log("\n" + "="*72)
    _log(f"--- REPORT DI REGIME COMPLETATO (v{VERSION}) ---")
    
    if not df_out.empty:
        counts = df_out["regime"].value_counts(dropna=False).to_dict()
        _log(f"[Summary Regimi] {counts}")
        
    _log(f"Salvato: {out_csv} | {out_json} (items: {len(df_out)})")
    _log(f"Hash del report (SHA1): {payload['meta']['sha1']}")
    _log("="*72)
    
# ... alla fine della funzione main()
    try:
        summary_lines = ["📊 ODIN - Report di Regime"] # Rimosso '*'
        summary_lines.append(f"Hash: {payload['meta']['sha1'][:10]}") # Rimosso '`'
        
        if not df_out.empty:
            counts = df_out["regime"].value_counts(dropna=False).to_dict()
            summary_lines.append("\nRiepilogo Regimi:") # Rimosso '*'
            for regime, count in counts.items():
                summary_lines.append(f"- {regime}: {count}")
        else:
            summary_lines.append("\nNessun asset analizzato.")
            
        trending_assets = df_out[df_out["regime"] == "TRENDING"]["asset"].tolist()
        ranging_assets = df_out[df_out["regime"].str.contains("LATERALE", na=False)]["asset"].tolist()

        if trending_assets:
            summary_lines.append("\nAsset in Trend:") # Rimosso '*'
            summary_lines.append(f"{', '.join(trending_assets)}") # Rimosso '`'
        
        if ranging_assets:
            summary_lines.append("\nAsset in Laterale:") # Rimosso '*'
            summary_lines.append(f"{', '.join(ranging_assets)}") # Rimosso '`'
            
        tg_send("\n".join(summary_lines))
    except Exception as e:
        _log(f"⚠️ Errore creazione report Telegram: {e!r}")

if __name__ == "__main__":
    if seed_env := os.getenv("ODIN_SEED_BACKOFF"):
        try:
            random.seed(int(seed_env))
            _log(f"[debug] Backoff RNG seed = {seed_env}")
        except: pass
    main()