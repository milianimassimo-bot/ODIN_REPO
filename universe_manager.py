# ============================================================
# ODIN - Universe Manager (full)
# Selezione intelligente dell'universo tradabile (FX+Metalli+Indici)
# - TwelveData OHLC (cache 6h)
# - Metriche: ATR%, EMA50/200, slope, BBW
# - Score 0..3 per WEEKLY (trend) e BB_MR (mean-reversion)
# - Salva in params/universe.json e invia report Telegram analitico
# ============================================================

import os, json, time, math, logging, random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ----------------- ENV -----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(ROOT, "params")
LOGS_DIR   = os.path.join(ROOT, "logs")
CACHE_DIR  = os.path.join(LOGS_DIR, "cache", "universe")

os.makedirs(PARAMS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

load_dotenv()
TD_API_KEY   = os.getenv("TWELVEDATA_API_KEY", "").strip()
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TTL_HOURS      = int(os.getenv("ODIN_UNIVERSE_TTL_H", "6"))
TOP_N          = int(os.getenv("ODIN_UNIVERSE_TOP_N", "16"))
SEND_TG        = (os.getenv("ODIN_UNIVERSE_TELEGRAM", "1") == "1")
MIN_BARS       = int(os.getenv("ODIN_UNIVERSE_MIN_BARS", "200"))

# ----------------- LOG -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("universe_manager")

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _utcnow_iso() -> str:
    return _utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# ----------------- TELEGRAM -----------------
def send_tg(text: str):
    if not SEND_TG or not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True}
        requests.post(url, json=payload, timeout=10).raise_for_status()
    except Exception as e:
        log.warning(f"Telegram send failed: {e!r}")

# ----------------- DATA: TwelveData -----------------
TD_BASE_URL = "https://api.twelvedata.com/time_series"

def fetch_ohlc_daily(symbol: str, outputsize: int = 400) -> pd.DataFrame:
    if not TD_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY mancante in .env")
    r = requests.get(
        TD_BASE_URL,
        params={"symbol": symbol, "interval": "1day", "outputsize": outputsize,
                "order": "ASC", "timezone": "UTC", "apikey": TD_API_KEY},
        timeout=12
    )
    r.raise_for_status()
    data = r.json()
    vals = data.get("values")
    if not vals:
        raise RuntimeError(f"Nessun dato per {symbol}")
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open","high","low","close"):
        df[c] = df[c].astype(float)
    return df

def get_cached_df(symbol_td: str, ttl_h: int = TTL_HOURS) -> Optional[pd.DataFrame]:
    p = os.path.join(CACHE_DIR, f"{symbol_td.replace('/','_')}.json")
    meta = _load_json(p, None)
    if not meta:
        return None
    ts = meta.get("_ts")
    if not ts:
        return None
    try:
        age = _utcnow() - datetime.fromisoformat(ts.replace("Z","+00:00"))
    except Exception:
        return None
    if age > timedelta(hours=ttl_h):
        return None
    df = pd.DataFrame(meta["data"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open","high","low","close"):
        df[c] = df[c].astype(float)
    return df

def put_cache_df(symbol_td: str, df: pd.DataFrame):
    p = os.path.join(CACHE_DIR, f"{symbol_td.replace('/','_')}.json")
    payload = {
        "_ts": _utcnow_iso(),
        "data": df[["datetime","open","high","low","close"]].assign(
            datetime=lambda x: x["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ).to_dict(orient="records")
    }
    _write_json(p, payload)

def get_df(symbol_td: str) -> Optional[pd.DataFrame]:
    try:
        df = get_cached_df(symbol_td)
        if df is not None:
            return df
        df = fetch_ohlc_daily(symbol_td, outputsize=max(400, MIN_BARS+50))
        if len(df) < MIN_BARS:
            log.info(f"{symbol_td}: dati insufficienti ({len(df)}<{MIN_BARS})")
            return None
        put_cache_df(symbol_td, df)
        return df
    except Exception as e:
        log.info(f"{symbol_td}: fetch error {e!r}")
        return None

# ----------------- METRICHE -----------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    h,l,c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    return float((atr / c.iloc[-1]) * 100.0)

def bbw(df: pd.DataFrame, length: int = 20, std: float = 2.0) -> float:
    close = df["close"]
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    up = ma + std*sd
    lo = ma - std*sd
    m = float(ma.iloc[-1]) if pd.notna(ma.iloc[-1]) else 0.0
    if m == 0.0 or pd.isna(up.iloc[-1]) or pd.isna(lo.iloc[-1]):
        return float("nan")
    return float((up.iloc[-1]-lo.iloc[-1])/m)

def weekly_score(df: pd.DataFrame) -> Tuple[int, Dict[str,float]]:
    # trend quality via EMA50/200 + slope 50; volatility window via ATR%
    c = df["close"]
    ema50 = ema(c, 50)
    ema200= ema(c, 200)
    if pd.isna(ema50.iloc[-1]) or pd.isna(ema200.iloc[-1]):
        return 0, {}
    slope50 = float(ema50.iloc[-1] - ema50.iloc[-6])  # ~1w slope (5 bars)
    spread = float(ema50.iloc[-1] - ema200.iloc[-1])
    atrp   = atr_percent(df, 14)
    # band ATR% "salute": 0.35–2.2 (più ampio per indici/metalli)
    atr_lo, atr_hi = 0.35, 2.2
    in_range = (atr_lo <= atrp <= atr_hi)
    sc = 0
    if in_range:
        sc += 1
    if spread > 0 and slope50 > 0:
        sc += 1
    if abs(spread) > 0.25 * (np.std(c.tail(100)) or 1.0):
        sc += 1
    metrics = {"ATR%": round(atrp,2), "ema50-ema200": round(spread,5), "slope50": round(slope50,5), "in_range": in_range}
    return sc, metrics

def bbmr_score(df: pd.DataFrame) -> Tuple[int, Dict[str,float]]:
    atrp = atr_percent(df, 14)
    bbw_now = bbw(df, 20, 2.0)
    # MR sano se: ATR% tra 0.25–1.6, BBW non troppo stretta e non esplosa
    ok_atr = (0.25 <= atrp <= 1.6)
    # euristica: BBW tra 0.02 e 0.12 (ca.)
    ok_bbw = (not pd.isna(bbw_now)) and (0.02 <= bbw_now <= 0.12)
    sc = 0
    if ok_atr: sc += 1
    if ok_bbw: sc += 1
    # bonus se volatilità “respira” (ATR% vicino al centro banda)
    if 0.45 <= atrp <= 1.2: sc += 1
    metrics = {"ATR%": round(atrp,2), "BBW": round(bbw_now,4) if not pd.isna(bbw_now) else float("nan"),
               "ok_atr": ok_atr, "ok_bbw": ok_bbw}
    return sc, metrics

# ----------------- UNIVERSO: elenco asset -----------------
# Nota: il campo "td" è il simbolo TwelveData (puoi adattarlo se usi varianti)
UNIVERSE: List[Dict[str,str]] = [
    # Majors
    {"asset":"EUR/USD","td":"EUR/USD"},{"asset":"GBP/USD","td":"GBP/USD"},
    {"asset":"USD/JPY","td":"USD/JPY"},{"asset":"AUD/USD","td":"AUD/USD"},
    {"asset":"NZD/USD","td":"NZD/USD"},{"asset":"USD/CHF","td":"USD/CHF"},
    {"asset":"USD/CAD","td":"USD/CAD"},
    # EUR crosses
    {"asset":"EUR/JPY","td":"EUR/JPY"},{"asset":"EUR/GBP","td":"EUR/GBP"},
    {"asset":"EUR/CHF","td":"EUR/CHF"},{"asset":"EUR/AUD","td":"EUR/AUD"},
    {"asset":"EUR/NZD","td":"EUR/NZD"},{"asset":"EUR/CAD","td":"EUR/CAD"},
    # GBP crosses
    {"asset":"GBP/JPY","td":"GBP/JPY"},{"asset":"GBP/CHF","td":"GBP/CHF"},
    {"asset":"GBP/AUD","td":"GBP/AUD"},{"asset":"GBP/NZD","td":"GBP/NZD"},
    {"asset":"GBP/CAD","td":"GBP/CAD"},
    # AUD & NZD crosses
    {"asset":"AUD/JPY","td":"AUD/JPY"},{"asset":"AUD/CHF","td":"AUD/CHF"},
    {"asset":"AUD/CAD","td":"AUD/CAD"},{"asset":"AUD/NZD","td":"AUD/NZD"},
    {"asset":"NZD/JPY","td":"NZD/JPY"},{"asset":"NZD/CHF","td":"NZD/CHF"},
    {"asset":"NZD/CAD","td":"NZD/CAD"},
    # CAD & CHF crosses
    {"asset":"CAD/JPY","td":"CAD/JPY"},{"asset":"CAD/CHF","td":"CAD/CHF"},
    {"asset":"CHF/JPY","td":"CHF/JPY"},
    # Metalli
    {"asset":"GOLD","td":"XAU/USD"},{"asset":"SILVER","td":"XAG/USD"},
    # Indici (usa simboli TD generici; adatta se necessario)
    {"asset":"SP500","td":"SPX"},{"asset":"NAS100","td":"NDX"},
    # Extra FX per arrivare a ~40
    {"asset":"USD/SGD","td":"USD/SGD"},{"asset":"USD/SEK","td":"USD/SEK"},
    {"asset":"USD/NOK","td":"USD/NOK"},{"asset":"EUR/SEK","td":"EUR/SEK"},
    {"asset":"EUR/NOK","td":"EUR/NOK"},
]

# ----------------- LOGICA SELEZIONE -----------------
def evaluate_asset(asset: str, td_symbol: str) -> Optional[Dict[str,Any]]:
    df = get_df(td_symbol)
    if df is None or len(df) < MIN_BARS:
        return None
    try:
        w_sc, w_m = weekly_score(df)
        m_sc, m_m = bbmr_score(df)
        entry = {
            "asset": asset,
            "td": td_symbol,
            "weekly": {"score": int(w_sc), "metrics": w_m},
            "bbmr":   {"score": int(m_sc), "metrics": m_m},
        }
        return entry
    except Exception as e:
        log.info(f"{asset}: eval error {e!r}")
        return None

def pick_universe(evals: List[Dict[str,Any]], top_n: int = TOP_N) -> Dict[str,Any]:
    # ranking primario: max(WEEKLY.score, BB_MR.score), poi media delle due
    def keyfun(e):
        m1 = e["weekly"]["score"]
        m2 = e["bbmr"]["score"]
        return (max(m1,m2), (m1+m2)/2.0)
    ranked = sorted(evals, key=keyfun, reverse=True)
    sel = ranked[:top_n]
    out_items = []
    for e in sel:
        strat = []
        if e["weekly"]["score"] >= 2: strat.append("WEEKLY")
        if e["bbmr"]["score"]   >= 2: strat.append("BB_MR")
        # se nessuna >=2, ma la migliore è 1, prendiamo la migliore sola (conservativo)
        if not strat:
            if e["weekly"]["score"] >= e["bbmr"]["score"] and e["weekly"]["score"] > 0:
                strat.append("WEEKLY")
            elif e["bbmr"]["score"] > 0:
                strat.append("BB_MR")
        out_items.append({
            "asset": e["asset"],
            "strategies": strat or [],
            "scores": {"WEEKLY": e["weekly"]["score"], "BB_MR": e["bbmr"]["score"]},
            "metrics": {"WEEKLY": e["weekly"]["metrics"], "BB_MR": e["bbmr"]["metrics"]},
            "td": e["td"]
        })
    return {"generated_utc": _utcnow_iso(), "items": out_items}

# ----------------- REPORT TG -----------------
def tg_report(payload: Dict[str,Any], all_evals: List[Dict[str,Any]]) -> str:
    lines = [f"🌌 *Universe Manager — Selezione Aggiornata*"]
    lines.append(f"_{payload.get('generated_utc','')}_")
    
    selected_items = payload.get("items", [])
    if not selected_items:
        lines.append("\nNessun asset selezionato.")
        return "\n".join(lines)
    
    lines.append("\n--- ✅ *Asset Selezionati* ---")
    for it in selected_items:
        asset = it['asset']
        scW = it["scores"]["WEEKLY"]
        scM = it["scores"]["BB_MR"]
        strategies = it.get("strategies", [])
        
        lines.append(f"• *{asset}*:")
        has_strat = False
        if "WEEKLY" in strategies:
            lines.append(f"  📈 *WEEKLY* (Punteggio: {scW}/3)")
            has_strat = True
        if "BB_MR" in strategies:
            lines.append(f"  🧘 *BB_MR* (Punteggio: {scM}/3)")
            has_strat = True
        if not has_strat:
            lines.append(f"  _(Nessuna strategia ha superato la soglia)_")

    # --- NUOVA SEZIONE PER GLI SCARTATI ---
    selected_assets = {it['asset'] for it in selected_items}
    all_assets_map = {e['asset']: e for e in all_evals}
    discarded_assets = sorted(list(set(all_assets_map.keys()) - selected_assets))

    if discarded_assets:
        lines.append("\n--- 🗑️ *Asset Scartati (Punteggio Basso)* ---")
        discard_lines = []
        for asset in discarded_assets:
            if eval_data := all_assets_map.get(asset):
                scW = eval_data["weekly"]["score"]
                scM = eval_data["bbmr"]["score"]
                max_score = max(scW, scM)
                discard_lines.append(f"{asset} (max score: {max_score})")
        lines.append(f"_{', '.join(discard_lines)}_")
            
    return "\n".join(lines)

# ----------------- MAIN -----------------
def main():
    log.info("=== Universe Manager: start ===")
    evals: List[Dict[str,Any]] = []
    random.seed(42)

    # mescola leggermente per distribuire le chiamate tra run (in caso di stop improvvisi)
    universe = UNIVERSE.copy()
    random.shuffle(universe)

    for row in universe:
        asset, td = row["asset"], row["td"]
        e = evaluate_asset(asset, td)
        if e:
            evals.append(e)

    if not evals:
        msg = "Universe: nessun asset valutabile (dati mancanti)."
        log.warning(msg)
        send_tg(f"🌌 Universe Manager\n{msg}")
        return

    sel = pick_universe(evals, TOP_N)

    # === ENRICH per ODIN: aggiungi picks e allow (compat totale) ===
    assets = [it["asset"] for it in sel["items"]]
    allow  = {it["asset"]: it.get("strategies", []) for it in sel["items"]}

    sel_enriched = dict(sel)
    sel_enriched["picks"] = [{"asset": a} for a in assets]
    sel_enriched["allow"] = allow

    out_path = os.path.join(PARAMS_DIR, "universe.json")
    _write_json(out_path, sel_enriched)
    log.info(f"Universe selezionato -> {out_path} ({len(sel_enriched['items'])} asset)")

    if SEND_TG:
        send_tg(tg_report(sel_enriched, evals))

    log.info("=== Universe Manager: done ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Universe crash: {e!r}")
        try:
            send_tg(f"❌ Universe Manager ERROR: {type(e).__name__}: {e}")
        except Exception:
            pass
