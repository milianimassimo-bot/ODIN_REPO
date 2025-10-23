# odin_retrain_params.py
# ======================================================
# ODIN - Retrain parametri (mensile/semestrale/"su evento")
# - Universe da params/strategy_roster.json (campo "universe") + (GOLD, SILVER) opzionali
# - Dati D1 da TwelveData con cache locale (fino a ODIN_DL_YEARS anni)
# - IS/OOS configurabili (default semestrale: 24/12)
# - Grid per WEEKLY (trend) e BB_MR (mean reversion)
# - Selezione su IS, validazione su OOS con vincoli
# - Update atomico di params/current.json (+ backup, + STATUS ACTIVE/PAUSED)
# - Report via Telegram
# - Panorama per asset (CSV) in ./data_cache/panorama
# - Filtri: RETRAIN_UNIVERSE_SOURCE, ODIN_INCLUDE_METALS, ODIN_RETRAIN_ONLY
# ======================================================

import os, json, math, time, shutil, logging
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()

# ---------- PATHS ----------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data_cache")
PARAMS_DIR = os.path.join(BASE_DIR, "params")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

ROSTER_PATH = os.path.join(PARAMS_DIR, "strategy_roster.json")

# ---------- ENV ----------
TD_API_KEY  = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_BASE_URL = "https://api.twelvedata.com/time_series"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Universe source
RETRAIN_UNIVERSE_SOURCE = os.getenv("RETRAIN_UNIVERSE_SOURCE", "ROSTER").upper()  # ROSTER | FIXED | ALL
ODIN_INCLUDE_METALS     = os.getenv("ODIN_INCLUDE_METALS", "1") == "1"            # <-- default: includi metalli
ODIN_RETRAIN_ONLY       = os.getenv("ODIN_RETRAIN_ONLY", "").strip()              # CSV labels (es. "GOLD,SILVER")

# finestre IS/OOS
IS_MONTHS  = int(os.getenv("RETRAIN_IS_MONTHS", "24"))
OOS_MONTHS = int(os.getenv("RETRAIN_OOS_MONTHS", "12"))

# vincoli OOS
MIN_TRADES_OOS = int(os.getenv("RETRAIN_MIN_TRADES_OOS", "15"))
MIN_PF_OOS     = float(os.getenv("RETRAIN_MIN_PF_OOS", "1.10"))
MAX_DD_OOS_R   = float(os.getenv("RETRAIN_MAX_DD_OOS_R", "15.0"))

# anni da scaricare/cache
ODIN_DL_YEARS = int(os.getenv("ODIN_DL_YEARS", "10"))

# dry run (non scrive current.json)
ODIN_RETRAIN_DRYRUN = os.getenv("ODIN_RETRAIN_DRYRUN", "0") == "1"

# ---------- LOG ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("odin_retrain")

log.info(f"RETRAIN_UNIVERSE_SOURCE={RETRAIN_UNIVERSE_SOURCE:>6} , INCLUDE_METALS={ODIN_INCLUDE_METALS}, "
         f"RETRAIN_ONLY='{ODIN_RETRAIN_ONLY or '# (opzionale) es: \"GOLD,SILVER\" o \"EUR/USD,USD/JPY\"'}'")

# ---------- Portfolio (label -> (symbol, kind)) ----------
ASSET_PORTFOLIO = {
    # Forex core
    "EUR/USD": ("EUR/USD", "forex"),
    "GBP/USD": ("GBP/USD", "forex"),
    "USD/JPY": ("USD/JPY", "forex"),
    "AUD/USD": ("AUD/USD", "forex"),
    "EUR/JPY": ("EUR/JPY", "forex"),
    "GBP/JPY": ("GBP/JPY", "forex"),
    # Metals (includibili a scelta)
    "GOLD":   ("XAU/USD", "metal"),
    "SILVER": ("XAG/USD", "metal"),
    # Altre coppie che spesso usi (se scegli FIXED/ALL o compaiono nel roster)
    "AUD/CAD": ("AUD/CAD", "forex"),
    "CHF/JPY": ("CHF/JPY", "forex"),
    "EUR/AUD": ("EUR/AUD", "forex"),
    "CAD/JPY": ("CAD/JPY", "forex"),
    "AUD/JPY": ("AUD/JPY", "forex"),
    "EUR/JPY": ("EUR/JPY", "forex"),
    "NZD/JPY": ("NZD/JPY", "forex"),
    "NZD/USD": ("NZD/USD", "forex"),
    "NZD/CAD": ("NZD/CAD", "forex"),
    "USD/SEK": ("USD/SEK", "forex"),
    "EUR/NZD": ("EUR/NZD", "forex"),
    "AUD/CHF": ("AUD/CHF", "forex"),
    "GBP/NZD": ("GBP/NZD", "forex"),
    "USD/NOK": ("USD/NOK", "forex"),
}

# ---------- Telegram ----------
def tg(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
    except Exception as e:
        log.warning(f"Telegram failed: {e!r}")

# ---------- Universe builder ----------
def load_roster_universe() -> list[str]:
    """Legge params/strategy_roster.json e ritorna la lista 'universe'."""
    if not os.path.exists(ROSTER_PATH):
        log.error(f"Roster non trovato: {ROSTER_PATH}")
        return []
    try:
        with open(ROSTER_PATH, "r", encoding="utf-8") as f:
            js = json.load(f)
        uni = js.get("universe", [])
        if not isinstance(uni, list):
            uni = []
        # normalizza: stringhe con slash, uppercase
        uni = [str(x).strip().upper() for x in uni if str(x).strip()]
        return uni
    except Exception as e:
        log.error(f"Errore lettura roster: {e!r}")
        return []

def build_universe() -> list[str]:
    """
    Costruisce l'universo finale in base a:
    - RETRAIN_UNIVERSE_SOURCE: ROSTER | FIXED | ALL
    - ODIN_INCLUDE_METALS: se True, aggiunge GOLD & SILVER
    - ODIN_RETRAIN_ONLY: se presente, filtra a subset CSV
    """
    source = RETRAIN_UNIVERSE_SOURCE
    out: list[str] = []

    if source == "ROSTER":
        base = load_roster_universe()
        out.extend(base)
    elif source == "FIXED":
        # Un set fisso minimale (puoi editarlo)
        out = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/USD"]
    elif source == "ALL":
        # Tutte le label note in ASSET_PORTFOLIO
        out = list(ASSET_PORTFOLIO.keys())
    else:
        log.warning(f"RETRAIN_UNIVERSE_SOURCE sconosciuto: {source}. Fallback=ROSTER.")
        out = load_roster_universe()

    # Aggiungi metalli se richiesto (tu li vuoi assolutamente)
    if ODIN_INCLUDE_METALS:
        for lab in ("GOLD", "SILVER"):
            if lab not in out:
                out.append(lab)

    # Filtra con ONLY se impostato
    if ODIN_RETRAIN_ONLY:
        only_set = {x.strip().upper() for x in ODIN_RETRAIN_ONLY.split(",") if x.strip()}
        out = [x for x in out if x.upper() in only_set]

    # pulizia finale
    out = [x for x in out if x in ASSET_PORTFOLIO]  # tiene solo label mappate
    out = sorted(set(out))
    log.info(f"[UNIVERSE] source={RETRAIN_UNIVERSE_SOURCE}, roster_path='{ROSTER_PATH}', "
             f"count={len(out)}, labels={out}")
    return out

# ---------- Data (cache incrementale, storico ampio) ----------
def cache_path(label: str) -> str:
    key = label.replace("/", "_")
    return os.path.join(DATA_DIR, f"{key}.csv")

def _td_call(symbol: str, interval: str = "1day", outputsize: int = 5000) -> pd.DataFrame:
    if not TD_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY mancante")
    r = requests.get(
        TD_BASE_URL,
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "order": "ASC",
            "timezone": "UTC",
            "apikey": TD_API_KEY
        },
        timeout=25
    )
    r.raise_for_status()
    data = r.json()
    vals = data.get("values")
    if not vals:
        raise RuntimeError(f"Nessun dato TD per {symbol} {interval}")
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open","high","low","close"):
        df[c] = df[c].astype(float)
    df = df.sort_values("datetime").set_index("datetime")
    # scarta barra odierna aperta
    if len(df) > 0 and df.index[-1].date() == datetime.now(timezone.utc).date():
        df = df.iloc[:-1]
    return df

def _years_to_outputsize(years: int) -> int:
    # stima: ~370 barre/anno
    return int(370 * max(1, years))

def get_data(label: str, symbol: str) -> pd.DataFrame:
    """
    Carica la cache locale se esiste, poi scarica storia ampia (ODIN_DL_YEARS),
    unisce e salva. Alla fine ritorna l'intera storia disponibile.
    """
    cp = cache_path(label)

    existing = None
    if os.path.exists(cp):
        try:
            existing = pd.read_csv(cp)
            existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)
            existing = existing.set_index("datetime")[["open","high","low","close"]].sort_index()
        except Exception:
            existing = None

    outputsize = _years_to_outputsize(ODIN_DL_YEARS)
    fresh = _td_call(symbol, interval="1day", outputsize=outputsize)

    # Se esiste cache più lunga, teniamola
    if existing is not None:
        merged = pd.concat([existing, fresh]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = fresh

    # salva cache completa
    merged.to_csv(cp, index=True)
    return merged

# ---------- Indicators ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def bbands(s: pd.Series, n: int = 20, k: float = 2.0):
    ma = s.rolling(n).mean(); sd = s.rolling(n).std(ddof=0)
    up = ma + k*sd; lo = ma - k*sd
    return lo, ma, up

def rsi(s: pd.Series, n: int = 14):
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, 1e-9)
    return 100 - (100/(1+rs))

# ---------- Simple backtest core ----------
def sim_weekly(df: pd.DataFrame, params: dict) -> list[float]:
    """
    WEEKLY (trend su D1):
    - direzione = EMA50 vs EMA200
    - entry quando |close-EMA10| <= anti_chase_mult * ATR(14)
    - SL = sl_atr_mult * ATR; TP a RR fisso; time_stop in giorni
    ritorna: lista di R per i trade chiusi
    """
    c = df["close"]; a = atr(df, 14)
    e10, e50, e200 = ema(c,10), ema(c,50), ema(c,200)
    rr = float(params["rr"])
    anti = float(params["anti_chase_mult"])
    slmult = float(params["sl_atr_mult"])
    tstop = int(params.get("time_stop", 30))
    Rs = []
    pos = None # dict: dir, entry, R, open_time

    for i in range(200, len(df)):
        # chiusura per time stop
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px - pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"] - px)
            Rs.append(fav / pos["R"])
            pos=None

        dirn = "LONG" if e50.iloc[i] > e200.iloc[i] else "SHORT"

        if not pos and abs(c.iloc[i]-e10.iloc[i]) <= anti * a.iloc[i]:
            entry = float(c.iloc[i]); R = slmult * a.iloc[i]
            if R > 1e-9:
                pos = {"dir":dirn,"entry":entry,"R":float(R),"open_time":df.index[i]}

        if pos:
            if pos["dir"]=="LONG":
                tp = pos["entry"] + rr*pos["R"]; sl = pos["entry"] - pos["R"]
                if c.iloc[i] >= tp: Rs.append(rr); pos=None
                elif c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                tp = pos["entry"] - rr*pos["R"]; sl = pos["entry"] + pos["R"]
                if c.iloc[i] <= tp: Rs.append(rr); pos=None
                elif c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def sim_bbmr(df: pd.DataFrame, params: dict) -> list[float]:
    """
    BB_MR (mean-reversion su D1):
    - BB (len/std), RSI (len), soglie buy/sell
    - exit: TP=MA (tp_to_ma=True) o RR fisso, SL = atr_stop_mult*ATR; time_stop
    """
    c = df["close"]; a = atr(df,14)
    lo, ma, up = bbands(c, int(params["bb_len"]), float(params["bb_std"]))
    r = rsi(c, int(params["rsi_len"]))
    rsi_buy  = float(params.get("rsi_buy", 35))
    rsi_sell = float(params.get("rsi_sell", 65))
    atr_stop = float(params.get("atr_stop_mult", 1.5))
    tp_to_ma = bool(params.get("tp_to_ma", True))
    tstop = int(params.get("time_stop", 20))
    Rs = []; pos=None

    for i in range(60, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px-pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"]-px)
            Rs.append(fav/pos["R"]); pos=None

        if not pos:
            if (c.iloc[i] <= lo.iloc[i]) and (r.iloc[i] <= rsi_buy):
                entry=float(c.iloc[i]); R=atr_stop*a.iloc[i]
                if R>1e-9: pos={"dir":"LONG","entry":entry,"R":float(R),"open_time":df.index[i]}
            elif (c.iloc[i] >= up.iloc[i]) and (r.iloc[i] >= rsi_sell):
                entry=float(c.iloc[i]); R=atr_stop*a.iloc[i]
                if R>1e-9: pos={"dir":"SHORT","entry":entry,"R":float(R),"open_time":df.index[i]}

        if pos:
            if pos["dir"]=="LONG":
                sl=pos["entry"]-pos["R"]
                if tp_to_ma and c.iloc[i] >= ma.iloc[i]:
                    Rs.append((ma.iloc[i]-pos["entry"]) / pos["R"]); pos=None
                elif (not tp_to_ma):
                    tp=pos["entry"]+1.5*pos["R"]
                    if c.iloc[i] >= tp: Rs.append(1.5); pos=None
                if pos and c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                sl=pos["entry"]+pos["R"]
                if tp_to_ma and c.iloc[i] <= ma.iloc[i]:
                    Rs.append((pos["entry"]-ma.iloc[i]) / pos["R"]); pos=None
                elif (not tp_to_ma):
                    tp=pos["entry"]-1.5*pos["R"]
                    if c.iloc[i] <= tp: Rs.append(1.5); pos=None
                if pos and c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def metrics_from_R(Rs: list[float]) -> dict:
    if not Rs:
        return {"trades":0,"pf":0.0,"exp":0.0,"maxdd":0.0}
    wins=[r for r in Rs if r>0]; losses=[-r for r in Rs if r<=0]
    gross_up=sum(wins); gross_dn=sum(losses)
    if gross_dn>0: pf=gross_up/gross_dn
    else: pf = float("inf") if gross_up>0 else 0.0
    exp=sum(Rs)/len(Rs)
    eq=[0.0]
    for r in Rs: eq.append(eq[-1]+r)
    peak=0.0; maxdd=0.0
    for x in eq:
        if x>peak: peak=x
        dd=peak-x
        if dd>maxdd: maxdd=dd
    return {"trades":len(Rs), "pf":pf, "exp":exp, "maxdd":maxdd}

# ---------- Grids ----------
GRID_WEEKLY = [
    {"adx_min":25,"r2_min":0.28,"anti_chase_mult":0.3,"sl_atr_mult":2.0,"rr":1.6,"time_stop":30},
    {"adx_min":25,"r2_min":0.34,"anti_chase_mult":0.5,"sl_atr_mult":2.5,"rr":1.8,"time_stop":30},
    {"adx_min":30,"r2_min":0.34,"anti_chase_mult":0.3,"sl_atr_mult":3.0,"rr":1.8,"time_stop":30},
]
GRID_BBMR = [
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":35,"rsi_sell":65,"atr_stop_mult":1.5,"tp_to_ma":True,"time_stop":20},
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":30,"rsi_sell":60,"atr_stop_mult":1.2,"tp_to_ma":True,"time_stop":20},
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":40,"rsi_sell":70,"atr_stop_mult":1.8,"tp_to_ma":True,"time_stop":20},
]

# ---------- Split IS/OOS ----------
def split_is_oos(df: pd.DataFrame, is_months: int, oos_months: int):
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame deve avere un DatetimeIndex")
    end = df.index[-1]
    oos_start = end - pd.DateOffset(months=oos_months-1)
    is_start  = oos_start - pd.DateOffset(months=is_months)
    mask_is  = (df.index >= is_start) & (df.index < oos_start)
    mask_oos = (df.index >= oos_start)
    return df.loc[mask_is].copy(), df.loc[mask_oos].copy()

# ---------- FIT selezione + validazione ----------
def choose_on_is_then_validate(label: str, df: pd.DataFrame):
    df_is, df_oos = split_is_oos(df, IS_MONTHS, OOS_MONTHS)
    out = {"WEEKLY": None, "BB_MR": None}

    # WEEKLY
    best = None
    for p in GRID_WEEKLY:
        Rs = sim_weekly(df_is, p)
        m  = metrics_from_R(Rs)
        if m["trades"] < 10:  # filtro minimo IS
            continue
        score = m["pf"] * max(0.1, m["exp"]+1.0)
        if (best is None) or (score > best["score"]):
            best = {"params":p,"is":m,"score":score}
    if best:
        Rs_oos = sim_weekly(df_oos, best["params"])
        m2 = metrics_from_R(Rs_oos)
        out["WEEKLY"] = {"params":best["params"],"IS":best["is"],"OOS":m2}

    # BB_MR
    best = None
    for p in GRID_BBMR:
        Rs = sim_bbmr(df_is, p)
        m  = metrics_from_R(Rs)
        if m["trades"] < 8:
            continue
        score = m["pf"] * max(0.1, m["exp"]+1.0)
        if (best is None) or (score > best["score"]):
            best = {"params":p,"is":m,"score":score}
    if best:
        Rs_oos = sim_bbmr(df_oos, best["params"])
        m2 = metrics_from_R(Rs_oos)
        out["BB_MR"] = {"params":best["params"],"IS":best["is"],"OOS":m2}
    return out

def passes_oos(m: dict) -> bool:
    return (m["OOS"]["trades"] >= MIN_TRADES_OOS and
            m["OOS"]["pf"] >= MIN_PF_OOS and
            m["OOS"]["exp"] > 0.0 and
            m["OOS"]["maxdd"] <= MAX_DD_OOS_R)

# ---------- Params I/O ----------
def params_current_path() -> str:
    return os.path.join(PARAMS_DIR, "current.json")

def params_backup_path(ts: str) -> str:
    return os.path.join(PARAMS_DIR, "archive", f"current_{ts}.json")

def load_params_current() -> dict:
    p = params_current_path()
    try:
        with open(p,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".new"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False)
    if os.path.exists(path):
        os.replace(tmp, path)
    else:
        shutil.move(tmp, path)

# ---------- Panorama ----------
def save_panorama(label: str, df: pd.DataFrame):
    pano_dir = os.path.join(DATA_DIR, "panorama")
    os.makedirs(pano_dir, exist_ok=True)
    c = df["close"]; h,l = df["high"], df["low"]
    prev = c.shift(1)
    tr = pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    atrp = (atr14 / c.replace(0, pd.NA)) * 100.0
    ma = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    bbw = (2*sd) / ma.replace(0, pd.NA)

    years = (df.index[-1] - df.index[0]).days / 365.25
    pano = {
        "bars": len(df),
        "years_covered": round(years, 2),
        "close_min": round(float(c.min()), 6),
        "close_max": round(float(c.max()), 6),
        "atrp_mean_%": round(float(atrp.mean()), 3),
        "atrp_median_%": round(float(atrp.median()), 3),
        "bbw_median": round(float(bbw.median()), 6),
    }
    out = pd.DataFrame([pano])
    out.to_csv(os.path.join(pano_dir, f"{label.replace('/','_')}.csv"), index=False)

# ---------- MAIN ----------
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("--- ODIN Retrain ---")
    tg("ODIN: retrain avviato…")

    # 1) costruisci universo
    UNIVERSE = build_universe()
    if not UNIVERSE:
        log.error("Universe vuoto. Controlla RETRAIN_UNIVERSE_SOURCE/ROSTER/ONLY.")
        tg("ODIN: retrain ABORTITO (universe vuoto)")
        return

    results = {}
    passes = 0
    total = 0

    for label in UNIVERSE:
        symbol, _kind = ASSET_PORTFOLIO[label]
        total += 1
        try:
            log.info(f"[{label}] cache TwelveData…")
            df = get_data(label, symbol)
            save_panorama(label, df)

            # taglia a IS+OOS per selezione
            cut = df.index[-1] - pd.DateOffset(months=IS_MONTHS+OOS_MONTHS)
            df_cut = df[df.index >= cut].copy()
            if len(df_cut) < 400:
                log.info(f"[{label}] dati insufficienti per IS/OOS (len={len(df_cut)})")
                continue

            res = choose_on_is_then_validate(label, df_cut)
            results[label] = res

            # decisione OOS (se almeno una strategia passa)
            val_ok = False
            if res["WEEKLY"] and passes_oos(res["WEEKLY"]): val_ok = True
            if res["BB_MR"] and passes_oos(res["BB_MR"]):   val_ok = True

            if val_ok:
                passes += 1
                log.info(f"[{label}] OOS PASSED")
            else:
                log.info(f"[{label}] OOS FAILED")

            time.sleep(0.3)
        except Exception as e:
            log.error(f"[{label}] errore retrain: {e!r}")

    # 2) update params/current.json
    cur = load_params_current()
    cur.setdefault("WEEKLY", {}).setdefault("per_asset", {})
    cur.setdefault("BB_MR", {}).setdefault("per_asset", {})
    cur.setdefault("STATUS", {})

    for label, res in results.items():
        upd = False
        if res.get("WEEKLY") and passes_oos(res["WEEKLY"]):
            cur["WEEKLY"]["per_asset"][label] = res["WEEKLY"]["params"]; upd=True
        if res.get("BB_MR") and passes_oos(res["BB_MR"]):
            cur["BB_MR"]["per_asset"][label] = res["BB_MR"]["params"]; upd=True
        cur["STATUS"][label] = "ACTIVE" if upd else "PAUSED"

    if not ODIN_RETRAIN_DRYRUN:
        arch_dir = os.path.join(PARAMS_DIR, "archive")
        os.makedirs(arch_dir, exist_ok=True)
        if os.path.exists(params_current_path()):
            shutil.copy2(params_current_path(), params_backup_path(ts))
        atomic_write_json(params_current_path(), cur)
        log.info(f"[WRITE] params/current.json aggiornato.")
    else:
        log.info("[DRYRUN] update saltato (ODIN_RETRAIN_DRYRUN=1)")

# --- AGGIUNTA: dump per AutoGate/Calibrator ---
RETRAIN_METRICS_JSON = os.path.join(PARAMS_DIR, "retrain_metrics.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def _why_failed(m: dict) -> list[str]:
    if not m or "OOS" not in m:
        return ["no_metrics"]
    o = m["OOS"]
    reasons = []
    if o["trades"] < MIN_TRADES_OOS:
        reasons.append(f"tr<{MIN_TRADES_OOS} ({o['trades']})")
    pf = o["pf"] if math.isfinite(o["pf"]) else 0.0
    if pf < MIN_PF_OOS:
        reasons.append(f"pf<{MIN_PF_OOS:.2f} ({pf:.2f})")
    if o["exp"] <= 0.0:
        reasons.append(f"exp<=0 ({o['exp']:.3f})")
    if o["maxdd"] > MAX_DD_OOS_R:
        reasons.append(f"dd>{MAX_DD_OOS_R:.1f} ({o['maxdd']:.1f})")
    return reasons or ["unknown"]

# 2bis) retrain_metrics.json (piatto, per ai_calibrator)
metrics = {}
for label, res in results.items():
    metrics[label] = {}
    for strat in ("WEEKLY", "BB_MR"):
        m = res.get(strat)
        if m and "OOS" in m:
            o = m["OOS"]
            pf = o["pf"] if math.isfinite(o["pf"]) else 999.0
            metrics[label][strat] = {
                "tr": int(o["trades"]),
                "pf": float(pf),
                "exp": float(o["exp"]),
                "dd": float(o["maxdd"]),
            }
        else:
            metrics[label][strat] = {"tr": 0, "pf": 0.0, "exp": 0.0, "dd": 0.0}

with open(RETRAIN_METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# 3bis) logs/last_retrain.txt (riassunto umano)
lines_txt = []
for label, res in results.items():
    w = res.get("WEEKLY")
    b = res.get("BB_MR")

    def _fmt(m):
        if not m:
            return "—"
        o = m["OOS"]
        pf = o["pf"] if math.isfinite(o["pf"]) else 999.0
        return f"OOS: tr={o['trades']}, PF={pf:.2f}, Exp={o['exp']:.3f}, DD={o['maxdd']:.1f}"

    why = []
    if w and not passes_oos(w):
        why.append("WEEKLY:" + ",".join(_why_failed(w)))
    if b and not passes_oos(b):
        why.append("BB_MR:" + ",".join(_why_failed(b)))

    st = cur["STATUS"].get(label, "ACTIVE")
    lines_txt.append(
        f"{label} [{st}]\n"
        f"  WEEKLY: {_fmt(w)}\n"
        f"  BB_MR : {_fmt(b)}\n"
        f"  WHY   : {'; '.join(why) if why else '—'}"
    )

with open(os.path.join(LOGS_DIR, "last_retrain.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines_txt))
log.info(f"[DUMP] retrain metrics -> {RETRAIN_METRICS_JSON} ; summary -> {os.path.join(LOGS_DIR, 'last_retrain.txt')}")


    # 3) report Telegram
    def fmt(m):
        if not m: return "—"
        o=m["OOS"]
        pf = o['pf'] if math.isfinite(o['pf']) else 999.0
        return f"tr={o['trades']}, PF={pf:.2f}, Exp={o['exp']:.3f}, DD={o['maxdd']:.1f}"

    lines = []
    for label, res in results.items():
        st = cur["STATUS"].get(label, "ACTIVE")
        lines.append(f"{label}: {st} | WEEKLY[{fmt(res.get('WEEKLY'))}] | BB_MR[{fmt(res.get('BB_MR'))}]")

    if lines:
        tg("ODIN retrain (IS/OOS):\n" + "\n".join(lines))
    log.info(f"Retrain completato. Assets processati: {total}, passed: {passes}, failed: {total-passes}")

if __name__ == "__main__":
    main()
