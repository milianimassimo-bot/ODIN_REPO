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

# >>> PATCH: files diagnostica retrain (JSON + TXT)
RETRAIN_METRICS_JSON = os.path.join(PARAMS_DIR, "retrain_metrics.json")
RETRAIN_REPORT_TXT   = os.path.join(BASE_DIR, "logs", "last_retrain.txt")
os.makedirs(os.path.dirname(RETRAIN_REPORT_TXT), exist_ok=True)
# <<< PATCH

# ---------- ENV ----------
TD_API_KEY  = os.getenv("TWELVEDATA_API_KEY", "").strip()

TD_BASE_URL = "https://api.twelvedata.com/time_series"
RETRAIN_UNIVERSE_SOURCE = os.getenv("RETRAIN_UNIVERSE_SOURCE", "ROSTER")
RETRAIN_UNIVERSE_SOURCE = (RETRAIN_UNIVERSE_SOURCE or "ROSTER").strip().upper()  # ROSTER | FIXED | ALL

def _truthy(x: str) -> bool:
    return str(x or "").strip().lower() in {"1","true","yes","on","y"}

ODIN_INCLUDE_METALS = _truthy(os.getenv("ODIN_INCLUDE_METALS", "0"))
ODIN_RETRAIN_ONLY   = (os.getenv("ODIN_RETRAIN_ONLY", "") or "").strip()  # CSV labels o vuoto


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# finestre IS/OOS
IS_MONTHS  = int(os.getenv("RETRAIN_IS_MONTHS", "24"))
OOS_MONTHS = int(os.getenv("RETRAIN_OOS_MONTHS", "12"))

# vincoli OOS
MIN_TRADES_OOS = int(os.getenv("RETRAIN_MIN_TRADES_OOS", "5"))
MIN_PF_OOS     = float(os.getenv("RETRAIN_MIN_PF_OOS", "1.05"))
MAX_DD_OOS_R   = float(os.getenv("RETRAIN_MAX_DD_OOS_R", "25.0"))

# anni da scaricare/cache
ODIN_DL_YEARS = int(os.getenv("ODIN_DL_YEARS", "10"))

# dry run (non scrive current.json)
ODIN_RETRAIN_DRYRUN = os.getenv("ODIN_RETRAIN_DRYRUN", "0") == "1"

# ---------- LOG ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("odin_retrain")

log.info(f"RETRAIN_UNIVERSE_SOURCE={RETRAIN_UNIVERSE_SOURCE}, INCLUDE_METALS={ODIN_INCLUDE_METALS}, RETRAIN_ONLY={ODIN_RETRAIN_ONLY or '(none)'}")



# ---------- Portfolio / mapping label -> (symbol, kind) ----------
# =====================================================
# 🔹 ASSET PORTFOLIO — Mappa completa per retrain & sentinel
# =====================================================
# Label = nome logico interno ODIN
# Value = (ticker TwelveData/MT5, tipo)
ASSET_PORTFOLIO = {
    # --- MAJORS ---
    "EUR/USD": ("EUR/USD", "forex"),
    "GBP/USD": ("GBP/USD", "forex"),
    "USD/JPY": ("USD/JPY", "forex"),
    "USD/CHF": ("USD/CHF", "forex"),
    "USD/CAD": ("USD/CAD", "forex"),
    "AUD/USD": ("AUD/USD", "forex"),
    "NZD/USD": ("NZD/USD", "forex"),

    # --- CROSS EUR ---
    "EUR/GBP": ("EUR/GBP", "forex"),
    "EUR/JPY": ("EUR/JPY", "forex"),
    "EUR/CHF": ("EUR/CHF", "forex"),
    "EUR/CAD": ("EUR/CAD", "forex"),
    "EUR/AUD": ("EUR/AUD", "forex"),
    "EUR/NZD": ("EUR/NZD", "forex"),

    # --- CROSS GBP ---
    "GBP/JPY": ("GBP/JPY", "forex"),
    "GBP/CHF": ("GBP/CHF", "forex"),
    "GBP/CAD": ("GBP/CAD", "forex"),
    "GBP/AUD": ("GBP/AUD", "forex"),
    "GBP/NZD": ("GBP/NZD", "forex"),

    # --- CROSS AUD ---
    "AUD/JPY": ("AUD/JPY", "forex"),
    "AUD/CHF": ("AUD/CHF", "forex"),
    "AUD/CAD": ("AUD/CAD", "forex"),
    "AUD/NZD": ("AUD/NZD", "forex"),

    # --- CROSS NZD ---
    "NZD/JPY": ("NZD/JPY", "forex"),
    "NZD/CHF": ("NZD/CHF", "forex"),
    "NZD/CAD": ("NZD/CAD", "forex"),

    # --- CROSS CAD ---
    "CAD/JPY": ("CAD/JPY", "forex"),
    "CAD/CHF": ("CAD/CHF", "forex"),

    # --- CROSS CHF ---
    "CHF/JPY": ("CHF/JPY", "forex"),

    # --- SCANDINAVI E EXOTICS ---
    "USD/SEK": ("USD/SEK", "forex"),
    "USD/NOK": ("USD/NOK", "forex"),
    "USD/DKK": ("USD/DKK", "forex"),
    "EUR/SEK": ("EUR/SEK", "forex"),
    "EUR/NOK": ("EUR/NOK", "forex"),
    "EUR/DKK": ("EUR/DKK", "forex"),

    # --- METALLI ---
    "GOLD":   ("XAU/USD", "metal"),
    "SILVER": ("XAG/USD", "metal"),

    # --- INDICI ---
    "SP500":  ("SPX/USD", "index"),      # alias TwelveData: SPX/USD o ^SPX
    "DAX":    ("GDAXI", "index"),
    "FTSE":   ("FTSE:FSI", "index"),
    "NAS100": ("NDX/USD", "index"),
    "US30":   ("DJI/USD", "index"),

    # --- CRIPTO (opzionale futuro) ---
    "BTC/USD": ("BTC/USD", "crypto"),
    "ETH/USD": ("ETH/USD", "crypto"),
}
# =====================================================



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
        # la chiave nel tuo file è "universe"
        uni = js.get("universe", [])
        if not isinstance(uni, list):
            raise ValueError("chiave 'universe' non è una lista")
        # normalizza ma senza cambiare il formato del label (già UPPER con slash)
        uni = [str(x).strip() for x in uni if str(x).strip()]
        return uni
    except Exception as e:
        log.error(f"Errore lettura roster: {e!r}")
        return []

def build_universe() -> list[str]:
    """
    Costruisce l'universo finale in base a:
    - RETRAIN_UNIVERSE_SOURCE: ROSTER | FIXED | ALL
    - ODIN_INCLUDE_METALS: se True, aggiunge GOLD & SILVER
    - ODIN_RETRAIN_ONLY: se presente, filtra a subset CSV (override)
    """
    src = (RETRAIN_UNIVERSE_SOURCE or "ROSTER").strip().upper()
    out: list[str] = []

    if src == "ROSTER":
        base = load_roster_universe()
        out.extend(base)

    elif src == "FIXED":
        # set fisso: qui uso le 16 del roster come base fissa
        out = [
            "AUD/CAD","CHF/JPY","EUR/AUD","CAD/JPY","GBP/JPY","AUD/JPY","USD/JPY","EUR/JPY",
            "NZD/JPY","NZD/USD","NZD/CAD","USD/SEK","EUR/NZD","AUD/CHF","GBP/NZD","USD/NOK"
        ]

    elif src == "ALL":
        # tutte le label che conosciamo (escludo metalli qui, li aggiungo sotto se flaggati)
        out = [k for k in ASSET_PORTFOLIO.keys() if k not in {"GOLD","SILVER"}]

    else:
        log.warning(f"RETRAIN_UNIVERSE_SOURCE sconosciuto: {src}. Fallback=ROSTER.")
        out = load_roster_universe()

    # Aggiungi metalli se richiesto
    if ODIN_INCLUDE_METALS:
        for lab in ("GOLD", "SILVER"):
            if lab not in out:
                out.append(lab)

    # Filtra con ONLY (override)
    if ODIN_RETRAIN_ONLY:
        only_set = {x.strip() for x in ODIN_RETRAIN_ONLY.split(",") if x.strip()}
        out = [x for x in out if x in only_set]
        # se ONLY include metalli, assicurati che restino
        for lab in ("GOLD","SILVER"):
            if lab in only_set and lab not in out:
                out.append(lab)

    # Mantieni solo label mappati
    out = [x for x in out if x in ASSET_PORTFOLIO]

    # Ordine stabile: come arriva (niente sort), evita duplicati preservando l’ordine
    seen = set(); ordered = []
    for x in out:
        if x not in seen:
            seen.add(x); ordered.append(x)

    log.info(f"[UNIVERSE] source={src}, roster_path='{ROSTER_PATH}', count={len(ordered)}, labels={ordered}")
    return ordered

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
# >>> PATCH (opzionale, più permissiva)
GRID_WEEKLY += [
    {"adx_min":25,"r2_min":0.25,"anti_chase_mult":0.8,"sl_atr_mult":2.0,"rr":1.4,"time_stop":45},
]
# <<< PATCH

GRID_BBMR = [
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":35,"rsi_sell":65,"atr_stop_mult":1.5,"tp_to_ma":True,"time_stop":20},
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":30,"rsi_sell":60,"atr_stop_mult":1.2,"tp_to_ma":True,"time_stop":20},
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":40,"rsi_sell":70,"atr_stop_mult":1.8,"tp_to_ma":True,"time_stop":20},
]
# >>> PATCH (opzionale, mean reversion più “morbida”)
GRID_BBMR += [
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":30,"rsi_sell":70,"atr_stop_mult":1.2,"tp_to_ma":True,"time_stop":30},
]
# <<< PATCH


# ---------- Split IS/OOS ----------
def split_is_oos(df: pd.DataFrame, is_months: int, oos_months: int):
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame deve avere un DatetimeIndex")

    end = df.index[-1]
    # fai OOS di lunghezza esatta 'oos_months'
    oos_start = end - pd.DateOffset(months=oos_months) + pd.Timedelta(days=1)
    is_start  = oos_start - pd.DateOffset(months=is_months)

    mask_is  = (df.index >= is_start) & (df.index < oos_start)
    mask_oos = (df.index >= oos_start)
    return df.loc[mask_is].copy(), df.loc[mask_oos].copy()


# ---------- FIT selezione + validazione ----------
def choose_on_is_then_validate(label: str, df: pd.DataFrame):
    # Split
    df_is, df_oos = split_is_oos(df, IS_MONTHS, OOS_MONTHS)

    # Usa un nome che non possa collidere: 'result' (non 'out')
    result = {"WEEKLY": None, "BB_MR": None}

    # WEEKLY
    best = None
    for p in GRID_WEEKLY:
        Rs = sim_weekly(df_is, p)
        m  = metrics_from_R(Rs)
        if m["trades"] < 10:
            continue
        score = m["pf"] * max(0.1, m["exp"] + 1.0)
        if (best is None) or (score > best["score"]):
            best = {"params": p, "is": m, "score": score}
    if best:
        Rs_oos = sim_weekly(df_oos, best["params"])
        m2 = metrics_from_R(Rs_oos)
        result["WEEKLY"] = {"params": best["params"], "IS": best["is"], "OOS": m2}

    # BB_MR
    best = None
    for p in GRID_BBMR:
        Rs = sim_bbmr(df_is, p)
        m  = metrics_from_R(Rs)
        if m["trades"] < 8:
            continue
        score = m["pf"] * max(0.1, m["exp"] + 1.0)
        if (best is None) or (score > best["score"]):
            best = {"params": p, "is": m, "score": score}
    if best:
        Rs_oos = sim_bbmr(df_oos, best["params"])
        m2 = metrics_from_R(Rs_oos)
        result["BB_MR"] = {"params": best["params"], "IS": best["is"], "OOS": m2}

    return result

    # WEEKLY
    best = None
    for p in GRID_WEEKLY:
        Rs = sim_weekly(df_is, p)
        m  = metrics_from_R(Rs)
        if m["trades"] < 10:  # filtro minimo IS
            continue
        score = m["pf"] * max(0.1, m["exp"] + 1.0)
        if (best is None) or (score > best["score"]):
            best = {"params": p, "is": m, "score": score}
    if best:
        Rs_oos = sim_weekly(df_oos, best["params"])
        m2 = metrics_from_R(Rs_oos)
        result["WEEKLY"] = {"params": best["params"], "IS": best["is"], "OOS": m2}

    # BB_MR
    best = None
    for p in GRID_BBMR:
        Rs = sim_bbmr(df_is, p)
        m  = metrics_from_R(Rs)
        if m["trades"] < 8:
            continue
        score = m["pf"] * max(0.1, m["exp"] + 1.0)
        if (best is None) or (score > best["score"]):
            best = {"params": p, "is": m, "score": score}
    if best:
        Rs_oos = sim_bbmr(df_oos, best["params"])
        m2 = metrics_from_R(Rs_oos)
        result["BB_MR"] = {"params": best["params"], "IS": best["is"], "OOS": m2}
    return result


# >>> PATCH: gate OOS centrale (usato nel main)
def passes_oos(m: dict) -> bool:
    """Applica i vincoli OOS globali al dict metrics {'OOS': {...}}."""
    if not m or "OOS" not in m:
        return False
    o = m["OOS"]
    tr_ok = o.get("trades", 0) >= MIN_TRADES_OOS
    pf    = o.get("pf", 0.0)
    pf_ok = math.isfinite(pf) and (pf >= MIN_PF_OOS)
    exp_ok = o.get("exp", 0.0) > 0.0
    dd_ok  = o.get("maxdd", 1e9) <= MAX_DD_OOS_R
    return tr_ok and pf_ok and exp_ok and dd_ok
# <<< PATCH


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
    pano_df = pd.DataFrame([pano])
    pano_df.to_csv(os.path.join(pano_dir, f"{label.replace('/','_')}.csv"), index=False)

# >>> PATCH: utility per spiegare perché una strategia fallisce i vincoli OOS
def why_failed(m: dict) -> list[str]:
    if not m or "OOS" not in m:
        return ["no_metrics"]
    o = m["OOS"]
    reasons = []
    # trades
    tr = o.get("trades", 0)
    if tr < MIN_TRADES_OOS:
        reasons.append(f"trades<{MIN_TRADES_OOS} ({tr})")
    # pf
    pf = o.get("pf", float("nan"))
    if (not math.isfinite(pf)) or pf < MIN_PF_OOS:
        shown = 0.0 if not math.isfinite(pf) else pf
        reasons.append(f"pf<{MIN_PF_OOS:.2f} ({shown:.2f})")
    # expectancy
    expv = o.get("exp", 0.0)
    if expv <= 0.0:
        reasons.append(f"exp<=0 ({expv:.3f})")
    # max drawdown in R
    dd = o.get("maxdd", float("inf"))
    if dd > MAX_DD_OOS_R:
        reasons.append(f"dd>{MAX_DD_OOS_R:.1f} ({dd:.1f})")
    return reasons or ["unknown"]

def reasons_for(res: dict) -> str:
    parts = []
    if res.get("WEEKLY"):
        r = why_failed(res["WEEKLY"])
        if r and r not in (["no_metrics"], ["unknown"]):
            parts.append("WEEKLY:" + "|".join(r))
    if res.get("BB_MR"):
        r = why_failed(res["BB_MR"])
        if r and r not in (["no_metrics"], ["unknown"]):
            parts.append("BB_MR:" + "|".join(r))
    return "; ".join(parts) if parts else "—"
# <<< PATCH

# ---------- MAIN ----------
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("--- ODIN Retrain ---")
    tg("ODIN: retrain avviato…")

    # 1) costruisci universo
    universe = build_universe()
    if not universe:
        log.error("Universe vuoto. Controlla RETRAIN_UNIVERSE_SOURCE/ROSTER/ONLY.")
        return

    results = {}
    passes = 0
    total = 0
    # >>> PATCH: raccoglitore diagnostica per dump finale
    all_metrics = {}
    # <<< PATCH
    for label in universe:
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
            
            try:
                res = choose_on_is_then_validate(label, df_cut)
            except Exception:
                import traceback
                log.error("choose_on_is_then_validate() crash:\n" + traceback.format_exc())
                raise
            results[label] = res


            # decisione OOS (se almeno una strategia passa)
            val_ok = False

            reasons = []
            if res.get("WEEKLY"):
                if passes_oos(res["WEEKLY"]):
                    val_ok = True
                else:
                    reasons.append("WEEKLY:" + "|".join(why_failed(res["WEEKLY"])))
            if res.get("BB_MR"):
                if passes_oos(res["BB_MR"]):
                    val_ok = True
                else:
                    reasons.append("BB_MR:" + "|".join(why_failed(res["BB_MR"])))

            if val_ok:
                passes += 1
                log.info(f"[{label}] OOS PASSED")
            else:
                log.info(f"[{label}] OOS FAILED — {'; '.join(reasons) if reasons else 'no reasons'}")

            time.sleep(0.3)
        except Exception as e:
            import traceback
            log.error(f"[{label}] errore retrain: {e!r}\n" + traceback.format_exc())


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

    # 3) report Telegram
    def fmt(m):
        if not m: return "—"
        o = m["OOS"]
        pf = o['pf'] if math.isfinite(o['pf']) else 999.0
        return f"tr={o['trades']}, PF={pf:.2f}, Exp={o['exp']:.3f}, DD={o['maxdd']:.1f}"

    # --- dump diagnostico su file (JSON + testo) ---
    try:
        dump_json = {}
        for label, res in results.items():
            dump_json[label] = {
                "status": cur["STATUS"].get(label, "ACTIVE"),
                "WEEKLY": res.get("WEEKLY"),
                "BB_MR": res.get("BB_MR")
            }

        os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

        out_json = os.getenv("AUTO_GATE_RETRAIN_JSON", os.path.join(PARAMS_DIR, "retrain_metrics.json"))
        out_txt  = os.getenv("AUTO_GATE_RETRAIN_TEXT", os.path.join(BASE_DIR, "logs", "last_retrain.txt"))

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(dump_json, f, indent=2, ensure_ascii=False)

        def _fmt_one(lbl, res):
            def _fmt_m(m):
                if not m: return "—"
                o = m["OOS"]; pf = o['pf'] if math.isfinite(o['pf']) else 999.0
                return f"OOS: tr={o['trades']}, PF={pf:.2f}, Exp={o['exp']:.3f}, DD={o['maxdd']:.1f}"
            def _reasons(m):
                if not m: return "—"
                bad = []
                o = m["OOS"]
                if o["trades"] < MIN_TRADES_OOS: bad.append(f"tr<{MIN_TRADES_OOS}")
                if (o["pf"] < MIN_PF_OOS) and math.isfinite(o["pf"]): bad.append(f"pf<{MIN_PF_OOS}")
                if o["exp"] <= 0.0: bad.append("exp<=0")
                if o["maxdd"] > MAX_DD_OOS_R: bad.append(f"dd>{MAX_DD_OOS_R}")
                return ",".join(bad) if bad else "—"
            lines = []
            lines.append(f"{lbl} [{cur['STATUS'].get(lbl,'ACTIVE')}]")
            lines.append("  WEEKLY: " + _fmt_m(res.get("WEEKLY")))
            lines.append("  BB_MR : " + _fmt_m(res.get("BB_MR")))
            lines.append("  WHY   : " + "; ".join(filter(None, [
                ("WEEKLY:"+_reasons(res.get("WEEKLY"))) if res.get("WEEKLY") else "",
                ("BB_MR:"+_reasons(res.get("BB_MR"))) if res.get("BB_MR") else ""
            ])))
            return "\n".join(lines)

        with open(out_txt, "w", encoding="utf-8") as f:
            for lbl, res in results.items():
                f.write(_fmt_one(lbl, res) + "\n")

        log.info(f"[DUMP] retrain metrics -> {out_json} ; summary -> {out_txt}")
    except Exception as e:
        log.warning(f"[DUMP] fallito: {e!r}")

    lines = []
    for label, res in results.items():
        st = cur["STATUS"].get(label, "ACTIVE")
        lines.append(
            f"{label}: {st} | "
            f"WEEKLY[{fmt(res.get('WEEKLY'))}] | "
            f"BB_MR[{fmt(res.get('BB_MR'))}] | "
            f"why: {reasons_for(res)}"
        )

    if lines:
        # Telegram: spezza in chunk per evitare limiti
        chunk = []
        acc = 0
        for ln in lines:
            if acc + len(ln) + 1 > 3500:
                tg("ODIN retrain (IS/OOS):\n" + "\n".join(chunk))
                chunk = [ln]
                acc = len(ln) + 1
            else:
                chunk.append(ln)
                acc += len(ln) + 1
        if chunk:
            tg("ODIN retrain (IS/OOS):\n" + "\n".join(chunk))

    log.info(f"Retrain completato. Assets processati: {total}, passed: {passes}, failed: {total-passes}")


if __name__ == "__main__":
    main()
