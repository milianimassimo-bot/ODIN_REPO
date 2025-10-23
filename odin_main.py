# ==============================================================================
# ODIN Project - Module 4: The Decision Room
# Version: 2.4.2 (Patched for Debugging and Indentation)
# ==============================================================================

import os
import sys
import json
import time
import math
import uuid
import hashlib
import logging
import random
import requests
import traceback
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Literal, Tuple
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from functools import lru_cache

import pandas as pd
import numpy as np
from numpy.linalg import lstsq
from dotenv import load_dotenv

from odin_ml_logger import log_signal # ML logger: segnali

# === Root & params paths ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(ROOT_DIR, "params")
MACRO_OVERRIDES_FILE = os.path.join(PARAMS_DIR, "macro_overrides.json")
MACRO_CONTEXT_FILE = os.path.join(PARAMS_DIR, "macro_context.json")
NEWS_BLACKOUT_FILE = os.path.join(PARAMS_DIR, "news_blackout.json")

# [UNIVERSE] file prodotto da Universe Manager
UNIVERSE_FILE = os.getenv("ODIN_UNIVERSE_JSON", os.path.join(PARAMS_DIR, "universe.json"))

# --- .env
load_dotenv()

from typing import List

# mapping categorie -> set di strategie Sentinel
TREND_STRATS = {
    "EMA_Trend_D1", "Keltner_Trend_D1", "VolTarget_Trend_D1",
    "Donchian_BO_D1", "EMA_Pullback_H4", "Regime_Switcher_D1", "SuperTrend"
}
MR_STRATS = {
    "BB_MR_D1", "EMA_Pullback_H4", "Keltner_MR_H4", "RSI_MR",
    "Regime_Switcher_D1"  # <-- AGGIUNGI QUESTA RIGA
}
BO_STRATS = {"Donchian_BO_D1", "BB_Squeeze_BO_D1"}  # se/quando userai i breakout

# ... (il resto del tuo codice rimane IDENTICO)

# dev’essere valorizzato in main():  _STRAT_ROSTER = _load_roster_safe()
_STRAT_ROSTER = {}
def is_strategy_active_for(asset_label: str, strat_name: str) -> bool:
    active = set(_STRAT_ROSTER.get("asset_strats", {}).get(asset_label, []))
    return strat_name in active

def _greenlight(asset_label: str, category: str) -> bool:
    """True se l’asset ha almeno UNA strategia attiva per la categoria."""
    # se non ho roster, non concedo nulla (preferisco prudenza)
    if not _STRAT_ROSTER:
        return False
    active = set(_STRAT_ROSTER.get("asset_strats", {}).get(asset_label, []))
    if not active:
        return False
    if category == "WEEKLY":
        return bool(active & TREND_STRATS)
    if category == "BB_MR":
        return bool(active & MR_STRATS)
    if category == "BREAKOUT":
        return bool(active & BO_STRATS)
    return False


# --- NewsGate (blackout) runtime toggles ---
NEWS_BLOCK_ENABLE = (os.getenv("NEWS_BLOCK_ENABLE", "1") == "1")
NEWS_BLOCK_PRE_MIN = int(os.getenv("NEWS_BLOCK_PRE_MIN", "60"))
NEWS_BLOCK_POST_MIN = int(os.getenv("NEWS_BLOCK_POST_MIN", "30"))

def _load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

# --- Universe loader (robusto a più formati) ----------------------------------
def load_universe_allowlist(path: str) -> dict[str, set]:
    """
    Ritorna una mappa: { "EUR/USD": {"WEEKLY","BB_MR"}, ... }
    Accetta formati equivalenti:
      - {"assets":[{"asset":"EUR/USD","strategies":["WEEKLY","BB_MR"]}, ...]}
      - {"selected":[{"asset":"EUR/USD","modes":["WEEKLY","BB_MR"]}, ...]}
      - {"items":[{"asset":"EUR/USD","strategies":["WEEKLY","BB_MR"]}, ...]} <-- compatibile col tuo
      - {"allow":{"EUR/USD":["WEEKLY","BB_MR"], ...}}
    """
    try:
        data = _load_json(path, default={})
        allow: dict[str, set] = {}
        if not data:
            return {}

        # --- direct dict form ---
        if isinstance(data.get("allow"), dict):
            for a, lst in data["allow"].items():
                s = set(str(x).upper() for x in (lst or []))
                if s:
                    allow[a.upper()] = s

        # --- generic list forms ---
        for key in ("assets", "items", "selected"):
            for row in (data.get(key) or []):
                a = str(row.get("asset", "")).strip().upper()
                if not a:
                    continue
                st = row.get("strategies") or row.get("modes") or []
                s = set(str(x).upper() for x in st if str(x).strip())
                if s:
                    allow.setdefault(a, set()).update(s)

        return allow
    except Exception as e:
        log.warning(f"[Universe] load error {e!r} — uso lista vuota.")
        return {}

# ------------------------------------------------------------------------------

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _clamp(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo

# --- cast robusti (PATCH) -----------------------------------------------------
def _to_float(x, default=0.0):
    """Cast robusto: se x è None o non numerico → default."""
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _to_int(x, default=0):
    try:
        if x is None:
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)

def _to_bool(x, default=False):
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)
# ------------------------------------------------------------------------------

# --- MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    mt5 = None
    MT5_AVAILABLE = False

# =============== CONFIG ===============
REGIME_REPORT_PATH = os.getenv("ODIN_REGIME_JSON", os.path.join(ROOT_DIR, "odin_regime_report_d1.json"))
ENV = os.getenv("ODIN_ENV", "DEV")
STRATS_JSON_PATH = os.getenv("ODIN_STRATS_JSON", os.path.join(ROOT_DIR, "strategies.json"))
VERSION = "2.4.2"

TD_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# FORZA IL DEBUG IGNORANDO IL FILE .env
# DEBUG_SIGNALS = (os.getenv("ODIN_DEBUG_SIGNALS", "0") == "1")
DEBUG_SIGNALS = True

def dbg(msg: str):
    if DEBUG_SIGNALS:
        log.info(f"[SIG-DBG] {msg}")

# --- Exploration (Stimolo) ENV vars ---
EXPL_ENABLE = (os.getenv("ODIN_EXPLORATION_ENABLE", "1") == "1")
EXPL_RATE = float(os.getenv("ODIN_EXPLORATION_RATE", "0.12"))
EXPL_MAX_PER_DAY = int(os.getenv("ODIN_EXPLORATION_MAX_PER_DAY", "2"))
EXPL_CD_HOURS = int(os.getenv("ODIN_EXPLORATION_COOLDOWN_H", "6"))
EXPL_RISK_MULT = float(os.getenv("ODIN_EXPLORATION_RISK_MULT", "0.5"))
EXPL_ALLOW_WK = (os.getenv("ODIN_EXPLORATION_ALLOW_WEEKLY", "1") == "1")
EXPL_ALLOW_BBMR = (os.getenv("ODIN_EXPLORATION_ALLOW_BBMR", "1") == "1")

EXPL_STATE_FILE = os.path.join(PARAMS_DIR, "exploration_state.json")

# --- NO-SIGNAL reason collector ---
_LAST_NOSIGNAL_REASON: dict[str, str] = {}
def dbg_reason(asset: str, msg: str):
    _LAST_NOSIGNAL_REASON[asset] = msg
    if DEBUG_SIGNALS:
        log.info(f"[SIG-DBG] {asset}: {msg}")

MAX_GLOBAL_RISK_PCT = _to_float(os.getenv("ODIN_RISK_GLOBAL", "3.0"))
MAX_BUCKET_RISK_PCT = _to_float(os.getenv("ODIN_RISK_BUCKET", "2.0"))
MAX_ASSET_RISK_PCT = _to_float(os.getenv("ODIN_RISK_ASSET", "1.0"))
MAX_TRADE_RISK_PCT = _to_float(os.getenv("ODIN_RISK_TRADE", "1.0"))
MIN_LOT_RISK_TOLERANCE_FACTOR = _to_float(os.getenv("ODIN_MIN_LOT_RISK_TOL", "1.5"))

MIN_EQUITY_TO_TRADE = _to_float(os.getenv("ODIN_MIN_EQUITY", "250.0"))
MARGIN_BUFFER_FACTOR = _to_float(os.getenv("ODIN_MARGIN_BUFFER", "0.20"))

ACCOUNT_EQUITY_FALLBACK = _to_float(os.getenv("ODIN_EQUITY", "10000.0"))
ACCOUNT_CCY = os.getenv("ODIN_ACCOUNT_CCY", "USD").upper()

ATR_METHOD = os.getenv("ODIN_ATR_METHOD", "EMA").upper()
TD_BASE_URL = "https://api.twelvedata.com/time_series"

BROKER_SYMBOL_MAP_DEFAULT = {
    "SP500": "Usa500",
    "GOLD": "GOLD",
    "SILVER": "SILVER",
    "OIL_WTI": "WTI",
}
BROKER_SYMBOL_MAP = {
    "SP500": os.getenv("ODIN_BROKER_SP500", BROKER_SYMBOL_MAP_DEFAULT["SP500"]),
    "GOLD": os.getenv("ODIN_BROKER_GOLD", BROKER_SYMBOL_MAP_DEFAULT["GOLD"]),
    "SILVER": os.getenv("ODIN_BROKER_SILVER", BROKER_SYMBOL_MAP_DEFAULT["SILVER"]),
    "OIL_WTI": os.getenv("ODIN_BROKER_OILWTI", BROKER_SYMBOL_MAP_DEFAULT["OIL_WTI"]),
}
MAGIC_NUMBER = 231120


# =============== LOGGING ===============
def setup_logger(name="odin"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
    fh = RotatingFileHandler(os.path.join(ROOT_DIR, "logs", f"{name}.log"),
                                 maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

log = setup_logger("odin_main")

# ======== ROSTER: percorso e loader (NUOVO BLOCCO) ========
ROSTER_PATH = os.getenv(
    "ODIN_STRATEGY_ROSTER",
    os.path.join("params", "strategy_roster.json")
)

def _load_roster_safe(path: str = ROSTER_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # log diagnostico sintetico
        log.info(f"[Sentinel] Roster caricato correttamente. "
                 f"{len(data.get('ranking_global', []))} strategie analizzate.")
        return data
    except Exception as e:
        log.warning(f"[Sentinel] Roster non disponibile: {e!r}. Uso fallback legacy.")
        return {}  # fallback: comportamento legacy



# =============== PARAMS LOADER ===============
BASE_DIR = ROOT_DIR
DEFAULT_PARAMS_PATH = os.path.join(BASE_DIR, "params", "current.json")
PARAMS_PATH = os.getenv("ODIN_PARAMS_FILE", DEFAULT_PARAMS_PATH)

def load_params(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"[PARAMS] Caricati da: {path}")
        return data
    except FileNotFoundError:
        log.warning(f"[PARAMS] File non trovato: {path} — uso default hard-coded.")
        return {}
    except Exception as e:
        log.error(f"[PARAMS] Errore caricamento ({path}): {e!r} — uso default hard-coded.")
        return {}

PARAMS = load_params(PARAMS_PATH)

def get_params(strategy: str, asset: str) -> dict:
    block = PARAMS.get(strategy, {}) or {}
    default = block.get("default", {}) or {}
    per_asset = (block.get("per_asset", {}) or {}).get(asset, {}) or {}
    merged = dict(default)
    merged.update(per_asset)
    return merged

def is_asset_disabled(asset: str) -> bool:
    return asset in set(PARAMS.get("disabled_assets", []) or [])

# === MacroGate: carica overrides + context, applica su PARAMS, calcola guardrails
def apply_macro_overrides_to_params(cur_params: dict, overrides: dict) -> dict:
    cur = dict(cur_params) if isinstance(cur_params, dict) else {}
    ov = overrides or {}
    mult = (ov.get("global", {}) or {}).get("risk_caps_multiplier")
    if isinstance(mult, (int, float)):
        g = cur.setdefault("global", {}).setdefault("risk_caps", {})
        for k in ("global", "bucket", "asset", "trade"):
            if k in g:
                g[k] = round(_clamp(g[k] * mult, 0.0005, 0.20), 6)
    aset = (ov.get("asset_overrides") or {})
    cur.setdefault("asset_overrides", {})
    for a, spec in aset.items():
        ane = spec.get("allow_new_entries")
        if isinstance(ane, bool):
            cur["asset_overrides"].setdefault(a, {})
            cur["asset_overrides"][a]["allow_new_entries"] = ane
    return cur

def compute_guardrails_from_macro(ctx: dict) -> dict:
    guard = {}
    try:
        usd = ((ctx or {}).get("usd_bias") or "").lower()
        comm = ((ctx or {}).get("commodities_bias") or "").lower()
        if comm in ("strong", "up", "bullish") and usd in ("weak", "neutral", "bearish", ""):
            guard["USD/CAD"] = {"block_long": True}
    except Exception:
        pass
    return guard

macro_over = _load_json(MACRO_OVERRIDES_FILE, default={})
macro_ctx = _load_json(MACRO_CONTEXT_FILE, default={})
PARAMS = apply_macro_overrides_to_params(PARAMS, macro_over)
_MACRO_GUARDRAILS = compute_guardrails_from_macro(macro_ctx)
try:
    applied_assets = ", ".join(sorted((macro_over.get("asset_overrides") or {}).keys())) or "—"
    mult = (macro_over.get("global") or {}).get("risk_caps_multiplier", "—")
    print(f"[MacroGate] risk_caps_multiplier={mult} | asset_overrides={applied_assets} | guardrails={_MACRO_GUARDRAILS}")
except Exception:
    pass

# === NewsGate livello 1 (json: global/assets/strategies con until_utc) ========
def _utcnow():
    return datetime.now(timezone.utc)

def _parse_utc(s: Optional[str]) -> Optional[datetime]:
    try:
        if not s:
            return None
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

NEWS_BLACKOUT = _load_json(NEWS_BLACKOUT_FILE, default={})
def is_news_blackout(asset: str, strategy: str) -> tuple[bool, str]:
    try:
        now = _utcnow()
        g_flag = bool(NEWS_BLACKOUT.get("global", False))
        g_until = _parse_utc(NEWS_BLACKOUT.get("until_utc"))
        if g_flag and (g_until is None or now < g_until):
            return True, f"NewsGate: blackout globale attivo fino a {g_until} (UTC)" if g_until else "NewsGate: blackout globale attivo"

        a_spec = (NEWS_BLACKOUT.get("assets", {}) or {}).get(asset, {})
        a_until = _parse_utc(a_spec.get("until_utc"))
        if a_until and now < a_until:
            return True, f"NewsGate: blackout {asset} fino a {a_until} (UTC)"

        s_spec = (NEWS_BLACKOUT.get("strategies", {}) or {}).get(strategy, {})
        s_until = _parse_utc(s_spec.get("until_utc"))
        if s_until and now < s_until:
            return True, f"NewsGate: blackout {strategy} fino a {s_until} (UTC)"

        return False, ""
    except Exception:
        return False, ""

# === NewsGate livello finestre/currency (da news_blackout.json generato) ======
def _load_news_blackout(path: str):
    try:
        with open(path,"r",encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    pre = timedelta(minutes=NEWS_BLOCK_PRE_MIN)
    post= timedelta(minutes=NEWS_BLOCK_POST_MIN)
    out = {}
    for ev in (data.get("windows") or []):
        cur = str(ev.get("currency","")).upper().strip()
        t = str(ev.get("time","")).strip()
        ttl = str(ev.get("event","")).strip()
        if not cur or not t:
            continue
        try:
            if t.endswith("Z"):
                when = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            else:
                when = datetime.fromisoformat(t)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        start = when - pre
        end = when + post
        out.setdefault(cur, []).append( (start, end, ttl) )
    return out

def _asset_currencies(asset: str):
    a = (asset or "").upper().replace(" ","")
    if "/" in a and len(a)==7:
        base,quote = a.split("/")
        return [base, quote]
    if a in ("XAUUSD","XAU/USD","GOLD"): return ["USD"]
    if a in ("XAGUSD","XAG/USD","SILVER"): return ["USD"]
    if a in ("CL","OIL_WTI","WTI","USOIL"):return ["USD"]
    if a in ("SP500","SPX","US500"): return ["USD"]
    if "/" in a:
        parts=a.split("/")
        return [p[:3] for p in parts if p]
    return []

def _news_blocked(asset: str, now_utc: datetime, blackout_map: dict):
    if not blackout_map:
        return (False, "")
    cur_list = _asset_currencies(asset)
    for ccy in cur_list:
        wins = blackout_map.get(ccy, [])
        for (start,end,title) in wins:
            if start <= now_utc <= end:
                return (True, f"{ccy} news '{title}'")
    return (False, "")

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def retry_backoff(fn, tries=4, base=0.4, cap=5.0):
    last_exc = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(min(cap, base * (2 ** i)) + random.uniform(0, 0.2))
    raise last_exc

def bbands(series: pd.Series, length: int = 20, std: float = 2.0):
    ma = series.rolling(length).mean()
    sd = series.rolling(length).std(ddof=0)
    upper = ma + std * sd
    lower = ma - std * sd
    return lower, ma, upper

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = up / down.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))

def atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return (atr / close) * 100.0

def rolling_r2(series: pd.Series, window: int = 50) -> pd.Series:
    y = series.values
    r2 = np.full_like(y, fill_value=np.nan, dtype=float)
    x_full = np.arange(len(y))
    for i in range(window - 1, len(y)):
        yw = y[i - window + 1:i + 1]
        xw = x_full[i - window + 1:i + 1]
        X = np.vstack([xw, np.ones_like(xw)]).T
        beta, _, _, _ = lstsq(X, yw, rcond=None)
        y_pred = X @ beta
        ss_res = float(np.sum((yw - y_pred) ** 2))
        ss_tot = float(np.sum((yw - np.mean(yw)) ** 2))
        r2[i] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return pd.Series(r2, index=series.index)

# =============== SYMBOL CAPS & TICKS ===============
@dataclass
class MT5SymbolCaps:
    symbol: str
    digits: int
    point: float
    volume_min: float
    volume_step: float
    volume_max: float
    trade_tick_value: float
    trade_contract_size: float
    stops_level_points: int

def resolve_broker_symbol(preferred: str) -> Optional[str]:
    if not MT5_AVAILABLE:
        return preferred
    candidates = [preferred]
    root = preferred.upper()
    variants = [preferred.upper(), preferred.lower(),
                preferred + ".i", preferred + "m", preferred + "-Var",
                preferred.replace("/", ""), preferred.replace("/", "") + ".i"]
    for v in variants:
        if v not in candidates:
            candidates.append(v)
    for sym in candidates:
        si = mt5.symbol_info(sym)
        if si is None:
            continue
        if not si.visible:
            mt5.symbol_select(sym, True)
            time.sleep(0.2)
            si = mt5.symbol_info(sym)
        if si and si.visible:
            return sym
    try:
        all_syms = mt5.symbols_get()
        if all_syms:
            root_short = root.replace("/", "")
            for s in all_syms:
                name = s.name
                if root_short in name.upper():
                    if not s.visible:
                        mt5.symbol_select(name, True)
                        time.sleep(0.2)
                        si = mt5.symbol_info(name)
                        if si and si.visible:
                            return name
                    else:
                        return name
    except Exception:
        pass
    return None

def get_mt5_symbol_caps(broker_symbol: str) -> Optional[MT5SymbolCaps]:
    if not MT5_AVAILABLE:
        return None
    si = mt5.symbol_info(broker_symbol)
    if si is None:
        log.error(f"MT5: simbolo inesistente/non visibile: {broker_symbol}")
        return None
    if not si.visible:
        log.info(f"MT5: simbolo {broker_symbol} non visibile, provo a selezionarlo...")
        mt5.symbol_select(broker_symbol, True)
        time.sleep(0.2)
        si = mt5.symbol_info(broker_symbol)
        if not si or not si.visible:
            log.error(f"MT5: impossibile rendere visibile {broker_symbol}")
            return None
    caps = MT5SymbolCaps(
        symbol=broker_symbol,
        digits=int(getattr(si, "digits", 0)),
        point=float(getattr(si, "point", 0.0)),
        volume_min=float(getattr(si, "volume_min", 0.01)),
        volume_step=float(getattr(si, "volume_step", 0.01)),
        volume_max=float(getattr(si, "volume_max", 100.0)),
        trade_tick_value=float(getattr(si, "trade_tick_value", 0.0)),
        trade_contract_size=float(getattr(si, "trade_contract_size", 0.0)),
        stops_level_points=int(getattr(si, "stops_level", 0) or 0)
    )
    log.info(f"MT5 caps {broker_symbol}: minVol={caps.volume_min}, step={caps.volume_step}, tickVal={caps.trade_tick_value}, digits={caps.digits}")
    return caps

def round_volume_to_caps(vol: float, caps: MT5SymbolCaps) -> float:
    if vol <= 0:
        return 0.0
    steps = max(0, int((vol - caps.volume_min) / caps.volume_step + 1e-9))
    vol_norm = caps.volume_min + steps * caps.volume_step
    vol_norm = max(caps.volume_min, min(vol_norm, caps.volume_max))
    return float(f"{vol_norm:.3f}")

def round_price_to_tick(price: float, caps: MT5SymbolCaps) -> float:
    if caps.point <= 0:
        return price
    ticks = round(price / caps.point)
    p = ticks * caps.point
    return float(("{:0." + str(caps.digits) + "f}").format(p))

def normalize_prices_for_mt5(entry: float, sl: float, tp: float, caps: MT5SymbolCaps) -> Tuple[float, float, float]:
    return (round_price_to_tick(entry, caps),
            round_price_to_tick(sl, caps),
            round_price_to_tick(tp, caps))

# =============== MT5 BRIDGE ===============
class MT5Bridge:
    def __init__(self):
        if not MT5_AVAILABLE or not mt5.initialize():
            log.error(f"Connessione a MetaTrader 5 fallita: {mt5.last_error() if MT5_AVAILABLE else 'Libreria non trovata'}")
            self.connected = False
        else:
            log.info("Connessione a MT5 stabilita.")
            self.connected = True

    def shutdown(self):
        if self.connected:
            mt5.shutdown()
            log.info("Connessione a MT5 chiusa.")

    def get_account_info(self) -> Optional[dict]:
        if not self.connected:
            return None
        info = mt5.account_info()
        if info is None:
            log.error(f"account_info() failed: {mt5.last_error()}")
            return None
        return {"equity": float(info.equity), "balance": float(info.balance),
                "free_margin": float(info.margin_free), "currency": info.currency}

    def execute_trade(self, plan: 'ActionPlan', caps: 'MT5SymbolCaps') -> Optional[int]:
        if not self.connected:
            log.error("MT5Bridge: non connesso.")
            return None
        s = plan.signal
        broker_symbol = BROKER_SYMBOL_MAP.get(s.asset, s.asset)
        if not mt5.symbol_select(broker_symbol, True):
            log.error(f"MT5: impossibile selezionare {broker_symbol} ({mt5.last_error()})")
            return None
        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            log.error(f"MT5: nessun tick per {broker_symbol}")
            return None
        order_type = mt5.ORDER_TYPE_BUY if s.direction.upper() == "LONG" else mt5.ORDER_TYPE_SELL
        mkt_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        vol = round_volume_to_caps(plan.risk.size_lots, caps)
        if vol < caps.volume_min:
            log.error(f"Volume {vol} sotto il minimo broker {caps.volume_min}")
            return None

        min_dist_price = float(caps.stops_level_points) * float(caps.point) if caps.stops_level_points > 0 else 0.0
        sl_raw, tp_raw = s.sl, s.tp
        if order_type == mt5.ORDER_TYPE_BUY:
            sl_adj = min(sl_raw, mkt_price - min_dist_price) if min_dist_price > 0 else sl_raw
            tp_adj = max(tp_raw, mkt_price + min_dist_price) if min_dist_price > 0 else tp_raw
        else:
            sl_adj = max(sl_raw, mkt_price + min_dist_price) if min_dist_price > 0 else sl_raw
            tp_adj = min(tp_raw, mkt_price - min_dist_price) if min_dist_price > 0 else tp_raw
        price_n, sl_n, tp_n = normalize_prices_for_mt5(mkt_price, sl_adj, tp_adj, caps)

        si = mt5.symbol_info(broker_symbol)
        filling = getattr(si, "filling_mode", mt5.ORDER_FILLING_FOK)
        if filling not in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
            filling = mt5.ORDER_FILLING_FOK

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "volume": vol,
            "type": order_type,
            "price": price_n,
            "sl": sl_n,
            "tp": tp_n,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": f"ODIN_{s.strategy}_{s.asset}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        log.info(f"MT5 order_send: {req}")
        res = mt5.order_send(req)
        if res is None:
            log.error(f"order_send NULL per {broker_symbol}: {mt5.last_error()}")
            return None
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = getattr(res, "order", None) or getattr(res, "deal", None)
            log.info(f"ORDINE ESEGUITO {broker_symbol}: ticket={ticket}, price={price_n}, sl={sl_n}, tp={tp_n}, vol={vol}")
            return int(ticket) if ticket is not None else None
        log.error(f"Esecuzione fallita {broker_symbol}: retcode={res.retcode}, commento={res.comment}")
        return None

# =============== DATACLASS SCHEMAS ===============
@dataclass
class DeterministicSignal:
    strategy: Literal["DAILY", "WEEKLY", "BB_MR"]
    asset: str
    symbol: str
    direction: Literal["LONG", "SHORT"]
    timeframe_entry: Literal["H4", "D1"]
    entry: float
    sl: float
    tp: float
    rr: float
    atr_d1: float
    basis: Dict[str, float]

@dataclass
class AIVeto:
    veto: bool
    score: float
    note: str

@dataclass
class RiskCaps:
    max_global: float = MAX_GLOBAL_RISK_PCT
    max_bucket: float = MAX_BUCKET_RISK_PCT
    max_asset: float = MAX_ASSET_RISK_PCT
    max_trade: float = MAX_TRADE_RISK_PCT

@dataclass
class RiskDecision:
    accept: bool
    reason: Optional[str]
    size_lots: float
    risk_pct: float
    caps_snapshot: Dict[str, float]

@dataclass
class ActionPlan:
    id: str
    asof_utc: str
    regime_sha1: str
    asset: str
    bucket: str
    status: Literal["PLANNED", "COMMITTED", "REJECTED"]
    signal: DeterministicSignal
    ai: AIVeto
    risk: RiskDecision
    notes: List[str]

# =============== REGIME REPORT I/O ===============
def load_regime_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    meta = payload.get("meta", {}); items = payload.get("items")
    if not meta or items is None or "sha1" not in meta:
        raise ValueError("Report malformato o meta.sha1 assente.")
    meta_wo = dict(meta); reported = meta_wo.pop("sha1", None)
    check = {"meta": meta_wo, "items": items}
    expected = hashlib.sha1(json.dumps(check, sort_keys=True).encode("utf-8")).hexdigest()
    if reported != expected:
        raise ValueError("Integrita' report fallita: hash non corrisponde.")
    return payload

# =============== DATA FETCH (TwelveData OHLC) ===============
def fetch_ohlc_twelvedata(symbol: str, interval: str, outputsize: int = 400) -> pd.DataFrame:
    from cache_ohlc import fetch_ohlc_cached
    return fetch_ohlc_cached(symbol, interval, outputsize)
    # Codice di fallback nel caso in cui cache_ohlc non sia disponibile
    if not TD_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY assente.")
    def _call():
        r = requests.get(
            TD_BASE_URL,
            params={"symbol": symbol, "interval": interval, "outputsize": outputsize,
                    "order": "ASC", "timezone": "UTC", "apikey": TD_API_KEY},
            timeout=10
        )
        r.raise_for_status()
        data = r.json(); vals = data.get("values")
        if not vals:
            raise RuntimeError(f"Nessun dato per {symbol} {interval}")
        return vals
    vals = retry_backoff(_call)
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df.iloc[:-1].copy() if len(df) > 1 else df

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr_from_df(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    if ATR_METHOD == "SMA":
        return tr.rolling(period).mean()
    return tr.ewm(alpha=1/period, adjust=False).mean()

# =============== STRATEGY PARAMS ===============
_DEFAULT_STRATS = {"DAILY": {"ema_period": 20, "atr_mult_sl": 2.0, "rr": 2.0}}
def load_strategy_params(path=STRATS_JSON_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        out = dict(_DEFAULT_STRATS)
        for k, v in (cfg or {}).items():
            if k in out and isinstance(v, dict):
                out[k].update(v)
        return out
    except Exception:
        return dict(_DEFAULT_STRATS)
STRATS = load_strategy_params()

# =============== SIGNAL BUILDERS ===============
def build_signal_daily(asset_item: dict) -> DeterministicSignal:
    p = STRATS["DAILY"]
    ema_per = _to_int(p.get("ema_period"), 20)
    atr_mult = _to_float(p.get("atr_mult_sl"), 2.0)
    rr_tgt = _to_float(p.get("rr"), 2.0)

    asset = asset_item["asset"]; symbol = asset_item["vendorSymbol"]
    h4 = fetch_ohlc_twelvedata(symbol, "4h", 300)
    d1 = fetch_ohlc_twelvedata(symbol, "1day", 260)
    if h4 is None or d1 is None or h4.empty or d1.empty:
        dbg_reason(asset, "DAILY: dati H4 o D1 assenti")
        return None
    h4["emaX"] = ema(h4["close"], ema_per)
    d1["atr14"] = atr_from_df(d1, 14)
    emaX = float(h4["emaX"].iloc[-1]); atr_d1 = float(d1["atr14"].iloc[-1])
    d1["ema50"] = ema(d1["close"], 50); d1["ema200"] = ema(d1["close"], 200)
    direction = "LONG" if d1["ema50"].iloc[-1] > d1["ema200"].iloc[-1] else "SHORT"
    entry = emaX
    if direction == "LONG":
        sl = entry - atr_mult * atr_d1; r = entry - sl; tp = entry + rr_tgt * r
    else:
        sl = entry + atr_mult * atr_d1; r = sl - entry; tp = entry - rr_tgt * r
    rr = abs((tp - entry) / (entry - sl)) if (entry - sl) != 0 else 0.0
    basis = {f"ema{ema_per}_h4": round(emaX,5),
             "ema50_d1": round(d1["ema50"].iloc[-1],5),
             "ema200_d1": round(d1["ema200"].iloc[-1],5)}
    return DeterministicSignal("DAILY", asset, symbol, direction, "H4",
                               round(entry,5), round(sl,5), round(tp,5), round(rr,2),
                               round(atr_d1,5), basis)

def build_signal_weekly(item: dict) -> Optional[DeterministicSignal]:
    asset = item["asset"]; symbol = item["vendorSymbol"]
    regime = item.get("regime", "INDEFINITO")
    metrics = item.get("metrics") or {}; ctx = item.get("context") or {}
    if regime != "TRENDING":
        dbg_reason(asset, f"WEEKLY: regime={regime} != TRENDING")
        return None

    p = get_params("WEEKLY", asset)

    adx = _to_float((metrics or {}).get("adx"), 0.0)
    r2 = (metrics or {}).get("r2_60", None)
    if r2 is None:
        dbg_reason(asset, "WEEKLY: r2_60 mancante nel report di regime")
        return None
    r2 = _to_float(r2, 0.0)

    adx_min = _to_float(p.get("adx_min"), 18.0)
    r2_min = _to_float(p.get("r2_min"), 0.25)
    ema_per = _to_int(p.get("ema_period"), 20)
    atrp_min = _to_float(p.get("atr_pct_min"), 0.30)
    atrp_max = _to_float(p.get("atr_pct_max"), 1.60)
    pull_min = _to_float(p.get("pullback_atr_min"), 0.4)
    pull_max = _to_float(p.get("pullback_atr_max"), 1.0)
    max_w_atr = _to_float(p.get("max_weekly_range_atr"), 1.6)
    anti_ch = _to_float(p.get("anti_chase_atr"), 1.3)
    sl_mult = _to_float(p.get("sl_atr_mult"), 2.5)
    rr_tgt = _to_float(p.get("rr"), 1.8)
    need_h4 = _to_bool(p.get("confirm_h4"), True)

    if (adx < adx_min) or (r2 < r2_min):
        dbg_reason(asset, f"WEEKLY: ADX={adx:.2f} o R2={r2:.2f} sotto soglia (min {adx_min}/{r2_min})")
        return None

    d1 = fetch_ohlc_twelvedata(symbol, "1day", 400)
    if d1 is None or d1.empty:
        dbg_reason(asset, "WEEKLY: dati D1 assenti/vuoti da TwelveData")
        return None
    
    d1["ema_fast"] = ema(d1["close"], ema_per); d1["ema50"] = ema(d1["close"], 50)
    d1["ema200"] = ema(d1["close"], 200); d1["atr14"] = atr_from_df(d1, 14)
    ema_fast = float(d1["ema_fast"].iloc[-1]); ema50=float(d1["ema50"].iloc[-1]); ema200=float(d1["ema200"].iloc[-1])
    atr14=float(d1["atr14"].iloc[-1]); close_=float(d1["close"].iloc[-1])

    atrp_now = float(atr_percent(d1, 14).iloc[-1]) if len(d1)>=15 else (atr14/close_)*100.0 if close_ else 0.0
    if not (atrp_min <= atrp_now <= atrp_max):
        dbg_reason(asset, f"WEEKLY: ATR% {atrp_now:.2f} fuori range [{atrp_min},{atrp_max}]")
        return None

    lo, ma, up = bbands(d1["close"], 20, 2.0)
    bbw = float(((up.iloc[-1]-lo.iloc[-1])/ma.iloc[-1]) if pd.notna(ma.iloc[-1]) and ma.iloc[-1]!=0 else 0.0)
    atrp_q75 = ctx.get("atrp_q75", None); bbw_q75 = ctx.get("bbw_q75", None)
    if (atrp_q75 is not None and atrp_now >= _to_float(atrp_q75, 0.0)) or (bbw_q75 is not None and bbw >= _to_float(bbw_q75, 9e9)):
        dbg_reason(asset, "WEEKLY: turbolenza alta vs contesto (ATR%/BBW)")
        return None

    ema_slope = d1["ema_fast"].diff(5).iloc[-1]
    direction = "LONG" if ema50 > ema200 else "SHORT"
    if (direction == "LONG" and not (ema_slope > 0)) or (direction == "SHORT" and not (ema_slope < 0)):
        dbg_reason(asset, f"WEEKLY: mismatch {direction} con slope EMA {float(ema_slope):.6f}")
        return None


    atr_abs = (atrp_now/100.0)*close_ if close_ else atr14
    dist_atr = abs(close_ - ema_fast) / max(atr_abs, 1e-9)

    # --- PATCH 2.A: Pullback Dinamico ---
    lo, hi = 0.35, 1.80
    if ctx.get("atrp_q75") and atrp_now >= ctx["atrp_q75"]:
        lo, hi = 0.30, 2.00
    if not (lo <= dist_atr <= hi):
        dbg_reason(asset, f"WEEKLY: pullback dist_atr={dist_atr:.2f} non in range dinamico [{lo},{hi}]")
        return None
    # --- FINE PATCH ---

    last_week_range = (d1['high'].rolling(5).max().iloc[-1] - d1['low'].rolling(5).min().iloc[-1])

    # --- PATCH 2.B: Range Dinamico ---
    threshold = 2.0
    if ctx.get("bbw_q75") and bbw >= ctx["bbw_q75"]:
        threshold = 2.4
    if (last_week_range / max(atr_abs, 1e-9)) > threshold:
        dbg_reason(asset, f"WEEKLY: range settimanale troppo ampio vs ATR (>{threshold:.1f}x)")
        return None
    # --- FINE PATCH ---


# ...subito dopo il controllo del pullback...
    last_week_range = (d1['high'].rolling(5).max().iloc[-1] - d1['low'].rolling(5).min().iloc[-1])
    
    # --- PATCH 1.B: Range Dinamico ---
    threshold = 2.0 # Baseline più permissiva
    last_bbw = float(((bbands(d1["close"], 20, 2.0)[2].iloc[-1] - bbands(d1["close"], 20, 2.0)[0].iloc[-1]) / bbands(d1["close"], 20, 2.0)[1].iloc[-1]))
    if ctx.get("bbw_q75") and last_bbw >= ctx["bbw_q75"]:
        threshold = 2.4 # Se la volatilità si sta espandendo, concedi un range più ampio
        
    if (last_week_range / max(atr_abs, 1e-9)) > threshold:
        dbg_reason(asset, f"WEEKLY: range settimanale troppo ampio vs ATR ({threshold:.1f}x)")
        return None
    # --- FINE PATCH ---

    if abs(close_ - ema_fast) > anti_ch * atr_abs:

        dbg_reason(asset, f"WEEKLY: anti-chase |close-EMA| > {anti_ch}*ATR")
        return None

    if need_h4:
        h4 = fetch_ohlc_twelvedata(symbol, "4h", 300)
        if h4 is None or h4.empty or len(h4) < 21:
            dbg_reason(asset, "WEEKLY: dati H4 assenti/insufficienti per conferma")
            return None
        ema_h4 = ema(h4['close'], 21)
        h4_ok = (ema_h4.iloc[-1] > ema_h4.iloc[-6]) if direction=="LONG" else (ema_h4.iloc[-1] < ema_h4.iloc[-6])
        if not h4_ok:
            dbg_reason(asset, "WEEKLY: conferma H4 negativa")
            return None

    entry = ema_fast
    if direction == "LONG":
        sl = entry - sl_mult*atr14; r = entry - sl; tp = entry + rr_tgt*r
    else:
        sl = entry + sl_mult*atr14; r = sl - entry; tp = entry - rr_tgt*r
    rr = abs((tp - entry) / (entry - sl)) if (entry - sl)!=0 else 0.0

    basis = {"ema_fast": round(ema_fast,5), "ema50_d1": round(ema50,5), "ema200_d1": round(ema200,5),
             "atr14_d1": round(atr14,5), "adx_d1": round(adx,2), "r2_60": round(r2,3),
             "atrp%": round(atrp_now,3), "bbw": round(bbw,4), "pullback_atr": round(dist_atr,3)}
    return DeterministicSignal("WEEKLY", asset, symbol, direction, "D1",
                               round(entry,5), round(sl,5), round(tp,5), round(rr,2),
                               round(atr14,5), basis)

def build_signal_bbmr_confirmed(item: dict) -> Optional[DeterministicSignal]:
    asset = item["asset"]; symbol = item["vendorSymbol"]
    regime = item.get("regime", "INDEFINITO")
    metrics = item.get("metrics") or {}; ctx = item.get("context") or {}
    if regime not in ("LATERALE","LATERALE (Compressione)"):
        dbg_reason(asset, f"BB_MR: regime={regime} non laterale")
        return None

    p = get_params("BB_MR", asset)

    adx = _to_float((metrics or {}).get("adx"), 0.0)
    adx_max = _to_float(p.get("adx_max"), 18.0)
    r2_m = (metrics or {}).get("r2_60", None)
    if r2_m is not None:
        r2_m = _to_float(r2_m, 0.0)

    d1 = fetch_ohlc_twelvedata(symbol, "1day", 260)
    if d1 is None or len(d1) < 25:
        dbg_reason(asset, "BB_MR: dati D1 insufficienti da TwelveData")
        return None

    if r2_m is None:
        r2_series = rolling_r2(d1["close"], window=50)
        r2_m = float(r2_series.iloc[-1]) if pd.notna(r2_series.iloc[-1]) else 0.0

    if adx >= adx_max:
        dbg_reason(asset, f"BB_MR: ADX={adx:.2f} >= adx_max={adx_max}")
        return None

    close_ = float(d1["close"].iloc[-1]); atr14 = float(atr_from_df(d1, 14).iloc[-1])
    atrp_now = float(atr_percent(d1, 14).iloc[-1]) if len(d1)>=15 else (atr14/close_)*100.0 if close_ else 0.0
    atrp_min = _to_float(p.get("atr_pct_min"), 0.25)
    atrp_max = _to_float(p.get("atr_pct_max"), 1.40)
    if not (atrp_min <= atrp_now <= atrp_max):
        dbg_reason(asset, f"BB_MR: ATR% {atrp_now:.2f} fuori range [{atrp_min},{atrp_max}]")
        return None

    bb_len = _to_int(p.get("bb_len"), 20)
    bb_std = _to_float(p.get("bb_std"), 2.0)
    rsi_len = _to_int(p.get("rsi_len"), 14)
    rsi_buy = _to_float(p.get("rsi_buy"), 30.0)
    rsi_sell = _to_float(p.get("rsi_sell"), 70.0)
    bbw_relax = _to_float(p.get("bbw_relax"), 1.0)
    tp_atr = p.get("tp_atr")
    sl_over = p.get("sl_over_band_atr")
    atr_stop = _to_float(p.get("atr_stop_mult"), 1.5)
    tp_to_ma = _to_bool(p.get("tp_to_ma"), True)

    lo, ma, up = bbands(d1["close"], bb_len, bb_std)
    rsi14 = rsi(d1["close"], rsi_len)

    bbw_q50 = (ctx or {}).get("bbw_q50", None)
    if (bbw_q50 is not None) and pd.notna(ma.iloc[-1]) and ma.iloc[-1] != 0:
        bbw_now = float((up.iloc[-1] - lo.iloc[-1]) / ma.iloc[-1])
        if bbw_now > _to_float(bbw_q50, 0.0) * bbw_relax:
            dbg_reason(asset, f"BB_MR: BBW {bbw_now:.3f} sopra soglia relativa (q50*relax)")
            return None

    last_close = float(d1["close"].iloc[-1]); prev_close = float(d1["close"].iloc[-2])
    last_low = float(lo.iloc[-1]); prev_low = float(lo.iloc[-2])
    last_up = float(up.iloc[-1]); prev_up = float(up.iloc[-2])
    last_ma = float(ma.iloc[-1])
    last_rsi = float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else 50.0

    confirm_inside = _to_bool(p.get("confirm_close_inside_band"), True)
    atr_abs = (atrp_now/100.0)*last_close if last_close else atr14

# ...dentro build_signal_bbmr_confirmed...
    # --- PATCH 2.C: Trigger BB_MR più permissivo ---
    cross_long = prev_close < prev_low and last_close > last_low
    touch_long = last_close <= last_low * 1.002
    
    cross_short = prev_close > prev_up and last_close < last_up
    touch_short = last_close >= last_up * 0.998

    # LONG
    if (cross_long or touch_long) and last_rsi <= rsi_buy:
        # ... (il resto della logica long rimane IDENTICO)
        entry = last_close
        sl = entry - (_to_float(sl_over, None)*atr_abs if sl_over is not None else atr_stop*atr14)
        if tp_atr is not None:
            tp_opt = entry + _to_float(tp_atr, 0.0)*atr_abs
            tp = min(last_ma, tp_opt) if tp_to_ma else tp_opt
        else:
            tp = last_ma if tp_to_ma else last_up
        rr = abs((tp - entry) / (entry - sl)) if (entry - sl)!=0 else 0.0
        return DeterministicSignal("BB_MR", asset, symbol, "LONG", "D1",
                                     round(entry,5), round(sl,5), round(tp,5), round(rr,2),
                                     round(atr14,5), {"bb_len":bb_len,"bb_std":bb_std,"rsi14":round(last_rsi,2)})

    # SHORT
    if (cross_short or touch_short) and last_rsi >= rsi_sell:
        # ... (il resto della logica short rimane IDENTICO)
        entry = last_close
        sl = entry + (_to_float(sl_over, None)*atr_abs if sl_over is not None else atr_stop*atr14)
        if tp_atr is not None:
            tp_opt = entry - _to_float(tp_atr, 0.0)*atr_abs
            tp = max(last_ma, tp_opt) if tp_to_ma else tp_opt
        else:
            tp = last_ma if tp_to_ma else last_low
        rr = abs((entry - tp) / (sl - entry)) if (sl - entry)!=0 else 0.0
        return DeterministicSignal("BB_MR", asset, symbol, "SHORT", "D1",
                                     round(entry,5), round(sl,5), round(tp,5), round(rr,2),
                                     round(atr14,5), {"bb_len":bb_len,"bb_std":bb_std,"rsi14":round(last_rsi,2)})
    # --- FINE PATCH ---
        
    dbg_reason(asset, "BB_MR: nessuna condizione di entry (incrocio/tocco bande + RSI)")
    return None

def build_signals_for_item(item: dict) -> List[DeterministicSignal]:
    asset  = item.get("asset", "N/A")
    regime = (item.get("regime") or "INDEFINITO").upper()
    signals: List[DeterministicSignal] = []

    if regime == "TRENDING":
        s = build_signal_weekly(item)
        if s:
            if _greenlight(asset, "WEEKLY"):
                signals.append(s)
            else:
                dbg_reason(asset, "WEEKLY bloccata: nessuna strategia TREND attiva nel roster.")

    if regime in ("LATERALE", "LATERALE (COMPRESSIONE)"):
        s = build_signal_bbmr_confirmed(item)
        if s:
            if _greenlight(asset, "BB_MR"):
                signals.append(s)
            else:
                dbg_reason(asset, "BB_MR bloccata: nessuna strategia MR attiva nel roster.")

    # Se in futuro vuoi gestire breakout in regimi 'VOLATILE', li aggiungi qui.

    return signals


# =============== AI VETO (stub) ===============
def build_ai_prompt(signal: DeterministicSignal, regime_item: dict) -> str:
    return "You are ODIN's risk sentry..."

def validate_ai_veto_json(payload: dict) -> dict:
    if not isinstance(payload, dict): raise ValueError("AI_VETO: payload non object.")
    for k in ["veto","score","note"]:
        if k not in payload: raise ValueError(f"AI_VETO: campo mancante {k}")
    if not isinstance(payload["veto"], bool): raise ValueError("AI_VETO: veto non boolean.")
    try: score = float(payload["score"])
    except Exception: raise ValueError("AI_VETO: score non numerico.")
    if not (0.0 <= score <= 1.0): raise ValueError("AI_VETO: score fuori range.")
    note = payload["note"]
    if not isinstance(note, str) or not (3 <= len(note) <= 180): raise ValueError("AI_VETO: note invalida.")
    payload["score"] = round(score, 2); return payload

def ai_veto(signal: DeterministicSignal, regime_item: dict) -> AIVeto:
    _ = build_ai_prompt(signal, regime_item)
    if not GOOGLE_API_KEY:
        return AIVeto(veto=False, score=0.5, note="AI_FALLBACK: no API key")
    try:
        raw = '{"veto": false, "score": 0.70, "note": "Trend pulito, nessun ostacolo evidente."}'
        payload = validate_ai_veto_json(json.loads(raw))
        return AIVeto(veto=payload["veto"], score=payload["score"], note=payload["note"])
    except Exception as e:
        log.error(f"Errore IA Veto per {signal.asset}: {e!r}")
        return AIVeto(veto=False, score=0.5, note=f"AI_FALLBACK: {type(e).__name__}")

# =============== POSITION SIZING / MARGIN ===============
def estimate_position_size(signal: DeterministicSignal, account_equity: float, risk_pct: float,
                            broker_caps: Optional[MT5SymbolCaps] = None) -> float:
    try:
        if broker_caps is None:
            log.error("estimate_position_size: broker_caps assenti"); return 0.0
        risk_amount = account_equity * (risk_pct/100.0)
        if risk_amount <= 0: return 0.0
        sl_dist = abs(signal.entry - signal.sl)
        if sl_dist <= 1e-9: return 0.0
        if broker_caps.point <= 0 or broker_caps.trade_tick_value <= 0: return 0.0
        tick_count = sl_dist / broker_caps.point
        risk_per_lot = tick_count * broker_caps.trade_tick_value
        if risk_per_lot <= 0: return 0.0
        lots_raw = risk_amount / risk_per_lot
        lots_rounded = round_volume_to_caps(lots_raw, broker_caps)
        if lots_rounded >= broker_caps.volume_min:
            return min(lots_rounded, float(f"{broker_caps.volume_max:.3f}"))
        risk_at_min = broker_caps.volume_min * risk_per_lot
        eff_risk_pct = (risk_at_min / account_equity) * 100.0
        tol = risk_pct * MIN_LOT_RISK_TOLERANCE_FACTOR
        if eff_risk_pct <= tol:
            log.info(f"Size {lots_raw:.4f} < min; forzo {broker_caps.volume_min} (eff_risk={eff_risk_pct:.2f}% <= tol={tol:.2f}%).")
            return float(f"{broker_caps.volume_min:.3f}")
        log.warning(f"Trade rifiutato: min-lot implica rischio {eff_risk_pct:.2f}% > tol {tol:.2f}% (target={risk_pct:.2f}%).")
        return 0.0
    except Exception as e:
        log.error(f"estimate_position_size: errore {e!r}"); return 0.0

def calc_required_margin(broker_symbol: str, direction: str, volume: float) -> float:
    if not MT5_AVAILABLE: return -1.0
    try:
        si = mt5.symbol_info(broker_symbol)
        if not si or not si.visible: return -1.0
        tick = mt5.symbol_info_tick(broker_symbol)
        if not tick: return -1.0
        order_type = mt5.ORDER_TYPE_BUY if direction.upper()=="LONG" else mt5.ORDER_TYPE_SELL
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        margin = mt5.order_calc_margin(order_type, broker_symbol, volume, price)
        return float(margin) if margin is not None else -1.0
    except Exception:
        return -1.0

def risk_manager(asset_item: dict, signal: DeterministicSignal, account_equity: float, caps: RiskCaps,
                 exposure_snapshot: Optional[Dict[str, float]], broker_caps: Optional[MT5SymbolCaps],
                 account_free_margin: float, broker_symbol: Optional[str]) -> RiskDecision:
    if account_equity < MIN_EQUITY_TO_TRADE:
        return RiskDecision(False, f"Equity troppo bassa ({account_equity:.2f}) < soglia ({MIN_EQUITY_TO_TRADE:.2f})",
                            0.0, 0.0, {})
    bucket = asset_item.get("bucket", "OTHER"); asset = asset_item["asset"]
    snap = exposure_snapshot or {}
    g_used = float(snap.get("GLOBAL", 0.0)); b_used = float(snap.get(f"BUCKET:{bucket}", 0.0)); a_used = float(snap.get(f"ASSET:{asset}", 0.0))
    room_g = max(0.0, caps.max_global - g_used); room_b = max(0.0, caps.max_bucket - b_used); room_a = max(0.0, caps.max_asset - a_used)
    proposed_risk = min(caps.max_trade, room_g, room_b, room_a)
    caps_snap = {"GLOBAL": g_used, f"BUCKET:{bucket}": b_used, f"ASSET:{asset}": a_used}
    if proposed_risk <= 0.0:
        return RiskDecision(False, "No room left (caps)", 0.0, 0.0, caps_snap)
    lots = estimate_position_size(signal, account_equity, proposed_risk, broker_caps)
    min_lot = (broker_caps.volume_min if broker_caps else 0.01)
    if lots < min_lot or lots <= 0.0:
        return RiskDecision(False, "Lot size too small or unsupported", 0.0, 0.0, caps_snap)
    if broker_symbol:
        required_margin = calc_required_margin(broker_symbol, signal.direction, lots)
        if required_margin < 0:
            return RiskDecision(False, "Margin calc failed (MT5)", 0.0, 0.0, caps_snap)
        allowed_free = max(0.0, account_free_margin * (1.0 - MARGIN_BUFFER_FACTOR))
        if required_margin > allowed_free:
            return RiskDecision(False, f"Insufficient margin: req {required_margin:.2f} > free*buffer {allowed_free:.2f}",
                                0.0, 0.0, caps_snap)
    return RiskDecision(True, None, lots, proposed_risk, caps_snap)

# =============== NOTIFICHE (Telegram) ===============
def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram non configurato; salto invio.")
        return
    def _call():
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}, timeout=10)
        r.raise_for_status()
    try:
        retry_backoff(_call, tries=3)
    except Exception as e:
        log.error(f"Telegram send failed: {e!r}")

def fmt_signal_for_telegram(plan: ActionPlan) -> str:
    s, r, ai = plan.signal, plan.risk, plan.ai
    broker_asset = resolve_broker_symbol(BROKER_SYMBOL_MAP.get(s.asset, s.asset)) or s.asset
    tag = " (EXPLORATION)" if s.basis.get("exploration") else ""
    lines = [
        f"ODIN / {s.strategy}{tag} / {broker_asset}",
        f"Dir: {s.direction} | TF: {s.timeframe_entry}",
        f"Entry: {s.entry:.5f} | SL: {s.sl:.5f} | TP: {s.tp:.5f} | RR: {s.rr}",
        f"Risk: {r.risk_pct:.2f}% | Size: {r.size_lots} lots",
        f"AI: veto={ai.veto}, score={ai.score:.2f}, note='{ai.note}'",
        f"PlanID: {plan.id} | RegimeSHA: {plan.regime_sha1[:10]}..."
    ]
    return "\n".join(lines)

def fmt_reject_for_telegram(asset: str, reason: str, ai: Optional[AIVeto] = None) -> str:
    parts = [f"ODIN / REJECT / {asset}", f"Motivo: {reason}"]
    if ai is not None:
        parts.append(f"AI: veto={ai.veto}, score={ai.score:.2f}, note='{ai.note}'")
    return "\n".join(parts)

def fmt_info_for_telegram(title: str, details: str) -> str:
    return f"ODIN / INFO / {title}\n{details}"

# =============== ACTION PLAN I/O ===============
def save_action_plan(plan: ActionPlan) -> str:
    os.makedirs(os.path.join(ROOT_DIR, "plans"), exist_ok=True)
    safe_asset = plan.asset.replace("/", "_")
    ts = plan.asof_utc.replace(':','').replace('-','')
    path = os.path.join(ROOT_DIR, "plans", f"{ts}_{safe_asset}_{plan.id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(plan), f, indent=2, ensure_ascii=False)
    return path

def update_plan_status(path: str, new_status: str, notes_append: Optional[str] = None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["status"] = new_status
    if notes_append:
        data.setdefault("notes", []).append(notes_append)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =============== Exploration state helpers ===============
def _expl_today_key(dt=None):
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d")

def _load_expl_state():
    st = _load_json(EXPL_STATE_FILE, default={})
    if not isinstance(st, dict):
        st = {}
    cur_day = _expl_today_key()
    if st.get("day") != cur_day:
        st = {"day": cur_day, "count": 0, "last": {}} # reset giornaliero
    st.setdefault("count", 0)
    st.setdefault("last", {})
    return st

def _save_expl_state(st):
    _save_json(EXPL_STATE_FILE, st)

def _can_explore(asset: str, st) -> bool:
    if not EXPL_ENABLE:
        return False
    if st.get("count", 0) >= EXPL_MAX_PER_DAY:
        return False
    last = st.get("last", {}).get(asset)
    if last:
        try:
            t_last = datetime.fromisoformat(last)
        except Exception:
            t_last = None
        if t_last and (datetime.now(timezone.utc) - t_last) < timedelta(hours=EXPL_CD_HOURS):
            return False
    return True

def _mark_explore(asset: str, st):
    st["count"] = int(st.get("count", 0)) + 1
    st.setdefault("last", {})[asset] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_expl_state(st)

# =============== Macro allow/guard helpers (fuori dalla classe) ===============
def _is_allowed_new_entry(params_obj: dict, asset_name: str) -> bool:
    try:
        ane = (params_obj.get("asset_overrides", {}).get(asset_name, {}).get("allow_new_entries", None))
        return True if ane is None else bool(ane)
    except Exception:
        return True

def _violates_guardrail(asset_name: str, direction: str) -> bool:
    g = _MACRO_GUARDRAILS.get(asset_name, {})
    if direction.upper() == "LONG" and g.get("block_long", False): return True
    if direction.upper() == "SHORT" and g.get("block_short", False): return True
    return False

# ==== helper robusti (UNIVERSE/REGIME → strategia) ====
def _norm_asset(a: str) -> str:
    return (a or "").upper().strip()

def _strategy_for_regime(r: str) -> Optional[str]:
    r = (r or "").upper()
    if "TREND" in r:
        return "WEEKLY"
    if any(k in r for k in ("LAT", "COMP", "SIDE", "SIDEWAYS", "RANGE")):
        return "BB_MR"
    return None

# =============== MAIN ORCHESTRATION ===============
def main():
    log.info(f"--- Avvio Sala Decisionale ODIN v{VERSION} ({ENV}) ---")

    # Recap accumulators
    recap = {
        "committed": 0, "reject_ai": 0, "reject_risk": 0, "reject_exec": 0,
        "reject_news": 0, "reject_macro": 0, "reject_guard": 0,
        "nosignal": [], "blocked_news_assets": [],
    }

    # Carico NewsGate finestre (currency-based). SAFE
    news_windows = {}
    if NEWS_BLOCK_ENABLE:
        try:
            news_windows = _load_news_blackout(NEWS_BLACKOUT_FILE)
            n_ev = sum(len(v) for v in news_windows.values())
            log.info(f"[NewsGate] enable={NEWS_BLOCK_ENABLE} | file={os.path.basename(NEWS_BLACKOUT_FILE)} | finestre={n_ev}")
        except Exception as e:
            log.warning(f"[NewsGate] impossibile caricare {NEWS_BLACKOUT_FILE}: {e!r} (nessun blocco)")
            news_windows = {}

    # Exploration state
    expl_state = _load_expl_state()

    bridge = MT5Bridge()
    try:
        if not bridge.connected:
            msg = "Impossibile connettersi a MT5. Terminale non avviato o non loggato."
            log.error(msg)
            try: send_telegram(fmt_info_for_telegram("MT5 non collegato", msg))
            except Exception: pass
            return

        account_info = bridge.get_account_info()
        if not account_info:
            msg = "Impossibile ottenere i dati del conto da MT5. Processo terminato."
            log.error(msg)
            try: send_telegram(fmt_info_for_telegram("Account Info", msg))
            except Exception: pass
            return

        account_equity = float(account_info.get("equity", 0.0))
        account_free_margin = float(account_info.get("free_margin", 0.0))
        account_ccy = account_info.get("currency", ACCOUNT_CCY)
        log.info(f"Equity attuale: {account_equity:.2f} {account_ccy}")

        try:
            regime = load_regime_report(REGIME_REPORT_PATH)
        except Exception as e:
            msg = f"Errore report regime: {e}"
            log.error(msg)
            try: send_telegram(fmt_info_for_telegram("Regime Report", msg))
            except Exception: pass
            return

        log.info(f"Regime v{regime['meta'].get('version')} OK (sha1 {regime['meta']['sha1'][:10]})")

        # Carica allowlist Universe (asset -> set strategie consentite oggi)
        allow_map: dict[str, set] = {}
        try:
            with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
                uj = json.load(f)
            for it in (uj.get("items") or []):
                a = str(it.get("asset","")).strip()
                ks = set(s.strip().upper() for s in (it.get("strategies") or []))
                if a and ks:
                    allow_map[a.upper()] = ks
            log.info(f"[Universe] Attivi {len(allow_map)} asset dal file universe.json")
        except Exception:
            log.info("[Universe] Nessun file allowlist — procedo senza filtro.")

        items = regime.get("items", [])
        tradables = [it for it in items if it.get('regime') in ("TRENDING", "LATERALE", "LATERALE (Compressione)")]

        # [PATCH] Universe Manager: se params/universe.json ha picks, limita agli asset selezionati
        try:
            um_path = os.path.join(PARAMS_DIR, "universe.json")
            if os.path.exists(UNIVERSE_FILE):
                um = _load_json(UNIVERSE_FILE, {})
                picks = [p.get("asset") for p in (um.get("picks") or [])]
                if picks:
                    tradables = [it for it in tradables if it.get("asset") in set(picks)]
                    log.info(f"[Universe] attivo: {len(picks)} asset selezionati nella giornata.")
        except Exception as _e:
            log.warning(f"[Universe] errore lettura picks: {_e!r}")

        if not tradables:
            info = "Nessun asset con setup valido. Nessuna azione."
            log.info(info)
            try: send_telegram(fmt_info_for_telegram("Nessuna Opportunita", info))
            except Exception: pass
            return

        # Snapshot esposizione (semplice, in assenza di lettura posizioni)
        exposure_snapshot: Dict[str, float] = {"GLOBAL": 0.0}

        # carica una sola volta (FIX: serve 'global' per aggiornare la variabile di modulo)
        global _STRAT_ROSTER
        _STRAT_ROSTER = _load_roster_safe()

        # =========================
        # MAIN LOOP sugli asset
        # =========================
        for item in tradables:
            asset = item.get("asset", "").strip()
            asset_norm = _norm_asset(asset)
            try:
                # --- [UNIVERSE] filtro per strategia consentita oggi ---
                allowed = allow_map.get(asset_norm)
                want = _strategy_for_regime(item.get("regime",""))

                if allowed:
                    if want is not None and want.upper() not in allowed:
                        log.info(f"[NO-SIGNAL] {asset}: Universe block (allowed {allowed}) — regime='{item.get('regime')}', want='{want}'")
                        _LAST_NOSIGNAL_REASON[asset] = f"Universe: strategy '{str(want).upper()}' non consentita oggi (allowed={sorted(list(allowed))})"
                        continue
                    else:
                        log.info(f"[Universe] {asset}: OK (allowed {allowed}) — regime='{item.get('regime')}', want='{want}'")
                else:
                    log.info(f"[Universe] {asset}: no allowlist entry — procedo normale")

                # --- NEWSGATE finestre/currency (prima del build segnali per risparmiare chiamate) ---
                if NEWS_BLOCK_ENABLE and news_windows:
                    _now_utc = datetime.now(timezone.utc)
                    blocked, why = _news_blocked(asset, _now_utc, news_windows)
                    if blocked:
                        msg = f"[NEWSGATE] BLOCK {asset} per finestra news: {why} (±{NEWS_BLOCK_PRE_MIN}/{NEWS_BLOCK_POST_MIN}m)"
                        log.info(msg)
                        recap["blocked_news_assets"].append(asset)
                        try: send_telegram(msg)
                        except Exception: pass
                        continue

                # --- Build segnali in base al regime ---
                signals = build_signals_for_item(item)

                # === Exploration Mode (segnale simulato) SOLO se non ci sono segnali reali ===
                if not signals and _can_explore(asset, expl_state) and random.random() < EXPL_RATE:
                    regime_lbl = item.get("regime", "")
                    fake_signal = None
                    if regime_lbl == "TRENDING" and EXPL_ALLOW_WK:
                        fake_signal = build_signal_weekly(item)
                    elif regime_lbl in ("LATERALE", "LATERALE (Compressione)") and EXPL_ALLOW_BBMR:
                        fake_signal = build_signal_bbmr_confirmed(item)
                    if fake_signal:
                        fake_signal.basis["exploration"] = True
                        signals = [fake_signal]
                        log.info(f"[EXPLORATION] {asset}: segnale simulato {fake_signal.strategy}/{fake_signal.direction}")

                # --- Volatility Gate (usa ATR% dal regime report) ---
                try:
                    metrics = item.get("metrics", {})
                    atrp_now = _to_float(metrics.get("atrp"), 0.0)
                    ctx = item.get("context", {})
                    atrp_q95 = _to_float(ctx.get("atrp_q95"), 0.0)
                    if atrp_q95 > 0 and atrp_now >= atrp_q95 * 1.1:
                        reason = f"VolGate: ATR% {atrp_now:.2f} > soglia {atrp_q95:.2f}"
                        log.info(f"[REJECT VOLGATE] {asset} — {reason}")
                        recap["reject_news"] += 1 # assimilabile a “volatility/news gate”
                        try: send_telegram(fmt_reject_for_telegram(asset, reason, None))
                        except Exception: pass
                        continue
                except Exception as e:
                    log.warning(f"VolGate check error {asset}: {e!r}")

                # --- Nessun segnale generato? log e continua ---
                if not signals:
                    reason = _LAST_NOSIGNAL_REASON.pop(asset, None)
                    if reason:
                        log.info(f"[NO-SIGNAL] {asset}: {reason}")
                        recap["nosignal"].append(f"{asset}: {reason}")
                    else:
                        log.info(f"[NO-SIGNAL] {asset}: nessun setup valido (regime={item.get('regime')}).")
                        recap["nosignal"].append(f"{asset}: nessun setup valido (regime={item.get('regime')})")
                    continue

                # --- Loop su ciascun segnale prodotto ---
                for signal in signals:
                    # NewsGate livello 1 (json base)
                    blk, blk_reason = is_news_blackout(asset, signal.strategy)
                    if blk:
                        log.info(f"[REJECT NEWS] {asset} — {blk_reason}")
                        recap["reject_news"] += 1
                        try: send_telegram(fmt_reject_for_telegram(asset, blk_reason, None))
                        except Exception: pass
                        continue

                    # MacroGate allow_new_entries
                    if not _is_allowed_new_entry(PARAMS, asset):
                        reason = "MacroGate: allow_new_entries=False"
                        log.info(f"[REJECT MACRO] {asset} — {reason}")
                        recap["reject_macro"] += 1
                        try: send_telegram(fmt_reject_for_telegram(asset, reason, None))
                        except Exception: pass
                        continue

                    # Macro guardrails
                    if _violates_guardrail(asset, signal.direction):
                        reason = f"MacroGuard: {signal.direction} bloccato"
                        log.info(f"[REJECT GUARD] {asset} — {reason}")
                        recap["reject_guard"] += 1
                        try: send_telegram(fmt_reject_for_telegram(asset, reason, None))
                        except Exception: pass
                        continue

                    # AI veto
                    ai = ai_veto(signal, item)

                    # Riduzione rischio per-trade (fallback AI / exploration)
                    is_expl = bool(signal.basis.get("exploration"))
                    trade_cap = MAX_TRADE_RISK_PCT
                    if "FALLBACK" in ai.note:
                        trade_cap *= 0.5
                    if is_expl:
                        trade_cap *= EXPL_RISK_MULT

                    local_caps = RiskCaps(
                        max_global=MAX_GLOBAL_RISK_PCT,
                        max_bucket=MAX_BUCKET_RISK_PCT,
                        max_asset=MAX_ASSET_RISK_PCT,
                        max_trade=trade_cap
                    )

                    preferred = BROKER_SYMBOL_MAP.get(signal.asset, signal.asset)
                    broker_symbol = resolve_broker_symbol(preferred)
                    if broker_symbol is None:
                        msg = f"Impossibile risolvere simbolo broker per {preferred}. Salto {asset}."
                        log.error(msg)
                        try: send_telegram(fmt_info_for_telegram("Simbolo non risolto", msg))
                        except Exception: pass
                        recap["reject_exec"] += 1
                        continue

                    broker_caps = get_mt5_symbol_caps(broker_symbol)
                    if broker_caps is None:
                        msg = f"Specifiche broker non disponibili per {broker_symbol}. Salto {asset}."
                        log.error(msg)
                        try: send_telegram(fmt_info_for_telegram("Cap broker mancanti", msg))
                        except Exception: pass
                        recap["reject_exec"] += 1
                        continue

                    risk = risk_manager(item, signal, account_equity, local_caps, exposure_snapshot,
                                        broker_caps, account_free_margin, broker_symbol)

                    plan = ActionPlan(
                        id=str(uuid.uuid4())[:8],
                        asof_utc=now_utc(),
                        regime_sha1=regime['meta']['sha1'],
                        asset=signal.asset,
                        bucket=item.get("bucket", "OTHER"),
                        status="PLANNED",
                        signal=signal,
                        ai=ai,
                        risk=risk,
                        notes=[]
                    )

                    # ML logger (anche se rifiutato)
                    try:
                        log_signal(plan, item)
                    except Exception as _e:
                        log.warning(f"ML logger (signal) error {asset}: {_e}")

                    plan_path = save_action_plan(plan)

                    if ai.veto:
                        reason = f"AI_VETO: {ai.note}"
                        update_plan_status(plan_path, "REJECTED", reason)
                        log.info(f"[REJECTED AI] {asset}: {reason}")
                        recap["reject_ai"] += 1
                        try: send_telegram(fmt_reject_for_telegram(asset, reason, ai))
                        except Exception: pass
                        continue

                    if not risk.accept:
                        reason = f"RISK_REJECT: {risk.reason}"
                        update_plan_status(plan_path, "REJECTED", reason)
                        log.info(f"[REJECTED RISK] {asset}: {reason}")
                        recap["reject_risk"] += 1
                        try: send_telegram(fmt_reject_for_telegram(asset, reason, ai))
                        except Exception: pass
                        continue

                    # Normalizza livelli per i tick del broker
                    entry_n, sl_n, tp_n = normalize_prices_for_mt5(signal.entry, signal.sl, signal.tp, broker_caps)
                    signal.entry, signal.sl, signal.tp = entry_n, sl_n, tp_n

                    ticket = bridge.execute_trade(plan, broker_caps)
                    if ticket is None:
                        reason = "EXECUTION_FAIL: ordine non eseguito su MT5"
                        update_plan_status(plan_path, "REJECTED", reason)
                        log.error(f"[REJECTED EXEC] {asset}: {reason}")
                        recap["reject_exec"] += 1
                        try: send_telegram(fmt_reject_for_telegram(asset, reason, ai))
                        except Exception: pass
                        continue

                    update_plan_status(plan_path, "COMMITTED", f"ORDER_TICKET={ticket}")
                    try: send_telegram(fmt_signal_for_telegram(plan))
                    except Exception: pass
                    log.info(f"[COMMITTED] {asset} eseguito. Ticket={ticket}")
                    recap["committed"] += 1

                    # Se exploration → marca stato (conteggio/cooldown)
                    if signal.basis.get("exploration"):
                        _mark_explore(asset, expl_state)

                    # Aggiorna snapshot esposizione (placeholder semplice)
                    exposure_snapshot["GLOBAL"] = exposure_snapshot.get("GLOBAL", 0.0) + risk.risk_pct
                    exposure_snapshot[f"BUCKET:{item.get('bucket','OTHER')}"] = \
                        exposure_snapshot.get(f"BUCKET:{item.get('bucket','OTHER')}", 0.0) + risk.risk_pct
                    exposure_snapshot[f"ASSET:{signal.asset}"] = \
                        exposure_snapshot.get(f"ASSET:{signal.asset}", 0.0) + risk.risk_pct

            except Exception as e:
                log.error(f"Errore su {asset}: {e}\n{traceback.format_exc()}")

        # [PATCH] Summary Telegram compatto (compatibile con recap finale)
        try:
            committed_count = recap["committed"]
            rejs = {"AI":recap["reject_ai"],"RISK":recap["reject_risk"],"EXEC":recap["reject_exec"],
                    "NEWS/VOL":recap["reject_news"],"MACRO":recap["reject_macro"],"GUARD":recap["reject_guard"]}
            
            no_signal_reasons = []
            for reason_text in recap["nosignal"]:
                no_signal_reasons.append(reason_text)

            summary = [
                "ODIN / SUMMARY / Decision Room",
                f"✅ COMMITTED: {committed_count}",
                "❌ REJECT — AI: {AI}, RISK: {RISK}, EXEC: {EXEC}, NEWS/VOL: {NEWS_VOL}, MACRO: {MACRO}, GUARD: {GUARD}".format(
                    AI=rejs["AI"], RISK=rejs["RISK"], EXEC=rejs["EXEC"], NEWS_VOL=rejs["NEWS/VOL"], MACRO=rejs["MACRO"], GUARD=rejs["GUARD"]
                ),
                f"ℹ️ NO-SIGNAL: " + (" | ".join(no_signal_reasons) if no_signal_reasons else "—"),
                "🕒 NEWS windows blocked: " + (" | ".join(recap["blocked_news_assets"]) if recap["blocked_news_assets"] else "—")
            ]
            # Unito i due recap per non mandare doppi messaggi
            # send_telegram("\n".join(summary))
        except Exception as _e:
            log.warning(f"Summary TG error: {_e!r}")

        log.info("--- Sessione decisionale completata ---")

        # ==== Telegram Recap Unificato ====
        try:
            # Usa le variabili già preparate dal blocco summary
            send_telegram("\n".join(summary))
        except Exception as e:
            log.warning(f"Recap telegram failed: {e!r}")

    finally:
        # Persist exploration state (già salvato alla marcatura, ma salvo comunque)
        try: _save_expl_state(expl_state)
        except Exception: pass
        bridge.shutdown()

# =============== ENTRYPOINT ===============
if __name__ == "__main__":
    print("ODIN bootstrap…", flush=True)
    try: logging.basicConfig(level=logging.INFO)
    except Exception: pass
    try: main()
    except Exception as e:
        print(f"ODIN CRASH: {e!r}", flush=True)
        traceback.print_exc()