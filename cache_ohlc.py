# cache_ohlc.py — robust cache per OHLC TwelveData (safe I/O + atomic write + TTL + bug fix)
import os
import json
import time
import logging
import random
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
log = logging.getLogger("cache_ohlc")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

CACHE_DIR = os.getenv("ODIN_CACHE_DIR", os.path.join("C:\\", "AUTOMAZIONE", "cache_ohlc_twelvedata"))
TD_API_KEY = (os.getenv("TWELVEDATA_API_KEY") or "").strip()
TD_BASE_URL = "https://api.twelvedata.com/time_series"

os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(symbol: str, interval: str, outputsize: int) -> str:
    safe_sym = symbol.replace("/", "_").replace(":", "-")
    fname = f"{safe_sym}__{interval}__{outputsize}.json"
    return os.path.join(CACHE_DIR, fname)

def _retry(fn, tries=4, base=0.5, cap=4.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(min(cap, base * (2 ** i)) + random.uniform(0, 0.15))
    raise last

def _fetch_live(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    if not TD_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY mancante (env/.env).")

    def _call():
        r = requests.get(
            TD_BASE_URL,
            params=dict(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                order="ASC",
                timezone="UTC",
                apikey=TD_API_KEY,
            ),
            headers={"User-Agent": "odin-cache/1.0"},
            timeout=12,
        )
        r.raise_for_status()
        js = r.json()
        # error handling esplicito TwelveData
        if isinstance(js, dict) and js.get("status") == "error":
            raise RuntimeError(f"TwelveData error: {js.get('message')}")
        vals = js.get("values")
        if not vals:
            raise RuntimeError(f"Nessun dato TwelveData per {symbol} {interval}")
        return vals

    vals = _retry(_call)
    df = pd.DataFrame(vals)
    # parse & tipizza
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # ordina + dedup (prendi l’ultimo in caso di duplicati)
    df = df.sort_values("datetime").dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last").reset_index(drop=True)
    return df


def fetch_ohlc_cached(symbol: str, interval: str, outputsize: int = 400, force: bool = False, max_age_seconds: int = 3600) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol, interval, outputsize)
    df = None # FIX 1: Inizializza df a None

    if os.path.exists(path) and not force:
        try:
            file_age = time.time() - os.path.getmtime(path)
            if file_age > max_age_seconds:
                raise ValueError("Cache scaduta")

            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                raise ValueError("File cache vuoto")

            data = json.loads(raw)
            df = pd.DataFrame(data)

            req = {"datetime", "open", "high", "low", "close"}
            if not req.issubset(df.columns):
                raise ValueError("Colonne mancanti nel file di cache")

            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            for c in ("open", "high", "low", "close"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

            if df["datetime"].isna().any():
                raise ValueError("Parsing datetime fallito")

            df = (df.sort_values("datetime")
                    .drop_duplicates("datetime", keep="last")
                    .reset_index(drop=True))

            log.info(f"[CACHE] HIT: {symbol} {interval}")
            
        except Exception as e:
            df = None # Resetta df se la cache è invalida
            try: os.remove(path)
            except Exception: pass
            log.warning(f"[CACHE] {symbol} {interval}: cache invalida ({e!r}). Refetch.")

    if df is None:
        log.info(f"[CACHE] MISS: {symbol} {interval}. Fetching da TwelveData...")
        df = _fetch_live(symbol, interval, outputsize)
        try:
            tmp_path = path + ".tmp"
            df_serial = df.copy()
            if "datetime" in df_serial.columns:
                # ISO tipo 2024-10-16T12:34:56Z (niente offset)
                df_serial["datetime"] = pd.to_datetime(df_serial["datetime"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(df_serial.to_dict(orient="records"), f, ensure_ascii=False)
            os.replace(tmp_path, path)
            log.info(f"[CACHE] Salvato {symbol} {interval} ({len(df)} barre)")
        except Exception as e:
            log.warning(f"[CACHE] Scrittura fallita per {path}: {e!r}")

    if df is not None and not df.empty:
        if df["datetime"].iloc[-1].date() == datetime.now(timezone.utc).date():
            df = df.iloc[:-1].copy()
    return df

def fetch_ohlc_twelvedata(symbol: str, interval: str, outputsize: int = 400) -> Optional[pd.DataFrame]:
    i = (interval or "").strip().lower()
    ttl = 15 * 60
    if i in ("4h", "4hour", "240min"):
        outputsize = max(outputsize, 1200)
        ttl = 3600
        interval = "4h"
    elif i in ("1day", "1d", "d"):
        outputsize = max(outputsize, 2000)   # ~8 anni
        ttl = 4 * 3600
        interval = "1day"
    return fetch_ohlc_cached(symbol, interval, outputsize, max_age_seconds=ttl)
