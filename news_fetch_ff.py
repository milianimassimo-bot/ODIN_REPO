# -*- coding: utf-8 -*-
"""
ODIN – NewsGate (ForexFactory JSON stable)
- Legge il calendario settimanale in JSON da ForexFactory (feed stabile)
- Filtra gli eventi ad alto impatto nella finestra temporale attorno a “adesso”
- Mappa gli eventi sugli asset ODIN e scrive params/news_blackout.json
- (opzionale) manda un riepilogo su Telegram

Dipendenze: requests, python-dotenv
Env utili (.env):
  NEWS_IMPACT_MIN=3                  # 1..3 (3 = high impact)
  NEWS_WINDOW_BEFORE_MIN=60          # minuti prima dell’evento
  NEWS_WINDOW_AFTER_MIN=15           # minuti dopo l’evento
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
"""

import os
import json
import time
import math
import traceback
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

# ============ CONFIG ============
ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(ROOT, "params")
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(PARAMS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

NEWS_BLACKOUT_FILE = os.path.join(PARAMS_DIR, "news_blackout.json")

# feed settimanale stabile (evita host “cdn-…/thisday” etc.)
FF_JSON_URL = "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"

# Env + default
load_dotenv()
NEWS_IMPACT_MIN = int(os.getenv("NEWS_IMPACT_MIN", "3"))
NEWS_WINDOW_BEFORE_MIN = int(os.getenv("NEWS_WINDOW_BEFORE_MIN", "60"))
NEWS_WINDOW_AFTER_MIN  = int(os.getenv("NEWS_WINDOW_AFTER_MIN",  "15"))

TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TG_CHAT  = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

# Mappa evento→asset che vogliamo “oscurare”
# (puoi aggiungere o togliere mapping liberamente)
ASSET_MAP = {
    "USD": ["EUR/USD", "GBP/USD", "AUD/USD", "USD/JPY", "GOLD", "SILVER", "SP500"],
    "EUR": ["EUR/USD", "EUR/JPY", "GOLD", "SILVER"],
    "GBP": ["GBP/USD", "GBP/JPY", "GOLD", "SILVER"],
    "JPY": ["USD/JPY", "EUR/JPY", "GBP/JPY"],
    "AUD": ["AUD/USD", "GOLD", "SILVER"],
    "NZD": ["NZD/USD", "GOLD", "SILVER"],
    "CAD": ["GOLD", "SILVER"],
    "CHF": ["EUR/USD", "GBP/USD", "USD/JPY"],
    # indici / commodities specifiche (usa USD come proxy per XAU/XAG/US equities)
}

# ============ HELPERS ============
def _now_utc():
    return datetime.now(timezone.utc)

def _within_window(evt_time: datetime, ref: datetime, before_m: int, after_m: int) -> bool:
    return (ref - timedelta(minutes=before_m)) <= evt_time <= (ref + timedelta(minutes=after_m))

def _retry_get_json(url: str, tries=3, timeout=15):
    last = None
    headers = {
        "User-Agent": "Mozilla/5.0 (ODIN-NewsGate; +https://example.local)",
        "Accept": "application/json",
        "Cache-Control": "no-cache",
    }
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.6 * (i + 1))
    raise last if last else RuntimeError("download failed")

def _save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _send_tg(text: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception:
        pass

# ============ PARSING ============
def _parse_events(raw_week: list[dict]) -> list[dict]:
    """
    Converte la lista JSON di FF in una lista normalizzata:
      {time: datetime(UTC), currency: 'USD', impact: 1..3, title: str}
    Note:
      - FF fornisce l'orario in UTC (“timestamp” o “dateTime”) — normalizziamo a datetime UTC.
      - Impact tipicamente: "Low", "Medium", "High" → map a 1/2/3.
    """
    out = []
    for ev in raw_week:
        try:
            # campi tipici: date, time, timestamp, impact, currency, title
            # timestamp è epoch ms o s a seconda del feed; gestiamo entrambi
            ts = ev.get("timestamp") or ev.get("dateTime")
            if ts is None:
                # fallback: concat date+time
                dt_s = f"{ev.get('date','')} {ev.get('time','')}".strip()
                try:
                    t = datetime.strptime(dt_s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                except Exception:
                    continue
            else:
                # epoch? può essere ms
                ts = int(ts)
                if ts > 10_000_000_000:  # ms
                    ts = ts // 1000
                t = datetime.fromtimestamp(ts, tz=timezone.utc)

            cur = (ev.get("currency") or "").strip().upper()
            imp_raw = (ev.get("impact") or ev.get("impactText") or "").lower()
            title = (ev.get("title") or ev.get("event") or "").strip()

            # map impact
            if isinstance(ev.get("impact"), int):
                impact = int(ev["impact"])  # già numerico 1..3 in alcuni dump
            else:
                if "high" in imp_raw or "!!!" in imp_raw:
                    impact = 3
                elif "med" in imp_raw or "!!" in imp_raw:
                    impact = 2
                else:
                    impact = 1

            out.append({
                "time": t,
                "currency": cur,
                "impact": impact,
                "title": title,
            })
        except Exception:
            continue
    return out

def _build_blackout(events: list[dict], ref_time: datetime) -> tuple[dict, list[str]]:
    """
    Dalla lista di eventi filtrati costruisce:
      blackout_map: {asset: True, ...}
      lines: righe riassuntive per log/TG
    """
    blackout = {}
    lines = []
    for ev in events:
        cur = ev["currency"]
        assets = ASSET_MAP.get(cur, [])
        if not assets:
            continue
        for a in assets:
            blackout[a] = True
        lines.append(f"- {cur} • {ev['title']} @ {ev['time'].strftime('%H:%MZ')} imp={ev['impact']} → {', '.join(assets)}")
    return blackout, lines

# ============ MAIN ============
def main():
    print("[FF] scarico calendario in JSON…")
    try:
        raw = _retry_get_json(FF_JSON_URL, tries=4, timeout=15)
    except Exception as e:
        print(f"[FF] ERRORE download: {e!r}")
        return

    # Il feed può essere un oggetto con la chiave "weekly" o direttamente una lista
    if isinstance(raw, dict):
        # prova alcune chiavi note
        if "weekly" in raw and isinstance(raw["weekly"], list):
            raw_week = raw["weekly"]
        elif "thisWeek" in raw and isinstance(raw["thisWeek"], list):
            raw_week = raw["thisWeek"]
        else:
            # fallback: se c'è "data"
            raw_week = raw.get("data", [])
            if not isinstance(raw_week, list):
                raw_week = []
    elif isinstance(raw, list):
        raw_week = raw
    else:
        raw_week = []

    if not raw_week:
        print("[FF] Nessun evento nel feed.")
        _save_json(NEWS_BLACKOUT_FILE, {})
        return

    events = _parse_events(raw_week)
    now = _now_utc()

    # Filtra per impatto e finestra temporale
    windowed = [
        e for e in events
        if e["impact"] >= NEWS_IMPACT_MIN and _within_window(
            e["time"], now, NEWS_WINDOW_BEFORE_MIN, NEWS_WINDOW_AFTER_MIN
        )
    ]

    print(f"[FF] eventi in finestra: {len(windowed)} | imp≥{NEWS_IMPACT_MIN} | win=-{NEWS_WINDOW_BEFORE_MIN}/+{NEWS_WINDOW_AFTER_MIN} min")

    blackout_map, lines = _build_blackout(windowed, now)
    _save_json(NEWS_BLACKOUT_FILE, blackout_map)

    # Log file umano
    log_txt = (
        f"[FF] asof={now.isoformat(timespec='seconds')}\n"
        f"impact_min={NEWS_IMPACT_MIN} window=-{NEWS_WINDOW_BEFORE_MIN}/+{NEWS_WINDOW_AFTER_MIN}min\n"
        + ("\n".join(lines) if lines else "(nessun evento rilevante)")
        + "\n"
    )
    with open(os.path.join(LOGS_DIR, "news_gate.log"), "a", encoding="utf-8") as f:
        f.write(log_txt)

    print(f"[FF] blackout scritto in: {NEWS_BLACKOUT_FILE}")

    # Telegram (opzionale)
    if TG_TOKEN and TG_CHAT:
        if lines:
            msg = "📰 NewsGate (FF)\n" + "\n".join(lines[:12])
        else:
            msg = "📰 NewsGate (FF)\nNessun evento ad alto impatto nella finestra impostata."
        _send_tg(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FF] FATAL: {e}\n{traceback.format_exc()}")
