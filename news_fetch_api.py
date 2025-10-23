# news_fetch_rapidapi.py — Economic Events -> params/news_blackout.json (robusto + finestre)
import os, json, requests
from datetime import datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.dirname(__file__))
PARAMS_DIR = os.path.join(ROOT, "params")
os.makedirs(PARAMS_DIR, exist_ok=True)

# --- ENV ---
API_KEY = (os.getenv("RAPIDAPI_KEY") or "").strip()
HOST = (os.getenv("RAPIDAPI_HOST") or "economic-events-calendar.p.rapidapi.com").strip()

# es: "US, GB, EU"
COUNTRIES = [x.strip() for x in (os.getenv("NEWS_COUNTRIES", "US,GB,EU").split(",")) if x.strip()]
MIN_IMPACT = (os.getenv("NEWS_MIN_IMPACT", "high") or "high").strip().lower()  # low|medium|high
PAD_MIN = int(os.getenv("NEWS_BLACKOUT_PAD_MIN", "30"))  # minuti prima/dopo per finestra blackout

if not API_KEY:
    raise SystemExit("❌ RAPIDAPI_KEY mancante (mettila nel .env o nell'ambiente).")

# --- Helpers ---
def _impact_pass(value: str) -> bool:
    v = (value or "").lower()
    order = {"low": 0, "medium": 1, "high": 2}
    return order.get(v, -1) >= order.get(MIN_IMPACT, 2)

def _parse_utc(s: str):
    """Prova a convertire stringa -> datetime UTC; se fallisce, None."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    # Prova alcuni formati comuni
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            # se manca il tempo, mettiti a mezzogiorno UTC per sicurezza
            if fmt == "%Y-%m-%d":
                dt = dt.replace(hour=12, minute=0, second=0)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None

def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

# --- API call ---
url = f"https://{HOST}/economic-events/tradingview"
headers = {"x-rapidapi-key": API_KEY, "x-rapidapi-host": HOST}
params = {"countries": ",".join(COUNTRIES)}

print(f"[FF] Scarico calendario economico per: {params['countries']} (min_impact={MIN_IMPACT}, pad={PAD_MIN}m)")
try:
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
except Exception as e:
    raise SystemExit(f"❌ Errore API: {e!r}")

# La risposta può essere { "result": [...] } oppure direttamente [...]
raw = []
if isinstance(data, dict):
    raw = data.get("result") or data.get("data") or []
elif isinstance(data, list):
    raw = data

if not isinstance(raw, list):
    raw = []

# --- Filtra + costruisci eventi e finestre ---
events = []
windows = []

for ev in raw:
    if not isinstance(ev, dict):
        continue
    impact = str(ev.get("impact", "")).lower()
    if not _impact_pass(impact):
        continue

    # prendi una data/ora plausibile
    when = _parse_utc(ev.get("date") or ev.get("time") or ev.get("datetime"))
    cur = (ev.get("currency") or "").strip().upper()
    name = (ev.get("event") or ev.get("title") or "").strip()

    item = {
        "currency": cur or None,
        "event": name or None,
        "impact": impact or None,
        "time_raw": ev.get("date") or ev.get("time") or ev.get("datetime"),
        "time_utc": _iso(when) if when else None,
    }
    events.append(item)

    if when:
        start = when - timedelta(minutes=PAD_MIN)
        end = when + timedelta(minutes=PAD_MIN)
        windows.append({
            "start_utc": _iso(start),
            "end_utc": _iso(end),
            "currency": cur or None,
            "event": name or None,
            "impact": impact or None,
        })

# ordina per tempo se disponibile
events.sort(key=lambda x: x.get("time_utc") or "Z")
windows.sort(key=lambda x: x.get("start_utc") or "Z")

# --- Salva ---
out = {
    "asof_utc": _iso(datetime.now(timezone.utc)),
    "countries": COUNTRIES,
    "min_impact": MIN_IMPACT,
    "pad_min": PAD_MIN,
    "events": events,
    "windows": windows,
}
out_path = os.path.join(PARAMS_DIR, "news_blackout.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"✅ news_blackout.json aggiornato: {len(events)} eventi, {len(windows)} finestre → {out_path}")
