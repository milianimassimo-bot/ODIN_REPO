# macro_knobs_advisor.py — 5 profili "manopole" giornalieri, con suggerimento Telegram
import os, json, requests
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(ROOT, "params")
MACRO_PROFILES = os.path.join(PARAMS_DIR, "macro_profiles.json")

# --- .env semplice (opzionale ma utile in Windows/script)
def load_env_file(path):
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        pass

load_env_file(os.path.join(ROOT, ".env"))

TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID   = (os.getenv("TELEGRAM_CHAT_ID")   or "").strip()

def utc_stamp(fmt="%Y-%m-%dT%H:%M:%SZ"):
    return datetime.now(timezone.utc).strftime(fmt)

def send_tg(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] Non configurato (TOKEN/CHAT_ID mancanti).")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
            timeout=15,
        )
        ok = (r.status_code == 200)
        if not ok:
            print(f"[TG] Errore HTTP {r.status_code}: {r.text[:200]}")
        return ok
    except Exception as e:
        print(f"[TG] eccezione: {e!r}")
        return False

def profiles(equity: float):
    # 5 profili base; se vuoi, puoi usare 'equity' per decidere dinamicamente
    return [
        ("UltraSoft", 0.6, ["EUR/USD","GBP/USD","USD/JPY","AUD/USD"]),
        ("Soft",      0.8, ["EUR/USD","GBP/USD","USD/JPY","AUD/USD"]),
        ("Neutral",   1.0, ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","GOLD","SILVER"]),
        ("Active",    1.2, ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","GOLD","SILVER","SP500"]),
        ("Push",      1.4, ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","GOLD","SILVER","SP500"]),
    ]

def atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def main():
    # equity robusto
    try:
        equity = float(os.getenv("ODIN_EQUITY", "10000"))
    except Exception:
        equity = 10000.0

    profs = profiles(equity)
    ts = utc_stamp()

    # euristica semplice per suggerire un profilo (puoi affinarla)
    # es: < 8k UltraSoft; < 12k Soft; < 20k Neutral; < 35k Active; altrimenti Push
    if equity < 8000:    idx = 0
    elif equity < 12000: idx = 1
    elif equity < 20000: idx = 2
    elif equity < 35000: idx = 3
    else:                idx = 4

    name, mult, allowed = profs[idx]
    sugg = {
        "suggested_utc": ts,
        "profile": name,
        "risk_caps_multiplier": mult,
        "allow_entries": allowed,
    }

    payload = {
        "profiles": [{"name": n, "mult": m, "allow": a} for n, m, a in profs],
        "suggestion": sugg,
    }
    atomic_write_json(MACRO_PROFILES, payload)

    lines = [
        f"🎛️ Macro Advisor ({ts})",
        "Profili: UltraSoft / Soft / Neutral / Active / Push",
        f"Suggerito: **{name}** (mult={mult})",
        "Allow entries: " + ", ".join(allowed),
        "Per applicare:  python macro_apply.py <UltraSoft|Soft|Neutral|Active|Push>",
    ]
    send_tg("\n".join(lines))
    print(f"Salvato suggerimento in {MACRO_PROFILES}")

if __name__ == "__main__":
    main()
