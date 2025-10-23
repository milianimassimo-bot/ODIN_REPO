# macro_context.py — Macro overview v1 per ODIN
# - Produce params/macro_context.json (sempre)
# - Se ENABLE_MACRO_OVERRIDES=true genera params/macro_overrides.json (consigli "freno a mano")
# - Facoltativo: manda un messaggio Telegram (usa TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID da .env)

import os, json
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.dirname(__file__))
PARAMS_DIR = os.path.join(ROOT, "params")
LOGS_DIR = os.path.join(ROOT, "logs")
os.makedirs(PARAMS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Carica .env se presente (senza dipendenze extra)
DOTENV_PATH = os.path.join(ROOT, ".env")
if os.path.exists(DOTENV_PATH):
    try:
        for line in open(DOTENV_PATH, "r", encoding="utf-8", errors="ignore"):
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line: 
                continue
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and (k not in os.environ):
                os.environ[k] = v
    except Exception:
        pass

ENABLE_MACRO_OVERRIDES = os.getenv("ENABLE_MACRO_OVERRIDES", "false").strip().lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def ts_utc(): 
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def log(msg):
    print(msg)

def pct_change(cur, prev):
    try:
        return (cur/prev - 1.0)
    except Exception:
        return None

def classify_risk(vix_last, spx_5d):
    if vix_last is None or spx_5d is None:
        return "normal"
    if vix_last >= 22 or spx_5d <= -0.02:
        return "high"
    if vix_last <= 14 and spx_5d >= 0.01:
        return "low"
    return "normal"

def classify_usd(eurusd_5d):
    if eurusd_5d is None: 
        return "neutral"
    if eurusd_5d <= -0.005:  # EUR scende → USD strong
        return "strong"
    if eurusd_5d >= 0.005:
        return "weak"
    return "neutral"

def classify_vol(vix_last):
    if vix_last is None:
        return "normal"
    if vix_last >= 22:
        return "elevated"
    if vix_last <= 14:
        return "subdued"
    return "normal"

def classify_commodities(gold_5d):
    if gold_5d is None:
        return "neutral"
    if gold_5d >= 0.01:
        return "strong"
    if gold_5d <= -0.01:
        return "soft"
    return "neutral"

def fetch_yf(symbol, start, end):
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
        if df is None or df.empty: 
            return None
        return df["Close"].dropna()
    except Exception:
        return None

def last(series, n=1):
    if series is None or len(series) < n: 
        return None
    return float(series.iloc[-n])

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("[Telegram] token/chat_id non configurati — skip")
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            log(f"[Telegram] Errore API: {r.text}")
        else:
            log("[Telegram] Messaggio inviato.")
    except Exception as e:
        log(f"[Telegram] Errore invio: {e}")

def main():
    end = datetime.utcnow()
    start = end - timedelta(days=120)

    syms = {
        "spx": "^GSPC",      # S&P500
        "vix": "^VIX",       # VIX
        "eurusd": "EURUSD=X",
        "gold": "GC=F"
    }

    data = {}
    for k, s in syms.items():
        data[k] = fetch_yf(s, start, end)

    spx_last   = last(data["spx"])
    spx_prev6  = last(data["spx"], 6)    # 5 giorni indietro -> indice 6 dall'ultimo
    spx_5d     = pct_change(spx_last, spx_prev6) if (spx_last and spx_prev6) else None

    vix_last   = last(data["vix"])

    eur_last   = last(data["eurusd"])
    eur_prev6  = last(data["eurusd"], 6)
    eur_5d     = pct_change(eur_last, eur_prev6) if (eur_last and eur_prev6) else None

    gold_last  = last(data["gold"])
    gold_prev6 = last(data["gold"], 6)
    gold_5d    = pct_change(gold_last, gold_prev6) if (gold_last and gold_prev6) else None

    risk_state = classify_risk(vix_last, spx_5d)
    usd_bias   = classify_usd(eur_5d)
    vol_state  = classify_vol(vix_last)
    com_bias   = classify_commodities(gold_5d)

    # confidence semplice (quante spie “accese”)
    score = 0
    if risk_state == "high": score += 1
    if vol_state  == "elevated": score += 1
    if usd_bias   in ("strong","weak"): score += 1
    if com_bias   in ("strong","soft"): score += 1
    confidence = min(0.5 + 0.1*score, 0.95)

    summary = f"Risk={risk_state} | USD={usd_bias} | Vol={vol_state} | Commd={com_bias}"

    macro = {
        "timestamp": ts_utc(),
        "risk_state": risk_state,               # low|normal|high
        "usd_bias": usd_bias,                   # weak|neutral|strong
        "vol_state": vol_state,                 # subdued|normal|elevated
        "commodities_bias": com_bias,           # strong|neutral|soft
        "confidence": round(confidence, 2),
        "bias_summary": summary,
        "inputs": {
            "spx_5d_ret":  round(spx_5d, 4) if spx_5d is not None else None,
            "vix_last":    round(vix_last, 2) if vix_last is not None else None,
            "eurusd_5d":   round(eur_5d, 4) if eur_5d is not None else None,
            "gold_5d":     round(gold_5d, 4) if gold_5d is not None else None
        }
    }

    out = os.path.join(PARAMS_DIR, "macro_context.json")
    save_json(out, macro)
    log(f"macro_context.json aggiornato: {out}")
    # log di cortesia
    try:
        save_json(os.path.join(LOGS_DIR, f"macro_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), macro)
    except Exception:
        pass

    # --- (OPZIONALE) Genera macro_overrides se abilitato ---
    if ENABLE_MACRO_OVERRIDES:
        overrides = {"notes": "auto from macro_context", "global": {}, "asset_overrides": {}}

        # 1) Volatilità elevata → riduci caps del 15%
        if vol_state == "elevated" or risk_state == "high":
            overrides["global"]["risk_caps_multiplier"] = 0.85

        # 2) USD strong → evita nuove entrate su GOLD/SILVER; USD weak → opposto
        if usd_bias == "strong":
            overrides["asset_overrides"]["GOLD"]   = {"allow_new_entries": False}
            overrides["asset_overrides"]["SILVER"] = {"allow_new_entries": False}
        elif usd_bias == "weak":
            # esempio: sblocca l'eventuale blocco su GOLD/SILVER
            overrides["asset_overrides"]["GOLD"]   = {"allow_new_entries": True}
            overrides["asset_overrides"]["SILVER"] = {"allow_new_entries": True}

        # 3) Commodities soft → ancora più prudenza su metalli
        if com_bias == "soft":
            overrides["asset_overrides"].setdefault("GOLD", {})["allow_new_entries"] = False
            overrides["asset_overrides"].setdefault("SILVER", {})["allow_new_entries"] = False

        # salva solo se c'è almeno una regola
        if (overrides.get("global") or overrides.get("asset_overrides")):
            mo_path = os.path.join(PARAMS_DIR, "macro_overrides.json")
            save_json(mo_path, overrides)
            log(f"macro_overrides.json generato: {mo_path}")

    # ---- Telegram (facoltativo) ----
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram(f"🌍 MacroContext\n{summary}\nconf={macro['confidence']}\n"
                      f"spx5d={macro['inputs']['spx_5d_ret']} vix={macro['inputs']['vix_last']} "
                      f"eur5d={macro['inputs']['eurusd_5d']} gold5d={macro['inputs']['gold_5d']}")

if __name__ == "__main__":
    main()
