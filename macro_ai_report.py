# macro_ai_report.py — Macro AI Report con Gemini (auto-discovery modelli)
# - Legge params/macro_context.json
# - Auto-scopre i modelli (se possibile) e/o usa GEMINI_MODEL se fornito
# - Chiama generateContent con fallback v1beta/v1 e piccolo retry
# - Log su logs/macro_ai_report_<timestamp>.txt
# - Telegram opzionale con chunking

import os, json, time, requests
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.dirname(__file__))
PARAMS_DIR = os.path.join(ROOT, "params")
LOGS_DIR   = os.path.join(ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

MACRO_PATH = os.path.join(PARAMS_DIR, "macro_context.json")

# ---------- .env minimal-robusto ----------
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
    except Exception as e:
        print("⚠️ Errore lettura .env:", e)

DOTENV_PATH = os.path.join(ROOT, ".env")
if os.path.exists(DOTENV_PATH):
    load_env_file(DOTENV_PATH)

# Variabili ambiente
GEMINI_KEY   = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
USER_MODEL   = (os.getenv("GEMINI_MODEL") or "").strip()  # es: "gemini-1.5-flash-latest" o "models/gemini-1.5-flash-latest"
TG_TOKEN     = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TG_CHAT_ID   = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
TG_ENABLE    = os.getenv("MACRO_AI_TG", "1") == "1"

def utc_stamp(fmt="%Y%m%d_%H%M%S"):
    return datetime.now(timezone.utc).strftime(fmt)

def read_macro_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("⚠️ Errore lettura macro_context.json:", e)
        return None

# ---------- Telegram helper (chunking) ----------
def send_telegram(text):
    if not TG_ENABLE:
        print("ℹ️ Telegram disabilitato (MACRO_AI_TG=0).")
        return
    if not TG_TOKEN or not TG_CHAT_ID:
        print("ℹ️ Telegram non configurato (TOKEN/CHAT_ID mancanti).")
        return

    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        max_len = 4096
        blocks = [text[i:i+max_len] for i in range(0, len(text), max_len)] or [text]
        for idx, chunk in enumerate(blocks, 1):
            payload = {"chat_id": TG_CHAT_ID, "text": chunk, "disable_web_page_preview": True}
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code != 200:
                print(f"⚠️ Telegram API error: {r.text}")
                break
            if len(blocks) > 1:
                time.sleep(0.8)  # piccolo rate-limit
        else:
            print("✅ Messaggio Telegram inviato.")
    except Exception as e:
        print("⚠️ Errore invio Telegram:", e)

# ---------- Auto-discovery modelli ----------
def list_models():
    """Ritorna lista di dict modello da /v1/models oppure [] se fallisce (403/401 inclusi)."""
    if not GEMINI_KEY:
        return []
    url = f"https://generativelanguage.googleapis.com/v1/models?key={GEMINI_KEY}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            # Molti account restituiscono 403 su ListModels: non è fatale.
            print("ℹ️ ListModels non disponibile:", r.status_code, r.text[:160])
            return []
        data = r.json()
        return data.get("models", []) or []
    except Exception as e:
        print("ℹ️ ListModels conn error:", e)
        return []

def supports_generate_content(model_obj):
    methods = model_obj.get("supportedGenerationMethods") or []
    return ("generateContent" in methods) or ("generateText" in methods)

def choose_model(models):
    """Preferenza: utente -> 1.5-flash -> 1.5-pro -> 1.0-pro -> gemini-pro -> text-bison."""
    names = [m.get("name","") for m in models if supports_generate_content(m)]

    def normalize(name):
        return name if name.startswith("models/") else f"models/{name}"

    # 1) Se l'utente ha indicato un modello, prova quello (anche se ListModels è vuoto)
    if USER_MODEL:
        um = normalize(USER_MODEL)
        if not names or um in names:
            return um  # prova diretto

    # 2) Altrimenti scegli il migliore tra quelli visti
    prefs = [
        "gemini-1.5-flash-latest", "gemini-1.5-flash",
        "gemini-1.5-pro-latest",   "gemini-1.5-pro",
        "gemini-1.0-pro",          "gemini-pro",
        "text-bison-001"
    ]
    for p in prefs:
        cand = f"models/{p}"
        if cand in names:
            return cand

    # 3) fallback qualunque supporti generateContent
    return names[0] if names else (normalize(USER_MODEL) if USER_MODEL else None)

# ---------- Query Gemini con fallback + piccolo retry ----------
def _gemini_request(model_name, prompt, api_ver):
    base = "https://generativelanguage.googleapis.com"
    url  = f"{base}/{api_ver}/models/{model_name}:generateContent?key={GEMINI_KEY}"
    headers = {"Content-Type": "application/json"}
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    return requests.post(url, headers=headers, json=body, timeout=30)

def query_gemini(prompt, retries=1):
    if not GEMINI_KEY:
        return "❌ Chiave Gemini mancante (GEMINI_API_KEY/GOOGLE_API_KEY)."

    chosen = choose_model(list_models())
    if not chosen:
        return ("❌ Nessun modello compatibile trovato/fornito.\n"
                "Tip: imposta GEMINI_MODEL=gemini-1.5-flash-latest (o pro) e verifica la tua chiave.")

    model_name = chosen.split("/", 1)[-1]
    attempts = [("v1beta", model_name), ("v1", model_name)]

    last_err = None
    for api_ver, mdl in attempts:
        for attempt in range(retries + 1):
            try:
                r = _gemini_request(mdl, prompt, api_ver)
                if r.status_code == 200:
                    out = r.json()
                    # parsing robusto
                    cand = (out.get("candidates") or [{}])[0]
                    content = (cand.get("content") or {})
                    parts = content.get("parts") or []
                    text = parts[0].get("text") if parts else None
                    if isinstance(text, str) and text.strip():
                        return text
                    # se niente testo, ritorna il JSON minimizzato
                    return json.dumps(out, ensure_ascii=False)
                elif r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(1.0 + 0.5 * attempt)  # backoff leggero
                    continue
                else:
                    last_err = f"[{api_ver}/{mdl}] {r.status_code}: {r.text[:400]}"
                    break
            except Exception as e:
                last_err = f"[{api_ver}/{mdl}] conn error: {e}"
                if attempt < retries:
                    time.sleep(1.0 + 0.5 * attempt)
                    continue
                break

    return (f"Errore API Gemini: {last_err}\n"
            f"(Modello scelto: {chosen})\n"
            "Tip: controlla i permessi della chiave o prova un altro modello (GEMINI_MODEL).")

# ---------- Main ----------
def main():
    data = read_macro_json(MACRO_PATH)
    if not data:
        print("⚠️ macro_context.json non trovato o vuoto. Esegui prima macro_context.py.")
        ts = utc_stamp()
        out_path = os.path.join(LOGS_DIR, f"macro_ai_report_{ts}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("⚠️ Nessun macro_context disponibile.")
        print(f"📝 Report minimale generato: {out_path}")
        return

    prompt = f"""
Analizza questi dati macroeconomici giornalieri e fornisci un commento operativo
in 3-4 righe, focalizzato su rischio, forza USD, metalli, e possibile effetto sul Forex.

Dati:
- Risk state: {data.get('risk_state')}
- USD bias: {data.get('usd_bias')}
- Volatility state: {data.get('vol_state')}
- Commodities bias: {data.get('commodities_bias')}
- Confidence: {data.get('confidence')}
- Inputs: {data.get('inputs')}

Scrivi in tono analitico, sintetico e in italiano.
""".strip()

    commento = query_gemini(prompt, retries=1)
    if not commento or not isinstance(commento, str):
        commento = "❌ Nessuna risposta valida da Gemini."

    ts = utc_stamp()
    out_path = os.path.join(LOGS_DIR, f"macro_ai_report_{ts}.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(commento)
        print(f"✅ macro_ai_report generato: {out_path}")
    except Exception as e:
        print("⚠️ Errore salvataggio report:", e)

    if commento.strip():
        send_telegram(f"🌐 Macro AI Report\n{commento}")

if __name__ == "__main__":
    main()
