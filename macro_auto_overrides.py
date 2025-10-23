# macro_auto_overrides.py — v1.4
# - Legge l'ultimo report IA (logs/macro_ai_report_*.txt)
# - Estrae keyword con sinonimi e normalizzazione (lowercase, senza accenti)
# - Arricchisce le keyword con segnali numerici da params/macro_context.json
# - Applica guardrail USD/CAD:
#     se (gold_up OR commods_up OR gold5d>=+2%) e USD NON weak → NON sbloccare USD/CAD
# - Scrive params/macro_overrides.json
# - Manda Telegram con elenco keyword e sorgente (IA / IA+Context / Fallback)

import os, re, json, glob, unicodedata
from datetime import datetime, timezone
import requests

ROOT = os.path.abspath(os.path.dirname(__file__))
LOGS = os.path.join(ROOT, "logs")
PARAMS = os.path.join(ROOT, "params")

# ---------- ENV ----------
def load_env():
    p = os.path.join(ROOT, ".env")
    try:
        with open(p, "r", encoding="utf-8-sig", errors="ignore") as f:
            for ln in f:
                if "=" in ln and not ln.strip().startswith("#"):
                    k, v = ln.split("=", 1)
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except:
        pass
load_env()

TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TG_CHAT  = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

def send_tg(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "disable_web_page_preview": True},
            timeout=12
        )
    except:
        pass

def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- IO ----------
def normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # toglie accenti
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"[^a-z0-9 %/.,+-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_latest_report():
    files = glob.glob(os.path.join(LOGS, "macro_ai_report_*.txt"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_macro_context():
    p = os.path.join(PARAMS, "macro_context.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

# ---------- Sinonimi ----------
SYN = {
    "risk_on": [
        "risk on", "risk-on", "propensione al rischio", "contesto favorevole",
        "fase positiva", "bias rialzista", "moderato risk on", "riskon"
    ],
    "risk_off": [
        "risk off", "risk-off", "avversione al rischio", "fase difensiva",
        "contesto avverso", "flight to quality", "riskoff"
    ],
    "usd_strong": [
        "usd forte", "dollaro forte", "rafforzamento usd", "apprezzamento del dollaro",
        "dxy in rialzo", "dollaro in rialzo", "usd strength", "usd up"
    ],
    "usd_weak": [
        "usd debole", "dollaro debole", "deprezzamento usd", "flessione del dollaro",
        "dxy in calo", "usd weakness", "usd down", "pressione ribassista sul dollaro"
    ],
    "usd_neutral": [
        "usd neutro", "dollaro neutro", "neutralita usd", "usd neutral"
    ],
    "gold_up": [
        "oro in rialzo", "rally dell oro", "oro forte", "gold up", "rialzo dell oro",
        "metalli forti", "bias rialzista dei metalli", "forza dei metalli"
    ],
    "gold_down": [
        "oro in calo", "oro debole", "gold down", "flessione dell oro"
    ],
    "commods_up": [
        "commodities forti", "bias sulle commodities", "forza sulle commodities",
        "commodities in rialzo", "materie prime in rialzo", "commodity currencies forti"
    ],
    "commods_down": [
        "commodities deboli", "materie prime in calo", "commodities in calo"
    ],
    "vol_high": [
        "volatilita elevata", "volatilita alta", "spike di volatilita", "vix alto"
    ],
    "vol_normal": [
        "volatilita normale", "volatilita stabile", "vix normale", "vix stabile"
    ],
    "risk_high": [
        "rischio elevato", "rischio alto", "contesto rischioso", "risk elevated"
    ],
    "risk_normal": [
        "rischio normale", "contesto normale", "risk normal", "risk stable"
    ],
}

# ---------- Parser keyword ----------
def extract_keywords(text: str) -> set:
    t = normalize_text(text or "")
    hits = set()
    for key, syns in SYN.items():
        for s in syns:
            if s in t:
                hits.add(key)
                break
    # euristica: "oro +x% 5gg"
    if re.search(r"oro[^0-9+%-]*\+?\d+(\.\d+)?%%?\s*(?:5gg|5g|5 giorni|5d)", t):
        hits.add("gold_up")
    return hits

def augment_hits_from_context(hits: set, ctx: dict) -> (set, float):
    """Arricchisce le keyword IA con regole numeriche semplici dal macro_context.
       Ritorna (added_set, gold5d_value) per debug/guardrail."""
    added = set()
    gold5d_val = None
    if not ctx:
        return added, gold5d_val
    try:
        # stati discreti
        comm = (ctx.get("commodities_bias") or "").lower()
        if comm in ("up", "strong", "bullish") and "commods_up" not in hits:
            hits.add("commods_up"); added.add("commods_up")

        usd = (ctx.get("usd_bias") or "").lower()
        if usd in ("weak", "bearish") and "usd_weak" not in hits:
            hits.add("usd_weak"); added.add("usd_weak")
        elif usd in ("strong", "bullish") and "usd_strong" not in hits:
            hits.add("usd_strong"); added.add("usd_strong")
        elif usd in ("neutral",) and "usd_neutral" not in hits:
            hits.add("usd_neutral"); added.add("usd_neutral")

        # numerico: gold5d
        raw = ctx.get("gold5d")
        try:
            gold5d_val = float(raw)
        except:
            gold5d_val = None
        if (gold5d_val is not None) and (gold5d_val >= 0.02) and ("gold_up" not in hits):
            hits.add("gold_up"); added.add("gold_up")

    except Exception:
        pass
    return added, gold5d_val

# ---------- Costruzione overrides ----------
def build_overrides_from_keywords(hits: set) -> dict:
    overrides = {"notes": "auto from macro_ai_report", "global": {}, "asset_overrides": {}}

    # 1) Risk caps
    if ("risk_off" in hits) or ("risk_high" in hits) or ("vol_high" in hits):
        overrides["global"]["risk_caps_multiplier"] = 0.85
    elif ("risk_on" in hits) or ("risk_normal" in hits) or ("vol_normal" in hits):
        overrides["global"]["risk_caps_multiplier"] = 1.0  # reset/nessuna stretta

    # 2) USD bias e commodities/metalli
    if ("usd_weak" in hits) or ("commods_up" in hits) or ("gold_up" in hits):
        # favorisci valute commodity e metalli
        for a in ("GOLD","SILVER","AUD/USD","NZD/USD"):
            overrides["asset_overrides"][a] = {"allow_new_entries": True}
        # opzionale USD/CAD (verrà filtrato dal guardrail se serve)
        overrides["asset_overrides"].setdefault("USD/CAD", {"allow_new_entries": True})

    if ("usd_strong" in hits) or ("commods_down" in hits) or ("gold_down" in hits):
        # blocca long contro USD e metalli
        for a in ("GOLD","SILVER","EUR/USD","GBP/USD","AUD/USD","NZD/USD"):
            overrides["asset_overrides"][a] = {"allow_new_entries": False}

    return overrides

# ---------- MAIN ----------
def main():
    os.makedirs(PARAMS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    latest = find_latest_report()
    source = "IA"
    text = ""
    if latest:
        with open(latest, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    # 1) Keyword da IA (se c'è testo)
    hits: set = extract_keywords(text) if text else set()

    # 2) Carica SEMPRE macro_context e arricchisci le keyword con segnali numerici
    ctx = load_macro_context()
    added, gold5d_val = augment_hits_from_context(hits, ctx)
    if added:
        source = "IA+Context" if text else "Fallback"
        print("[DEBUG] augment from context:", sorted(added))
    if gold5d_val is not None:
        print(f"[DEBUG] gold5d from context: {gold5d_val:+.4f}")

    # 3) Costruisci gli overrides sulle hit finali
    if not hits:
        overrides, hits_fb = fallback_from_context(ctx)
        hits = hits_fb
        source = "Fallback"
    else:
        overrides = build_overrides_from_keywords(hits)

    # 4) 🔒 GUARDRAIL USD/CAD robusto:
    #    se (gold_up OR commods_up OR gold5d>=+2%) e USD NON weak → rimuovi USD/CAD
    gold_signal = ("gold_up" in hits) or ("commods_up" in hits) or ((gold5d_val is not None) and (gold5d_val >= 0.02))
    usd_not_weak = ("usd_weak" not in hits)
    if gold_signal and usd_not_weak:
        ao = overrides.get("asset_overrides", {})
        if "USD/CAD" in ao:
            ao.pop("USD/CAD", None)
            overrides["asset_overrides"] = ao
            print("[Overrides] Guardrail: (gold_up/commods_up/num) + USD not weak → USD/CAD rimosso")

    # 5) Scrittura file
    outp = os.path.join(PARAMS, "macro_overrides.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)

    # 6) Telegram
    hits_str = ", ".join(sorted(hits)) if hits else "—"
    msg = (
        "✅ Macro overrides aggiornati\n"
        f"Sorgente: {source}\n"
        f"Keyword: {hits_str}\n"
        f"Global caps: {overrides.get('global',{})}\n"
        f"Asset overrides: {', '.join(overrides.get('asset_overrides',{}).keys()) or 'nessuno'}"
    )
    send_tg(msg)

    print(f"macro_overrides.json aggiornato: {outp}")
    print(msg)

# ---------- Fallback numerico ----------
def fallback_from_context(ctx: dict) -> (dict, set):
    hits = set()
    overrides = {"notes": "fallback from macro_context", "global": {}, "asset_overrides": {}}
    if not ctx:
        return overrides, hits

    risk = (ctx.get("risk_state") or "").lower()
    vol  = (ctx.get("vol_state") or "").lower()
    usd  = (ctx.get("usd_bias") or "").lower()
    comm = (ctx.get("commodities_bias") or "").lower()

    if risk in ("high","elevated") or vol in ("high","elevated"):
        overrides["global"]["risk_caps_multiplier"] = 0.85; hits |= {"risk_high","vol_high"}
    elif risk in ("normal","low") and vol in ("normal","low"):
        overrides["global"]["risk_caps_multiplier"] = 1.0; hits |= {"risk_normal","vol_normal"}

    if usd in ("weak","bearish"):
        hits.add("usd_weak")
        for a in ("GOLD","SILVER","AUD/USD","NZD/USD"):
            overrides["asset_overrides"][a] = {"allow_new_entries": True}
    elif usd in ("strong","bullish"):
        hits.add("usd_strong")
        for a in ("GOLD","SILVER","EUR/USD","GBP/USD","AUD/USD","NZD/USD"):
            overrides["asset_overrides"][a] = {"allow_new_entries": False}

    if comm in ("up","strong","bullish"):
        hits.add("commods_up")
        for a in ("GOLD","SILVER","AUD/USD","NZD/USD"):
            overrides["asset_overrides"][a] = {"allow_new_entries": True}
    elif comm in ("down","weak","bearish"):
        hits.add("commods_down")
        for a in ("GOLD","SILVER","EUR/USD","GBP/USD","AUD/USD","NZD/USD"):
            overrides["asset_overrides"][a] = {"allow_new_entries": False}

    # euristica oro numerico
    raw = ctx.get("gold5d")
    try:
        g = float(raw)
    except:
        g = None
    if g is not None and g >= 0.02:
        hits.add("gold_up")
        overrides["asset_overrides"].setdefault("GOLD", {"allow_new_entries": True})
        overrides["asset_overrides"].setdefault("SILVER", {"allow_new_entries": True})

    return overrides, hits

if __name__ == "__main__":
    main()
