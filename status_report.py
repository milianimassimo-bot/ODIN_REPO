# status_report.py — v1.0
# Snapshot operativo ODIN (11:55 / 21:55):
# - Equity/Expo/Trades (da MT5 se disponibile, altrimenti N/A)
# - Asset sbloccati (macro_overrides + current.json)
# - Regimi attuali per universo principale (se odin_regime_report_d1.json presente)
# - MacroContext (risk/usd/vol/commods + numerici chiave)
# - Watchdog policy (estratto)
# - Salva log e invia Telegram

import os, json, glob
from datetime import datetime, timezone
from textwrap import shorten

# --- env loader minimale (senza dipendenze)
ROOT = os.path.abspath(os.path.dirname(__file__))
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
    except: pass
load_env()

import requests

TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TG_CHAT  = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

def send_tg(text: str):
    if not TG_TOKEN or not TG_CHAT:
        return False, "no-token-or-chatid"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "disable_web_page_preview": True},
            timeout=12
        )
        ok = (r.status_code == 200)
        return ok, (None if ok else r.text)
    except Exception as e:
        return False, str(e)

def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

PARAMS = os.path.join(ROOT, "params")
LOGS   = os.path.join(ROOT, "logs")

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def find_latest_regime_json():
    # usa odin_regime_report_d1.json se presente; altrimenti il più recente matching *_regime*.json
    p = os.path.join(ROOT, "odin_regime_report_d1.json")
    if os.path.exists(p): return p
    cands = glob.glob(os.path.join(ROOT, "*regime*.json"))
    return max(cands, key=os.path.getmtime) if cands else None

def mt5_snapshot():
    """Raccoglie equity / posizioni se MetaTrader5 disponibile. Soft-fail."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None
        acc_info = mt5.account_info()
        equity = float(acc_info.equity) if acc_info else None
        positions = mt5.positions_get()
        expo = 0.0
        npos = 0
        if positions:
            npos = len(positions)
            # TV = somma valore nominale rischiato non banale → qui semplifichiamo
            # expo come numero posizioni (placeholder)
            expo = float(npos)
        mt5.shutdown()
        return {"equity": equity, "positions": npos, "expo_hint": expo}
    except Exception:
        return None

def main():
    os.makedirs(LOGS, exist_ok=True)

    curr = load_json(os.path.join(PARAMS, "current.json")) or {}
    macro_over = load_json(os.path.join(PARAMS, "macro_overrides.json")) or {}
    macro_ctx = load_json(os.path.join(PARAMS, "macro_context.json")) or {}

    # global policy estratto
    g = curr.get("global", {})
    wd = (g.get("watchdog_policy") or {})
    risk_caps = (g.get("risk_caps") or {})
    paused = bool(g.get("paused", False))

    # asset sbloccati
    ao_curr = (curr.get("asset_overrides") or {})
    ao_macro = (macro_over.get("asset_overrides") or {})
    allow_assets = sorted({a for a, v in ao_curr.items() if v.get("allow_new_entries")}
                          | {a for a, v in ao_macro.items() if v.get("allow_new_entries")})

    # regime (se file)
    regimes_line = "N/A"
    rpath = find_latest_regime_json()
    if rpath:
        rj = load_json(rpath) or {}
        items = rj.get("items") or []
        key_assets = ("GOLD","SILVER","EUR/USD","AUD/USD","NZD/USD","USD/JPY","SP500")
        pairs = []
        for it in items:
            a = it.get("asset")
            if a in key_assets:
                reg = it.get("regime")
                m = it.get("metrics") or {}
                atrp = m.get("atr_pct")
                if atrp is not None:
                    pairs.append(f"{a}:{reg} (ATR% {atrp:.2f})")
                else:
                    pairs.append(f"{a}:{reg}")
        if pairs:
            regimes_line = " | ".join(pairs)

    # macro quick
    risk_state = macro_ctx.get("risk_state")
    vol_state  = macro_ctx.get("vol_state")
    usd_bias   = macro_ctx.get("usd_bias")
    comm_bias  = macro_ctx.get("commodities_bias")
    spx5d = macro_ctx.get("spx5d"); vix = macro_ctx.get("vix"); eur5d = macro_ctx.get("eur5d"); gold5d = macro_ctx.get("gold5d")

    # mt5
    mt5_info = mt5_snapshot()
    if mt5_info:
        eq = f"€{mt5_info['equity']:.2f}" if mt5_info['equity'] is not None else "N/A"
        pos = str(mt5_info['positions'])
        expo = f"{mt5_info['expo_hint']:.0f} pos"
    else:
        eq = "N/A"; pos = "N/A"; expo = "N/A"

    # compose message
    ts = utcnow()
    msg = (
        f"🛰️ ODIN / STATUS — {ts}\n"
        f"Equity={eq} | Expo={expo} | Trades open={pos}\n"
        f"Paused={'YES' if paused else 'NO'} | RiskCaps g/b/a/t={risk_caps.get('global', '—')}/{risk_caps.get('bucket','—')}/{risk_caps.get('asset','—')}/{risk_caps.get('trade','—')}\n"
        f"Allow entries: {', '.join(allow_assets) if allow_assets else '—'}\n"
        f"Regimes: {shorten(regimes_line, width=180, placeholder='…')}\n"
        f"Macro: Risk={risk_state or '—'} | USD={usd_bias or '—'} | Vol={vol_state or '—'} | Commd={comm_bias or '—'}\n"
        f"Macro nums: spx5d={spx5d if spx5d is not None else '—'} vix={vix if vix is not None else '—'} eur5d={eur5d if eur5d is not None else '—'} gold5d={gold5d if gold5d is not None else '—'}\n"
        f"WD: BE={wd.get('breakeven',{})} | TRL={wd.get('trailing',{})} | CooloffH={wd.get('cooloff_hours_after_regime_mismatch','—')} | MaxAct={wd.get('max_daily_actions','—')}"
    )

    # save log
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    outp = os.path.join(LOGS, f"status_{stamp}.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(msg)

    ok, err = send_tg(msg)
    if ok:
        print(f"✅ Status inviato e salvato: {outp}")
    else:
        print(f"⚠️ Status salvato ma Telegram NO: {err}\nFile: {outp}")

if __name__ == "__main__":
    main()
