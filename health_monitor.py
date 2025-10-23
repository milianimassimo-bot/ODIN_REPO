# health_monitor.py — ODIN Health Monitor v1.0
# Controlli:
# - MT5 raggiungibile e account info
# - Staleness file chiave (regime, params, macro_overrides)
# - .env variabili critiche presenti
# - Log rotation: avvisa se log supera soglia
# - Spazio disco, tempo macchina
# - Telegram report (auto-chunk)

import os, sys, json, time, shutil, fnmatch
from datetime import datetime, timezone, timedelta

import requests
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ROOT = os.path.abspath(os.path.dirname(__file__))
if load_dotenv:
    try:
        load_dotenv(dotenv_path=os.path.join(ROOT, ".env"), override=True)
    except Exception:
        pass

PARAMS_DIR = os.path.join(ROOT, "params")
LOGS_DIR   = os.path.join(ROOT, "logs")

REGIME_JSON = os.getenv("ODIN_REGIME_JSON", os.path.join(ROOT, "odin_regime_report_d1.json"))
PARAMS_CURR = os.getenv("ODIN_PARAMS_FILE", os.path.join(PARAMS_DIR, "current.json"))
MACRO_OVR   = os.path.join(PARAMS_DIR, "macro_overrides.json")

TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TG_CHAT  = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

# soglie configurabili via env
STALE_REGIME_MIN = int(os.getenv("HEALTH_STALE_REGIME_MIN", "180"))   # 3 ore
STALE_PARAMS_MIN = int(os.getenv("HEALTH_STALE_PARAMS_MIN", "1440"))  # 24 ore
STALE_MACRO_MIN  = int(os.getenv("HEALTH_STALE_MACRO_MIN",  "1440"))  # 24 ore
LOG_MAX_MB       = float(os.getenv("HEALTH_LOG_MAX_MB", "5"))         # 5 MB
USE_TG           = os.getenv("HEALTH_TG", "1") == "1"                 # 0 per non inviare
EXIT_ON_FAIL     = os.getenv("HEALTH_EXIT_ON_FAIL", "1") == "1"       # exit code !=0 se problemi


def _utcnow():
    return datetime.now(timezone.utc)


def _age_minutes(path):
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        return (_utcnow() - mtime).total_seconds() / 60.0
    except Exception:
        return None


def _exists(path):
    return os.path.exists(path)


def _tg_send(text):
    global TG_TOKEN, TG_CHAT
    if not USE_TG:
        print("[TG] disabilitato (HEALTH_TG=0)")
        return False

    TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or TG_TOKEN or "").strip()
    TG_CHAT  = (os.getenv("TELEGRAM_CHAT_ID")   or TG_CHAT  or "").strip()

    if not TG_TOKEN or not TG_CHAT:
        print(f"[TG] missing token/chat (token_len={len(TG_TOKEN)}, chat='{TG_CHAT}')")
        return False
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        ok = r.ok
        print(f"[TG] http={r.status_code} ok={ok} resp='{r.text[:200]}'")
        return ok
    except Exception as e:
        print(f"[TG] exception: {e!r}")
        return False


def _tg_report(title, lines):
    CH = 3600
    head = f"🩺 ODIN / HEALTH — {title}\n"
    block = head + "\n".join(lines)
    if len(block) <= CH:
        _tg_send(block)
        return
    buf = head
    for ln in lines:
        if len(buf) + len(ln) + 1 > CH:
            _tg_send(buf.rstrip())
            buf = "…continua…\n" + ln + "\n"
        else:
            buf += ln + "\n"
    if buf.strip():
        _tg_send(buf.rstrip())


def _mt5_check():
    try:
        import MetaTrader5 as mt5
    except Exception:
        return False, "MetaTrader5 lib non importabile"
    try:
        if not mt5.initialize():
            return False, f"mt5.initialize() fallita: {mt5.last_error()}"
        ai = mt5.account_info()
        if not ai:
            mt5.shutdown()
            return False, "account_info() None"
        eq = float(ai.equity)
        ccy = ai.currency
        mt5.shutdown()
        return True, f"MT5 OK — equity={eq:.2f} {ccy}"
    except Exception as e:
        try:
            mt5.shutdown()
        except Exception:
            pass
        return False, f"MT5 errore: {type(e).__name__}: {e}"


def chk(path, max_min, label, warnings):
    if not _exists(path):
        warnings.append(f"{label}: MANCANTE ({path})")
        return False
    age = _age_minutes(path)
    if age is None:
        warnings.append(f"{label}: età sconosciuta ({path})")
        return False
    if age > max_min:
        warnings.append(f"{label}: STALE {age:.0f} min (> {max_min}) — {path}")
        return False
    return True


def chk_json(path, label, warnings):
    if not _exists(path):
        warnings.append(f"{label}: MANCANTE ({path})")
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception as e:
        warnings.append(f"{label}: JSON non valido ({e})")
        return False


def main():
    lines = []
    ok = True

    # 1) MT5
    mt5_ok, mt5_msg = _mt5_check()
    lines.append(f"• MT5: {mt5_msg}")
    ok = ok and mt5_ok

    # 2) .env variables critiche
    required_env = ["TWELVEDATA_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ODIN_MAGIC"]
    missing = [k for k in required_env if not (os.getenv(k) or "").strip()]
    if missing:
        lines.append(f"• ENV: MANCANO {missing}")
        ok = False
    else:
        lines.append("• ENV: OK")

    # 3) Staleness + 3b) Validità JSON
    stale_warn = []
    ok &= chk(REGIME_JSON, STALE_REGIME_MIN, "Regime report D1", stale_warn)
    ok &= chk(PARAMS_CURR, STALE_PARAMS_MIN, "Params current.json", stale_warn)
    ok &= chk(MACRO_OVR,   STALE_MACRO_MIN,  "Macro overrides.json", stale_warn)

    ok &= chk_json(PARAMS_CURR, "Params current.json", stale_warn)
    ok &= chk_json(MACRO_OVR,   "Macro overrides.json", stale_warn)

    if stale_warn:
        lines.append("• STALENESS:")
        lines += [f"   - {w}" for w in stale_warn]
    else:
        lines.append("• STALENESS: OK")

    # 4) Log size & rotation hint
    big = []
    os.makedirs(LOGS_DIR, exist_ok=True)
    patterns = ("odin_main.log", "wd_actions.csv", "status_*.txt", "macro_ai_report_*.txt")
    for fn in os.listdir(LOGS_DIR):
        p = os.path.join(LOGS_DIR, fn)
        if not os.path.isfile(p):
            continue
        if any(fnmatch.fnmatch(fn, pat) for pat in patterns):
            if os.path.getsize(p) > LOG_MAX_MB * 1024 * 1024:
                big.append(p)
    if big:
        lines.append(f"• LOG: file grandi (>{LOG_MAX_MB:.0f}MB):")
        lines += [f"   - {p}" for p in big]
    else:
        lines.append("• LOG: OK")

    # 5) Disco
    total, used, free = shutil.disk_usage(ROOT)
    free_gb = free / (1024 ** 3)
    if free_gb < 2.0:
        lines.append(f"• DISK: Low free space {free_gb:.2f} GB")
        ok = False
    else:
        lines.append(f"• DISK: {free_gb:.2f} GB liberi")

    # 6) Orario macchina
    lines.append(f"• System UTC: {_utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    title = "OK" if ok else "ATTENZIONE"
    _tg_report(title, lines)
    print("\n".join([f"[HEALTH] {l}" for l in lines]))

    if EXIT_ON_FAIL and not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
