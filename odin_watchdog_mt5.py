# odin_watchdog_mt5.py
# ======================================================
# ODIN Watchdog (Sentinella MT5) — Report Telegram + Watchdog++ + ML Logger
# ======================================================

import os, re, json, logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

import requests

# ML logger (file: odin_ml_logger.py)
try:
    from odin_ml_logger import log_outcome   # scrive su logs/ml_trades_log.jsonl
except Exception:
    def log_outcome(**kwargs):
        pass  # fallback silenzioso

load_dotenv()

# ---------- ENV & CONFIG ----------
MAGIC_NUMBER = int(os.getenv("ODIN_MAGIC", "231120"))
ENV          = os.getenv("ODIN_ENV", "DEV")

ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.getenv("ODIN_PARAMS_FILE",
                        os.path.join(ROOT_DIR, "params", "current.json"))
REGIME_JSON = os.getenv("ODIN_REGIME_JSON",
                        os.path.join(ROOT_DIR, "odin_regime_report_d1.json"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

WD_STATE_FILE        = os.path.join(ROOT_DIR, "params", "wd_state.json")
WD_ACTIONS_CSV       = os.path.join(ROOT_DIR, "logs", "wd_actions.csv")
WD_COOLDOWN_HOURS    = int(os.getenv("WD_COOLDOWN_HOURS", "8"))
WD_MAX_DAILY_ACTIONS = int(os.getenv("WD_MAX_DAILY_ACTIONS", "6"))
WD_ENABLE_BREAKEVEN  = os.getenv("WD_ENABLE_BREAKEVEN", "true").lower() == "true"
WD_ENABLE_TRAILING   = os.getenv("WD_ENABLE_TRAILING", "true").lower() == "true"
WD_TIME_STOP_DAYS    = float(os.getenv("WD_TIME_STOP_DAYS", "6"))

# ---------- LOG ----------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("odin_watchdog")

# ---------- MT5 ----------
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    mt5 = None
    MT5_AVAILABLE = False

# ======================================================
# UTILS
# ======================================================
def _utcnow():
    return datetime.now(timezone.utc)

def _load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _append_csv(path, row: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    need_hdr = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if need_hdr:
            f.write("ts,symbol,action,detail,before,after,R,regime\n")
        f.write(row.rstrip("\n") + "\n")

def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

# Telegram helpers
def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID,
                                 "text": text, "disable_web_page_preview": True},
                      timeout=10)
    except Exception as e:
        log.warning(f"[TG] errore invio: {e!r}")

def send_telegram_report(header: str, lines: list[str]):
    msg = header + "\n" + ("\n".join(lines) if lines else "—")
    send_telegram(msg)

# regime lookup dal JSON di regime (se disponibile)
def get_regime_label(asset: str) -> str:
    try:
        data = _load_json(REGIME_JSON, {})
        for it in data.get("items", []):
            if it.get("asset") == asset:
                return it.get("regime", "N/A")
    except Exception:
        pass
    return "N/A"

# interpreta commento ordine "ODIN_<STRAT>_<ASSET>" o "ODIN / STRAT / ASSET"
def parse_comment(comment: str):
    if not comment:
        return ("?", "?")
    m = re.search(r"ODIN[_/ ]+([A-Z_]+)[_/ ]+([\w/]+)", comment.strip(), re.I)
    if m:
        return (m.group(1).upper(), m.group(2).upper())
    # fallback
    parts = comment.strip().split("_")
    if len(parts) >= 3:
        return (parts[1].upper(), parts[2].upper())
    return ("?", "?")

# chiusura a mercato
def close_position_market(ticket: int, symbol: str, volume: float, direction: str) -> bool:
    try:
        if not MT5_AVAILABLE:
            return False
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        order_type = (mt5.ORDER_TYPE_SELL if direction.upper() == "LONG"
                      else mt5.ORDER_TYPE_BUY)
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": f"WD_CLOSE_{ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        res = mt5.order_send(req)
        return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)
    except Exception as e:
        log.warning(f"close_position_market error: {e!r}")
        return False

# SL update
def update_sl(symbol: str, position_ticket: int, new_sl: float) -> bool:
    try:
        if not MT5_AVAILABLE:
            return False
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position_ticket,
            "symbol": symbol,
            "sl": float(new_sl),
            "tp": 0.0
        }
        res = mt5.order_send(req)
        return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)
    except Exception as e:
        log.warning(f"update_sl error: {e!r}")
        return False

# ======================================================
# MISMATCH HANDLER (fix: niente elif “volante”, struttura corretta)
# ======================================================
def wd_handle_mismatch(ticket, symbol, pos, entry, cur_sl, volume,
                       direction, regime, step_prev, r_mult):
    """
    step 1: avvisa e mette cooldown
    step 2: alza SL verso BE se consentito
    step 3+: se pnl<0 oppure R<0.3 → chiudi (ML log + CSV + Telegram)
    """
    step = max(1, int(step_prev or 0) + 1)
    changed = False

    if step == 1:
        send_telegram(f"#MISMATCH 1/3 | {symbol} | regime={regime}")
        changed = True

    elif step == 2 and WD_ENABLE_BREAKEVEN:
        # porta SL a break-even se possibile
        if direction == "LONG":
            be = entry
            if cur_sl is None or be > cur_sl:
                ok = update_sl(symbol, int(pos.ticket), be)
                if ok:
                    _append_csv(WD_ACTIONS_CSV,
                                f"{_utcnow().isoformat()},{symbol},MISMATCH_BE,step2,{cur_sl},{be},,{regime}")
                    send_telegram(f"#MISMATCH 2/3 | {symbol} | SL→BE")
                    changed = True
        else:
            be = entry
            if cur_sl is None or be < cur_sl:
                ok = update_sl(symbol, int(pos.ticket), be)
                if ok:
                    _append_csv(WD_ACTIONS_CSV,
                                f"{_utcnow().isoformat()},{symbol},MISMATCH_BE,step2,{cur_sl},{be},,{regime}")
                    send_telegram(f"#MISMATCH 2/3 | {symbol} | SL→BE")
                    changed = True

    else:  # step >= 3
        pnl = _to_float(pos.profit, 0.0)
        if (pnl < 0.0) or (r_mult is not None and r_mult < 0.3):
            ok = close_position_market(int(pos.ticket), symbol, float(volume), direction)
            if ok:
                _append_csv(WD_ACTIONS_CSV,
                            f"{_utcnow().isoformat()},{symbol},MISMATCH_EXIT,step>=3,,CLOSE,,{regime}")
                send_telegram(f"#MISMATCH 3/3 | {symbol} | EXIT (pnl<0 or R<0.3)")
                try:
                    log_outcome(strategy="WEEKLY",  # best effort (non conosco la strategia dal ticket)
                                asset=symbol, reason="mismatch_exit",
                                pnl=pnl, equity=None,
                                metrics={"regime": regime})
                except Exception as e:
                    log.warning(f"[ML_LOG] mismatch_exit errore: {e}")
                changed = True

    return step, changed

# ======================================================
# FUNZIONE PRINCIPALE
# ======================================================
def watchdog():
    if not MT5_AVAILABLE or not mt5.initialize():
        log.error("MT5 non disponibile")
        return

    st = _load_json(WD_STATE_FILE, {"cooloffs": {}, "mismatch_step": {}, "actions_count": {}}) or \
         {"cooloffs": {}, "mismatch_step": {}, "actions_count": {}}

    lines = []
    ts_now = _utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        acc = mt5.account_info()
        if not acc:
            log.error("account_info() non disponibile")
            return

        header = f"ODIN Watchdog — {ts_now} — equity={acc.equity:.2f} {acc.currency}"
        log.info(header)

        positions = mt5.positions_get()
        if not positions:
            send_telegram_report(header, ["Nessuna posizione aperta."])
            return

        odin_positions = [p for p in positions if int(getattr(p, "magic", 0)) == MAGIC_NUMBER]
        if not odin_positions:
            send_telegram_report(header, ["Nessuna posizione ODIN trovata."])
            return

        # contatori azioni giornaliere
        day_key = _utcnow().date().isoformat()
        actions_today = int(st.get("actions_count", {}).get(day_key, 0))

        for p in odin_positions:
            ticket = int(p.ticket)
            symbol = p.symbol
            direction = "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT"
            volume = float(p.volume)
            entry  = _to_float(p.price_open)
            cur_sl = None if _to_float(p.sl, None) is None else _to_float(p.sl)
            pnl    = _to_float(p.profit, 0.0)

            strat, asset = parse_comment(p.comment)
            regime_lbl = get_regime_label(asset)

            time_open = datetime.fromtimestamp(int(p.time), tz=timezone.utc)
            days_open = (_utcnow() - time_open).total_seconds() / 86400.0

            # R approssimato (se c'è SL)
            r_mult = None
            if cur_sl is not None and entry not in (0, None):
                dist = abs(entry - cur_sl)
                if dist > 0:
                    last = _to_float(p.price_current, entry)
                    r_mult = abs(last - entry) / dist

            # TIME-STOP
            if days_open >= WD_TIME_STOP_DAYS:
                if actions_today < WD_MAX_DAILY_ACTIONS and close_position_market(ticket, symbol, volume, direction):
                    msg = f"→ CLOSE time_stop | {symbol} ticket={ticket} days={days_open:.2f}>={WD_TIME_STOP_DAYS}"
                    lines.append(msg); log.info(msg)
                    try:
                        log_outcome(strategy=strat, asset=asset, reason="time_stop",
                                    pnl=pnl, equity=acc.equity, metrics={"regime": regime_lbl})
                    except Exception as e:
                        log.warning(f"[ML_LOG] time_stop errore: {e}")
                    actions_today += 1
                    continue
                else:
                    lines.append(f"× CLOSE time_stop FAILED/QUOTA | {symbol} ticket={ticket}")
                    continue

            # INVALID STRUCTURE (placeholder: qui potresti leggere da file params un trigger)
            invalid_structure = False
            if invalid_structure:
                if actions_today < WD_MAX_DAILY_ACTIONS and close_position_market(ticket, symbol, volume, direction):
                    msg = f"→ CLOSE invalid_structure | {symbol} ticket={ticket} regime={regime_lbl}"
                    lines.append(msg); log.info(msg)
                    try:
                        log_outcome(strategy=strat, asset=asset, reason="invalid_structure",
                                    pnl=pnl, equity=acc.equity, metrics={"regime": regime_lbl})
                    except Exception as e:
                        log.warning(f"[ML_LOG] invalid_structure errore: {e}")
                    actions_today += 1
                    continue

            # MISMATCH STEP MACHINE (esempio: se regime è “LATERALE” ma strategia = WEEKLY)
            mismatch = (strat == "WEEKLY" and "LATERALE" in regime_lbl)
            prev = int(_to_float(st["mismatch_step"].get(symbol, 0), 0))
            if mismatch:
                step, changed = wd_handle_mismatch(ticket, symbol, p, entry, cur_sl, volume,
                                                   direction, regime_lbl, prev, r_mult)
                if changed:
                    st["mismatch_step"][symbol] = step
                    actions_today += 1
                    continue
            else:
                if prev:
                    st["mismatch_step"][symbol] = 0  # reset

            # BE/TRAILING (semplificati)
            if WD_ENABLE_BREAKEVEN and r_mult is not None and r_mult >= 1.0:
                # Aggiorna SL a BE se manca
                if direction == "LONG":
                    be = entry
                    if cur_sl is None or be > cur_sl:
                        if update_sl(symbol, ticket, be):
                            lines.append(f"→ SL→BE | {symbol}")
                else:
                    be = entry
                    if cur_sl is None or be < cur_sl:
                        if update_sl(symbol, ticket, be):
                            lines.append(f"→ SL→BE | {symbol}")

        st.setdefault("actions_count", {})[day_key] = actions_today
        _save_json(WD_STATE_FILE, st)
        send_telegram_report(header, lines)

    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

# ======================================================
# ENTRYPOINT
# ======================================================
if __name__ == "__main__":
    watchdog()
