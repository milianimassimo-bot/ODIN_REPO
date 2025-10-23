# ============================================================
# ODIN - Trainer Suggestor v3.4 (Full Auto + Telegram + Summary)
# ------------------------------------------------------------
# Legge logs/odin_ml.db (outcomes con pnl_abs)
# Analizza risultati per (asset, strategia) con decay temporale
# Filtra con n_min; genera params/softeners_plan.json
# Scrive logs/trainer_report_*.txt e invia report Telegram
# Ora con SUMMARY di portafoglio (winrate, media pesata, top/bottom)
# ============================================================

import os, json, sqlite3, statistics, requests, time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')


load_dotenv()

# --- Telegram setup ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def tg_send(msg: str):
    """Invia messaggio Telegram (3 retry, non blocca)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "disable_web_page_preview": True}
    for i in range(3):
        try:
            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            return
        except Exception:
            time.sleep(0.8 * (2 ** i))

# --- Percorsi principali ---
ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
PARAMS = os.path.join(ROOT, "params")
DB_PATH = os.path.join(LOGS, "odin_ml.db")
CURRENT_PARAMS = os.getenv("ODIN_PARAMS_FILE", os.path.join(PARAMS, "current.json"))
SOFTENERS_OUT = os.path.join(PARAMS, "softeners_plan.json")
REPORT_OUT = os.path.join(LOGS, f"trainer_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt")

# --- Parametri di training ---
N_MIN = int(os.getenv("TRAINER_MIN_TRADES", "4"))
TTL_MIN = int(os.getenv("SOFTENER_TTL_MIN", "720"))  # 12h
DECAY_DAYS = int(os.getenv("TRAINER_DECAY_DAYS", "30"))
STEP_STRICT = 0.10
STEP_RELAX = 0.10

# --- Utility ---
def _read_json(p, default=None):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def _write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _clamp(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except:
        return lo

def _now_utc():
    return datetime.now(timezone.utc)

# ============================================================
# 1) carica outcomes dal DB
# ============================================================
def load_outcomes():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB non trovato: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    rows = c.execute("""
        SELECT ts_utc, asset, strategy, pnl_abs, exit_reason
        FROM outcomes
        WHERE pnl_abs IS NOT NULL
    """).fetchall()
    con.close()
    return rows

# ============================================================
# 2) calcola metriche con decay temporale
# ============================================================
def compute_metrics(rows):
    stats = {}
    now = _now_utc()
    for ts, asset, strat, pnl, reason in rows:
        try:
            pnl = float(pnl)
            t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            age_days = (now - t).days
            w = max(0.1, 1.0 - (age_days / DECAY_DAYS))
        except Exception:
            continue
        k = (asset, strat)
        bucket = stats.setdefault(k, {"p": [], "w": [], "reasons": []})
        bucket["p"].append(pnl)
        bucket["w"].append(w)
        bucket["reasons"].append(reason)

    for k, b in stats.items():
        if not b["p"]:
            continue
        try:
            wavg = sum(p * w for p, w in zip(b["p"], b["w"])) / sum(b["w"])
        except ZeroDivisionError:
            wavg = statistics.mean(b["p"])
        b["avg"] = wavg
        b["n"] = len(b["p"])
        b["win"] = len([x for x in b["p"] if x > 0])
        b["loss"] = len([x for x in b["p"] if x <= 0])
        b["winrate"] = b["win"] / b["n"] if b["n"] else 0
    return stats

# ============================================================
# 3) baseline parametri
# ============================================================
def load_baseline():
    return _read_json(CURRENT_PARAMS, {})

def get_weekly_vals(cur, asset):
    wk = ((cur.get("WEEKLY") or {}).get("per_asset") or {}).get(asset, {}) or {}
    return {
        "pullback_atr_max": float(wk.get("pullback_atr_max", 1.4)),
        "atr_pct_max": float(wk.get("atr_pct_max", 2.0))
    }

def get_bbm_vals(cur, asset):
    mm = ((cur.get("BB_MR") or {}).get("per_asset") or {}).get(asset, {}) or {}
    return {
        "adx_max": float(mm.get("adx_max", 18.0)),
        "r2_max": float(mm.get("r2_max", 0.20))
    }

# ============================================================
# 4) costruisci piano softeners
# ============================================================
def build_softeners(stats, cur):
    plan = {"WEEKLY": {"per_asset": {}}, "BB_MR": {"per_asset": {}}}
    lines = []
    for (asset, strat), m in stats.items():
        if m["n"] < N_MIN:
            lines.append(f"⚪️ {asset}/{strat}: solo {m['n']} trade, serve ≥{N_MIN}")
            continue
        wr, avg = m["winrate"], m["avg"]

        if strat == "WEEKLY":
            base = get_weekly_vals(cur, asset)
            pb, ap = base["pullback_atr_max"], base["atr_pct_max"]
            if wr < 0.45 or avg < 0:
                new_pb = _clamp(pb * (1 - STEP_STRICT), 0.8, 2.5)
                new_ap = _clamp(ap * (1 - STEP_STRICT), 0.6, 3.0)
                plan["WEEKLY"]["per_asset"].setdefault(asset, {})
                plan["WEEKLY"]["per_asset"][asset]["pullback_atr_max"] = {"set": new_pb, "ttl_min": TTL_MIN}
                plan["WEEKLY"]["per_asset"][asset]["atr_pct_max"] = {"set": new_ap, "ttl_min": TTL_MIN}
                lines.append(f"🔻 WEEKLY {asset}: wr={wr:.2f} avg={avg:.2f} → tighten pb {pb:.2f}->{new_pb:.2f}, atr% {ap:.2f}->{new_ap:.2f}")
            elif wr > 0.55 and avg > 0:
                new_pb = _clamp(pb * (1 + STEP_RELAX), 0.8, 3.5)
                new_ap = _clamp(ap * (1 + STEP_RELAX), 0.6, 4.0)
                plan["WEEKLY"]["per_asset"].setdefault(asset, {})
                plan["WEEKLY"]["per_asset"][asset]["pullback_atr_max"] = {"set": new_pb, "ttl_min": TTL_MIN}
                plan["WEEKLY"]["per_asset"][asset]["atr_pct_max"] = {"set": new_ap, "ttl_min": TTL_MIN}
                lines.append(f"🟢 WEEKLY {asset}: wr={wr:.2f} avg={avg:.2f} → relax pb {pb:.2f}->{new_pb:.2f}, atr% {ap:.2f}->{new_ap:.2f}")

        elif strat == "BB_MR":
            base = get_bbm_vals(cur, asset)
            ax, r2 = base["adx_max"], base["r2_max"]
            if avg < 0:
                new_ax = _clamp(ax - 1.0, 12.0, 30.0)
                new_r2 = _clamp(r2 - 0.02, 0.10, 0.60)
                plan["BB_MR"]["per_asset"].setdefault(asset, {})
                plan["BB_MR"]["per_asset"][asset]["adx_max"] = {"set": new_ax, "ttl_min": TTL_MIN}
                plan["BB_MR"]["per_asset"][asset]["r2_max"] = {"set": new_r2, "ttl_min": TTL_MIN}
                lines.append(f"🔻 BB_MR {asset}: avg={avg:.2f} → tighten adx {ax}->{new_ax}, r2 {r2}->{new_r2}")
            elif avg > 0 and wr > 0.55:
                new_ax = _clamp(ax + 1.0, 10.0, 35.0)
                new_r2 = _clamp(r2 + 0.02, 0.10, 0.70)
                plan["BB_MR"]["per_asset"].setdefault(asset, {})
                plan["BB_MR"]["per_asset"][asset]["adx_max"] = {"set": new_ax, "ttl_min": TTL_MIN}
                plan["BB_MR"]["per_asset"][asset]["r2_max"] = {"set": new_r2, "ttl_min": TTL_MIN}
                lines.append(f"🟢 BB_MR {asset}: wr={wr:.2f} avg={avg:.2f} → relax adx {ax}->{new_ax}, r2 {r2}->{new_r2}")

    if not plan["WEEKLY"]["per_asset"]:
        del plan["WEEKLY"]
    if not plan["BB_MR"]["per_asset"]:
        del plan["BB_MR"]

    return plan, lines

# ============================================================
# 4bis) SUMMARY di portafoglio per Telegram/file
# ============================================================
def summarize_portfolio(stats):
    """Ritorna righe di sintesi globale + per strategia + top/bottom asset."""
    if not stats:
        return ["(nessun dato)"]

    # globali
    n_tot = sum(v["n"] for v in stats.values())
    wins  = sum(v["win"] for v in stats.values())
    losses= sum(v["loss"] for v in stats.values())
    wr_glob = (wins / n_tot) if n_tot else 0.0
    # media PnL pesata per numero trades
    avg_glob = (sum(v["avg"] * v["n"] for v in stats.values()) / n_tot) if n_tot else 0.0

    # per strategia
    per_strat = {}
    for (asset, strat), v in stats.items():
        b = per_strat.setdefault(strat, {"n":0,"wins":0,"loss":0,"avg_sum":0.0})
        b["n"]     += v["n"]
        b["wins"]  += v["win"]
        b["loss"]  += v["loss"]
        b["avg_sum"] += v["avg"] * v["n"]

    strat_lines = []
    for strat, b in per_strat.items():
        wr = (b["wins"]/b["n"]) if b["n"] else 0.0
        avg = (b["avg_sum"]/b["n"]) if b["n"] else 0.0
        strat_lines.append(f"• {strat}: n={b['n']} | WR={wr:.2f} | avg={avg:.2f}")

    # top/bottom asset per media (aggrego tra strategie)
    per_asset = {}
    for (asset, strat), v in stats.items():
        a = per_asset.setdefault(asset, {"n":0, "avg_sum":0.0})
        a["n"] += v["n"]
        a["avg_sum"] += v["avg"] * v["n"]
    for a in per_asset.values():
        a["avg"] = (a["avg_sum"]/a["n"]) if a["n"] else 0.0

    tops = sorted(per_asset.items(), key=lambda kv: kv[1]["avg"], reverse=True)[:3]
    bots = sorted(per_asset.items(), key=lambda kv: kv[1]["avg"])[:3]

    lines = [
        f"Tot trades={n_tot} | WR={wr_glob:.2f} | PnL avg={avg_glob:.2f}",
        *(strat_lines if strat_lines else []),
    ]
    if tops:
        lines.append("Top ⬆️: " + ", ".join(f"{a}({d['avg']:.2f})" for a,d in tops))
    if bots:
        lines.append("Bottom ⬇️: " + ", ".join(f"{a}({d['avg']:.2f})" for a,d in bots))
    return lines

# ============================================================
# 5) main
# ============================================================
def main():
    tg_send("⚙️ ODIN Trainer — avvio analisi ML outcomes…")
    try:
        rows = load_outcomes()
        if not rows:
            tg_send("ℹ️ Trainer: nessun outcome trovato nel DB.")
            return

        stats = compute_metrics(rows)
        cur = load_baseline()
        plan, lines = build_softeners(stats, cur)
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        out = {"generated_utc": ts, "notes": "auto from trainer_suggest v3.4", **plan}
        _write_json(SOFTENERS_OUT, out)

        # --- summary ---
        summary_lines = summarize_portfolio(stats)

        # --- report file completo ---
        with open(REPORT_OUT, "w", encoding="utf-8") as f:
            f.write(f"=== ODIN Trainer Report {ts} ===\nDB: {DB_PATH}\n\n")
            f.write("[SUMMARY]\n")
            for s in summary_lines:
                f.write(s + "\n")
            f.write("\n[DETTAGLIO ASSET/STRAT]\n")
            for k, v in stats.items():
                f.write(f"{k[0]}/{k[1]} | n={v['n']} winrate={v['winrate']:.2f} avg={v['avg']:.2f}\n")
            f.write("\n[DECISIONI]\n")
            if lines:
                for ln in lines: f.write(ln + "\n")
            else:
                f.write("Nessuna modifica proposta.\n")

        # --- Telegram report finale ---
        header = f"🤖 ODIN Trainer Report ({ts})"
        smry   = "\n".join(summary_lines[:5])  # compattato
        if lines:
            body = "\n".join(lines[:12])  # limite sicurezza
            tg_send(f"{header}\n{smry}\n\nModifiche: {len(lines)}\n{body}")
        else:
            tg_send(f"{header}\n{smry}\n\nNessuna modifica proposta.")

        print(f"✅ Report completo: {REPORT_OUT}")
        print(f"🧠 Softeners generati in: {SOFTENERS_OUT}")

    except Exception as e:
        tg_send(f"❌ Trainer ERROR: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    main()
