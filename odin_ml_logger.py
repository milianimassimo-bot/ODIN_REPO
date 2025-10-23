# ============================================================
# ODIN ML Logger (v2.1) — compatibile con Watchdog e Odin Main
# - Salva segnali e outcome su DB SQLite
# - log_signal() → chiamato da odin_main
# - log_outcome() → chiamato da watchdog per trade chiusi
# ============================================================

import os
import json
import sqlite3
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT, "logs")
DB_PATH = os.path.join(LOGS_DIR, "odin_ml.db")


# ---------- DB setup ----------
def _ensure_db():
    os.makedirs(LOGS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS signals(
        id TEXT PRIMARY KEY,
        ts_utc TEXT,
        asset TEXT,
        strategy TEXT,
        direction TEXT,
        timeframe TEXT,
        entry REAL,
        sl REAL,
        tp REAL,
        rr REAL,
        atr_d1 REAL,
        basis_json TEXT,
        regime_json TEXT,
        ai_json TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS outcomes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT,
        asset TEXT,
        strategy TEXT,
        pnl_abs REAL,
        equity REAL,
        exit_reason TEXT,
        regime_json TEXT,
        metrics_json TEXT
    )""")
    conn.commit()
    conn.close()


# ---------- log_signal (invariato) ----------
def log_signal(plan, regime_item):
    """Chiamato da odin_main per registrare un nuovo setup."""
    try:
        _ensure_db()
        d = plan.signal
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO signals
        (id, ts_utc, asset, strategy, direction, timeframe, entry, sl, tp, rr, atr_d1, basis_json, regime_json, ai_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            plan.id, plan.asof_utc, d.asset, d.strategy, d.direction, d.timeframe_entry,
            float(d.entry), float(d.sl), float(d.tp), float(d.rr), float(d.atr_d1),
            json.dumps(d.basis, ensure_ascii=False),
            json.dumps(regime_item or {}, ensure_ascii=False),
            json.dumps(plan.ai.__dict__, ensure_ascii=False)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ML_LOGGER] log_signal error: {e!r}")


# ---------- log_outcome (nuovo, compatibile Watchdog) ----------
def log_outcome(strategy, asset, reason, pnl, equity=None, metrics=None):
    """
    Registra un outcome generico (chiusura, stop, mismatch, ecc.)
    strategy: "WEEKLY", "BB_MR"...
    asset: es. "EUR/USD"
    reason: es. "time_stop", "invalid_structure", "mismatch_exit"
    pnl: profit/loss assoluto
    equity: equity account al momento (facoltativo)
    metrics: dict con indicatori (ADX, ATR%, regime, ecc.)
    """
    try:
        _ensure_db()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT INTO outcomes (ts_utc, asset, strategy, pnl_abs, equity, exit_reason, regime_json, metrics_json)
        VALUES (?,?,?,?,?,?,?,?)
        """, (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            asset,
            strategy,
            float(pnl) if pnl is not None else None,
            float(equity) if equity is not None else None,
            reason,
            json.dumps({"regime": metrics.get("regime")} if metrics else {}, ensure_ascii=False),
            json.dumps(metrics or {}, ensure_ascii=False)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ML_LOGGER] log_outcome error: {e!r}")
