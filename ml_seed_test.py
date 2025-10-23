from odin_ml_logger import _ensure_db, log_outcome, DB_PATH

_ensure_db()

# Esempio 1: chiusura per time_stop su EUR/USD WEEKLY con +35€
log_outcome(
    strategy="WEEKLY",
    asset="EUR/USD",
    reason="time_stop",
    pnl=35.0,
    equity=5636.72,
    metrics={"R": 0.7, "regime": "TRENDING"}
)

# Esempio 2: stop strutturale su GOLD BB_MR con -12.5€
log_outcome(
    strategy="BB_MR",
    asset="GOLD",
    reason="invalid_structure",
    pnl=-12.5,
    equity=5630.10,
    metrics={"R": -0.2, "regime": "VOLATILE"}
)

print(f"Seed OK -> {DB_PATH}")
