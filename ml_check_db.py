# check_ml_db.py — verifica rapida del DB ODIN (read-only, robusta)
import os, sys, sqlite3
from pathlib import Path

DEF_DB = Path("logs") / "odin_ml.db"

def connect_ro(db_path: Path) -> sqlite3.Connection:
    # connessione read-only per evitare lock/incidenti
    uri = "file:" + db_path.resolve().as_posix() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)

def table_exists(con: sqlite3.Connection, name: str) -> bool:
    cur = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
    return cur.fetchone() is not None

def count_rows(con: sqlite3.Connection, table: str) -> int | None:
    if not table_exists(con, table):
        return None
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

def list_tables(con: sqlite3.Connection) -> list[str]:
    return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

def table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []

def print_last_outcomes(con: sqlite3.Connection, limit: int = 5):
    if not table_exists(con, "outcomes"):
        print("Ultimi outcomes: tabella 'outcomes' assente.")
        return
    cols_needed = {"ts_utc","asset","strategy","pnl_abs","exit_reason"}
    cols_have = set(table_columns(con, "outcomes"))
    missing = cols_needed - cols_have
    if missing:
        print(f"⚠️  Colonne mancanti in outcomes: {sorted(missing)}")
        print("   Colonne disponibili:", sorted(cols_have))

    q = """
        SELECT ts_utc, asset, strategy, pnl_abs, exit_reason
        FROM outcomes
        ORDER BY rowid DESC
        LIMIT ?
    """
    print(f"Ultimi outcomes (max {limit}):")
    for row in con.execute(q, (limit,)):
        # row = (ts_utc, asset, strategy, pnl_abs, exit_reason)
        print(" -", row)

def main():
    db_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEF_DB
    limit = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    print("\n=== CHECK ODIN_ML.DB ===")
    if not db_path.exists():
        print(f"❌ DB non trovato: {db_path}")
        sys.exit(1)

    try:
        with connect_ro(db_path) as con:
            sig = count_rows(con, "signals")
            out = count_rows(con, "outcomes")

            if sig is None and out is None:
                print("⚠️  Tabelle 'signals' e 'outcomes' non trovate.")
                print("   Tabelle presenti:", ", ".join(list_tables(con)) or "—")
                sys.exit(2)

            print(f"signals: {sig if sig is not None else '—'} | outcomes: {out if out is not None else '—'}\n")
            print_last_outcomes(con, limit)

    except sqlite3.Error as e:
        print("❌ Errore SQLite:", e)

    print("\n=========================\n")

if __name__ == "__main__":
    main()
