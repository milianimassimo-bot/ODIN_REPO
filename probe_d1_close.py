# probe_d1_close.py — stampa orari D1 (open/close) del broker in UTC e locale
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

SYMBOL = "EURUSD"   # puoi cambiare con GOLD, GBPUSD, ecc.

def fmt(dt_utc: datetime) -> str:
    local = dt_utc.astimezone()  # orario locale Windows
    return f"UTC={dt_utc.strftime('%Y-%m-%d %H:%M')} | Local={local.strftime('%Y-%m-%d %H:%M %Z')}"

# --- connetti a MT5 ---
if not mt5.initialize():
    print("⚠️ Impossibile connettersi a MT5. Apri MT5 e assicurati che sia in esecuzione.")
    quit()

rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, 2)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("⚠️ Nessun dato ricevuto da MT5. Verifica il simbolo o la connessione.")
else:
    for i, r in enumerate(rates, 1):
        open_utc = datetime.fromtimestamp(int(r['time']), tz=timezone.utc)
        close_utc = open_utc + timedelta(days=1)
        print(f"\nBar #{i}")
        print(f"  OPEN : {fmt(open_utc)}")
        print(f"  CLOSE: {fmt(close_utc)}")

    print("\n👉 Schedula odin_main.py circa 5 minuti PRIMA della CLOSE (ora locale).")
