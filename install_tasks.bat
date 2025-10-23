@echo off
setlocal
set "ROOT=%~dp0"

REM ====== PULIZIA TASK VECCHI ======
for %%T in (
  "ODIN\Sentinel"
  "ODIN\Regime D1"
  "ODIN\Universe"
  "ODIN\Retrain (night)"
  "ODIN\Calibrator (night)"
  "ODIN\Decision Room"
  "ODIN\Status (morning)"
  "ODIN\Watchdog 15m"
  "ODIN\Trainer (daily)"
  "ODIN\Calibrator (evening)"
  "ODIN\Status (evening)"
) do schtasks /Delete /TN %%T /F >nul 2>&1

REM ====== CREAZIONE NUOVI TASK ======
REM Tutti gli orari sono ora locale (Italia). D1 close ~ 02:00.

REM 01:30 Sentinel (prepara il roster)
schtasks /Create /TN "ODIN\Sentinel" /TR "\"%ROOT%01_sentinel.bat\"" /SC DAILY /ST 01:30 /RL LIMITED

REM 02:05 Regime (barre chiuse)
schtasks /Create /TN "ODIN\Regime D1" /TR "\"%ROOT%02_regime.bat\"" /SC DAILY /ST 02:05 /RL LIMITED

REM 02:10 Universe (selezione paniere del giorno)
schtasks /Create /TN "ODIN\Universe" /TR "\"%ROOT%03_universe.bat\"" /SC DAILY /ST 02:10 /RL LIMITED

REM 02:15 Retrain (parametri da roster)
schtasks /Create /TN "ODIN\Retrain (night)" /TR "\"%ROOT%04_retrain.bat\"" /SC DAILY /ST 02:15 /RL LIMITED

REM 02:20 Calibrator (ultimo tassello prima di tradare)
schtasks /Create /TN "ODIN\Calibrator (night)" /TR "\"%ROOT%05_calibrator_night.bat\"" /SC DAILY /ST 02:20 /RL LIMITED

REM 02:25 Decision Room (apre eventuali trade)
schtasks /Create /TN "ODIN\Decision Room" /TR "\"%ROOT%06_odin_main.bat\"" /SC DAILY /ST 02:25 /RL LIMITED

REM 02:35 Status mattina (health & riepilogo)
schtasks /Create /TN "ODIN\Status (morning)" /TR "\"%ROOT%07_status_morning.bat\"" /SC DAILY /ST 02:35 /RL LIMITED

REM Watchdog ogni 15 minuti
schtasks /Create /TN "ODIN\Watchdog 15m" /TR "\"%ROOT%watchdog_15m.bat\"" /SC MINUTE /MO 15 /RL LIMITED

REM 11:30 Trainer (softeners plan)
schtasks /Create /TN "ODIN\Trainer (daily)" /TR "\"%ROOT%trainer_daily.bat\"" /SC DAILY /ST 11:30 /RL LIMITED

REM 18:00 Calibrator serale (macro+autogate)
schtasks /Create /TN "ODIN\Calibrator (evening)" /TR "\"%ROOT%calibrator_evening.bat\"" /SC DAILY /ST 18:00 /RL LIMITED

REM 18:05 Status serale
schtasks /Create /TN "ODIN\Status (evening)" /TR "\"%ROOT%status_evening.bat\"" /SC DAILY /ST 18:05 /RL LIMITED

echo.
echo ✅ Task Scheduler ODIN installato.
echo Apri Utilita' di pianificazione per verificare: Libreria Utilita' -> ODIN
endlocal
