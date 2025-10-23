@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM === ENV comune (BOT/CHAT, flags ecc.) ===
call "%~dp0env_common.bat"

set "BASEDIR=%~dp0"
cd /d "%BASEDIR%"
if not exist "logs" mkdir "logs"

REM === Timestamp compatibile (YYYYMMDD_HHMM)
for /f "tokens=2 delims==." %%i in ('wmic os get localdatetime /value 2^>nul') do set "LDT=%%i"
if not defined LDT (
  for /f "tokens=1-4 delims=/ " %%a in ("%date%") do (
    set "D=%%d%%b%%c"
  )
  for /f "tokens=1-2 delims=:." %%h in ("%time%") do (
    set "T=%%h%%i"
  )
  set "STAMP=%D%_%T%"
) else (
  set "STAMP=%LDT:~0,4%%LDT:~4,2%%LDT:~6,2%_%LDT:~8,2%%LDT:~10,2%"
)

REM === fallback ulteriore se wmic non esiste del tutto
if "%STAMP%"=="" (
  for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set "D=%%d%%b%%c"
  for /f "tokens=1-2 delims=:." %%h in ("%time%") do set "T=%%h%%i"
  set "STAMP=%D%_%T%"
)

set "LOGFILE=logs\sentinel_%STAMP%.log"

REM === Esegui Sentinel
py "sentinel.py" >> "%LOGFILE%" 2>&1
set "ERR=%ERRORLEVEL%"

REM === Messaggio breve (sempre)
if "%ERR%"=="0" (
  py "notify_telegram.py" "[Sentinel (scoring)] exit=0 | log=%CD%\%LOGFILE%"
) else (
  py "notify_telegram.py" "[Sentinel (scoring)] exit=%ERR% | log=%CD%\%LOGFILE%"
)

REM === Costruisci riepilogo leggibile dal log
setlocal EnableDelayedExpansion
set NL=^


set "MSG=🌠 Sentinel — Riepilogo scoring"
set /a CNT=0

REM Prende le righe con [SCORE]
for /f "usebackq delims=" %%L in (`findstr /L /C:"[SCORE]" "%LOGFILE%"`) do (
  set /a CNT+=1
  if !CNT! LEQ 18 (
    set "MSG=!MSG!!NL!%%L"
  )
)

REM Aggiunge medie globali
for /f "usebackq delims=" %%L in (`findstr /L /C:"-> Punteggio medio globale" "%LOGFILE%"`) do (
  set "MSG=!MSG!!NL!%%L"
)

REM Aggiunge eventuali avvisi
for /f "usebackq delims=" %%L in (`findstr /L /C:"-> ATTENZIONE" "%LOGFILE%"`) do (
  set "MSG=!MSG!!NL!%%L"
)

REM Invia riepilogo se c’è qualcosa
if not "!MSG!"=="🌠 Sentinel — Riepilogo scoring" (
  py "notify_telegram.py" "!MSG!"
)

endlocal & endlocal
