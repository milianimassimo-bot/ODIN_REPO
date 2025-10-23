@echo off & setlocal
set "BASEDIR=%~dp0"
call "%BASEDIR%env_common.bat"
call "%BASEDIR%venv\Scripts\activate.bat"
cd /d "%BASEDIR%"

rem Non spammare Telegram ad ogni giro: manda solo se errore o ogni N giri (qui: sempre ma ok per test)
for /f "tokens=1-5 delims=/:. " %%a in ("%date% %time%") do set TS=%%c%%b%%a_%%d%%e
set "TS=%TS: =0%"
set "LOG=logs\watchdog_%TS%.log"

py odin_watchdog_mt5.py > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
py notify_telegram.py "[Watchdog 15m] exit=%EC% | log=%CD%\%LOG%"
endlocal & exit /b %EC%
