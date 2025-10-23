@echo off & setlocal
set "BASEDIR=%~dp0"
call "%BASEDIR%env_common.bat"
call "%BASEDIR%venv\Scripts\activate.bat"
cd /d "%BASEDIR%"
py notify_telegram.py "[ENV PING] RETRAIN_TR=%RETRAIN_MIN_TR% PF=%RETRAIN_MIN_PF% EXP=%RETRAIN_MIN_EXP% DD=%RETRAIN_MAX_DD% | STRATS=%SENTINEL_STRATS%"
endlocal
