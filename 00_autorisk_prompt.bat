@echo off
setlocal
set "BASEDIR=%~dp0"
cd /d "%BASEDIR%"
if not exist "logs" mkdir "logs"
call "%BASEDIR%venv\Scripts\activate.bat"
if exist "%BASEDIR%env_common.bat" call "%BASEDIR%env_common.bat"
set PYTHONUTF8=1

rem Default fallback (se non rispondi)
set "AUTORISK_FALLBACK=high"
rem Timeout (minuti) per attendere risposta
set "AUTORISK_TIMEOUT_MIN=15"

py autorisk_prompt.py --timeout %AUTORISK_TIMEOUT_MIN% --fallback %AUTORISK_FALLBACK% >> "logs\autorisk_prompt.log" 2>&1
py notify_telegram.py --title "🎚️ Autorisk Prompt" --text "Log: C:\AUTOMAZIONE\logs\autorisk_prompt.log" --code %ERRORLEVEL% --log "logs\autorisk_prompt.log"
endlocal
