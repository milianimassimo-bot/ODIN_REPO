@echo off
setlocal
set "BASEDIR=%~dp0"
if not exist "%BASEDIR%logs" mkdir "%BASEDIR%logs"
call "%BASEDIR%env_common.bat"
set "PYTHON=%BASEDIR%venv\Scripts\python.exe"
set "TEXT=%*"
if "%TEXT%"=="" set "TEXT=[ODIN] ping"
"%PYTHON%" "%BASEDIR%notify_telegram.py" --text "%TEXT%" >> "%BASEDIR%logs\notify.log" 2>&1
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 echo [notify] exit %ERR% >> "%BASEDIR%logs\notify.log"
endlocal & exit /b %ERR%
