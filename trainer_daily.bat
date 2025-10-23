@echo off & setlocal
set "BASEDIR=%~dp0"
call "%BASEDIR%env_common.bat"
call "%BASEDIR%venv\Scripts\activate.bat"
cd /d "%BASEDIR%"

for /f "tokens=1-5 delims=/:. " %%a in ("%date% %time%") do set TS=%%c%%b%%a_%%d%%e
set "TS=%TS: =0%"
set "LOG=logs\trainer_%TS%.log"

py trainer_suggest.py > "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"
py notify_telegram.py "[Trainer (daily)] exit=%EC% | log=%CD%\%LOG%"
endlocal & exit /b %EC%
