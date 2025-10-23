@echo off
setlocal enableextensions enabledelayedexpansion
cd /d C:\AUTOMAZIONE || (echo [FAIL] cd in C:\AUTOMAZIONE & exit /b 1)
if not exist logs mkdir logs
echo [%06/10/2025% %23:32:36,36%] START Macro AI Report >> logs\macro_ai_report_schedule.log
set "PYEXE=C:\AUTOMAZIONE\venv\Scripts\python.exe"
if not exist "%%PYEXE%%" (
  echo [FAIL] python non trovato: %%PYEXE%% >> logs\macro_ai_report_schedule.log
  exit /b 2
)
"%%PYEXE%%" "C:\AUTOMAZIONE\macro_ai_report.py" >> logs\macro_ai_report_schedule.log 2>&1
echo [%06/10/2025% %23:32:36,36%] END Macro AI Report >> logs\macro_ai_report_schedule.log
exit /b 0
