@echo off
rem ======= ENV COMUNE (EDITA QUI I TOKEN TELEGRAM) =======
set "TELEGRAM_BOT_TOKEN=7822023821:AAEcQGt0dS4og9Z1wtJIoXYGPNIsEDWRkB8"
set "TELEGRAM_CHAT_ID=-1002752265278"
set "ODIN_TELEGRAM_SUMMARY=1"
set "ODIN_TELEGRAM_VERBOSE=1"
set "PYTHONUTF8=1"
set "TWELVEDATA_API_KEY=7dcad4a0ef2c4c78bbd096988919f96f"



REM === Percorsi base / file usati da tutti ===
set "ODIN_ROOT=%~dp0"
set "ODIN_PARAMS_FILE=%ODIN_ROOT%params\current.json"
set "ODIN_REGIME_JSON=%ODIN_ROOT%odin_regime_report_d1.json"


REM === News gate (se vuoi tenerlo spento lascia 0) ===
set "NEWS_BLACKOUT_FILE=%ODIN_ROOT%params\news_blackout.json"
set "NEWS_BLOCK_ENABLE=0"
set "NEWS_BLOCK_PRE_MIN=60"
set "NEWS_BLOCK_POST_MIN=30"

REM === Modalità esecuzione (demo reale / niente stub) ===
set "ODIN_PAPER=0"
set "ODIN_DRYRUN=0"

:: === RISK / AUTORISK ===
set "ODIN_AUTORISK_ENABLE=1"
set "ODIN_AUTORISK_DEFAULT=HIGH"
set "ODIN_AUTORISK_TIMEOUT_SEC=900"

rem — compat Autorisk (oltre ai tuoi ODIN_AUTORISK_*) —
set "ODIN_RISK_MODE=AUTO"
set "AUTORISK_INTERACTIVE=1"
set "AUTORISK_DEFAULT=HIGH"
set "AUTORISK_TIMEOUT_SEC=900"

:: === STRATEGIE abilitate nel sistema ===
:: (Sentinel le valuta; Universe le può proporre; Retrain/Calibrator le considerano)
set "SENTINEL_STRATS=WEEKLY,BB_MR,REGIME_SWITCHER_D1,KELTNER_MR_H4,DONCHIAN_BO_D1,EMA_PULLBACK_H4"
set "RETRAIN_STRATEGIES=%SENTINEL_STRATS%"

rem Profondità cache dati (per avere più trade OOS)
set "ODIN_CACHE_D1_BARS=2200"
set "ODIN_CACHE_H4_BARS=2200"

:: === Soglie RETRAIN (OOS) rilassate ===
set "RETRAIN_MIN_TR=4"
set "RETRAIN_MIN_PF=1.05"
set "RETRAIN_MIN_EXP=-0.05"
set "RETRAIN_MAX_DD=25"
set "RETRAIN_LOOKBACK_YEARS=3"

REM === Qualche extra utile ===
set "ODIN_TELEGRAM_PREFIX=[ODIN]"

:: === Soglie AutoGate (Calibrator) ===
set "AUTO_GATE_ENABLE=1"
set "AUTO_GATE_MIN_TR=4"
set "AUTO_GATE_MIN_PF=1.05"
set "AUTO_GATE_MIN_EXP=-0.05"
set "AUTO_GATE_MAX_DD=25"

:: === Timeframes ===
set "ODIN_TIMEFRAME_D1=D1"
set "ODIN_TIMEFRAME_H4=H4"

rem (opzionale) rischio di default (se non rispondi al prompt autorisk)
set "ODIN_RISK_TRADE=0.50"
set "ODIN_RISK_ASSET=1.00"
