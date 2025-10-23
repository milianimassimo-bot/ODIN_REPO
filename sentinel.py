# ==============================================================================
# ODIN - Strategy Sentinel (Hybrid D1+H4)
# Version: 1.2.2 (Telegram Fix & Full Integration)
# ==============================================================================

import os
import json
import importlib
import requests  # <-- AGGIUNTO
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from strats.base import Strategy, BacktestResult

load_dotenv()

# --- PATCH: safe _clamp fallback ---
import builtins
if not hasattr(builtins, "_clamp"):
    def _clamp(x, lo, hi):
        try:
            if not np.isfinite(x): return lo
            x = float(x)
            if x < lo: return lo
            if x > hi: return hi
            return x
        except (ValueError, TypeError):
            return lo
    builtins._clamp = _clamp
# --- END PATCH ---

# --- CONFIG ---
PARAMS_DIR = "params"
UNIVERSE_FILE = os.getenv("ODIN_UNIVERSE_JSON", os.path.join(PARAMS_DIR, "universe.json"))
ROSTER_FILE = os.getenv("ODIN_STRATEGY_ROSTER", os.path.join(PARAMS_DIR, "strategy_roster.json"))
ALIASES_FILE = os.path.join(PARAMS_DIR, "symbol_aliases.json")

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Elenco strategie da testare
STRATEGY_CLASSES = [
    "strats.ema_trend_d1.Strat_EMA_Trend_D1",
    "strats.ema_pullback_h4.Strat_EMA_Pullback_H4",
    "strats.bb_meanrev_d1.Strat_BB_MR_D1",
    "strats.donchian_breakout_d1.Strat_Donchian_BO_D1",
    "strats.keltner_trend_d1.Strat_Keltner_Trend_D1",
    "strats.keltner_mr_h4.Strat_Keltner_MR_H4",
    "strats.bb_squeeze_breakout_d1.Strat_BB_Squeeze_BO_D1",
    "strats.vol_target_trend_d1.Strat_VolTarget_Trend_D1",
    "strats.regime_switcher_d1.Strat_Regime_Switcher_D1",
]

# --- UTILS ---
def _log(msg: str): print(msg, flush=True)

def tg_send(msg: str):
    if not TG_TOKEN or not TG_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": msg, "disable_web_page_preview": True, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10).raise_for_status()
    except Exception as e:
        _log(f"⚠️ Telegram send failed: {e!r}")

def _load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: return default if default is not None else {}

def _save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_universe_labels()->List[Dict[str,Any]]:
    aliases = _load_json(ALIASES_FILE, {})
    try:
        with open(UNIVERSE_FILE,"r",encoding="utf-8") as f: u = json.load(f)
        out, raw_labels = [], []
        for row in (u.get("picks") or u.get("items") or []):
            if asset := (row.get("asset") or "").strip():
                raw_labels.append({"asset": asset, "vendorSymbol": row.get("vendorSymbol") or asset})
        if not raw_labels and isinstance(u.get("allow"), dict):
            for asset in u["allow"].keys():
                if asset.strip(): raw_labels.append({"asset": asset, "vendorSymbol": asset})
        
        for item in raw_labels:
            asset, vendor = item["asset"], item["vendorSymbol"]
            if alias_data := aliases.get(asset):
                vendor = alias_data.get("twelvedata", vendor)
            out.append({"asset": asset, "vendorSymbol": vendor})
        return out
    except Exception:
        return []

def _instantiate(path: str)->Strategy:
    mod, cls = path.rsplit(".",1)
    M = importlib.import_module(mod)
    C = getattr(M, cls)
    return C()

# --- SCORING ROBUSTO ---
MIN_TRADES = 3
W_PF, W_SH, W_DD = 0.45, 0.35, 0.20

def _score(result: BacktestResult, asset: str) -> Optional[float]:
    trades = int(result.trades) if result.trades is not None else 0
    if trades < MIN_TRADES:
        _log(f"  - [SCORE] {result.name} @ {asset}: scartato, pochi trade ({trades} < {MIN_TRADES})")
        return None

    pf = result.pf if np.isfinite(result.pf) else 0.0
    sharpe = result.sharpe if np.isfinite(result.sharpe) else 0.0
    dd = result.dd if np.isfinite(result.dd) else 1.0

    pf_norm = _clamp(pf, 0.0, 3.0) / 3.0
    sharpe_norm = (_clamp(sharpe, -1.0, 3.0) + 1.0) / 4.0
    dd_norm = 1.0 - _clamp(dd, 0.0, 0.30) / 0.30

    score = W_PF * pf_norm + W_SH * sharpe_norm + W_DD * dd_norm
    _log(f"  - [SCORE] {result.name} @ {asset}: score={score:.3f} (PF={pf:.2f}, Sh={sharpe:.2f}, DD={dd:.2f}, T={trades})")
    return max(0.0, min(score, 1.0))

def main():
    _log("--- Avvio Strategy Sentinel ---")
    labels = _load_universe_labels()
    if not labels:
        _log(f"Universe vuoto o non trovato: {UNIVERSE_FILE}")
        return

    strategies: List[Strategy] = [_instantiate(p) for p in STRATEGY_CLASSES]
    results, per_asset = [], {}

    for s in strategies:
        _log(f"\nAnalizzando strategia: {s.name} ({s.timeframe})...")
        strat_scores = []
        for lab in labels:
            asset, vendor = lab["asset"], lab["vendorSymbol"]
            try:
                if s.name == "Regime_Switcher_D1": setattr(s, "_asset_label", asset)
                
                df = s.fetch_data(vendor)
                df = df.tail(252).copy()
                
                if df.empty or len(df) < 100:
                    _log(f"  - {asset}: dati insufficienti per il test.")
                    continue
                
                res = s.run(df)
                score = _score(res, asset)
                
                if score is not None:
                    strat_scores.append(score)
                    per_asset.setdefault(asset, []).append({
                        "strategy": s.name, "score": score, "pf": res.pf,
                        "dd": res.dd, "sharpe": res.sharpe, "trades": res.trades
                    })
            except Exception as e:
                _log(f"  - ERRORE su {asset}: {e}")

        avg_score = round(sum(strat_scores)/len(strat_scores), 4) if strat_scores else 0.0
        if not strat_scores:
             _log(f"-> ATTENZIONE: Nessun asset valido trovato per {s.name}.")
        
        results.append({"strategy": s.name, "avg_score": avg_score})
        _log(f"-> Punteggio medio globale per {s.name}: {avg_score}")

    strat_rank = sorted(results, key=lambda x: x["avg_score"], reverse=True)
    TOP_N = int(os.getenv("ODIN_STRATS_TOP_N","3"))
    active_map = {}
    for asset, lst in per_asset.items():
        good = [r for r in lst if r.get("score",0.0) > 0.1]
        top = sorted(good, key=lambda x: x["score"], reverse=True)[:TOP_N]
        active_map[asset] = {"active": [t["strategy"] for t in top], "candidates": lst}

    roster = {
        "asof_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe": [x["asset"] for x in labels],
        "ranking_global": strat_rank,
        "per_asset": active_map
    }
    _save_json(ROSTER_FILE, roster)
    _log(f"\n✅ Sentinel completato. Salvato roster in {ROSTER_FILE}")

    # --- BLOCCO TELEGRAM ---
    try:
        summary_lines = ["⚔️ *ODIN - Roster Strategie Aggiornato*"]
        summary_lines.append(f"_{roster.get('asof_utc', '')}_")
        
        summary_lines.append("\n*🏆 Ranking Globale Strategie (Top 5):*")
        for i, row in enumerate(roster.get('ranking_global', [])[:5], 1):
            score = row.get('avg_score', 0)
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "▪️"
            summary_lines.append(f"{emoji} {row['strategy']}: `{score:.4f}`")
            
        summary_lines.append("\n*🎯 Strategie Attive per Asset:*")
        active_count = 0
        for asset, data in sorted(roster.get("per_asset", {}).items()):
            active = data.get("active", [])
            if active:
                summary_lines.append(f"- *{asset}*: {', '.join(active)}")
                active_count += 1
        
        if active_count == 0:
            summary_lines.append("Nessuna strategia attiva per nessun asset.")
            
        tg_send("\n".join(summary_lines))
    except Exception as e:
        _log(f"⚠️ Errore creazione report Telegram per Sentinel: {e!r}")

if __name__ == "__main__":
    main()