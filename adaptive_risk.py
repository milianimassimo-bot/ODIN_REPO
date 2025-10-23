# adaptive_risk.py — risk caps dinamici in base a equity + vol
import os, json
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(ROOT, "params")
MACRO_OVR = os.path.join(PARAMS_DIR, "macro_overrides.json")

def _write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def bands(equity: float) -> float:
    # banda risk caps multiplier
    if equity < 300:   return 0.50
    elif equity < 600: return 0.75
    elif equity < 1200:return 1.00
    elif equity < 3000:return 1.15
    else:              return 1.25

def compute(equity: float, vol_state: str = "normal"):
    mult = bands(equity)
    if   vol_state == "high": mult = 0.8
    elif vol_state == "low":  mult = 1.1
    return round(mult, 2)

def main():
    equity = float(os.getenv("ODIN_EQUITY", "10000"))
    vol_state = os.getenv("ADAPTIVE_VOL_STATE", "normal")
    mult = compute(equity, vol_state)
    out = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "global": {"risk_caps_multiplier": mult}
    }
    _write_json(MACRO_OVR, out)
    print(f"[AdaptiveRisk] risk_caps_multiplier={mult} scritto in {MACRO_OVR}")

if __name__ == "__main__":
    main()
