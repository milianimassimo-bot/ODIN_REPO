# ============================================================
# ODIN - AI Calibrator v2.2 + AUTOGATE DEBUG PATCH (clean boot)
# - Applica macro_overrides.json a params/current.json
# - Applica Softeners (TTL + revert)
# - Auto-Gate (accende/spegne allow_new_entries in base alle metriche retrain)
# - Opzionale retrain via CALIBRATOR_RUN_RETRAIN_CMD
# - Report Telegram
# - DEBUG_AUTOGATE: diagnostica completa lettura metriche e decisioni
# ============================================================

import os, json, time, subprocess, re
from datetime import datetime, timezone, timedelta
from copy import deepcopy
import requests
from dotenv import load_dotenv

# ============================================================
# 1️⃣ Carica .env immediatamente (prima di qualsiasi getenv)
# ============================================================
load_dotenv(override=True)

# ============================================================
# 2️⃣ Debug e AutoTune placeholders
# ============================================================
DEBUG_AUTOGATE = True  # metti False per silenziare i log di diagnostica

AUTO_TUNE_ENABLE    = os.getenv("AUTO_TUNE_ENABLE", "0").strip() in ("1", "true", "True")
AUTO_TUNE_MAX_DELTA = float(os.getenv("AUTO_TUNE_MAX_DELTA", "0.2"))
AUTO_TUNE_TTL_MIN   = int(os.getenv("AUTO_TUNE_TTL_MIN", "1440"))

# ============================================================
# 3️⃣ Paths principali
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(ROOT, "params")
LOGS_DIR   = os.path.join(ROOT, "logs")

CURRENT_FILE    = os.getenv("ODIN_PARAMS_FILE", os.path.join(PARAMS_DIR, "current.json"))
MACRO_OV_FILE   = os.path.join(PARAMS_DIR, "macro_overrides.json")
SOFT_PLAN_FILE  = os.path.join(PARAMS_DIR, "softeners_plan.json")
SOFT_STATE_FILE = os.path.join(PARAMS_DIR, "softeners_state.json")

# ============================================================
# 4️⃣ ENV principali del Calibrator
# ============================================================
CALIBRATOR_MODE = os.getenv("CALIBRATOR_MODE", "AUTO").upper().strip()
CALIBRATOR_RUN_RETRAIN_CMD = os.getenv("CALIBRATOR_RUN_RETRAIN_CMD", "").strip()
CALIBRATOR_MAX_AGE_MIN = int(os.getenv("CALIBRATOR_MAX_AGE_MIN", "1440"))
CALIBRATOR_MACRO_APPLY_IF_STALE = os.getenv("CALIBRATOR_MACRO_APPLY_IF_STALE", "true").lower().strip() == "true"
DO_RETRAIN_FLAG = os.getenv("CALIBRATOR_DO_RETRAIN", "0").strip() in ("1", "true", "True")

TG_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TG_CHAT  = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

# ============================================================
# 5️⃣ AUTO-GATE (attivazione/disattivazione asset automatica)
# ============================================================
AUTO_GATE_ENABLE = os.getenv("AUTO_GATE_ENABLE", "0").strip() in ("1", "true", "True")
AUTO_GATE_RETRAIN_JSON = os.getenv("AUTO_GATE_RETRAIN_JSON", os.path.join(PARAMS_DIR, "retrain_metrics.json"))
AUTO_GATE_RETRAIN_TEXT = os.getenv("AUTO_GATE_RETRAIN_TEXT", os.path.join(LOGS_DIR, "last_retrain.txt"))

AUTO_GATE_STRATS = {s.strip() for s in os.getenv("AUTO_GATE_STRATS", "BB_MR,WEEKLY").split(",") if s.strip()}
AUTO_GATE_MIN_TR  = int(os.getenv("AUTO_GATE_MIN_TR", "5"))
AUTO_GATE_MIN_PF  = float(os.getenv("AUTO_GATE_MIN_PF", "1.3"))
AUTO_GATE_MIN_EXP = float(os.getenv("AUTO_GATE_MIN_EXP", "0.0"))
AUTO_GATE_MAX_DD  = float(os.getenv("AUTO_GATE_MAX_DD", "10.0"))
AUTO_GATE_COOLDOWN_DAYS = int(os.getenv("AUTO_GATE_COOLDOWN_DAYS", "3"))

AUTO_GATE_STATE_FILE = os.path.join(PARAMS_DIR, "auto_gate_state.json")

# ---------------- Utils ----------------
def send_tg(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "disable_web_page_preview": True},
            timeout=15
        )
    except Exception:
        pass

def read_json(p, default=None):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _clamp(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo

def _mtime_minutes(path: str) -> float:
    try:
        age_sec = time.time() - os.path.getmtime(path)
        return max(0.0, age_sec/60.0)
    except Exception:
        return 1e9

# ---------------- Macro overrides → current.json ----------------
def apply_macro_overrides(cur_params: dict, overrides: dict) -> dict:
    """
    - risk_caps_multiplier (clamp 0.0005–0.20)
    - asset_overrides.allow_new_entries
    """
    out = deepcopy(cur_params or {})
    ov  = overrides or {}

    mult = (ov.get("global", {}) or {}).get("risk_caps_multiplier")
    if isinstance(mult, (int, float)):
        g = out.setdefault("global", {}).setdefault("risk_caps", {})
        for k in ("global","bucket","asset","trade"):
            if k in g and isinstance(g[k], (int,float)):
                g[k] = round(_clamp(g[k]*float(mult), 0.0005, 0.20), 6)

    aset = (ov.get("asset_overrides") or {})
    if aset:
        out.setdefault("asset_overrides", {})
        for a, spec in aset.items():
            if isinstance(spec, dict) and "allow_new_entries" in spec:
                out["asset_overrides"].setdefault(a, {})
                out["asset_overrides"][a]["allow_new_entries"] = bool(spec["allow_new_entries"])

    return out

# ---------------- Softeners (TTL + revert) ----------------
def _set_path(d: dict, path: list[str], value):
    ref = d
    for k in path[:-1]:
        ref = ref.setdefault(k, {})
    ref[path[-1]] = value

def _get_path(d: dict, path: list[str], default=None):
    ref = d
    for k in path:
        if not isinstance(ref, dict) or k not in ref:
            return default
        ref = ref[k]
    return ref

def apply_softeners(current: dict, plan: dict, state: dict) -> tuple[dict, list[str], list[str]]:
    cur = deepcopy(current)
    changes, reverted = [], []
    now = datetime.now(timezone.utc)

    applied = state.get("applied", {})
    for path_str, info in list(applied.items()):
        try:
            ts  = datetime.fromisoformat(info["ts"])
        except Exception:
            ts  = now
        ttl = int(info.get("ttl_min", 0))
        if ttl > 0 and (now - ts) >= timedelta(minutes=ttl):
            path = path_str.split("::")
            orig_value = info.get("orig", None)
            _set_path(cur, path, orig_value)
            reverted.append(f"{path_str}: revert -> {orig_value}")
            del applied[path_str]

    for section, per_asset in (plan or {}).items():
        if not isinstance(per_asset, dict):
            continue
        for asset, fields in per_asset.items():
            if not isinstance(fields, dict):
                continue
            for key, spec in fields.items():
                if not isinstance(spec, dict) or "set" not in spec:
                    continue
                target_path = [section, "per_asset", asset, key]
                path_str = "::".join(target_path)
                if path_str in applied:
                    continue
                current_val = _get_path(cur, target_path, default=None)
                new_val = spec["set"]
                _set_path(cur, target_path, new_val)
                applied[path_str] = {
                    "ts": now.isoformat(),
                    "ttl_min": int(spec.get("ttl_min", 0)),
                    "orig": current_val
                }
                changes.append(f"{path_str}: {current_val} -> {new_val}")

    state["applied"] = applied
    state["asof_utc"] = utcnow()
    return cur, changes, reverted

# ----------------- Auto-Gate helpers -----------------
def _parse_block_stats(bl: str) -> dict | None:
    bl = (bl or "").strip()
    if bl in ("—","-",""):
        return None
    d = {}
    m = re.search(r"tr\s*=\s*(\d+)", bl, re.I)
    d["tr"] = int(m.group(1)) if m else 0
    m = re.search(r"PF\s*=\s*([0-9.]+)", bl, re.I)
    d["pf"] = float(m.group(1)) if m else 0.0
    m = re.search(r"Exp\s*=\s*([-+]?[0-9.]+)", bl, re.I)
    d["exp"] = float(m.group(1)) if m else 0.0
    m = re.search(r"DD\s*=\s*([-+]?[0-9.]+)", bl, re.I)
    d["dd"] = float(m.group(1)) if m else 0.0
    return d

def _load_retrain_text(p: str) -> dict:
    out = {}
    if not os.path.exists(p):
        if DEBUG_AUTOGATE:
            print(f"[AG] text not found: {p}")
        return out
    if DEBUG_AUTOGATE:
        print(f"[AG] load text: {p}")
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"^([A-Z/]+):.*?WEEKLY\[(.*?)\].*?\|\s*BB_MR\[(.*?)\]", line)
            if not m:
                continue
            asset = m.group(1).strip()
            wk = _parse_block_stats(m.group(2))
            bb = _parse_block_stats(m.group(3))
            out[asset] = {"WEEKLY": wk, "BB_MR": bb}
    if DEBUG_AUTOGATE:
        print(f"[AG] text assets: {len(out)}")
    return out

def _load_retrain_json(p: str) -> dict:
    raw = read_json(p, default={})
    if DEBUG_AUTOGATE:
        print(f"[AG] load json: {p} exists={os.path.exists(p)} type={type(raw).__name__}")
    if not isinstance(raw, dict):
        return {}

    def norm(m):
        if not isinstance(m, dict):
            return None
        out = {}
        out["tr"]  = int(m.get("tr", m.get("trades", 0)) or 0)
        out["pf"]  = float(m.get("pf", 0.0) or 0.0)
        out["exp"] = float(m.get("exp", 0.0) or 0.0)
        out["dd"]  = float(m.get("dd", 0.0) or 0.0)
        return out

    out = {}
    for asset, stratmap in raw.items():
        if not isinstance(stratmap, dict):
            continue
        out[asset] = {
            "WEEKLY": norm(stratmap.get("WEEKLY")),
            "BB_MR": norm(stratmap.get("BB_MR")),
        }
    if DEBUG_AUTOGATE:
        try:
            print(f"[AG] assets loaded: {len(out)} → {list(out.keys())}")
        except Exception:
            pass
    return out

def load_retrain_metrics() -> dict:
    if os.path.exists(AUTO_GATE_RETRAIN_JSON):
        return _load_retrain_json(AUTO_GATE_RETRAIN_JSON)
    if os.path.exists(AUTO_GATE_RETRAIN_TEXT):
        return _load_retrain_text(AUTO_GATE_RETRAIN_TEXT)
    if DEBUG_AUTOGATE:
        print(f"[AG] no metrics files found. JSON={AUTO_GATE_RETRAIN_JSON}, TEXT={AUTO_GATE_RETRAIN_TEXT}")
    return {}

def _qualifies(m: dict | None) -> bool:
    if not m: 
        return False
    try:
        return (m.get("tr",0) >= AUTO_GATE_MIN_TR and
                float(m.get("pf",0.0)) >= AUTO_GATE_MIN_PF and
                float(m.get("exp",0.0)) >= AUTO_GATE_MIN_EXP and
                float(m.get("dd",999.0)) <= AUTO_GATE_MAX_DD)
    except Exception:
        return False

def _severe_negative(m: dict | None) -> bool:
    if not m:
        return False
    try:
        pf = float(m.get("pf", 0.0))
        exp = float(m.get("exp", 0.0))
        dd = float(m.get("dd", 0.0))
        return (exp < 0.0) or (pf < 0.9) or (dd > (AUTO_GATE_MAX_DD * 1.2))
    except Exception:
        return False

def _cooldown_ok(ag_state: dict, asset: str, new_allow: bool) -> bool:
    rec = (ag_state.get("assets", {}) or {}).get(asset)
    if not rec:
        return True
    last_allow = bool(rec.get("allow", True))
    last_ts = rec.get("ts")
    try:
        dt_last = datetime.fromisoformat(last_ts.replace("Z",""))
    except Exception:
        return True
    if last_allow == new_allow:
        return True
    return (datetime.now(timezone.utc) - dt_last) >= timedelta(days=AUTO_GATE_COOLDOWN_DAYS)

def apply_auto_gate(current: dict, ag_state: dict) -> tuple[dict, list[str], dict]:
    params = deepcopy(current or {})
    ag_state = deepcopy(ag_state or {"assets":{}})
    metrics = load_retrain_metrics()
    if DEBUG_AUTOGATE:
        print(f"[AG] ENABLE={AUTO_GATE_ENABLE}  METRICS={AUTO_GATE_RETRAIN_JSON}")
        print(f"[AG] strategies considered: {sorted(list(AUTO_GATE_STRATS))}")
        print(f"[AG] thresholds: TR≥{AUTO_GATE_MIN_TR}, PF≥{AUTO_GATE_MIN_PF}, EXP≥{AUTO_GATE_MIN_EXP}, DD≤{AUTO_GATE_MAX_DD}")

    changes = []
    if not metrics:
        if DEBUG_AUTOGATE:
            print("[AG] metrics empty → no changes")
        return params, changes, ag_state

    for asset, strat_map in metrics.items():
        if DEBUG_AUTOGATE:
            print(f"[AG] asset {asset}: {strat_map}")
        m_list = []
        if isinstance(strat_map, dict):
            for s in AUTO_GATE_STRATS:
                m_list.append((s, strat_map.get(s)))
        allow = None
        if any(_qualifies(m) for (_,m) in m_list):
            allow = True
        elif any(_severe_negative(m) for (_,m) in m_list):
            allow = False

        if allow is None:
            if DEBUG_AUTOGATE:
                print(f"[AG] {asset}: no decision (no qualifiers and no severe negatives)")
            continue

        if not allow and any(_severe_negative(m) for (_,m) in m_list):
            pass
        else:
            if not _cooldown_ok(ag_state, asset, allow):
                if DEBUG_AUTOGATE:
                    print(f"[AG] {asset}: cooldown blocks flip")
                continue

        cur_allow = (params.get("asset_overrides", {})
                           .get(asset, {})
                           .get("allow_new_entries", True))

        if bool(cur_allow) == bool(allow):
            if DEBUG_AUTOGATE:
                print(f"[AG] {asset}: already allow={allow}, no change")
            continue

        params.setdefault("asset_overrides", {}).setdefault(asset, {})
        params["asset_overrides"][asset]["allow_new_entries"] = bool(allow)

        ag_state.setdefault("assets", {})[asset] = {
            "allow": bool(allow),
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        why = "qualifica" if allow else "negativo"
        changes.append(f"AutoGate {asset}: allow_new_entries {cur_allow} -> {allow} ({why})")
        if DEBUG_AUTOGATE:
            print(f"[AG] CHANGE {asset}: allow {cur_allow} -> {allow} ({why})")

    if DEBUG_AUTOGATE and not changes:
        print("[AG] no changes computed. Either no severe negatives/qualifiers, or flags already match.")
    return params, changes, ag_state

# ---------------- Retrain/Refresh (opzionale) ----------------
def maybe_run_retrain() -> tuple[bool, str]:
    cmd = CALIBRATOR_RUN_RETRAIN_CMD.strip()
    if not cmd:
        return False, "no_cmd"
    try:
        proc = subprocess.run(cmd, shell=True, cwd=ROOT, timeout=1800)
        return (proc.returncode == 0), f"retcode={proc.returncode}"
    except Exception as e:
        return False, f"exc={e!r}"

# ---------------- Main ----------------
def main():
    load_dotenv()

    os.makedirs(PARAMS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    params   = read_json(CURRENT_FILE, default={})
    plan     = read_json(SOFT_PLAN_FILE, default={})
    state    = read_json(SOFT_STATE_FILE, default={"applied":{}})
    macro_ov = read_json(MACRO_OV_FILE, default={})

    print(f"\n=== ODIN AI CALIBRATOR BOOT ===")
    print(f"AUTO_GATE_ENABLE={AUTO_GATE_ENABLE} | MODE={CALIBRATOR_MODE} | RETRAIN_JSON={AUTO_GATE_RETRAIN_JSON}")
    print(f"PARAMS_FILE={CURRENT_FILE}")
    print(f"--------------------------------\n")

    apply_macro = (CALIBRATOR_MODE in ("AUTO","MACRO","MACRO+RETRAIN","FRESH_RETRAIN"))
    apply_soft  = (CALIBRATOR_MODE in ("AUTO","SOFT_ONLY","FRESH_RETRAIN"))

    # 1️⃣ Macro
    macro_age_min = _mtime_minutes(MACRO_OV_FILE)
    if CALIBRATOR_MODE == "MACRO" and not CALIBRATOR_MACRO_APPLY_IF_STALE and macro_age_min > CALIBRATOR_MAX_AGE_MIN:
        apply_macro = False

    params_after_macro = deepcopy(params)
    macro_changed = False
    if apply_macro:
        params_after_macro = apply_macro_overrides(params_after_macro, macro_ov)
        macro_changed = (params_after_macro != params)

    # 2️⃣ Softeners
    params_after_soft, soft_changed, soft_reverted = params_after_macro, [], []
    if apply_soft:
        params_after_soft, soft_changed, soft_reverted = apply_softeners(params_after_macro, plan, state)

    # 3️⃣ Auto-Gate (forzato se abilitato)
    auto_changes = []
    auto_state = read_json(AUTO_GATE_STATE_FILE, default={"assets":{}})
    if AUTO_GATE_ENABLE:
        print("[BOOT] AutoGate attivo → applico logica su retrain_metrics.json\n")
        params_after_soft, auto_changes, auto_state = apply_auto_gate(params_after_soft, auto_state)
    else:
        print("[BOOT] AutoGate disattivo (AUTO_GATE_ENABLE != 1)\n")

    # 4️⃣ Persist
    anything_changed = (params_after_soft != params)
    if anything_changed:
        print("[WRITE] Salvo modifiche in current.json")
        write_json(CURRENT_FILE, params_after_soft)
    else:
        print("[WRITE] Nessuna modifica ai parametri.")
    write_json(SOFT_STATE_FILE, state)
    if AUTO_GATE_ENABLE:
        write_json(AUTO_GATE_STATE_FILE, auto_state)

    # 5️⃣ Determina etichetta azione
    action_parts = []
    if CALIBRATOR_MODE == "FRESH_RETRAIN":
        action_parts.append("FRESH_RETRAIN")
    elif CALIBRATOR_MODE == "MACRO+RETRAIN":
        action_parts.append("MACRO+RETRAIN")
    else:
        if apply_macro and macro_changed: action_parts.append("MACRO")
        if apply_soft and soft_changed:   action_parts.append("SOFT_APPLY")
        if apply_soft and (soft_reverted and not soft_changed): action_parts.append("SOFT_REVERT_ONLY")
        if AUTO_GATE_ENABLE and auto_changes: action_parts.append("AUTOGATE")
        if not action_parts:
            action_parts.append("NOOP")

    # 6️⃣ Retrain
    retrain_flag = False
    retrain_info = ""
    trigger_retrain = DO_RETRAIN_FLAG or ("FRESH_RETRAIN" in action_parts) or ("MACRO+RETRAIN" in action_parts)
    if trigger_retrain:
        retrain_flag, retrain_info = maybe_run_retrain()
        if "NOOP" in action_parts and retrain_flag:
            action_parts = ["FRESH_RETRAIN"]

    # 7️⃣ Telegram
    changed_lines = []
    if macro_changed: changed_lines.append("…macro caps/allow entries aggiornati…")
    if soft_changed:  changed_lines.extend(soft_changed)
    if AUTO_GATE_ENABLE and auto_changes: changed_lines.extend(auto_changes)

    autogate_line = ("AutoGate: " + "; ".join(auto_changes[:8]) + ("" if len(auto_changes) <= 8 else " …")) \
                    if (AUTO_GATE_ENABLE and auto_changes) else "AutoGate: —"
    softeners_line = ("Softeners: " + "; ".join(soft_changed[:8]) + ("" if len(soft_changed) <= 8 else " …")) \
                    if soft_changed else "Softeners: —"
    reverted_line  = ("Expired/Reverted: " + "; ".join(soft_reverted[:8]) + ("" if len(soft_reverted) <= 8 else " …")) \
                    if soft_reverted else "Expired/Reverted: —"

    msg = (
        "🤖 AI Calibrator\n"
        f"Azione: {'+'.join(action_parts)}\n"
        f"Changed: {('; '.join(changed_lines)) if changed_lines else '—'}\n"
        f"{autogate_line}\n{softeners_line}\n{reverted_line}"
    )
    if retrain_info:
        msg += f"\nRetrain: {retrain_flag} ({retrain_info})"

    print("\n--- REPORT ---")
    print(msg)
    print("--------------\n")
    send_tg(msg)


if __name__ == "__main__":
    main()
