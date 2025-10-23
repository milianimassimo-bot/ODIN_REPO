# macro_apply.py — applica uno dei profili su macro_overrides.json (safe & handy)
import os, sys, json
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(ROOT, "params")
PROFILES_PATH = os.path.join(PARAMS_DIR, "macro_profiles.json")
OUT_PATH = os.path.join(PARAMS_DIR, "macro_overrides.json")

def utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load_profiles(path):
    if not os.path.exists(path):
        sys.exit(f"[ERR] File profili non trovato: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
    except Exception as e:
        sys.exit(f"[ERR] Lettura profili fallita: {e!r}")
    profs = js.get("profiles")
    if not isinstance(profs, list):
        sys.exit("[ERR] macro_profiles.json: campo 'profiles' mancante o non lista")
    return profs

def main():
    args = sys.argv[1:]
    if not args or args[0] in {"-h","--help"}:
        print("Uso:")
        print("  python macro_apply.py --list")
        print("  python macro_apply.py <Profilo> [--dry-run] [--disallow-others]")
        return

    if args[0] == "--list":
        profs = load_profiles(PROFILES_PATH)
        names = [p.get("name","?") for p in profs]
        print("Profili disponibili:", ", ".join(names))
        return

    name = args[0]
    dry = "--dry-run" in args
    disallow_others = "--disallow-others" in args

    profs = load_profiles(PROFILES_PATH)
    prof = next((p for p in profs if str(p.get("name","")).lower() == name.lower()), None)
    if not prof:
        sys.exit(f"[ERR] Profilo '{name}' non trovato. Usa --list per vedere i nomi.")

    try:
        mult = float(prof["mult"])
    except Exception:
        sys.exit("[ERR] Profilo: 'mult' mancante o non numerico.")

    allow = prof.get("allow", [])
    if not isinstance(allow, list):
        sys.exit("[ERR] Profilo: 'allow' deve essere una lista di asset.")

    # Costruisci output
    out = {
        "generated_utc": utc_iso(),
        "global": {"risk_caps_multiplier": mult},
        "asset_overrides": {a: {"allow_new_entries": True} for a in allow}
    }

    # Opzione: disallow per tutti gli altri, se già presenti nel file attuale
    if disallow_others and os.path.exists(OUT_PATH):
        try:
            cur = json.load(open(OUT_PATH, "r", encoding="utf-8"))
            cur_ao = cur.get("asset_overrides", {})
            for a in cur_ao.keys():
                if a not in out["asset_overrides"]:
                    out["asset_overrides"][a] = {"allow_new_entries": False}
        except Exception:
            pass  # se fallisce, pazienza: non blocchiamo l'apply

    # Esito
    if dry:
        print(f"[DRY-RUN] Scriverei in {OUT_PATH}:")
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    os.makedirs(PARAMS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OK] Applicato profilo '{name}': mult={mult} allow={len(allow)} asset → {OUT_PATH}")
    if disallow_others:
        print("[INFO] --disallow-others: gli asset non in 'allow' sono stati impostati a allow_new_entries=False (se esistevano già).")

if __name__ == "__main__":
    main()
