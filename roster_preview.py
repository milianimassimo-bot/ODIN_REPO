# roster_preview.py
import os, json, sys

ROSTER = os.getenv("ODIN_STRATEGY_ROSTER", os.path.join("params","strategy_roster.json"))

def main():
    try:
        with open(ROSTER,"r",encoding="utf-8") as f:
            r = json.load(f)
    except Exception as e:
        print(f"Errore a leggere {ROSTER}: {e}")
        sys.exit(1)

    print(f"Roster @ {r.get('asof_utc')}\nUniverse: {', '.join(r.get('universe',[]))}\n")
    print("== Ranking globale strategie (top 10) ==")
    for i, row in enumerate(sorted(r.get('ranking_global',[]), key=lambda x: x.get('avg_score', 0), reverse=True)[:10], 1):
        print(f"{i:>2}. {row['strategy']:<28} score={row.get('avg_score', 0):.4f}")
    print("\n== Strategie attive per ogni asset ==")
    per = r.get("per_asset", {})
    for a in sorted(per.keys()):
        active = per[a].get("active", [])
        print(f"- {a}: {', '.join(active) if active else '—'}")
    print("\nOK.")

if __name__ == "__main__":
    main()