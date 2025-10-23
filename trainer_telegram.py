import os, json

def send_trainer_tg(report_path: str, stats: dict, lines: list):
    tok = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    cid = os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not tok or not cid: return
    # messaggio ASCII-safe
    head = "ODIN Trainer Report\n"
    body = []
    body.append(f"File: {os.path.basename(report_path)}")
    if stats:
        ks = list(stats.items())[:8]
        for (asset, strat), v in ks:
            body.append(f"{asset}/{strat}: n={v.get('n')} wr={v.get('winrate',0):.2f} avg={v.get('avg',0):.2f}")
    if lines:
        body.append("--- Decisioni ---")
        for ln in lines[:12]:
            body.append(f"- {ln}")
    text = head + "\n".join(body)
    import requests
    requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                  json={"chat_id": cid, "text": text, "disable_web_page_preview": True}, timeout=10)
