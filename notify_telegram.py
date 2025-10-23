import os, sys, json, urllib.request, urllib.parse, ssl

def die(msg, code=2):
    print("[notify] ERROR:", msg)
    sys.exit(code)

def main():
    text = None
    # Accetta --text "..." oppure tutto il resto come testo
    if "--text" in sys.argv:
        i = sys.argv.index("--text")
        if i+1 < len(sys.argv):
            text = sys.argv[i+1]
            args = sys.argv[:i] + sys.argv[i+2:]
        else:
            die("missing text after --text")
    else:
        # se non c'è --text prendi tutto unito
        args = sys.argv[1:]
        text = " ".join(args).strip() if args else None

    if not text:
        text = "[ODIN] ping"

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat  = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        die("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID non settati nell'ambiente", 3)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")

    # per alcune reti con ispezione SSL
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
            raw = r.read().decode("utf-8", "ignore")
            j = json.loads(raw)
            if not j.get("ok"):
                die(f"Telegram response not ok: {raw}", 4)
            print("[notify] OK")
            sys.exit(0)
    except Exception as e:
        die(f"Exception contacting Telegram: {e}", 5)

if __name__ == "__main__":
    main()
