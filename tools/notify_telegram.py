import os, sys, requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

token = os.getenv("TELEGRAM_BOT_TOKEN", "")
chat  = os.getenv("TELEGRAM_CHAT_ID", "")
msg   = " ".join(sys.argv[1:]) or "[ODIN] job finito"

if token and chat:
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": msg},
            timeout=10
        )
    except Exception as e:
        # Silenzioso: non blocca il .bat
        pass
