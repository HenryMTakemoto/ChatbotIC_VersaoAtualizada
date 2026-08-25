import asyncio
import threading
import streamlit as st
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from .bot import (
    telegram_start,
    telegram_nova_conversa,
    telegram_status,
    telegram_vincular,
    telegram_handle_message,
)


def _telegram_enabled() -> bool:
    value = st.secrets.get("ENABLE_TELEGRAM_BOT", True)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "sim"}
    return bool(value)


def run_telegram_bot():
    """Executa o bot do Telegram em thread separada com seu próprio event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    token = st.secrets.get("TELEGRAM_TOKEN", "")
    if not token or token == "seu_telegram_token_aqui":
        print("[Telegram] Token não configurado. Bot não iniciado.")
        return

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", telegram_start))
    app.add_handler(CommandHandler("nova", telegram_nova_conversa))
    app.add_handler(CommandHandler("status", telegram_status))
    app.add_handler(CommandHandler("vincular", telegram_vincular))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, telegram_handle_message)
    )

    print("[Telegram] Bot iniciado em background.")
    app.run_polling(stop_signals=None, drop_pending_updates=True)


@st.cache_resource
def start_background_bot():
    """Inicia o bot em daemon thread — executado uma única vez pelo Streamlit."""
    if not _telegram_enabled():
        print("[Telegram] Desativado pela configuração.", flush=True)
        return None

    t = threading.Thread(target=run_telegram_bot, daemon=True)
    t.start()
    return t
