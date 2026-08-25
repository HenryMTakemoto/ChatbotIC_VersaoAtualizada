import hashlib
from telegram import Update
from telegram.ext import ContextTypes
from langchain_core.messages import HumanMessage, AIMessage
from db.supabase_client import get_supabase
from db.conversations import (
    create_conversation, save_message, get_conversation_messages
)
from rag.retriever import hybrid_search
from llm.chat import build_messages_with_rag, invoke_llm

def get_user_id_by_telegram(telegram_user_id: int) -> str | None:
    """Retorna o user_id do Supabase vinculado ao telegram_user_id."""
    supabase = get_supabase()
    result = supabase.table("profiles").select("id").eq(
        "telegram_user_id", telegram_user_id
    ).execute()
    return result.data[0]["id"] if result.data else None

async def telegram_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = get_user_id_by_telegram(update.effective_user.id)

    if user_id:
        msg = "✅ Conta vinculada! Pode perguntar sobre os documentos."
    else:
        msg = (
            "Olá! 🚜 Sou o Assistente Cia Agro.\n\n"
            "Para acessar seus documentos pessoais, vincule sua conta:\n"
            "1. Acesse o painel web\n"
            "2. Vá em Configurações → Vincular Telegram\n"
            "3. Envie /vincular <seu_codigo> aqui\n\n"
            "Ou pergunte sem vincular — usarei apenas a base global."
        )

    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

async def telegram_vincular(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /vincular <codigo>"""
    if not context.args:
        await update.message.reply_text("Use: /vincular <seu_codigo>")
        return

    code = context.args[0]
    supabase = get_supabase()

    # Verifica se o código existe (é o user_id ou um hash dele)
    result = supabase.table("profiles").select("id").eq("id", code).execute()

    if not result.data:
        await update.message.reply_text("❌ Código inválido.")
        return

    user_id = result.data[0]["id"]
    telegram_user_id = update.effective_user.id

    supabase.table("profiles").update({
        "telegram_user_id": telegram_user_id
    }).eq("id", user_id).execute()

    await update.message.reply_text("✅ Conta vinculada com sucesso!")

async def telegram_nova_conversa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /nova — inicia uma nova conversa, descartando o histórico atual."""
    telegram_uid = update.effective_user.id
    session_key = f"tg_conv_{telegram_uid}"

    # Remove a conversa ativa do cache para forçar criação de uma nova
    if session_key in context.user_data:
        del context.user_data[session_key]

    await update.message.reply_text("🆕 Nova conversa iniciada! Pode perguntar.")

async def telegram_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /status — mostra se a conta Telegram está vinculada."""
    user_id = get_user_id_by_telegram(update.effective_user.id)

    if user_id:
        await update.message.reply_text(
            "✅ Sua conta está vinculada.\n"
            "Você tem acesso aos documentos globais e pessoais."
        )
    else:
        await update.message.reply_text(
            "❌ Conta não vinculada.\n"
            "Use /vincular <codigo> para vincular.\n"
            "Enquanto isso, só posso consultar a base global."
        )

async def telegram_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_uid = update.effective_user.id
    text = update.message.text
    user_id = get_user_id_by_telegram(telegram_uid)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        # Busca ou cria conversa ativa para este usuário Telegram
        session_key = f"tg_conv_{telegram_uid}"
        if session_key not in context.user_data:
            if user_id:
                conv_id = create_conversation(user_id, source="telegram")
            else:
                # Usuário não vinculado: conversa sem user_id (anônima)
                conv_id = create_conversation("anonymous", source="telegram")
            context.user_data[session_key] = conv_id
        else:
            conv_id = context.user_data[session_key]

        # Busca histórico recente
        msgs = get_conversation_messages(conv_id)
        history = []
        for m in msgs[-10:]:  # últimas 5 turns
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            else:
                history.append(AIMessage(content=m["content"]))

        # RAG + LLM
        messages, rag_used = build_messages_with_rag(text, history, user_id)
        reply = invoke_llm(messages)
        if rag_used:
            reply += "\n\n📄 Contexto recuperado da base documental."
        else:
            reply += (
                "\n\n⚠️ Nenhum contexto relevante foi recuperado; "
                "a resposta não está fundamentada na base documental."
            )

        # Salva no banco
        if user_id:
            save_message(conv_id, "user", text)
            save_message(conv_id, "assistant", reply, rag_used)

        # Envia resposta
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=reply,
                parse_mode="Markdown"
            )
        except Exception:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=reply
            )

    except Exception as e:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"⚠️ Erro: {str(e)[:100]}"
        )
