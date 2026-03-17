import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from auth.authenticator import get_current_user_id
from db.conversations import (
    create_conversation,
    get_user_conversations,
    save_message,
    get_conversation_messages,
    update_conversation_title,
)
from llm.chat import build_messages_with_rag, invoke_llm


def show_chat_page():
    user_id = get_current_user_id()

    # --- Área principal do chat ---
    st.markdown("### 💬 Chat — Assistente Cia Agro")

    conversations = get_user_conversations(user_id)

    # Inicializa conversa ativa se necessário
    if "active_conversation" not in st.session_state or not st.session_state.active_conversation:
        if conversations:
            st.session_state.active_conversation = conversations[0]["id"]
        else:
            conv_id = create_conversation(user_id)
            st.session_state.active_conversation = conv_id

    active_conv = st.session_state.active_conversation

    # Carrega histórico do banco
    if "chat_messages" not in st.session_state or not st.session_state.chat_messages:
        db_messages = get_conversation_messages(active_conv)
        st.session_state.chat_messages = [
            {
                "role": m["role"],
                "content": m["content"],
                "rag_used": m.get("rag_context_used", False),
            }
            for m in db_messages
        ]

    # Exibe mensagens
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("rag_used") and msg["role"] == "assistant":
                st.caption("📄 Resposta baseada em documentos indexados")

    # Input do usuário
    if prompt := st.chat_input("Digite sua pergunta..."):
        # Exibe mensagem do usuário
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt, "rag_used": False}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Atualiza título na 1ª mensagem
        if len(st.session_state.chat_messages) == 1:
            update_conversation_title(active_conv, prompt)

        # Salva mensagem do usuário
        save_message(active_conv, "user", prompt)

        # Monta histórico para o LLM
        history = []
        for m in st.session_state.chat_messages[:-1]:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            else:
                history.append(AIMessage(content=m["content"]))

        # Gera resposta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                messages, rag_used = build_messages_with_rag(
                    prompt, history, user_id
                )
                reply = invoke_llm(messages)

            st.markdown(reply)
            if rag_used:
                st.caption("📄 Resposta baseada em documentos indexados")

        # Salva resposta
        save_message(active_conv, "assistant", reply, rag_used)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": reply, "rag_used": rag_used}
        )
