import streamlit as st
from auth.authenticator import get_current_user_id
from db.conversations import get_user_conversations, get_conversation_messages


def show_history_page():
    user_id = get_current_user_id()
    st.markdown("### 📚 Histórico de Conversas")

    conversations = get_user_conversations(user_id)

    if not conversations:
        st.info("Você ainda não tem conversas. Vá para o Chat e faça sua primeira pergunta!")
        return

    st.caption(f"{len(conversations)} conversa(s) encontrada(s)")
    st.markdown("---")

    for conv in conversations:
        source_icon = "💬" if conv["source"] == "web" else "📱"
        created = conv["created_at"][:10]  # YYYY-MM-DD

        with st.expander(f"{source_icon} {conv['title']}  •  {created}"):
            msgs = get_conversation_messages(conv["id"])
            if not msgs:
                st.caption("Conversa vazia.")
                continue

            for msg in msgs:
                role_label = "🧑 Você" if msg["role"] == "user" else "🤖 Assistente"
                rag_badge = " 📄" if msg.get("rag_context_used") else ""
                st.markdown(f"**{role_label}{rag_badge}**")
                st.markdown(msg["content"])
                st.markdown("---")
