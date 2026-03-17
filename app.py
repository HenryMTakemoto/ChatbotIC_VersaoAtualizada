import streamlit as st
from auth.authenticator import init_session, is_authenticated, get_current_user, get_current_user_id, logout
from auth.login_page import show_login_page
from telegram_bot.runner import start_background_bot
from db.conversations import get_user_conversations, create_conversation

# Configuração da página
st.set_page_config(
    page_title="Cia Agro",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS adicional para ocultar navegação padrão do streamlit
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# Inicializa sessão de auth
init_session()

# Inicia Telegram bot em background
start_background_bot()

# Roteamento principal
if not is_authenticated():
    show_login_page()
else:
    user = get_current_user()
    user_id = get_current_user_id()

    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"

    def navigate_to(page):
        st.session_state.current_page = page

    # Sidebar com navegação
    with st.sidebar:
        st.markdown(f"**👤 {user['email']}**")
        
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.current_page == "chat" else "secondary"):
                navigate_to("chat")
                st.rerun()
        with col_nav2:
            if st.button("📁 Upload", use_container_width=True, type="primary" if st.session_state.current_page == "admin" else "secondary"):
                navigate_to("admin")
                st.rerun()
                
        if st.button("📚 Ver Todo o Histórico", use_container_width=True, type="primary" if st.session_state.current_page == "history" else "secondary"):
            navigate_to("history")
            st.rerun()

        st.markdown("---")

        if st.button("➕ Nova Conversa", use_container_width=True, type="primary"):
            conv_id = create_conversation(user_id)
            st.session_state.active_conversation = conv_id
            st.session_state.chat_messages = []
            st.session_state.current_page = "chat"
            st.rerun()

        st.markdown("#### 🕒 Conversas Recentes")
        conversations = get_user_conversations(user_id)

        for conv in conversations[:15]:  # Mostrar até 15 recentes
            col1, col2 = st.columns([5, 1])
            with col1:
                # Truncate title se for muito longo
                display_title = conv['title'] if len(conv['title']) < 25 else conv['title'][:22] + "..."
                if st.button(
                    f"📝 {display_title}",
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                ):
                    st.session_state.active_conversation = conv["id"]
                    st.session_state.chat_messages = []
                    st.session_state.current_page = "chat"
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_conv_{conv['id']}"):
                    from db.supabase_client import get_supabase
                    supabase = get_supabase()
                    supabase.table("conversations").delete().eq(
                        "id", conv["id"]
                    ).execute()
                    if st.session_state.get("active_conversation") == conv["id"]:
                        st.session_state.active_conversation = None
                        st.session_state.chat_messages = []
                    st.rerun()

        st.markdown("---")
        if st.button("🚪 Sair", use_container_width=True):
            logout()

    # Renderiza página selecionada
    if st.session_state.current_page == "chat":
        from pages.chat import show_chat_page
        show_chat_page()
    elif st.session_state.current_page == "history":
        from pages.history import show_history_page
        show_history_page()
    elif st.session_state.current_page == "admin":
        from pages.admin import show_admin_page
        show_admin_page()