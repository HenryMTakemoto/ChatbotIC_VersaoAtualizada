import streamlit as st
from auth.authenticator import init_session, is_authenticated, get_current_user, logout
from auth.login_page import show_login_page
from telegram_bot.runner import start_background_bot

# Configuração da página
st.set_page_config(
    page_title="Cia Agro",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializa sessão de auth
init_session()

# Inicia Telegram bot em background
start_background_bot()

# Roteamento principal
if not is_authenticated():
    show_login_page()
else:
    user = get_current_user()

    # Sidebar com navegação
    with st.sidebar:
        st.markdown(f"### 👤 {user['email']}")
        st.markdown("---")

        page = st.radio(
            "Navegação",
            ["💬 Chat", "📚 Histórico", "⚙️ Admin"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        if st.button("🚪 Sair", use_container_width=True):
            logout()

    # Renderiza página selecionada
    if page == "💬 Chat":
        from pages.chat import show_chat_page
        show_chat_page()
    elif page == "📚 Histórico":
        from pages.history import show_history_page
        show_history_page()
    elif page == "⚙️ Admin":
        from pages.admin import show_admin_page
        show_admin_page()