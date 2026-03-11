import streamlit as st
from .authenticator import login, register


def show_login_page():
    """Página de login e registro com abas."""
    st.markdown("## 🚜 Cia Agro — Assistente Agronômico")
    st.markdown("---")

    tab_login, tab_register = st.tabs(["🔑 Entrar", "📝 Criar Conta"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Senha", type="password", key="login_password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)

            if submitted:
                if not email or not password:
                    st.error("Preencha todos os campos.")
                else:
                    success, msg = login(email, password)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    with tab_register:
        with st.form("register_form"):
            name = st.text_input("Nome completo", key="reg_name")
            email_reg = st.text_input("Email", key="reg_email")
            password_reg = st.text_input("Senha", type="password", key="reg_password")
            password_confirm = st.text_input(
                "Confirmar senha", type="password", key="reg_password_confirm"
            )
            submitted_reg = st.form_submit_button(
                "Criar Conta", use_container_width=True
            )

            if submitted_reg:
                if not name or not email_reg or not password_reg:
                    st.error("Preencha todos os campos.")
                elif password_reg != password_confirm:
                    st.error("As senhas não coincidem.")
                elif len(password_reg) < 6:
                    st.error("A senha deve ter pelo menos 6 caracteres.")
                else:
                    success, msg = register(email_reg, password_reg, name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
