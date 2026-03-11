import streamlit as st
from db.supabase_client import get_supabase

def init_session():
    """Inicializa variáveis de sessão de autenticação."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None

def is_authenticated() -> bool:
    return st.session_state.get("user") is not None

def get_current_user() -> dict | None:
    return st.session_state.get("user")

def get_current_user_id() -> str | None:
    user = get_current_user()
    return user["id"] if user else None

def is_admin() -> bool:
    user = get_current_user()
    if not user:
        return False
    supabase = get_supabase()
    result = supabase.table("profiles").select("is_admin").eq(
        "id", user["id"]
    ).single().execute()
    return result.data.get("is_admin", False) if result.data else False

def login(email: str, password: str) -> tuple[bool, str]:
    """Faz login via Supabase Auth. Retorna (sucesso, mensagem)."""
    supabase = get_supabase()
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        st.session_state.user = {
            "id": response.user.id,
            "email": response.user.email,
        }
        st.session_state.access_token = response.session.access_token
        return True, "Login realizado com sucesso!"
    except Exception as e:
        return False, f"Erro de login: {str(e)}"

def register(email: str, password: str, full_name: str) -> tuple[bool, str]:
    """Registra novo usuário via Supabase Auth."""
    from db.supabase_client import get_supabase_admin
    try:
        # Verifica diretamente no auth.users se o email já existe
        admin = get_supabase_admin()
        users = admin.auth.admin.list_users()
        if any(u.email == email for u in users):
            return False, "Este email já está cadastrado. Faça login ou use outro email."
    except Exception:
        pass  # Se a verificação falhar, tenta registrar normalmente

    supabase = get_supabase()
    try:
        supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"full_name": full_name}}
        })
        return True, "Conta criada! Verifique seu email para confirmar."
    except Exception as e:
        error_msg = str(e).lower()
        if "already registered" in error_msg or "already exists" in error_msg:
            return False, "Este email já está cadastrado. Faça login ou use outro email."
        return False, f"Erro ao registrar: {str(e)}"

def logout():
    """Limpa sessão."""
    supabase = get_supabase()
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.user = None
    st.session_state.access_token = None
    st.rerun()