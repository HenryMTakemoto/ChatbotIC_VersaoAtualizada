from functools import lru_cache
import streamlit as st
from supabase import create_client, Client, ClientOptions


def _query_timeout_seconds() -> float:
    try:
        value = float(st.secrets.get("SUPABASE_QUERY_TIMEOUT_SECONDS", 20))
    except (TypeError, ValueError):
        return 20
    return value if value > 0 else 20


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Singleton do cliente Supabase — reutilizado em toda a aplicação."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_PUBLISHABLE_KEY"]
    options = ClientOptions(
        postgrest_client_timeout=_query_timeout_seconds(),
    )
    return create_client(url, key, options=options)


def get_supabase_admin() -> Client:
    """
    Cliente com service role key — usar APENAS em operações admin do backend.
    Nunca expor no frontend.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SECRET_KEY"]
    return create_client(url, key)
