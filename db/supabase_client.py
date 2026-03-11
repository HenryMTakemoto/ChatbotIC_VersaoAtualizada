from functools import lru_cache
import streamlit as st
from supabase import create_client, Client


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Singleton do cliente Supabase — reutilizado em toda a aplicação."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_PUBLISHABLE_KEY"]
    return create_client(url, key)


def get_supabase_admin() -> Client:
    """
    Cliente com service role key — usar APENAS em operações admin do backend.
    Nunca expor no frontend.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SECRET_KEY"]
    return create_client(url, key)
