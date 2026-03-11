from functools import lru_cache
import streamlit as st


@lru_cache(maxsize=1)
def get_llm():
    """
    Singleton do LLM. Tenta NVIDIA NIM primeiro, faz fallback para Groq.
    """
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            nvidia_api_key=st.secrets["NVIDIA_API_KEY"],
            model="meta/llama-3.1-405b-instruct",
            temperature=0.5,
            max_tokens=1024
        )
    except Exception as e:
        print(f"[LLM] NVIDIA falhou ({e}), tentando Groq...")

    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"],
            model_name="llama-3.3-70b-versatile",
            temperature=0.5
        )
    except Exception as e:
        print(f"[LLM] Groq também falhou: {e}")
        return None
