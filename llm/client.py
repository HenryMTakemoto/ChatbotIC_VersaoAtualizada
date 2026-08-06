from functools import lru_cache
import streamlit as st


class LLMWithFallback:
    """Invoca os provedores em ordem até que um deles responda."""

    def __init__(self, providers):
        self.providers = providers

    def invoke(self, messages):
        errors = []

        for index, (provider_name, llm) in enumerate(self.providers):
            try:
                return llm.invoke(messages)
            except Exception as e:
                errors.append(f"{provider_name}: {str(e)[:200]}")
                has_fallback = index < len(self.providers) - 1
                if has_fallback:
                    print(
                        f"[LLM] {provider_name} falhou durante a chamada "
                        f"({str(e)[:200]}), tentando próximo provedor..."
                    )

        raise RuntimeError(
            "Todos os provedores de LLM falharam: " + " | ".join(errors)
        )


@lru_cache(maxsize=1)
def get_llm():
    """
    Singleton do LLM. Tenta NVIDIA NIM primeiro e usa Groq como fallback,
    inclusive quando a falha acontece durante a chamada ao modelo.
    """
    providers = []

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        nvidia = ChatNVIDIA(
            nvidia_api_key=st.secrets["NVIDIA_API_KEY"],
            model="meta/llama-3.3-70b-instruct",
            temperature=0.5,
            max_tokens=1024
        )
        providers.append(("NVIDIA", nvidia))
    except Exception as e:
        print(f"[LLM] NVIDIA não pôde ser inicializada ({str(e)[:200]}).")

    try:
        from langchain_groq import ChatGroq
        groq = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"],
            model_name="llama-3.3-70b-versatile",
            temperature=0.5
        )
        providers.append(("Groq", groq))
    except Exception as e:
        print(f"[LLM] Groq não pôde ser inicializado: {str(e)[:200]}")

    return LLMWithFallback(providers) if providers else None
