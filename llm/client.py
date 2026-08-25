from functools import lru_cache
from time import perf_counter
import streamlit as st


def _positive_int_setting(name: str, default: int) -> int:
    try:
        value = int(st.secrets.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class LLMWithFallback:
    """Invoca os provedores em ordem até que um deles responda."""

    def __init__(self, providers):
        self.providers = providers

    def invoke(self, messages):
        errors = []

        for index, (provider_name, llm) in enumerate(self.providers):
            started_at = perf_counter()
            try:
                response = llm.invoke(messages)
                print(
                    f"[Performance] LLM {provider_name} respondeu em "
                    f"{perf_counter() - started_at:.2f}s",
                    flush=True,
                )
                return response
            except Exception as e:
                errors.append(f"{provider_name}: {str(e)[:200]}")
                has_fallback = index < len(self.providers) - 1
                if has_fallback:
                    print(
                        f"[LLM] {provider_name} falhou durante a chamada "
                        f"({str(e)[:200]}), tentando próximo provedor...",
                        flush=True,
                    )

        raise RuntimeError(
            "Todos os provedores de LLM falharam: " + " | ".join(errors)
        )


@lru_cache(maxsize=2)
def get_llm(purpose: str = "answer"):
    """
    Singleton por finalidade. O modelo principal redige respostas; o modelo
    utilitário executa apenas tarefas curtas de recuperação.

    O Groq usa modelos de produção atuais. Os IDs podem ser sobrescritos nos
    secrets sem alteração de código.
    """
    if purpose not in {"answer", "utility"}:
        raise ValueError(f"Finalidade de LLM desconhecida: {purpose}")

    providers = []
    is_utility = purpose == "utility"

    try:
        from langchain_groq import ChatGroq
        default_model = "openai/gpt-oss-20b" if is_utility else "openai/gpt-oss-120b"
        model_name = st.secrets.get(
            "GROQ_UTILITY_MODEL" if is_utility else "GROQ_ANSWER_MODEL",
            default_model,
        )
        groq_kwargs = {}
        if model_name.startswith("openai/gpt-oss"):
            reasoning_effort = str(st.secrets.get(
                "GROQ_UTILITY_REASONING_EFFORT" if is_utility
                else "GROQ_ANSWER_REASONING_EFFORT",
                "low",
            )).lower()
            if reasoning_effort not in {"low", "medium", "high"}:
                reasoning_effort = "low"
            groq_kwargs.update(
                reasoning_effort=reasoning_effort,
                reasoning_format="hidden",
            )
        max_tokens = _positive_int_setting(
            "LLM_UTILITY_MAX_TOKENS" if is_utility else "LLM_ANSWER_MAX_TOKENS",
            400 if is_utility else 1_200,
        )
        groq = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"],
            model_name=model_name,
            temperature=0 if is_utility else 0.1,
            max_tokens=max_tokens,
            timeout=45,
            max_retries=1,
            **groq_kwargs,
        )
        providers.append((f"Groq/{model_name}", groq))
    except Exception as e:
        print(f"[LLM] Groq não pôde ser inicializado: {str(e)[:200]}")

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        model_name = st.secrets.get(
            "NVIDIA_UTILITY_MODEL" if is_utility else "NVIDIA_ANSWER_MODEL",
            "meta/llama-3.3-70b-instruct",
        )
        max_tokens = _positive_int_setting(
            "LLM_UTILITY_MAX_TOKENS" if is_utility else "LLM_ANSWER_MAX_TOKENS",
            400 if is_utility else 1_200,
        )
        nvidia = ChatNVIDIA(
            nvidia_api_key=st.secrets["NVIDIA_API_KEY"],
            model=model_name,
            temperature=0 if is_utility else 0.1,
            max_tokens=max_tokens,
            timeout=45,
        )
        providers.append((f"NVIDIA/{model_name}", nvidia))
    except Exception as e:
        print(f"[LLM] NVIDIA não pôde ser inicializada ({str(e)[:200]}).")

    return LLMWithFallback(providers) if providers else None
