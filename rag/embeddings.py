from functools import lru_cache
from time import perf_counter


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _model_auth_kwargs() -> dict:
    """Usa o token do Hugging Face quando configurado no Streamlit."""
    import streamlit as st

    token = st.secrets.get("HF_TOKEN", "")
    return {"token": token} if token else {}


@lru_cache(maxsize=1)
def _get_model():
    """Singleton do modelo de embeddings."""
    from sentence_transformers import SentenceTransformer

    started_at = perf_counter()
    model = SentenceTransformer(MODEL_NAME, **_model_auth_kwargs())
    print(
        f"[Performance] Modelo de embeddings carregado em "
        f"{perf_counter() - started_at:.2f}s",
        flush=True,
    )
    return model


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Singleton do modelo de re-ranking."""
    from sentence_transformers import CrossEncoder

    started_at = perf_counter()
    model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, **_model_auth_kwargs())
    print(
        f"[Performance] Cross-Encoder carregado em "
        f"{perf_counter() - started_at:.2f}s",
        flush=True,
    )
    return model


def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """Retorna o score de relevância para pares (query, documento)."""
    if not pairs:
        return []

    started_at = perf_counter()
    model = _get_cross_encoder()
    scores = model.predict(
        pairs,
        batch_size=min(16, len(pairs)),
        show_progress_bar=False,
    )
    print(
        f"[Performance] Re-ranking de {len(pairs)} trechos em "
        f"{perf_counter() - started_at:.2f}s",
        flush=True,
    )
    # Garante que retorne uma lista de floats em vez de numpy array
    try:
        return scores.tolist()
    except AttributeError:
        # Caso o retorno já seja lista
        return list(scores)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos.
    Retorna lista de vetores (384 dimensões).
    """
    if not texts:
        return []

    started_at = perf_counter()
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=min(32, len(texts)),
        show_progress_bar=False,
    )
    print(
        f"[Performance] Embeddings de {len(texts)} textos em "
        f"{perf_counter() - started_at:.2f}s",
        flush=True,
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """
    Gera embedding para uma única query de busca.
    Retorna vetor de 384 dimensões.
    """
    model = _get_model()
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()
