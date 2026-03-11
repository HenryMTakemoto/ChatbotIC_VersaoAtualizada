from functools import lru_cache
from sentence_transformers import SentenceTransformer


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Singleton do modelo de embeddings."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos.
    Retorna lista de vetores (384 dimensões).
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """
    Gera embedding para uma única query de busca.
    Retorna vetor de 384 dimensões.
    """
    model = _get_model()
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()
