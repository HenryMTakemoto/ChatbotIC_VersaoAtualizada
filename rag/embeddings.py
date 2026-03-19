from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Singleton do modelo de embeddings."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    """Singleton do modelo de re-ranking."""
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """Retorna o score de relevância para pares (query, documento)."""
    model = _get_cross_encoder()
    scores = model.predict(pairs, show_progress_bar=False)
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
