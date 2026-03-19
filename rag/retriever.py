from .embeddings import embed_query, score_pairs
from db.supabase_client import get_supabase


def hybrid_search(query: str, user_id: str | None = None, top_k: int = 5) -> str:
    """
    Busca híbrida usando similaridade de cosseno via pgvector.
    Combina documentos globais + pessoais do usuário.

    Retorna contexto formatado como string, ou string vazia se nada encontrado.
    """
    supabase = get_supabase()
    query_embedding = embed_query(query)

    # Overfetching: busca mais resultados iniciais via similaridade de cosseno
    # Aumentado para 20 para termos boa diversidade antes do re-reranking
    fetch_count = max(top_k * 4, 20)
    
    # Chama função SQL search_chunks_hybrid no Supabase
    result = supabase.rpc("search_chunks_hybrid", {
        "query_embedding": query_embedding,
        "match_count": fetch_count,
        "p_user_id": user_id
    }).execute()

    if not result.data:
        return ""

    chunks = result.data

    # RE-RANKING (Cross-Encoder)
    # Prepara os pares (Pergunta, Trecho) para avaliação rigorosa
    pairs = []
    for chunk in chunks:
        content = chunk.get("content", "")
        pairs.append((query, content))

    # O Cross-Encoder avalia a relevância semântica verdadeira
    scores = score_pairs(pairs)

    # Associa a nota rigorosa de volta aos chunks
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    # Ordena os resultados decrescentemente pela nota do Cross-Encoder
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Mantém apenas os top_k melhores
    top_chunks = chunks[:top_k]

    # Formata contexto para o LLM
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_name", "Desconhecido")
        page = metadata.get("page", "?")
        content = chunk.get("content", "")
        cosine_sim = chunk.get("similarity", 0)
        rerank_score = chunk.get("rerank_score", 0)

        context_parts.append(
            f"[Trecho {i}] (Fonte: {source}, Página: {page}, "
            f"Cross-Score: {rerank_score:.2f}, Vector-Sim: {cosine_sim:.2f})\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)
