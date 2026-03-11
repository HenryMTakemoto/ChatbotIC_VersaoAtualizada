from .embeddings import embed_query
from db.supabase_client import get_supabase


def hybrid_search(query: str, user_id: str | None = None, top_k: int = 5) -> str:
    """
    Busca híbrida usando similaridade de cosseno via pgvector.
    Combina documentos globais + pessoais do usuário.

    Retorna contexto formatado como string, ou string vazia se nada encontrado.
    """
    supabase = get_supabase()
    query_embedding = embed_query(query)

    # Chama função SQL search_chunks_hybrid no Supabase
    result = supabase.rpc("search_chunks_hybrid", {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "p_user_id": user_id
    }).execute()

    if not result.data:
        return ""

    # Formata contexto para o LLM
    context_parts = []
    for i, chunk in enumerate(result.data, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_name", "Desconhecido")
        page = metadata.get("page", "?")
        content = chunk.get("content", "")
        similarity = chunk.get("similarity", 0)

        context_parts.append(
            f"[Trecho {i}] (Fonte: {source}, Página: {page}, "
            f"Similaridade: {similarity:.2f})\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)
