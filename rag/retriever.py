from .embeddings import embed_query, score_pairs
from db.supabase_client import get_supabase


# --- SELF-QUERYING: Prompt para extrair filtros de metadados ---
FILTER_EXTRACTION_PROMPT = """Analise a pergunta do usuário abaixo e determine se ele está solicitando informações de um arquivo específico pelo nome.

Exemplos de perguntas com filtro:
- "Resuma o arquivo relatorio_2024.pdf" → {"source_name": "relatorio_2024.pdf"}
- "O que diz o manual de trigo sobre adubação?" → {"source_name": "trigo"}
- "Busque no documento soja.pdf" → {"source_name": "soja.pdf"}

Exemplos de perguntas SEM filtro (busca geral):
- "Quais são as pragas da soja?" → {}
- "Como fazer adubação?" → {}
- "Explique o ciclo do milho" → {}

Responda APENAS com um objeto JSON válido, sem texto adicional, sem markdown.
Se houver filtro, retorne: {"source_name": "nome_detectado"}
Se não houver filtro, retorne: {}

Pergunta do usuário: {question}

JSON:"""


def extract_metadata_filter(question: str) -> dict:
    """
    Usa o LLM para detectar se o usuário quer um documento específico.
    Retorna um dicionário com filtros (ex: {"source_name": "manual.pdf"})
    ou um dict vazio se a busca deve ser geral.
    """
    import json
    from langchain_core.messages import HumanMessage

    # Importa lazy para evitar circular imports
    try:
        from llm.client import get_llm
        llm = get_llm()
        if not llm:
            return {}

        prompt = FILTER_EXTRACTION_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Remove markdown code fences se o LLM as incluiu
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        filters = json.loads(raw)
        if isinstance(filters, dict) and filters:
            print(f"[SelfQuery] Filtro detectado: {filters}")
        return filters if isinstance(filters, dict) else {}

    except Exception as e:
        print(f"[SelfQuery] Filtro não detectado (usando busca geral): {e}")
        return {}


def apply_metadata_filter(chunks: list, filters: dict) -> list:
    """
    Filtra chunks em memória com base nos metadados.
    Se nenhum chunk bater com o filtro, retorna a lista original (fallback seguro).
    """
    if not filters:
        return chunks

    source_filter = filters.get("source_name", "").lower()
    if not source_filter:
        return chunks

    filtered = [
        chunk for chunk in chunks
        if source_filter in chunk.get("metadata", {}).get("source_name", "").lower()
    ]

    if filtered:
        print(f"[SelfQuery] {len(filtered)}/{len(chunks)} chunks após filtro por '{source_filter}'")
        return filtered

    # Fallback: se o filtro eliminou tudo, retorna tudo (melhor responder errado do que não responder)
    print(f"[SelfQuery] Nenhum chunk bateu com '{source_filter}'. Usando busca geral.")
    return chunks


def hybrid_search(query: str, user_id: str | None = None, top_k: int = 5) -> str:
    """
    Pipeline completo de busca RAG:
    1. Self-Querying: detecta filtros de metadados via LLM
    2. Overfetching: busca Top 20 via pgvector/cosine
    3. Filtro em memória: aplica filtro de arquivo se detectado
    4. Re-ranking: Cross-Encoder reordena pelos mais relevantes semanticamente
    5. Formatação: serve parent_content (se disponível) para a IA

    Retorna contexto formatado como string, ou string vazia se nada encontrado.
    """
    supabase = get_supabase()
    query_embedding = embed_query(query)

    # Passo 1: Self-Querying — detecta filtros de metadados
    filters = extract_metadata_filter(query)

    # Passo 2: Overfetching — busca inicial ampla via similaridade de cosseno
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

    # Passo 3: Filtro em memória por metadados (Self-Querying)
    chunks = apply_metadata_filter(chunks, filters)

    # Passo 4: RE-RANKING (Cross-Encoder)
    # Prepara os pares (Pergunta, Trecho) para avaliação rigorosa
    pairs = [(query, chunk.get("content", "")) for chunk in chunks]
    scores = score_pairs(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    # Ordena decrescentemente pela nota do Cross-Encoder
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Mantém apenas os top_k melhores
    top_chunks = chunks[:top_k]

    # Passo 5: Formata contexto para o LLM
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_name", "Desconhecido")
        page = metadata.get("page", "?")
        cosine_sim = chunk.get("similarity", 0)
        rerank_score = chunk.get("rerank_score", 0)

        # Parent Document Retriever: serve o trecho pai (1500 chars, rico em contexto)
        # se disponível. Caso contrário, usa o conteúdo original do chunk (compatibilidade).
        parent_content = metadata.get("parent_content", "")
        content_to_serve = parent_content if parent_content else chunk.get("content", "")

        context_parts.append(
            f"[Trecho {i}] (Fonte: {source}, Página: {page}, "
            f"Cross-Score: {rerank_score:.2f}, Vector-Sim: {cosine_sim:.2f})\n{content_to_serve}"
        )

    return "\n\n---\n\n".join(context_parts)
