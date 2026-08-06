from .embeddings import embed_query, score_pairs
from db.supabase_client import get_supabase


# --- SELF-QUERYING: Prompt para extrair filtros de metadados ---
FILTER_EXTRACTION_PROMPT = """Analise a pergunta do usuário abaixo e determine se ele está solicitando informações de um arquivo específico pelo nome.

Exemplos de perguntas com filtro:
- "Resuma o arquivo relatorio_2024.pdf" → {{"source_name": "relatorio_2024.pdf"}}
- "O que diz o manual de trigo sobre adubação?" → {{"source_name": "trigo"}}
- "Busque no documento soja.pdf" → {{"source_name": "soja.pdf"}}

Exemplos de perguntas SEM filtro (busca geral):
- "Quais são as pragas da soja?" → {{}}
- "Como fazer adubação?" → {{}}
- "Explique o ciclo do milho" → {{}}

Responda APENAS com um objeto JSON válido, sem texto adicional, sem markdown.
Se houver filtro, retorne: {{"source_name": "nome_detectado"}}
Se não houver filtro, retorne: {{}}

Pergunta do usuário: {question}

JSON:"""


# --- MULTI-QUERY: Prompt para gerar variações da query ---
MULTI_QUERY_PROMPT = """Você é um especialista em busca semântica. Dada a pergunta abaixo, gere exatamente 3 formas alternativas de perguntar a mesma coisa, usando vocabulário diferente que possa existir em documentos técnicos de agronomia.

Regras:
- Cada variação deve ter um significado equivalente à pergunta original
- Use sinônimos técnicos, termos do campo, e diferentes estruturas de frase
- NÃO responda a pergunta, apenas gere as variações
- Retorne APENAS as 3 variações, uma por linha, sem numeração, sem bullet points

Pergunta original: {question}

Variações:"""


def extract_metadata_filter(question: str) -> dict:
    """
    Usa o LLM para detectar se o usuário quer um documento específico.
    Retorna um dicionário com filtros (ex: {"source_name": "manual.pdf"})
    ou um dict vazio se a busca deve ser geral.
    """
    import json
    from langchain_core.messages import HumanMessage

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

    print(f"[SelfQuery] Nenhum chunk bateu com '{source_filter}'. Usando busca geral.")
    return chunks


def generate_query_variations(question: str) -> list[str]:
    """
    Multi-Query Retriever: usa o LLM para gerar 3 variações semânticas da pergunta.
    Sempre retorna ao menos a pergunta original como fallback.
    """
    from langchain_core.messages import HumanMessage

    try:
        from llm.client import get_llm
        llm = get_llm()
        if not llm:
            return [question]

        prompt = MULTI_QUERY_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Quebra por linha e limpa espaços/vazios
        variations = [v.strip() for v in raw.split("\n") if v.strip()]

        # Garante no máximo 3 variações + a original (sem duplicatas)
        seen = set()
        unique = []
        for v in [question] + variations[:3]:
            if v.lower() not in seen:
                seen.add(v.lower())
                unique.append(v)

        print(f"[MultiQuery] {len(unique)} queries geradas: {unique}")
        return unique

    except Exception as e:
        print(f"[MultiQuery] Fallback para query original: {e}")
        return [question]


def fetch_chunks_for_query(supabase, query: str, fetch_count: int, user_id: str | None) -> list:
    """Busca chunks no Supabase para uma única query."""
    embedding = embed_query(query)
    result = supabase.rpc("search_chunks_hybrid", {
        "query_embedding": embedding,
        "match_count": fetch_count,
        "p_user_id": user_id
    }).execute()
    return result.data or []


def deduplicate_chunks(all_chunks: list) -> list:
    """Remove chunks duplicados baseado no conteúdo (content), mantendo o de maior similaridade."""
    seen_content = {}
    for chunk in all_chunks:
        content = chunk.get("content", "")
        # Mantém o chunk com maior similaridade de cosseno se o mesmo conteúdo aparecer mais de uma vez
        existing = seen_content.get(content)
        if not existing or chunk.get("similarity", 0) > existing.get("similarity", 0):
            seen_content[content] = chunk

    return list(seen_content.values())


def hybrid_search(query: str, user_id: str | None = None, top_k: int = 5) -> str:
    """
    Pipeline completo de busca RAG com todas as 5 técnicas avançadas:
    1. Self-Querying:  detecta filtros de metadados via LLM
    2. Multi-Query:    gera variações semânticas para ampliar o recall
    3. Overfetching:   busca Top 20 por variação via pgvector/cosine
    4. Deduplicação:   remove resultados repetidos entre variações
    5. Filtro:         aplica filtro de arquivo se detectado (Self-Querying)
    6. Re-ranking:     Cross-Encoder reordena pelos mais relevantes semanticamente
    7. Formatação:     serve parent_content (se disponível) para a IA

    Retorna contexto formatado como string, ou string vazia se nada encontrado.
    """
    supabase = get_supabase()

    # Passo 1: Self-Querying — detecta filtros de metadados
    filters = extract_metadata_filter(query)

    # Passo 2: Multi-Query — gera variações semânticas da pergunta
    query_variations = generate_query_variations(query)

    # Passo 3: Overfetching — busca ampla para CADA variação
    fetch_count = max(top_k * 4, 20)
    all_chunks = []
    for variation in query_variations:
        variation_chunks = fetch_chunks_for_query(supabase, variation, fetch_count, user_id)
        all_chunks.extend(variation_chunks)

    if not all_chunks:
        return ""

    # Passo 4: Deduplicação — remove chunks repetidos entre as variações
    chunks = deduplicate_chunks(all_chunks)
    print(f"[MultiQuery] {len(all_chunks)} chunks brutos → {len(chunks)} após deduplicação")

    # Passo 5: Filtro em memória por metadados (Self-Querying)
    chunks = apply_metadata_filter(chunks, filters)

    # Passo 6: RE-RANKING (Cross-Encoder)
    pairs = [(query, chunk.get("content", "")) for chunk in chunks]
    scores = score_pairs(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_chunks = chunks[:top_k]

    # Passo 7: Formata contexto para o LLM
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_name", "Desconhecido")
        page = metadata.get("page", "?")
        cosine_sim = chunk.get("similarity", 0)
        rerank_score = chunk.get("rerank_score", 0)

        # Parent Document Retriever: serve o trecho pai (1500 chars, rico em contexto)
        parent_content = metadata.get("parent_content", "")
        content_to_serve = parent_content if parent_content else chunk.get("content", "")

        context_parts.append(
            f"[Trecho {i}] (Fonte: {source}, Página: {page}, "
            f"Cross-Score: {rerank_score:.2f}, Vector-Sim: {cosine_sim:.2f})\n{content_to_serve}"
        )

    return "\n\n---\n\n".join(context_parts)
