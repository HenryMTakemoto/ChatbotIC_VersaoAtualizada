import re
from time import perf_counter

from .embeddings import embed_texts, score_pairs
from db.supabase_client import get_supabase


QUERY_VARIATION_COUNT = 2
RERANK_CANDIDATE_COUNT = 15
DOCUMENT_REFERENCE_PATTERN = re.compile(
    r"(?:\b(?:arquivo|documento|pdf|manual|relatório|relatorio|apostila|cartilha)\b|\.pdf\b)",
    re.IGNORECASE,
)


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
MULTI_QUERY_PROMPT = """Você é um especialista em busca semântica. Dada a pergunta abaixo, gere exatamente {variation_count} formas alternativas de perguntar a mesma coisa, usando vocabulário diferente que possa existir em documentos técnicos de agronomia.

Regras:
- Cada variação deve ter um significado equivalente à pergunta original
- Use sinônimos técnicos, termos do campo, e diferentes estruturas de frase
- NÃO responda a pergunta, apenas gere as variações
- Retorne APENAS as {variation_count} variações, uma por linha, sem numeração, sem bullet points

Pergunta original: {question}

Variações:"""


def extract_metadata_filter(question: str) -> dict:
    """
    Usa o LLM para detectar se o usuário quer um documento específico.
    Retorna um dicionário com filtros (ex: {"source_name": "manual.pdf"})
    ou um dict vazio se a busca deve ser geral.
    """
    # A maioria das perguntas agronômicas não cita um documento. Evita uma
    # chamada de LLM sem alterar o comportamento das perguntas sobre arquivos.
    if not DOCUMENT_REFERENCE_PATTERN.search(question):
        print("[SelfQuery] Ignorado: pergunta sem referência a documento.", flush=True)
        return {}

    import json
    from langchain_core.messages import HumanMessage

    try:
        from llm.client import get_llm
        llm = get_llm()
        if not llm:
            return {}

        started_at = perf_counter()
        prompt = FILTER_EXTRACTION_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        print(
            f"[Performance] Self-Querying concluído em "
            f"{perf_counter() - started_at:.2f}s",
            flush=True,
        )

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
    Multi-Query Retriever: usa o LLM para gerar variações semânticas da pergunta.
    Sempre retorna ao menos a pergunta original como fallback.
    """
    from langchain_core.messages import HumanMessage

    try:
        from llm.client import get_llm
        llm = get_llm()
        if not llm:
            return [question]

        started_at = perf_counter()
        prompt = MULTI_QUERY_PROMPT.format(
            question=question,
            variation_count=QUERY_VARIATION_COUNT,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        print(
            f"[Performance] Multi-Query concluído em "
            f"{perf_counter() - started_at:.2f}s",
            flush=True,
        )

        # Quebra por linha e limpa espaços/vazios
        variations = [v.strip() for v in raw.split("\n") if v.strip()]

        # Garante o limite de variações + a original (sem duplicatas)
        seen = set()
        unique = []
        for v in [question] + variations[:QUERY_VARIATION_COUNT]:
            if v.lower() not in seen:
                seen.add(v.lower())
                unique.append(v)

        print(f"[MultiQuery] {len(unique)} queries geradas: {unique}")
        return unique

    except Exception as e:
        print(f"[MultiQuery] Fallback para query original: {e}")
        return [question]


def fetch_chunks_for_query(
    supabase,
    embedding: list[float],
    fetch_count: int,
    user_id: str | None,
) -> list:
    """Busca chunks no Supabase para um embedding de query."""
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
        chunk_similarity = float(chunk.get("similarity") or 0)
        existing_similarity = float(existing.get("similarity") or 0) if existing else 0
        if not existing or chunk_similarity > existing_similarity:
            seen_content[content] = chunk

    return list(seen_content.values())


def select_unique_parent_chunks(chunks: list, top_k: int) -> list:
    """Evita enviar o mesmo parent_content repetido ao LLM."""
    selected = []
    seen_content = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        content_key = metadata.get("parent_content") or chunk.get("content", "")
        if content_key in seen_content:
            continue

        seen_content.add(content_key)
        selected.append(chunk)
        if len(selected) == top_k:
            break

    return selected


def hybrid_search(query: str, user_id: str | None = None, top_k: int = 5) -> str:
    """
    Pipeline completo de busca RAG com todas as 5 técnicas avançadas:
    1. Self-Querying:  detecta filtros de metadados via LLM
    2. Multi-Query:    gera variações semânticas para ampliar o recall
    3. Overfetching:   busca Top 10 por variação via pgvector/cosine
    4. Deduplicação:   remove resultados repetidos entre variações
    5. Filtro:         aplica filtro de arquivo se detectado (Self-Querying)
    6. Re-ranking:     Cross-Encoder reordena até 15 candidatos
    7. Formatação:     serve parent_content (se disponível) para a IA

    Retorna contexto formatado como string, ou string vazia se nada encontrado.
    """
    supabase = get_supabase()

    # Passo 1: Self-Querying — detecta filtros de metadados
    filters = extract_metadata_filter(query)

    # Passo 2: Multi-Query — gera variações semânticas da pergunta
    query_variations = generate_query_variations(query)

    # Passo 3: gera todos os embeddings em um único lote e faz a busca.
    # Dez candidatos por query preservam recall suficiente antes do re-ranking.
    fetch_count = max(top_k * 2, 10)
    search_started_at = perf_counter()
    query_embeddings = embed_texts(query_variations)
    all_chunks = []
    for embedding in query_embeddings:
        variation_chunks = fetch_chunks_for_query(
            supabase,
            embedding,
            fetch_count,
            user_id,
        )
        all_chunks.extend(variation_chunks)

    print(
        f"[Performance] Busca de {len(query_variations)} queries no Supabase em "
        f"{perf_counter() - search_started_at:.2f}s",
        flush=True,
    )

    if not all_chunks:
        return ""

    # Passo 4: Deduplicação — remove chunks repetidos entre as variações
    chunks = deduplicate_chunks(all_chunks)
    print(f"[MultiQuery] {len(all_chunks)} chunks brutos → {len(chunks)} após deduplicação")

    # Passo 5: Filtro em memória por metadados (Self-Querying)
    chunks = apply_metadata_filter(chunks, filters)

    # Passo 6: pré-seleciona por similaridade e limita o Cross-Encoder. Em caso
    # de falha do modelo local, a busca vetorial continua entregando resultados.
    chunks.sort(key=lambda x: float(x.get("similarity") or 0), reverse=True)
    rerank_candidates = chunks[:RERANK_CANDIDATE_COUNT]

    try:
        pairs = [
            (query, chunk.get("content", ""))
            for chunk in rerank_candidates
        ]
        scores = score_pairs(pairs)

        for chunk, score in zip(rerank_candidates, scores):
            chunk["rerank_score"] = float(score)

        rerank_candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )
    except Exception as e:
        print(
            f"[Rerank] Cross-Encoder indisponível; usando similaridade vetorial: "
            f"{str(e)[:200]}",
            flush=True,
        )
        for chunk in rerank_candidates:
            chunk["rerank_score"] = float(chunk.get("similarity") or 0)

    top_chunks = select_unique_parent_chunks(rerank_candidates, top_k)

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
