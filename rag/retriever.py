import re
from time import perf_counter

from .embeddings import embed_texts, score_pairs
from .relevance import filter_by_vector_similarity, is_domain_query


QUERY_VARIATION_COUNT = 1
RERANK_CANDIDATE_COUNT = 15
MIN_VECTOR_SIMILARITY = 0.30
MIN_FILTERED_VECTOR_SIMILARITY = 0.20
DOCUMENT_REFERENCE_PATTERN = re.compile(
    r"(?:\b(?:arquivos?|documentos?|pdfs?|manuais?|relatórios?|relatorios?|apostilas?|cartilhas?)\b|\.pdf\b)",
    re.IGNORECASE,
)


def _feature_enabled(name: str, default: bool = False) -> bool:
    """Lê uma feature flag dos secrets do Streamlit de forma tolerante."""
    try:
        import streamlit as st
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "sim"}
    return bool(value)


def _float_setting(name: str, default: float) -> float:
    """Lê um número dos secrets e preserva um padrão seguro se for inválido."""
    try:
        import streamlit as st
        return float(st.secrets.get(name, default))
    except Exception:
        return default


# SELF-QUERYING: Prompt para extrair filtros de metadados
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


# MULTI-QUERY: Prompt para gerar variações da query
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
        llm = get_llm("utility")
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
    Se o arquivo pedido não for encontrado, não usa outro documento no lugar.
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

    print(f"[SelfQuery] Nenhum chunk bateu com '{source_filter}'. Busca interrompida.")
    return []


def generate_query_variations(question: str) -> list[str]:
    """
    Multi-Query Retriever: usa o LLM para gerar variações semânticas da pergunta.
    Sempre retorna ao menos a pergunta original como fallback.
    """
    # Desabilitado por padrão: em um corpus pequeno, o reranker já recebe
    # candidatos suficientes e uma chamada de LLM por consulta custa latência e
    # cota. Pode ser reativado com ENABLE_MULTI_QUERY=true após medir ganho.
    if not _feature_enabled("ENABLE_MULTI_QUERY"):
        return [question]

    from langchain_core.messages import HumanMessage

    try:
        from llm.client import get_llm
        llm = get_llm("utility")
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


def rank_candidates(query: str, chunks: list) -> list:
    """Ordena candidatos, usando o Cross-Encoder somente quando habilitado.
    """
    ranked = sorted(
        chunks,
        key=lambda item: float(item.get("similarity") or 0),
        reverse=True,
    )[:RERANK_CANDIDATE_COUNT]

    if not _feature_enabled("ENABLE_RERANKER"):
        print(
            "[Rerank] Desativado; mantendo ordem da similaridade vetorial multilíngue.",
            flush=True,
        )
        return ranked

    try:
        pairs = [(query, chunk.get("content", "")) for chunk in ranked]
        scores = score_pairs(pairs)

        for chunk, score in zip(ranked, scores):
            chunk["rerank_score"] = float(score)

        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    except Exception as e:
        print(
            f"[Rerank] Cross-Encoder indisponível; usando similaridade vetorial: "
            f"{str(e)[:200]}",
            flush=True,
        )

    return ranked


def hybrid_search(query: str, user_id: str | None = None, top_k: int = 5) -> str:
    """
    Pipeline de busca RAG especializado:
    1. Roteamento de domínio: evita consultar artigos agrícolas para perguntas alheias
    2. Self-Querying: detecta filtros explícitos de documento
    3. Multi-Query opcional: só é ativado quando medido/configurado
    4. Busca vetorial, deduplicação e limiar mínimo de relevância
    5. Re-ranking opcional de até 15 candidatos (desligado por padrão)
    6. Formatação do parent com fonte e página verificáveis

    Retorna contexto formatado como string, ou string vazia se nada encontrado.
    """
    mentions_document = bool(DOCUMENT_REFERENCE_PATTERN.search(query))
    if not is_domain_query(query, mentions_document=mentions_document):
        print("[RAG] Busca ignorada: pergunta fora do domínio da base.", flush=True)
        return ""

    from db.supabase_client import get_supabase

    supabase = get_supabase()

    # Self-Querying — detecta filtros de metadados
    filters = extract_metadata_filter(query)

    # Multi-Query — gera variações semânticas da pergunta
    query_variations = generate_query_variations(query)

    # gera todos os embeddings em um único lote e faz a busca.
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

    # Deduplicação — remove chunks repetidos entre as variações
    chunks = deduplicate_chunks(all_chunks)
    search_label = "MultiQuery" if len(query_variations) > 1 else "RAG"
    print(
        f"[{search_label}] {len(all_chunks)} chunks brutos → "
        f"{len(chunks)} após deduplicação"
    )

    # Filtro em memória por metadados (Self-Querying)
    chunks = apply_metadata_filter(chunks, filters)
    if not chunks:
        return ""

    # pgvector sempre consegue devolver vizinhos, mesmo para perguntas sem
    # resposta. O limiar impede que "vizinho mais próximo" seja confundido com
    # evidência suficiente. Filtros explícitos toleram similaridade um pouco menor.
    minimum_similarity = (
        _float_setting("RAG_MIN_FILTERED_VECTOR_SIMILARITY", MIN_FILTERED_VECTOR_SIMILARITY)
        if filters
        else _float_setting("RAG_MIN_VECTOR_SIMILARITY", MIN_VECTOR_SIMILARITY)
    )
    chunks = filter_by_vector_similarity(chunks, minimum_similarity)
    if not chunks:
        print(
            f"[RAG] Nenhum candidato atingiu Vector-Sim >= {minimum_similarity:.2f}.",
            flush=True,
        )
        return ""

    ranked_candidates = rank_candidates(query, chunks)
    top_chunks = select_unique_parent_chunks(ranked_candidates, top_k)

    # Formata contexto para o LLM
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_name", "Desconhecido")
        page = metadata.get("page", "?")
        # Documentos indexados antes da correção guardavam índice iniciado em 0.
        if metadata.get("page_numbering") != "pdf_1_based":
            try:
                page = int(page) + 1
            except (TypeError, ValueError):
                pass
        cosine_sim = chunk.get("similarity", 0)

        # Parent Document Retriever: serve o trecho pai (1500 chars, rico em contexto)
        parent_content = metadata.get("parent_content", "")
        content_to_serve = parent_content if parent_content else chunk.get("content", "")

        context_parts.append(
            f"[Trecho {i}] [Fonte: {source}, página do PDF: {page}]\n"
            f"{content_to_serve}"
        )

        ranking_details = f"Vector={cosine_sim:.2f}"
        if "rerank_score" in chunk:
            ranking_details = (
                f"Cross={chunk['rerank_score']:.2f}, {ranking_details}"
            )
        print(
            f"[RAG] Trecho {i}: {source}, p. {page}, {ranking_details}",
            flush=True,
        )

    return "\n\n---\n\n".join(context_parts)
