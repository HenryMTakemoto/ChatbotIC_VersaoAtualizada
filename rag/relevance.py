"""Regras puras de roteamento e relevância para o pipeline de RAG."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SEARCH_STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "entre", "essa", "esse", "esta", "este", "foi",
    "mais", "na", "nas", "no", "nos", "o", "os", "ou", "para", "por",
    "qual", "quais", "que", "se", "sem", "ser", "um", "uma", "the",
    "of", "and", "in", "to", "for", "is", "are", "on", "with", "from",
    "by", "can", "does", "what", "which", "how",
}

CORE_DOMAIN_TERMS = {
    "cigarrinha", "cigarrinhas", "dalbulus", "maidis", "enfezamento",
    "enfezamentos", "fitoplasma", "fitoplasmas", "espiroplasma",
    "espiroplasmas", "molicute", "molicutes", "maize", "corn", "leafhopper",
    "stunt", "tiguera",
}
AGRONOMIC_TERMS = {
    "agricultura", "agronomia", "aplicacao", "aplicacoes", "bula", "controle",
    "cultivar", "cultivares", "cultivo", "doenca", "doencas", "dose", "hibrido",
    "hibridos", "inseticida", "inseticidas", "lavoura", "manejo", "milho",
    "monitoramento", "patogeno", "patogenos", "plantio", "populacao", "praga",
    "pragas", "produtividade", "pulverizacao", "resistencia", "safra", "safrinha",
    "semente", "sementes", "semeadura", "severidade", "tolerancia", "transmissao",
    "vetor",
}
CLEARLY_OUT_OF_SCOPE_TERMS = {
    "bolo", "campeonato", "futebol", "gol", "jogo", "placar", "receita",
    "selecao", "sobremesa",
}
HISTORY_REFERENCE_PATTERN = re.compile(
    r"\b(ele|ela|eles|elas|isso|isto|esse|essa|esses|essas|dele|dela|disso|"
    r"anterior|acima|tamb[eé]m)\b|^e\s+(qual|quais|como|quando|quanto|por que)\b",
    re.IGNORECASE,
)

QUERY_EXPANSION_GROUPS = (
    (
        {
            "agente", "agentes", "causa", "causador", "causadores",
            "etiologia", "etiologico", "etiologicos",
        },
        (
            "enfezamento pálido Spiroplasma kunkelii enfezamento vermelho "
            "maize bushy stunt phytoplasma MBSP agentes causais molicutes "
            "corn pale stunt red stunt causal agents"
        ),
    ),
    (
        {
            "adquire", "adquirir", "aquisicao", "infectiva", "infectivo",
            "inocula", "inocular", "inoculacao", "latencia", "patogeno",
            "patogenos", "persistencia", "retencao", "transmite", "transmitir",
            "transmissao",
        },
        (
            "aquisição inoculação período latente persistência retenção "
            "transmissão persistente-propagativa infectividade por toda a vida "
            "acquisition inoculation latent period persistent-propagative "
            "transmission lifetime retention infective leafhopper"
        ),
    ),
    (
        {
            "confirmacao", "confirmar", "deficiencia", "diagnosticar",
            "diagnostico", "diferenciar", "sintoma", "sintomas", "visual",
        },
        (
            "sintomas sobrepostos infecção mista diagnóstico diferencial "
            "confirmação laboratorial PCR visual symptoms mixed infection "
            "precise diagnosis field samples laboratory detection"
        ),
    ),
    (
        {
            "controle", "inseticida", "inseticidas", "pulverizacao",
            "pulverizar", "quimico", "quimicos",
        },
        (
            "tempo de inoculação mortalidade ação rápida ação lenta migração "
            "reinfestação cigarrinhas infectantes inoculation access period "
            "infective leafhoppers migration take longer to kill prevent inoculation"
        ),
    ),
    (
        {
            "hibrido", "hibridos", "resistencia", "resistente", "suscetivel",
            "tolerancia", "tolerante",
        },
        (
            "resistência ao inseto tolerância à doença severidade produtividade "
            "susceptible resistant maize hybrids disease tolerance probing behavior"
        ),
    ),
    (
        {"amostragem", "densidade", "monitoramento", "populacao"},
        (
            "amostragem de adultos densidade populacional horário de avaliação "
            "sampling technique adult leafhopper population density time of day"
        ),
    ),
)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in value if not unicodedata.combining(char))


def tokens(value: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(normalize(value)))


def is_domain_query(question: str, mentions_document: bool = False) -> bool:
    """Decide se vale consultar a base especializada.

    Referências explícitas a documentos sempre são aceitas. Termos muito
    específicos do domínio bastam por si; termos agronômicos amplos exigem ao
    menos duas pistas para evitar recuperar artigos para perguntas triviais.
    """
    if mentions_document:
        return True

    query_tokens = tokens(question)
    if query_tokens & CLEARLY_OUT_OF_SCOPE_TERMS and not query_tokens & CORE_DOMAIN_TERMS:
        return False
    if query_tokens & CORE_DOMAIN_TERMS:
        return True
    return len(query_tokens & AGRONOMIC_TERMS) >= 2


def question_requires_history(question: str) -> bool:
    """Detecta perguntas que provavelmente dependem do turno anterior."""
    compact = question.strip()
    if not compact:
        return False
    if HISTORY_REFERENCE_PATTERN.search(compact):
        return True
    return len(tokens(compact)) <= 3


def build_domain_search_queries(question: str) -> list[str]:
    """Cria uma consulta técnica bilíngue sem consumir uma chamada de LLM."""
    query_tokens = tokens(question)
    expansions = [
        expansion
        for triggers, expansion in QUERY_EXPANSION_GROUPS
        if query_tokens & triggers
    ]
    if not expansions:
        return [question]

    return [question, *expansions]


def filter_by_vector_similarity(chunks: list, minimum: float) -> list:
    """Remove candidatos abaixo do limiar absoluto de similaridade."""
    accepted = []
    for chunk in chunks:
        try:
            similarity = float(chunk.get("similarity") or 0)
        except (TypeError, ValueError):
            similarity = 0.0
        if similarity >= minimum:
            accepted.append(chunk)
    return accepted


def is_probable_bibliography(text: str) -> bool:
    """Detecta páginas de referências que não devem sustentar respostas."""
    if not text:
        return False

    normalized = normalize(text)
    if re.search(
        r"(?m)^\s*(references|bibliografia|referencias bibliograficas|"
        r"tables and figures \(captions\))\s*$",
        normalized,
    ):
        return True

    doi_mentions = max(
        len(re.findall(r"\bdoi\s*:", normalized)),
        len(re.findall(r"doi\.org/", normalized)),
    )
    url_mentions = len(re.findall(r"https?\s*:\s*//", normalized))
    year_mentions = len(re.findall(r"\b(?:19|20)\d{2}\b", normalized))
    author_entries = len(re.findall(
        r"\b[A-ZÀ-Ý][A-ZÀ-Ý'-]{2,},\s*[A-ZÀ-Ý]",
        text,
    ))
    return (
        doi_mentions >= 2
        or url_mentions >= 3
        or (author_entries >= 3 and year_mentions >= 3)
    )


def filter_probable_bibliography(chunks: list) -> list:
    """Remove candidatos cujo conteúdo pai aparenta ser bibliografia."""
    accepted = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        text = metadata.get("parent_content") or chunk.get("content", "")
        if not is_probable_bibliography(text):
            accepted.append(chunk)
    return accepted


def add_lexical_retrieval_scores(
    chunks: list,
    queries: list[str],
    lexical_weight: float = 0.15,
) -> list:
    """Combina o vetor com correspondência lexical técnica nos candidatos."""
    query_terms = set()
    for query in queries:
        query_terms.update(
            token for token in tokens(query)
            if len(token) > 2 and token not in SEARCH_STOPWORDS
        )
    if not chunks or not query_terms:
        return chunks

    chunk_terms = [
        {
            token for token in tokens(chunk.get("content", ""))
            if len(token) > 2 and token not in SEARCH_STOPWORDS
        }
        for chunk in chunks
    ]
    document_frequency = Counter()
    for terms_in_chunk in chunk_terms:
        document_frequency.update(terms_in_chunk & query_terms)

    total = len(chunks)
    raw_scores = []
    for terms_in_chunk in chunk_terms:
        score = sum(
            math.log(1 + (total - document_frequency[term] + 0.5)
                     / (document_frequency[term] + 0.5))
            for term in terms_in_chunk & query_terms
        )
        raw_scores.append(score)

    maximum = max(raw_scores, default=0.0)
    for chunk, raw_score in zip(chunks, raw_scores):
        lexical_score = raw_score / maximum if maximum else 0.0
        vector_score = float(chunk.get("similarity") or 0)
        chunk["lexical_score"] = lexical_score
        chunk["retrieval_score"] = vector_score + lexical_weight * lexical_score
    return chunks
