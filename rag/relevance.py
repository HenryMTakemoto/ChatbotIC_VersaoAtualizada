"""Regras puras de roteamento e relevância para o pipeline de RAG."""

from __future__ import annotations

import re
import unicodedata


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

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
        r"(?m)^\s*(references|bibliografia|referencias bibliograficas)\s*$",
        normalized,
    ):
        return True

    doi_mentions = max(
        len(re.findall(r"\bdoi\s*:", normalized)),
        len(re.findall(r"doi\.org/", normalized)),
    )
    url_mentions = len(re.findall(r"https?\s*:\s*//", normalized))
    return doi_mentions >= 2 or url_mentions >= 3


def filter_probable_bibliography(chunks: list) -> list:
    """Remove candidatos cujo conteúdo pai aparenta ser bibliografia."""
    accepted = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        text = metadata.get("parent_content") or chunk.get("content", "")
        if not is_probable_bibliography(text):
            accepted.append(chunk)
    return accepted
