import re


CONTEXT_CITATION_PATTERN = re.compile(
    r"DOCUMENTO RECUPERADO\s+(\d+)\s*\n"
    r"CITAÇÃO OBRIGATÓRIA:\s*(\[[^\]\n]+\.pdf,\s*p\.\s*\d+\])",
    re.IGNORECASE,
)
CONTEXT_DOCUMENT_PATTERN = re.compile(
    r"DOCUMENTO RECUPERADO\s+(\d+)\s*\n"
    r"CITAÇÃO OBRIGATÓRIA:\s*(\[[^\]\n]+\.pdf,\s*p\.\s*\d+\])\s*\n"
    r"CONTEÚDO:\s*\n(.*?)"
    r"(?=\n\n---\n\nDOCUMENTO RECUPERADO|\Z)",
    re.IGNORECASE | re.DOTALL,
)
LEGACY_CITATION_PATTERN = re.compile(
    r"\[(?:Trecho|Documento(?:\s+recuperado)?|Fonte)\s*(\d+)"
    r"(?:\s*,[^\]]*)?\]",
    re.IGNORECASE,
)
UNICODE_CITATION_PATTERN = re.compile(
    r"【\s*([^】\n]+\.pdf,\s*p\.\s*\d+(?:\s*;\s*[^】\n]+\.pdf,\s*p\.\s*\d+)*)\s*】",
    re.IGNORECASE,
)
RESPONSE_CITATION_PATTERN = re.compile(
    r"\[([^\]\n]+\.pdf,\s*p\.\s*\d+"
    r"(?:\s*;\s*[^\]\n]+\.pdf,\s*p\.\s*\d+)*)\]",
    re.IGNORECASE,
)
SINGLE_CITATION_PATTERN = re.compile(
    r"^\s*(.+\.pdf),\s*p\.\s*(\d+)\s*$",
    re.IGNORECASE,
)
SHORTHAND_PAGE_CITATION_PATTERN = re.compile(
    r"\[([^\]\n;]+\.pdf),\s*p\.\s*(\d+)"
    r"((?:\s*;\s*p\.\s*\d+)+)\]",
    re.IGNORECASE,
)
SHORTHAND_PAGE_PATTERN = re.compile(r"p\.\s*(\d+)", re.IGNORECASE)
NUTRITION_GAP_POLICY_MARKER = (
    "Os trechos recuperados não fornecem critérios de diagnóstico de "
    "deficiência nutricional"
)


def extract_context_citations(context: str) -> dict[int, str]:
    """Relaciona o número interno do documento à citação auditável."""
    return {
        int(document_number): citation
        for document_number, citation in CONTEXT_CITATION_PATTERN.findall(context)
    }


def extract_context_documents(context: str) -> list[dict]:
    """Extrai citação e conteúdo de cada bloco entregue ao gerador."""
    return [
        {
            "number": int(document_number),
            "citation": citation,
            "content": content.strip(),
        }
        for document_number, citation, content in CONTEXT_DOCUMENT_PATTERN.findall(context)
    ]


def _citation_identity(value: str) -> tuple[str, int] | None:
    match = SINGLE_CITATION_PATTERN.match(value)
    if not match:
        return None
    filename = match.group(1).translate(str.maketrans({
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    }))
    filename = " ".join(filename.split()).casefold()
    return filename, int(match.group(2))


def remove_unknown_response_citations(response: str, context: str) -> str:
    """Impede que o modelo exponha arquivos ou páginas não recuperados."""
    allowed = {}
    for citation in extract_context_citations(context).values():
        identity = _citation_identity(citation[1:-1])
        if identity:
            allowed[identity] = citation[1:-1]

    if not allowed:
        return response

    def replace(match: re.Match) -> str:
        accepted = []
        for component in match.group(1).split(";"):
            original = allowed.get(_citation_identity(component))
            if original and original not in accepted:
                accepted.append(original)
        return f"[{'; '.join(accepted)}]" if accepted else ""

    return RESPONSE_CITATION_PATTERN.sub(replace, response)


def expand_shorthand_page_citations(response: str) -> str:
    """Repete o arquivo quando o modelo abrevia páginas da mesma fonte."""
    def replace(match: re.Match) -> str:
        filename = match.group(1).strip()
        pages = [match.group(2), *SHORTHAND_PAGE_PATTERN.findall(match.group(3))]
        return "[" + "; ".join(
            f"{filename}, p. {page}" for page in pages
        ) + "]"

    return SHORTHAND_PAGE_CITATION_PATTERN.sub(replace, response)


def remove_forbidden_nutrition_complement(response: str, context: str) -> str:
    """Remove complemento nutricional quando a própria base declarou a lacuna."""
    if NUTRITION_GAP_POLICY_MARKER not in context:
        return response

    match = re.search(
        r"(?im)^.*Complemento de conhecimento geral[^\n]*\n?",
        response,
    )
    if not match:
        return response

    conclusion = re.search(
        r"(?im)^\s*(?:\*{0,2})(?:Em resumo|Portanto|Conclusão)\b",
        response[match.end():],
    )
    if conclusion:
        end = match.end() + conclusion.start()
        return (response[:match.start()].rstrip() + "\n\n"
                + response[end:].lstrip())
    return response[:match.start()].rstrip()


def normalize_response_citations(response: str, context: str) -> str:
    """Converte referências internas em nomes reais de arquivo e página."""
    response = UNICODE_CITATION_PATTERN.sub(r"[\1]", response)
    response = expand_shorthand_page_citations(response)
    citation_map = extract_context_citations(context)
    if not citation_map:
        return response

    def replace_legacy(match: re.Match) -> str:
        return citation_map.get(int(match.group(1)), match.group(0))

    response = LEGACY_CITATION_PATTERN.sub(replace_legacy, response)
    return remove_unknown_response_citations(response, context)


def apply_response_guards(response: str, context: str) -> str:
    """Aplica somente barreiras determinísticas que não exigem outra LLM."""
    response = remove_forbidden_nutrition_complement(response, context)
    return normalize_response_citations(response, context)


def response_was_truncated(response) -> bool:
    """Detecta os motivos de parada por limite usados pelos provedores."""
    metadata = getattr(response, "response_metadata", {}) or {}
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    if not reason and isinstance(metadata.get("choices"), list):
        choices = metadata["choices"]
        if choices and isinstance(choices[0], dict):
            reason = choices[0].get("finish_reason")
    return str(reason).lower() in {
        "length", "max_tokens", "max_output_tokens", "token_limit",
    }
