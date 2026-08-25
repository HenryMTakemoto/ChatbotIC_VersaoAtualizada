#!/usr/bin/env python3
"""Avaliação lexical, local e sem dependências da cobertura do corpus.

Este script não substitui a avaliação do RAG semântico em produção. Ele oferece
um baseline reproduzível: extrai páginas com ``pdftotext``, cria trechos e usa
BM25 para verificar se perguntas cegas encontram evidência textual plausível.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do",
    "dos", "e", "é", "em", "entre", "essa", "esse", "esta", "este",
    "foi", "mais", "na", "nas", "no", "nos", "o", "os", "ou", "para",
    "por", "porque", "qual", "quais", "que", "se", "sem", "ser", "um",
    "uma", "the", "of", "and", "in", "to", "for", "is", "are", "on",
    "with", "from", "by", "can", "does", "what", "which", "how",
}


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    content: str


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in value if not unicodedata.combining(char))


def tokenize(value: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(normalize(value))
        if len(token) > 1 and token not in STOPWORDS
    ]


def split_with_overlap(text: str, size: int = 1_500, overlap: int = 150) -> list[str]:
    compact = re.sub(r"[ \t]+", " ", text).strip()
    if not compact:
        return []

    chunks = []
    start = 0
    while start < len(compact):
        end = min(start + size, len(compact))
        if end < len(compact):
            boundary = max(
                compact.rfind("\n", start + size // 2, end),
                compact.rfind(". ", start + size // 2, end),
            )
            if boundary > start:
                end = boundary + 1
        chunks.append(compact[start:end].strip())
        if end >= len(compact):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_chunks(pdf_dir: Path) -> list[Chunk]:
    chunks = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        )
        for page_number, page_text in enumerate(completed.stdout.split("\f"), 1):
            for content in split_with_overlap(page_text):
                chunks.append(Chunk(pdf_path.name, page_number, content))
    return chunks


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.token_counts = [Counter(tokenize(chunk.content)) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.token_counts]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)

        document_frequency = Counter()
        for counts in self.token_counts:
            document_frequency.update(counts.keys())

        total = len(chunks)
        self.idf = {
            token: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[float, Chunk]]:
        query_tokens = tokenize(query)
        ranked = []
        for index, counts in enumerate(self.token_counts):
            score = 0.0
            length = self.lengths[index]
            normalization = self.k1 * (
                1 - self.b + self.b * length / max(self.avg_length, 1)
            )
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                score += self.idf.get(token, 0) * (
                    frequency * (self.k1 + 1) / (frequency + normalization)
                )
            if score:
                ranked.append((score, self.chunks[index]))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions_blind.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    question_set = json.loads(args.questions.read_text(encoding="utf-8"))
    chunks = extract_chunks(args.pdf_dir)
    index = BM25(chunks)
    report = {
        "metadata": {
            "method": "BM25 lexical baseline",
            "pdf_count": len(list(args.pdf_dir.glob("*.pdf"))),
            "chunk_count": len(chunks),
            "warning": "Não representa o embedding nem o reranker usados em produção.",
        },
        "results": [],
    }

    for item in question_set["questions"]:
        hits = index.search(item["question"], args.top_k)
        result = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "hits": [
                {
                    "score": round(score, 4),
                    "source": chunk.source,
                    "page": chunk.page,
                    "preview": re.sub(r"\s+", " ", chunk.content)[:360],
                }
                for score, chunk in hits
            ],
        }
        report["results"].append(result)

        top = hits[0] if hits else None
        top_label = (
            f"{top[1].source}, p. {top[1].page}, BM25={top[0]:.2f}"
            if top else "sem resultado lexical"
        )
        print(f"{item['id']}: {top_label}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Relatório salvo em {args.json_out}")


if __name__ == "__main__":
    main()
