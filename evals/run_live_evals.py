#!/usr/bin/env python3
"""Executa as perguntas cegas contra o mesmo RAG usado pela aplicação."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.citations import extract_context_citations
from rag.retriever import hybrid_search


def load_questions(path: Path, selected_ids: set[str]) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))["questions"]
    if not selected_ids:
        return questions
    selected = [item for item in questions if item["id"] in selected_ids]
    found = {item["id"] for item in selected}
    missing = selected_ids - found
    if missing:
        raise ValueError(f"IDs desconhecidos: {', '.join(sorted(missing))}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("questions_blind.json"),
    )
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--answers", action="store_true")
    parser.add_argument(
        "--allow-all-answers",
        action="store_true",
        help="Confirma conscientemente a geração das respostas de todo o conjunto.",
    )
    parser.add_argument(
        "--answer-delay",
        type=float,
        default=20.0,
        help="Espera entre respostas para respeitar a cota de tokens por minuto.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("/tmp/chatbot_live_evals.json"),
    )
    args = parser.parse_args()

    if args.answers and not args.ids and not args.allow_all_answers:
        parser.error(
            "--answers exige --ids para evitar consumo acidental de toda a cota; "
            "use --allow-all-answers somente se quiser gerar as 24 respostas."
        )

    questions = load_questions(args.questions, set(args.ids))
    report = {
        "metadata": {
            "method": "RAG de produção local",
            "answers_generated": args.answers,
        },
        "results": [],
    }

    for index, item in enumerate(questions):
        if args.answers and index and args.answer_delay > 0:
            time.sleep(args.answer_delay)
        if args.answers:
            from llm.chat import build_messages_with_rag, invoke_llm

            retrieval_started_at = perf_counter()
            messages, rag_used = build_messages_with_rag(
                item["question"], history=[], user_id=None
            )
            retrieval_seconds = perf_counter() - retrieval_started_at
            context = str(messages[-1].content)
            generation_started_at = perf_counter()
            answer = invoke_llm(messages)
            generation_seconds = perf_counter() - generation_started_at
        else:
            retrieval_started_at = perf_counter()
            context = hybrid_search(item["question"], user_id=None)
            retrieval_seconds = perf_counter() - retrieval_started_at
            rag_used = bool(context)
            answer = None
            generation_seconds = None

        citations = list(extract_context_citations(context).values())
        report["results"].append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_concepts": item["expected_concepts"],
            "risk": item["risk"],
            "rag_used": rag_used,
            "citations": citations,
            "context": context,
            "answer": answer,
            "timing_seconds": {
                "retrieval": round(retrieval_seconds, 3),
                "generation": (
                    round(generation_seconds, 3)
                    if generation_seconds is not None else None
                ),
            },
        })
        status = ", ".join(citations) if citations else "sem contexto aceito"
        print(f"{item['id']}: {status}", flush=True)

    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Relatório salvo em {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
