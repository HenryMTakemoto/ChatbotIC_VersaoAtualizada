import unittest
from types import SimpleNamespace

from llm.citations import (
    apply_response_guards,
    expand_shorthand_page_citations,
    extract_context_citations,
    extract_context_documents,
    normalize_response_citations,
    response_was_truncated,
)


CONTEXT = """DOCUMENTO RECUPERADO 1
CITAÇÃO OBRIGATÓRIA: [artigo_a.pdf, p. 5]
CONTEÚDO:
Texto A.

---

DOCUMENTO RECUPERADO 2
CITAÇÃO OBRIGATÓRIA: [artigo_b.pdf, p. 9]
CONTEÚDO:
Texto B.
"""


class CitationNormalizationTests(unittest.TestCase):
    def test_extracts_document_content_with_its_citation(self):
        context = (
            "DOCUMENTO RECUPERADO 1\n"
            "CITAÇÃO OBRIGATÓRIA: [artigo.pdf, p. 6]\n"
            "CONTEÚDO:\nPCR com primers específicos."
        )
        self.assertEqual(
            extract_context_documents(context),
            [{
                "number": 1,
                "citation": "[artigo.pdf, p. 6]",
                "content": "PCR com primers específicos.",
            }],
        )

    def test_converts_unicode_citation_brackets(self):
        response = "Afirmação【artigo.pdf, p. 6】."
        self.assertEqual(
            normalize_response_citations(response, ""),
            "Afirmação[artigo.pdf, p. 6].",
        )

    def test_extracts_auditable_citations_from_context(self):
        self.assertEqual(
            extract_context_citations(CONTEXT),
            {1: "[artigo_a.pdf, p. 5]", 2: "[artigo_b.pdf, p. 9]"},
        )

    def test_replaces_legacy_trecho_citation_and_its_page(self):
        response = "Afirmação [Trecho 2, p. 3]. Outra [Trecho 1]."
        self.assertEqual(
            normalize_response_citations(response, CONTEXT),
            "Afirmação [artigo_b.pdf, p. 9]. Outra [artigo_a.pdf, p. 5].",
        )

    def test_preserves_unknown_legacy_reference(self):
        response = "Afirmação [Trecho 7]."
        self.assertEqual(normalize_response_citations(response, CONTEXT), response)

    def test_removes_citation_not_present_in_retrieved_context(self):
        response = "Afirmação [arquivo_inventado.pdf, p. 99]."
        self.assertEqual(
            normalize_response_citations(response, CONTEXT),
            "Afirmação .",
        )

    def test_keeps_valid_part_and_removes_invented_part(self):
        response = (
            "Afirmação [artigo_a.pdf, p. 5; arquivo_inventado.pdf, p. 99]."
        )
        self.assertEqual(
            normalize_response_citations(response, CONTEXT),
            "Afirmação [artigo_a.pdf, p. 5].",
        )

    def test_expands_abbreviated_pages_from_same_document(self):
        response = "Afirmação [artigo_a.pdf, p. 5; p. 9]."
        self.assertEqual(
            expand_shorthand_page_citations(response),
            "Afirmação [artigo_a.pdf, p. 5; artigo_a.pdf, p. 9].",
        )

    def test_removes_nutrition_complement_when_policy_declares_gap(self):
        prompt = (
            CONTEXT
            + "\nOs trechos recuperados não fornecem critérios de diagnóstico "
            "de deficiência nutricional."
        )
        response = (
            "A base não permite diferenciar nutrientes.\n\n"
            "**Complemento de conhecimento geral (não localizado na base "
            "consultada):**\nPadrões inventados de nutrientes.\n\n"
            "Em resumo, sintomas não confirmam o agente."
        )
        self.assertEqual(
            apply_response_guards(response, prompt),
            "A base não permite diferenciar nutrientes.\n\n"
            "Em resumo, sintomas não confirmam o agente.",
        )

    def test_removes_general_complement_for_high_risk_answer(self):
        prompt = CONTEXT + "\nCONHECIMENTO GERAL PROIBIDO NESTA RESPOSTA."
        response = (
            "O estudo avaliou um tratamento [artigo_a.pdf, p. 5].\n\n"
            "Complemento de conhecimento geral (não localizado na base "
            "consultada):\nProduto inventado e supostamente registrado.\n\n"
            "Em resumo, o registro vigente precisa de fonte oficial."
        )
        self.assertEqual(
            apply_response_guards(response, prompt),
            "O estudo avaliou um tratamento [artigo_a.pdf, p. 5].\n\n"
            "Em resumo, o registro vigente precisa de fonte oficial.",
        )


class TruncationDetectionTests(unittest.TestCase):
    def test_detects_length_finish_reason(self):
        response = SimpleNamespace(response_metadata={"finish_reason": "length"})
        self.assertTrue(response_was_truncated(response))

    def test_accepts_normal_stop(self):
        response = SimpleNamespace(response_metadata={"finish_reason": "stop"})
        self.assertFalse(response_was_truncated(response))


if __name__ == "__main__":
    unittest.main()
