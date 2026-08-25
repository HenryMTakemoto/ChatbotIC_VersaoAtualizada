import unittest
from unittest.mock import patch

from rag.relevance import (
    filter_by_vector_similarity,
    is_domain_query,
    question_requires_history,
)
from rag.retriever import rank_candidates


class DomainRoutingTests(unittest.TestCase):
    def test_specific_corn_stunt_query_is_in_domain(self):
        self.assertTrue(
            is_domain_query("Como a cigarrinha transmite o enfezamento do milho?")
        )

    def test_broad_agronomic_query_needs_multiple_signals(self):
        self.assertTrue(is_domain_query("Como manejar pragas na lavoura de milho?"))
        self.assertFalse(is_domain_query("Qual é o preço do milho hoje?"))

    def test_clearly_unrelated_query_does_not_search(self):
        self.assertFalse(is_domain_query("Qual foi o placar do último jogo?"))
        self.assertFalse(is_domain_query("Como fazer bolo de milho?"))

    def test_explicit_document_request_is_searchable(self):
        self.assertTrue(
            is_domain_query("Resuma o documento relatorio.pdf", mentions_document=True)
        )


class ConversationalQueryTests(unittest.TestCase):
    def test_pronoun_or_short_follow_up_requires_history(self):
        self.assertTrue(question_requires_history("E por que isso acontece?"))
        self.assertTrue(question_requires_history("E quanto tempo?"))

    def test_standalone_question_does_not_require_history(self):
        self.assertFalse(
            question_requires_history(
                "Como Dalbulus maidis transmite os agentes dos enfezamentos?"
            )
        )


class SimilarityGateTests(unittest.TestCase):
    def test_only_chunks_at_or_above_threshold_are_kept(self):
        chunks = [
            {"content": "baixo", "similarity": 0.29},
            {"content": "limite", "similarity": "0.30"},
            {"content": "alto", "similarity": 0.71},
            {"content": "ausente"},
        ]
        accepted = filter_by_vector_similarity(chunks, 0.30)
        self.assertEqual([item["content"] for item in accepted], ["limite", "alto"])


class CandidateRankingTests(unittest.TestCase):
    @patch("rag.retriever._feature_enabled", return_value=False)
    @patch("rag.retriever.score_pairs")
    def test_disabled_reranker_preserves_vector_order(
        self, score_pairs_mock, _feature_enabled_mock
    ):
        chunks = [
            {"content": "menor", "similarity": 0.61},
            {"content": "maior", "similarity": 0.81},
            {"content": "intermediário", "similarity": 0.74},
        ]

        ranked = rank_candidates("pergunta em português", chunks)

        self.assertEqual(
            [item["content"] for item in ranked],
            ["maior", "intermediário", "menor"],
        )
        score_pairs_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
