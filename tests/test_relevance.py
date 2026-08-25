import unittest
from unittest.mock import patch

from rag.relevance import (
    add_lexical_retrieval_scores,
    build_domain_search_queries,
    filter_by_vector_similarity,
    filter_probable_bibliography,
    is_probable_bibliography,
    is_domain_query,
    question_requires_history,
)
from rag.retriever import extract_metadata_filter
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


class SelfQueryCostTests(unittest.TestCase):
    @patch("builtins.print")
    def test_explicit_pdf_name_keeps_self_query_available(self, _mock_print):
        with patch("llm.client.get_llm", return_value=None) as get_llm:
            result = extract_metadata_filter(
                "O que diz 2022-Pozebon_CornStunt.pdf?"
            )

        self.assertEqual(result, {})
        get_llm.assert_called_once_with("utility")

    @patch("builtins.print")
    def test_generic_document_mention_does_not_call_llm(self, _mock_print):
        with patch("llm.client.get_llm") as get_llm:
            result = extract_metadata_filter(
                "Responda mesmo que os documentos não tragam comparação direta."
            )

        self.assertEqual(result, {})
        get_llm.assert_not_called()

    @patch("builtins.print")
    def test_question_without_document_does_not_call_llm(self, _mock_print):
        with patch("llm.client.get_llm") as get_llm:
            result = extract_metadata_filter("Explique o enfezamento pálido.")

        self.assertEqual(result, {})
        get_llm.assert_not_called()


class QueryExpansionTests(unittest.TestCase):
    def test_etiology_question_gets_causal_agent_terms(self):
        queries = build_domain_search_queries(
            "Quais agentes causam os enfezamentos do milho?"
        )
        self.assertEqual(len(queries), 2)
        self.assertIn("Spiroplasma kunkelii", queries[1])
        self.assertIn("maize bushy stunt phytoplasma", queries[1])

    def test_transmission_question_gets_bilingual_technical_terms(self):
        queries = build_domain_search_queries(
            "Uma cigarrinha infectiva deixa de transmitir após alguns dias?"
        )
        self.assertEqual(len(queries), 2)
        self.assertIn("persistent-propagative", queries[1])
        self.assertIn("período latente", queries[1])

    def test_diagnosis_question_gets_laboratory_terms(self):
        queries = build_domain_search_queries(
            "Dá para confirmar o diagnóstico somente pelos sintomas?"
        )
        self.assertIn("confirmação laboratorial", queries[1])
        self.assertIn("mixed infection", queries[1])

    def test_query_without_known_intent_is_not_expanded(self):
        question = "Explique Dalbulus maidis"
        self.assertEqual(build_domain_search_queries(question), [question])


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


class HybridRetrievalScoreTests(unittest.TestCase):
    def test_technical_lexical_match_can_rescue_semantic_candidate(self):
        chunks = [
            {
                "content": "controle químico de insetos na cultura",
                "similarity": 0.71,
            },
            {
                "content": (
                    "persistent-propagative transmission with lifetime retention "
                    "after acquisition and latent period"
                ),
                "similarity": 0.65,
            },
        ]
        queries = [
            "persistent-propagative transmission lifetime retention acquisition latent"
        ]

        scored = add_lexical_retrieval_scores(chunks, queries)
        ranked = sorted(scored, key=lambda item: item["retrieval_score"], reverse=True)

        self.assertIn("persistent-propagative", ranked[0]["content"])
        self.assertGreater(ranked[0]["retrieval_score"], ranked[1]["retrieval_score"])


class BibliographyFilterTests(unittest.TestCase):
    def test_reference_page_with_multiple_dois_is_detected(self):
        text = (
            "AUTOR. Título. Revista, 2019. DOI: https://doi.org/10.1000/a. "
            "OUTRO. Título. Revista, 2022. DOI: https://doi.org/10.1000/b."
        )
        self.assertTrue(is_probable_bibliography(text))

    def test_article_abstract_with_single_doi_is_preserved(self):
        text = (
            "DOI: https://doi.org/10.1000/artigo. Resumo: Dalbulus maidis "
            "transmite fitopatógenos e os resultados demonstraram diferenças."
        )
        self.assertFalse(is_probable_bibliography(text))

    def test_reference_list_without_dois_is_detected(self):
        text = (
            "OLIVEIRA, C. M. Controle químico. Revista, 2007. "
            "PERFECTO, I. Maize pest system. Ecology, 1990. "
            "PICANÇO, M. C. Fatores de perdas. Acta, 2003."
        )
        self.assertTrue(is_probable_bibliography(text))

    def test_tables_and_figures_caption_page_is_detected(self):
        self.assertTrue(
            is_probable_bibliography(
                "TABLES AND FIGURES (CAPTIONS)\nTable 1. Visual vigor scale."
            )
        )

    def test_filter_uses_parent_content(self):
        chunks = [
            {
                "content": "resultado relevante",
                "metadata": {"parent_content": "Resultados do experimento em milho."},
            },
            {
                "content": "referência",
                "metadata": {
                    "parent_content": (
                        "REFERÊNCIAS\nDOI: https://doi.org/10.1/a\n"
                        "DOI: https://doi.org/10.1/b"
                    )
                },
            },
        ]
        accepted = filter_probable_bibliography(chunks)
        self.assertEqual([item["content"] for item in accepted], ["resultado relevante"])


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
