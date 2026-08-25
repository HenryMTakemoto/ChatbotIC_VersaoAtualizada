import unittest

from llm.answer_policy import build_answer_policy


class AnswerPolicyTests(unittest.TestCase):
    def test_all_answers_receive_evidence_hierarchy(self):
        policy = build_answer_policy("Explique a cigarrinha-do-milho")
        self.assertIn("Complemento de conhecimento geral", policy)
        self.assertIn("declare a limitação", policy)
        self.assertIn("Não mencione números internos", policy)

    def test_diagnosis_requires_visual_uncertainty_and_laboratory_confirmation(self):
        policy = build_answer_policy(
            "Como diferenciar pelos sintomas? Dá para confirmar no campo?"
        )
        self.assertIn("não confirmação etiológica", policy)
        self.assertIn("método laboratorial", policy)
        self.assertIn("Use somente sintomas explícitos", policy)

    def test_diagnosis_marks_missing_nutritional_evidence(self):
        policy = build_answer_policy(
            "Como diferenciar enfezamento de deficiência nutricional?",
            "Os sintomas dos enfezamentos podem se sobrepor. A confirmação usou PCR.",
        )
        self.assertIn("não fornecem critérios", policy)
        self.assertIn("Não descreva padrões de nutrientes", policy)
        self.assertIn("não crie seção de conhecimento geral", policy)

    def test_diagnosis_does_not_mark_gap_when_context_has_nutritional_evidence(self):
        policy = build_answer_policy(
            "Como diferenciar enfezamento de deficiência nutricional?",
            "O estudo descreve deficiência nutricional e seu diagnóstico diferencial.",
        )
        self.assertNotIn("não fornecem critérios de diagnóstico", policy)

    def test_diagnosis_routes_pcr_citation_to_its_exact_scope(self):
        context = (
            "DOCUMENTO RECUPERADO 1\n"
            "CITAÇÃO OBRIGATÓRIA: [pcr.pdf, p. 5]\n"
            "CONTEÚDO:\nPCR CSSF2 CSSR6 para Spiroplasma kunkelii."
        )
        policy = build_answer_policy("Como confirmar o diagnóstico?", context)
        self.assertIn("[pcr.pdf, p. 5]", policy)
        self.assertIn("não a use como fonte de sintomas", policy)

    def test_management_excludes_unnecessary_latency_tangent(self):
        policy = build_answer_policy(
            "Por que pulverizar inseticida nem sempre reduz o enfezamento?"
        )
        self.assertIn("mortalidade do vetor", policy)
        self.assertIn("não acrescente aquisição, latência", policy)
        self.assertIn("Não invente estádio da cultura", policy)

    def test_transmission_preserves_lifetime_qualifier(self):
        context = (
            "DOCUMENTO RECUPERADO 1\n"
            "CITAÇÃO OBRIGATÓRIA: [transmissao.pdf, p. 6]\n"
            "CONTEÚDO:\nThe vector remains bacterialiferous often its entire lifetime."
        )
        policy = build_answer_policy(
            "Uma cigarrinha infectiva deixa de transmitir depois de alguns dias?",
            context,
        )
        self.assertIn("Preserve qualificadores", policy)
        self.assertIn("não fale de inseticidas", policy)
        self.assertIn("[transmissao.pdf, p. 6]", policy)
        self.assertIn("não diga 'sempre'", policy)

    def test_transmission_routes_latency_to_exact_source(self):
        context = (
            "DOCUMENTO RECUPERADO 1\n"
            "CITAÇÃO OBRIGATÓRIA: [ciclo.pdf, p. 19]\n"
            "CONTEÚDO:\nMultiplicação até colonizarem as glândulas salivares "
            "após o período de lat ência de 17 a 28 dias."
        )
        policy = build_answer_policy("Como ocorre a transmissão?", context)
        self.assertIn("[ciclo.pdf, p. 19]", policy)
        self.assertIn("latência de 17 a 28 dias", policy)

    def test_management_routes_transmission_time_to_exact_page(self):
        context = (
            "DOCUMENTO RECUPERADO 1\n"
            "CITAÇÃO OBRIGATÓRIA: [controle.pdf, p. 4]\n"
            "CONTEÚDO:\nNecessitam alimentar-se cerca de 0,5 e 1 hora para "
            "transmissão do fitoplasma e espiroplasma."
        )
        policy = build_answer_policy(
            "Por que pulverizar inseticida nem sempre evita o enfezamento?",
            context,
        )
        self.assertIn("Somente [controle.pdf, p. 4]", policy)
        self.assertIn("0,5 e 1 hora", policy)

    def test_etiology_avoids_unasked_ecology(self):
        policy = build_answer_policy(
            "Quais agentes causam os enfezamentos e qual o papel do vetor?"
        )
        self.assertIn("Não acrescente sintomas", policy)
        self.assertIn("Não diga que Dalbulus maidis atua exclusivamente", policy)
        self.assertIn("Cite imediatamente após cada associação", policy)
        self.assertIn("nem enumere todo o corn stunt complex", policy)


if __name__ == "__main__":
    unittest.main()
