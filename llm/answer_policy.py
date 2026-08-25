from rag.relevance import tokens
from .citations import extract_context_documents


DIAGNOSIS_TERMS = {
    "confirmacao", "confirmar", "deficiencia", "diagnosticar", "diagnostico",
    "diferenciar", "sintoma", "sintomas", "visual",
}
TRANSMISSION_TERMS = {
    "adquire", "adquirir", "aquisicao", "infectiva", "infectivo", "inocular",
    "inoculacao", "latencia", "persistencia", "retencao", "transmite",
    "transmitir", "transmissao",
}
MANAGEMENT_TERMS = {
    "aplicacao", "aplicacoes", "controle", "controlar", "inseticida",
    "inseticidas", "manejo", "pulverizacao", "pulverizar", "pulverizacoes",
    "tratamento", "tratamentos",
}
ETIOLOGY_TERMS = {
    "agente", "agentes", "causa", "causador", "causadores", "etiologia",
    "etiologico", "etiologicos",
}
CHEMICAL_CONTROL_TERMS = {
    "agrotoxico", "agrotoxicos", "ativo", "ativos", "bula", "defensivo",
    "defensivos", "dose", "doses", "ingrediente", "ingredientes",
    "inseticida", "inseticidas", "produto", "produtos", "pulverizacao",
    "pulverizar", "pulverizacoes", "tratamento", "tratamentos",
}
REGULATORY_TERMS = {
    "agrofit", "autorizado", "autorizados", "legal", "permitido",
    "permitidos", "proibido", "proibidos", "recomendado", "recomendados",
    "registrado", "registrados", "registro", "vigente", "vigentes",
}
HYBRID_ENTITY_TERMS = {
    "cultivar", "cultivares", "genotipo", "genotipos", "hibrido", "hibridos",
}
HYBRID_TOLERANCE_TERMS = {
    "suscetibilidade", "suscetivel", "suscetiveis", "tolerancia", "tolerante",
    "tolerantes",
}


def build_answer_policy(question: str, documentary_context: str = "") -> str:
    """Define limites de resposta conforme a intenção técnica da pergunta."""
    question_tokens = tokens(question)
    context_tokens = tokens(documentary_context)
    is_hybrid_question = bool(question_tokens & HYBRID_ENTITY_TERMS) or bool(
        question_tokens & HYBRID_TOLERANCE_TERMS
        and question_tokens & {"enfezamento", "enfezamentos", "doenca", "doencas"}
    )
    rules = [
        "Responda diretamente em até 220 palavras, sem tabela, citações textuais ou tópicos laterais.",
        "Para fatos sustentados pela base, use somente as citações fornecidas no contexto e coloque cada uma após a afirmação que ela sustenta.",
        "Conhecimento geral pode complementar apenas conceitos de baixo risco e deve ficar em seção separada iniciada por 'Complemento de conhecimento geral (não localizado na base consultada):'; uma proibição específica abaixo prevalece.",
        "Sem evidência, apenas declare a limitação para doses, registros, diagnóstico definitivo, ranking de híbridos, garantias, custos ou recomendações locais.",
        "Um artigo científico sustenta somente o que foi avaliado nas condições do estudo. Não transforme ingrediente estudado em produto atualmente registrado, resultado experimental em recomendação de campo, nem associação em mecanismo comprovado.",
        "Não mencione números internos como 'Trecho 1' ou 'Documento 2' na resposta.",
    ]

    if question_tokens & (CHEMICAL_CONTROL_TERMS | REGULATORY_TERMS):
        rules.extend([
            "CONHECIMENTO GERAL PROIBIDO NESTA RESPOSTA.",
            "Use apenas os trechos recuperados para nomes de ingredientes, grupos químicos, duração, eficácia e formas de aplicação; se não estiverem sustentados, declare a lacuna.",
            "Não afirme que um ingrediente é registrado, permitido, recomendado ou vigente com base em artigo científico. Sem consulta oficial atual ao AGROFIT, informe que o registro atual não pode ser confirmado.",
            "Ao mencionar um ingrediente presente no contexto, descreva-o somente como avaliado ou relatado no estudo, preservando data, ambiente e limitações disponíveis; não prescreva seu uso.",
        ])

    if is_hybrid_question:
        rules.extend([
            "CONHECIMENTO GERAL PROIBIDO NESTA RESPOSTA.",
            "Diferencie resistência à cigarrinha, resistência ao patógeno e tolerância aos danos da doença; não trate esses fenótipos como sinônimos.",
            "Tolerância ao enfezamento deve ser descrita pela manutenção relativa de desenvolvimento ou produtividade sob doença. Não atribua antixenose, antibiose, tricomas, cutícula, alteração da sondagem ou menor transmissão sem evidência direta para esse mecanismo e material.",
            "Um estudo de comportamento de sondagem em híbridos resistentes ao inseto não demonstra, por si só, tolerância ao enfezamento.",
            "Ao explicar resistência à cigarrinha, descreva apenas os efeitos medidos no contexto e preserve termos como 'pode' ou 'tipicamente'. Não invente uma causa física, química ou genética para o efeito observado.",
        ])

    if question_tokens & DIAGNOSIS_TERMS:
        rules.extend([
            "Comece dizendo que sintomas permitem suspeita, não confirmação etiológica; explique brevemente sobreposição e infecção mista.",
            "Use somente sintomas explícitos no contexto, sem tratar 'cinta-roja' como sinônimo automático nem juntar fatos de páginas diferentes sob uma só citação.",
            "Não trate aparência visual ou resposta a fertilizante como confirmação e não extrapole o alvo de um método laboratorial.",
        ])
        asks_about_nutritional_deficiency = bool(
            question_tokens & {"deficiencia", "deficiencias", "nutricional", "nutricionais"}
        )
        context_has_nutritional_evidence = bool(
            context_tokens & {"deficiencia", "deficiencias", "nutricional", "nutricionais"}
        )
        if asks_about_nutritional_deficiency and not context_has_nutritional_evidence:
            rules.extend([
                "Os trechos recuperados não fornecem critérios de diagnóstico de deficiência nutricional; declare apenas essa lacuna.",
                "Não descreva padrões de nutrientes, não afirme que um sintoma está ausente na deficiência e não crie seção de conhecimento geral nem sugira análise de solo ou tecido.",
            ])

        for document in extract_context_documents(documentary_context):
            document_tokens = tokens(document["content"])
            citation = document["citation"]
            if {"cssf2", "cssr6", "pcr"} <= document_tokens:
                rules.append(
                    f"A fonte {citation} sustenta PCR especificamente para "
                    "Spiroplasma kunkelii; não a use como fonte de sintomas nem "
                    "como confirmação documentada de MBSP."
                )
            normalized_content = " ".join(document["content"].lower().split())
            if (
                "differentiation of the corn stunting diseases" in normalized_content
                and "visual symptoms is impractical" in normalized_content
            ):
                rules.append(
                    f"A fonte {citation} sustenta somente, para esta resposta: "
                    "folhas amarelas ou vermelhas no enfezamento vermelho; "
                    "estrias cloróticas iniciadas na base das folhas no pálido; "
                    "e diferenciação visual impraticável pela semelhança e "
                    "possível infecção simultânea. Não atribua a ela sintomas "
                    "mais detalhados vindos de outra fonte."
                )
    elif question_tokens & MANAGEMENT_TERMS:
        rules.extend([
            "Concentre-se na diferença entre mortalidade do vetor e prevenção da inoculação.",
            "Priorize tempo de inoculação, velocidade de ação, efeito residual e chegada de indivíduos infectantes; não acrescente aquisição, latência ou recomendações não pedidas.",
            "Não invente estádio da cultura, janela de proteção ou duração residual ausente do contexto.",
            "Não afirme que existe limiar econômico ou nível de ação universal validado para Dalbulus maidis. A abundância do vetor, isoladamente, não determina o risco de enfezamento porque também importam infectividade, estádio da planta e pressão de inóculo.",
        ])
        for document in extract_context_documents(documentary_context):
            normalized_content = " ".join(document["content"].lower().split())
            citation = document["citation"]
            if "0,5 e 1 hora" in normalized_content:
                rules.append(
                    f"Somente {citation} sustenta os tempos aproximados de 0,5 "
                    "e 1 hora para transmissão; coloque essa citação imediatamente "
                    "após esses números."
                )
            if (
                "diluição dos inseticidas" in normalized_content
                or "diluicao dos inseticidas" in normalized_content
            ) and "ação lenta" in normalized_content:
                rules.append(
                    f"A fonte {citation} sustenta a perda de efeito com o "
                    "crescimento da planta e a ação lenta, mas não os tempos "
                    "exatos de transmissão."
                )
            if "fluxo migratório de cigarrinhas infectantes" in normalized_content:
                rules.append(
                    f"A fonte {citation} sustenta que, no campo, o fluxo "
                    "migratório de cigarrinhas infectantes pode explicar a falta "
                    "de redução da incidência."
                )
    elif question_tokens & TRANSMISSION_TERMS:
        rules.extend([
            "Diferencie aquisição, latência, inoculação e retenção sem abrir tópicos de manejo.",
            "Explique que ausência de novo contato com milho doente não equivale a ausência de alimentação ou sobrevivência.",
            "Preserve qualificadores sobre retenção por toda a vida; não fale de inseticidas ou controle.",
        ])
        for document in extract_context_documents(documentary_context):
            document_tokens = tokens(document["content"])
            normalized_content = " ".join(document["content"].lower().split())
            citation = document["citation"]
            if {"17", "28", "glandulas"} <= document_tokens:
                rules.append(
                    f"Somente {citation} sustenta aquisição, multiplicação, "
                    "colonização das glândulas e latência de 17 a 28 dias; cite-a "
                    "imediatamente após esses fatos."
                )
            if "often its entire lifetime" in normalized_content:
                rules.append(
                    f"A fonte {citation} diz que a retenção ocorre frequentemente "
                    "por toda a vida. Preserve 'frequentemente'; não diga 'sempre', "
                    "'só deixa quando morre' ou que a perda ocorre raramente."
                )
    elif question_tokens & ETIOLOGY_TERMS:
        rules.extend([
            "Limite-se aos agentes, às doenças correspondentes e ao papel de Dalbulus maidis como vetor.",
            "Cite imediatamente após cada associação entre agente e doença; não deixe uma citação agrupada distante das afirmações.",
            "Não acrescente sintomas, dano direto, híbridos, controle ou ecologia nem enumere todo o corn stunt complex.",
            "Não diga que Dalbulus maidis atua exclusivamente como vetor e não descreva aquisição, latência ou colonização; basta informar seu papel na transmissão.",
        ])

    return "\n".join(f"- {rule}" for rule in rules)
