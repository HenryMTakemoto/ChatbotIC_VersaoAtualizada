from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rag.retriever import hybrid_search
from .client import get_llm

SYSTEM_PROMPT = """
Você é o Assistente Virtual Oficial do projeto 'Cia Agro'.
Sua persona: Um agrônomo especialista e prestativo.
Seu objetivo: Ajudar usuários com dúvidas sobre o projeto e agricultura geral.

REGRAS IMPORTANTES:
- Se um CONTEXTO DE DOCUMENTOS for fornecido abaixo, priorize ABSOLUTAMENTE essas informações.
- Ao usar informações do contexto, cite a fonte (nome do arquivo e página).
- Se a informação NÃO estiver no contexto, avise o usuário e responda com seu conhecimento geral.
- Seja conciso, técnico e use emojis relevantes.
"""

CONDENSE_QUESTION_PROMPT = """
Dado o histórico de conversa abaixo e uma nova pergunta do usuário, reescreva a pergunta de forma que ela seja totalmente independente e autossuficiente, sem depender do contexto anterior. Se a pergunta já for clara e independente, retorne-a sem alterações.

Não responda a pergunta. Apenas reescreva-a se necessário.

Histórico da conversa:
{history}

Pergunta do usuário: {question}

Pergunta reescrita:"""


def rephrase_to_standalone_question(question: str, history: list) -> str:
    """
    Usa o LLM para reescrever a pergunta do usuário como uma Standalone Question,
    considerando o histórico da conversa. Se não houver histórico ou o LLM falhar,
    retorna a pergunta original sem modificação.
    """
    # Sem histórico: não é necessário reescrever
    if not history:
        return question

    llm = get_llm()
    if not llm:
        return question

    try:
        # Formata o histórico como texto legível
        history_text = ""
        for msg in history:
            if isinstance(msg, HumanMessage):
                history_text += f"Usuário: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # Trunca respostas longas do assistente para economizar tokens
                content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                history_text += f"Assistente: {content}\n"

        prompt = CONDENSE_QUESTION_PROMPT.format(
            history=history_text.strip(),
            question=question
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        rephrased = response.content.strip()

        # Fallback: se o LLM retornar algo vazio ou muito longo, usa original
        if rephrased and len(rephrased) < 500:
            return rephrased
        return question

    except Exception as e:
        print(f"[ConversationalRAG] Falha ao reescrever pergunta: {e}")
        return question

def build_messages_with_rag(
    question: str,
    history: list,
    user_id: str | None = None
) -> tuple[list, bool]:
    """
    Monta mensagens para o LLM com contexto RAG híbrido e Conversational Retrieval.
    
    Fluxo:
    1. Reescreve a pergunta como Standalone Question (leva o histórico em conta).
    2. Usa a Standalone Question para buscar no RAG (busca mais precisa).
    3. Monta o prompt final com o contexto recuperado e a pergunta ORIGINAL do usuário.
    
    Retorna (messages, rag_was_used).
    """
    # Conversational Retrieval — reescreve para uma query autossuficiente
    standalone_question = rephrase_to_standalone_question(question, history)
    
    # Log para debug (aparece nos logs do Streamlit Cloud)
    if standalone_question != question:
        print(f"[ConversationalRAG] Pergunta original: '{question}'")
        print(f"[ConversationalRAG] Pergunta reescrita: '{standalone_question}'")

    # Busca RAG usando a Standalone Question (mais eficaz)
    context = hybrid_search(standalone_question, user_id)
    rag_used = bool(context)

    # Monta o prompt com a pergunta ORIGINAL (melhor experiência para o usuário)
    if context:
        augmented_question = (
            f"CONTEXTO DE DOCUMENTOS (use como fonte primária):\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"PERGUNTA DO USUÁRIO: {question}"
        )
    else:
        augmented_question = question

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages += history
    messages.append(HumanMessage(content=augmented_question))

    return messages, rag_used

def invoke_llm(messages: list) -> str:
    """Invoca o LLM e retorna o texto da resposta."""
    llm = get_llm()
    if not llm:
        return "❌ LLM não disponível. Verifique as credenciais."
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"❌ Erro ao consultar IA: {str(e)[:200]}"