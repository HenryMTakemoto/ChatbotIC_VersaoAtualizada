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

def build_messages_with_rag(
    question: str,
    history: list,
    user_id: str | None = None
) -> tuple[list, bool]:
    """
    Monta mensagens para o LLM com contexto RAG híbrido.
    Retorna (messages, rag_was_used).
    """
    context = hybrid_search(question, user_id)
    rag_used = bool(context)

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