import uuid
from datetime import datetime
from .supabase_client import get_supabase

def create_conversation(user_id: str, title: str = "Nova Conversa", source: str = "web") -> str:
    """Cria nova conversa. Retorna o ID."""
    supabase = get_supabase()
    conv_id = str(uuid.uuid4())
    supabase.table("conversations").insert({
        "id": conv_id,
        "user_id": user_id,
        "title": title,
        "source": source
    }).execute()
    return conv_id

def get_user_conversations(user_id: str, limit: int = 50) -> list[dict]:
    """Lista conversas do usuário, mais recentes primeiro."""
    supabase = get_supabase()
    result = supabase.table("conversations").select("*").eq(
        "user_id", user_id
    ).order("updated_at", desc=True).limit(limit).execute()
    return result.data or []

def save_message(conversation_id: str, role: str, content: str, rag_used: bool = False):
    """Salva uma mensagem na conversa."""
    supabase = get_supabase()
    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "rag_context_used": rag_used
    }).execute()

    # Atualiza updated_at da conversa
    supabase.table("conversations").update({
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", conversation_id).execute()

def get_conversation_messages(conversation_id: str) -> list[dict]:
    """Retorna todas as mensagens de uma conversa ordenadas."""
    supabase = get_supabase()
    result = supabase.table("messages").select("*").eq(
        "conversation_id", conversation_id
    ).order("created_at").execute()
    return result.data or []

def update_conversation_title(conversation_id: str, first_message: str):
    """Gera título automático a partir da 1ª mensagem (primeiras 50 chars)."""
    title = first_message[:50].strip()
    if len(first_message) > 50:
        title += "..."
    supabase = get_supabase()
    supabase.table("conversations").update({"title": title}).eq(
        "id", conversation_id
    ).execute()