from .supabase_client import get_supabase


def list_documents(user_id: str) -> list[dict]:
    """
    Retorna documentos visíveis ao usuário:
    - Seus próprios documentos pessoais
    - Todos os documentos globais
    """
    supabase = get_supabase()
    result = (
        supabase.table("documents")
        .select("*")
        .or_(f"user_id.eq.{user_id},is_global.eq.true")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def list_global_documents() -> list[dict]:
    """Retorna apenas documentos globais (para painel admin)."""
    supabase = get_supabase()
    result = (
        supabase.table("documents")
        .select("*")
        .eq("is_global", True)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def document_already_exists(file_name: str, user_id: str | None, is_global: bool) -> bool:
    """Verifica se um PDF já foi indexado para evitar duplicatas."""
    supabase = get_supabase()
    query = supabase.table("documents").select("id").eq("file_name", file_name)

    if is_global:
        query = query.eq("is_global", True)
    else:
        query = query.eq("user_id", user_id)

    result = query.execute()
    return bool(result.data)


def delete_document(document_id: str):
    """
    Remove documento e todos os seus chunks (CASCADE no banco).
    Chunks = embeddings também são deletados.
    """
    supabase = get_supabase()
    supabase.table("documents").delete().eq("id", document_id).execute()
