import tempfile
import os
import uuid
from .embeddings import embed_texts
from db.supabase_client import get_supabase
from db.documents import document_already_exists


def process_and_store_pdf(
    pdf_file,
    user_id: str,
    is_global: bool = False
) -> tuple[bool, str]:
    """
    Pipeline completo: PDF → chunks → embeddings → Supabase.

    Args:
        pdf_file: UploadedFile do Streamlit
        user_id: UUID do usuário que fez o upload
        is_global: True = visível para todos (apenas admin)

    Returns:
        (sucesso: bool, mensagem: str)
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    supabase = get_supabase()
    pdf_name = pdf_file.name

    # Verifica duplicata
    if document_already_exists(pdf_name, user_id, is_global):
        return False, f"⚠️ '{pdf_name}' já está indexado."

    try:
        # Salva em arquivo temporário (PyPDFLoader precisa de path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        # Carrega páginas
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)

        if not pages:
            return False, f"❌ Nenhuma página extraída de '{pdf_name}'."

        # Divide em chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(pages)

        if not chunks:
            return False, f"❌ Nenhum texto extraído de '{pdf_name}'."

        # Salva metadados do documento
        doc_id = str(uuid.uuid4())
        supabase.table("documents").insert({
            "id": doc_id,
            "user_id": None if is_global else user_id,
            "file_name": pdf_name,
            "file_size_bytes": pdf_file.size,
            "total_chunks": len(chunks),
            "is_global": is_global,
            "uploaded_by": user_id
        }).execute()

        # Gera todos os embeddings de uma vez (mais eficiente)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embed_texts(texts)

        # Monta linhas para inserção
        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            rows.append({
                "document_id": doc_id,
                "user_id": None if is_global else user_id,
                "is_global": is_global,
                "content": chunk.page_content,
                "metadata": {
                    "source_name": pdf_name,
                    "page": chunk.metadata.get("page", 0)
                },
                "embedding": embedding
            })

        # Insere em batches de 100 (limite seguro do Supabase)
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            supabase.table("document_chunks").insert(rows[i:i + batch_size]).execute()

        scope = "global 🌐" if is_global else "pessoal 👤"
        return True, f"✅ '{pdf_name}' indexado como documento {scope}! ({len(chunks)} chunks)"

    except Exception as e:
        return False, f"❌ Erro ao processar '{pdf_name}': {e}"
