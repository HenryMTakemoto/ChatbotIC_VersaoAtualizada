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

    Usa a técnica de Parent Document Retriever:
    - Child chunks (300 chars): usados para busca vetorial de alta precisão.
    - Parent chunks (1500 chars): armazenados no metadata do child, servidos para a IA.

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

        # --- PARENT DOCUMENT RETRIEVER ---

        # Passo 1: Chunks GRANDES (Parent) — contexto rico para a IA
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        parent_chunks = parent_splitter.split_documents(pages)

        if not parent_chunks:
            return False, f"❌ Nenhum texto extraído de '{pdf_name}'."

        # Passo 2: Chunks PEQUENOS (Child) — alta precisão na busca vetorial
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Para cada parent, gera os children e vincula pelo metadata
        all_child_chunks = []
        for parent in parent_chunks:
            children = child_splitter.split_documents([parent])
            for child in children:
                # Guarda o texto COMPLETO do parent no metadata do child.
                # O retriever irá servir esse texto rico para a IA em vez do child pequeno.
                child.metadata["parent_content"] = parent.page_content
                all_child_chunks.append(child)

        if not all_child_chunks:
            return False, f"❌ Nenhum child chunk gerado de '{pdf_name}'."

        # Salva metadados do documento
        doc_id = str(uuid.uuid4())
        supabase.table("documents").insert({
            "id": doc_id,
            "user_id": None if is_global else user_id,
            "file_name": pdf_name,
            "file_size_bytes": pdf_file.size,
            "total_chunks": len(all_child_chunks),
            "is_global": is_global,
            "uploaded_by": user_id
        }).execute()

        # Gera embeddings apenas para os child chunks (textos pequenos e precisos)
        texts = [chunk.page_content for chunk in all_child_chunks]
        embeddings = embed_texts(texts)

        # Monta linhas para inserção
        rows = []
        for chunk, embedding in zip(all_child_chunks, embeddings):
            rows.append({
                "document_id": doc_id,
                "user_id": None if is_global else user_id,
                "is_global": is_global,
                "content": chunk.page_content,   # child: buscado pelo vetor
                "metadata": {
                    "source_name": pdf_name,
                    "page": chunk.metadata.get("page", 0),
                    "parent_content": chunk.metadata.get("parent_content", "")  # parent: servido para IA
                },
                "embedding": embedding
            })

        # Insere em batches de 100 (limite seguro do Supabase)
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            supabase.table("document_chunks").insert(rows[i:i + batch_size]).execute()

        scope = "global 🌐" if is_global else "pessoal 👤"
        return True, f"✅ '{pdf_name}' indexado como documento {scope}! ({len(all_child_chunks)} chunks via Parent-Child)"

    except Exception as e:
        return False, f"❌ Erro ao processar '{pdf_name}': {e}"
