import streamlit as st
import json
from pathlib import Path

from auth.authenticator import is_admin, get_current_user_id
from rag.pdf_processor import process_and_store_pdf
from db.documents import list_documents, list_global_documents, delete_document


def show_admin_page():
    user_id = get_current_user_id()
    admin = is_admin()

    st.title("📁 Upload de Documentos")

    # Upload de PDFs pessoais (todos os usuários)
    st.subheader("📁 Meus Documentos")
    st.caption("PDFs visíveis apenas para você nas respostas do assistente.")

    uploaded = st.file_uploader(
        "Enviar PDF pessoal",
        type="pdf",
        accept_multiple_files=True,
        key="upload_personal"
    )

    if uploaded:
        for pdf_file in uploaded:
            session_key = f"indexed_personal_{pdf_file.name}_{pdf_file.size}"
            if session_key not in st.session_state:
                with st.spinner(f"Indexando {pdf_file.name}..."):
                    success, msg = process_and_store_pdf(
                        pdf_file, user_id, is_global=False
                    )
                st.write(msg)
                if success:
                    st.session_state[session_key] = True

    # Lista documentos pessoais
    all_docs = list_documents(user_id)
    personal_docs = [d for d in all_docs if not d["is_global"]]

    if personal_docs:
        st.markdown("**Seus documentos indexados:**")
        for doc in personal_docs:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"📄 {doc['file_name']} — {doc['total_chunks']} chunks")
            with col2:
                if st.button("🗑", key=f"del_doc_{doc['id']}"):
                    delete_document(doc["id"])
                    st.rerun()
    else:
        st.info("Nenhum documento pessoal indexado ainda.")

    # Upload global (apenas admin)
    if admin:
        st.markdown("---")
        st.subheader("🌐 Base Global (Admin)")
        st.caption("PDFs disponíveis para TODOS os usuários.")

        uploaded_global = st.file_uploader(
            "Enviar PDF global",
            type="pdf",
            accept_multiple_files=True,
            key="upload_global"
        )

        if uploaded_global:
            for pdf_file in uploaded_global:
                session_key = f"indexed_global_{pdf_file.name}_{pdf_file.size}"
                if session_key not in st.session_state:
                    with st.spinner(f"Indexando {pdf_file.name} como global..."):
                        success, msg = process_and_store_pdf(
                            pdf_file, user_id, is_global=True
                        )
                    st.write(msg)
                    if success:
                        st.session_state[session_key] = True

        # Lista documentos globais
        global_docs = list_global_documents()
        if global_docs:
            st.markdown("**Documentos globais indexados:**")
            for doc in global_docs:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"🌐 {doc['file_name']} — {doc['total_chunks']} chunks")
                with col2:
                    if st.button("🗑", key=f"del_global_{doc['id']}"):
                        delete_document(doc["id"])
                        st.rerun()
        else:
            st.info("Nenhum documento global indexado ainda.")

        st.markdown("---")
        st.subheader("🧪 Diagnóstico da recuperação")
        st.caption(
            "Executa uma pergunta cega somente até a recuperação dos trechos, "
            "sem gastar tokens da LLM de resposta."
        )

        question_file = Path(__file__).resolve().parents[1] / "evals" / "questions_blind.json"
        try:
            question_items = json.loads(
                question_file.read_text(encoding="utf-8")
            )["questions"]
            selected_id = st.selectbox(
                "Pergunta de avaliação",
                options=[item["id"] for item in question_items],
                format_func=lambda item_id: next(
                    f"{item['id']} — {item['question']}"
                    for item in question_items
                    if item["id"] == item_id
                ),
            )
            selected = next(
                item for item in question_items if item["id"] == selected_id
            )

            if st.button("Testar recuperação", type="secondary"):
                from rag.retriever import hybrid_search

                with st.spinner("Buscando evidências na base..."):
                    retrieved_context = hybrid_search(
                        selected["question"], user_id
                    )

                if retrieved_context:
                    st.success("O RAG encontrou contexto acima do limiar de relevância.")
                    st.text_area(
                        "Trechos que seriam enviados ao modelo",
                        value=retrieved_context,
                        height=420,
                    )
                else:
                    st.warning(
                        "Nenhum contexto foi aceito. Para perguntas fora do escopo, "
                        "este é o comportamento esperado."
                    )
        except (OSError, KeyError, ValueError) as exc:
            st.warning(f"Perguntas de avaliação indisponíveis: {exc}")

    # Vinculação Telegram 
    st.markdown("---")
    st.subheader("📱 Vincular Telegram")
    st.caption("Use este código para vincular sua conta ao bot do Telegram.")

    st.code(user_id, language=None)
    st.info(
        "Envie o comando abaixo no Telegram:\n\n"
        f"`/vincular {user_id}`"
    )
