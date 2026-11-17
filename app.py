"""Interface Streamlit: upload de PDFs (US01) e chat com fontes (US07)."""
import streamlit as st

import config
from ingest import IngestError, get_vectorstore, ingest_pdf
from rag import answer

st.set_page_config(page_title="Assistente RAG", page_icon="📚", layout="wide")
st.title("📚 Assistente de PDFs — Codex + RAG")

if not config.OPENAI_API_KEY:
    st.warning("Defina OPENAI_API_KEY no arquivo .env (veja .env.example).")
    st.stop()


@st.cache_resource
def store():
    return get_vectorstore()


with st.sidebar:
    st.header("Documentos")
    files = st.file_uploader("Envie um ou mais PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Indexar", type="primary"):
        for f in files:
            try:
                with st.spinner(f"Indexando {f.name}..."):
                    r = ingest_pdf(f.getvalue(), f.name, store())
                st.success(f"{r.source}: {r.pages} páginas, {r.chunks} trechos")
            except IngestError as exc:
                st.error(str(exc))

if "history" not in st.session_state:
    st.session_state.history = []

def show_sources(sources):
    """US07: trechos usados, com arquivo, página e texto."""
    with st.expander("Trechos usados"):
        for d in sources:
            score = d.get("score")
            rel = f" · relevância {score:.2f}" if score is not None else ""
            st.markdown(f"**{d['source']} — p. {d['page']}**{rel}")
            st.caption(d["text"][:500] + "…")


with st.sidebar:
    if st.session_state.history and st.button("Limpar conversa"):
        st.session_state.history = []
        st.rerun()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if question := st.chat_input("Pergunte algo sobre os documentos"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        result = answer(question, store())
        st.markdown(result.text)
        sources = [
            {"source": d.metadata["source"], "page": d.metadata["page"],
             "score": d.metadata.get("score"), "text": d.page_content}
            for d in result.sources
        ]
        if sources:
            show_sources(sources)
    st.session_state.history.append({"role": "assistant", "content": result.text, "sources": sources})
