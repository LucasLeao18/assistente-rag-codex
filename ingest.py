"""Épico A — Ingestão: PDF -> texto com página -> trechos -> Chroma (US01, US02, US03)."""
from __future__ import annotations

import io
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

import config


class IngestError(Exception):
    """Erro amigável exibido na interface (PDF vazio, corrompido etc.)."""


@dataclass
class IngestResult:
    source: str
    pages: int
    chunks: int


def extract_pages(data: bytes, source: str) -> list[Document]:
    """US02: extrai o texto de cada página preservando arquivo e número da página."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, ValueError, OSError) as exc:
        raise IngestError(f"Não foi possível ler '{source}': arquivo corrompido ou não é PDF.") from exc

    docs: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())  # limpeza básica de espaços
        if text:
            docs.append(Document(page_content=text, metadata={"source": source, "page": number}))

    if not docs:
        raise IngestError(f"'{source}' não tem texto extraível (vazio ou escaneado).")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """US03: divide em trechos de ~650 tokens com sobreposição de ~120, mantendo metadados."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#p{chunk.metadata['page']}-{i}"
    return chunks


def get_vectorstore(embeddings=None, persist_directory: str | None = None):
    """US03: índice Chroma persistente (sobrevive a reinício) com embeddings da OpenAI."""
    from langchain_chroma import Chroma

    if embeddings is None:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_directory or config.CHROMA_DIR,
    )


def search(question: str, store, k: int = config.TOP_K) -> list[Document]:
    """US03: busca direta no índice (usada para conferir a indexação)."""
    return store.similarity_search(question, k=k)


def ingest_pdf(data: bytes, source: str, store=None) -> IngestResult:
    """US01 + US02 + US03: fluxo completo de ingestão de um PDF enviado."""
    pages = extract_pages(data, source)
    chunks = split_documents(pages)
    store = store or get_vectorstore()
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return IngestResult(source=source, pages=len(pages), chunks=len(chunks))
