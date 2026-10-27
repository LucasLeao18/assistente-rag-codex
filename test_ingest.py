import io

import pytest
from pypdf import PdfWriter

from ingest import IngestError, extract_pages, split_documents
from langchain_core.documents import Document


def blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_corrompido_gera_erro_amigavel():
    with pytest.raises(IngestError, match="corrompido"):
        extract_pages(b"isto nao e um pdf", "ruim.pdf")


def test_pdf_sem_texto_gera_erro_amigavel():
    with pytest.raises(IngestError, match="sem texto|não tem texto"):
        extract_pages(blank_pdf(), "vazio.pdf")


def test_split_preserva_arquivo_e_pagina():
    doc = Document(page_content="palavra " * 2000, metadata={"source": "aula.pdf", "page": 3})
    chunks = split_documents([doc])
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "aula.pdf" and c.metadata["page"] == 3 for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
