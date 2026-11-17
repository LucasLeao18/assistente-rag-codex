import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

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


def text_pdf(pages: list[str]) -> bytes:
    """Gera um PDF simples com uma linha de texto por página."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=300, height=200)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extrai_texto_com_arquivo_e_pagina():
    docs = extract_pages(text_pdf(["Primeira pagina", "Segunda pagina"]), "aula.pdf")
    assert [d.metadata for d in docs] == [
        {"source": "aula.pdf", "page": 1},
        {"source": "aula.pdf", "page": 2},
    ]
    assert "Segunda" in docs[1].page_content


def test_arquivo_que_nao_e_pdf_e_recusado():
    with pytest.raises(IngestError, match="não é um PDF"):
        extract_pages(b"%PDF qualquer", "notas.txt")
