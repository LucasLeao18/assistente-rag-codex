"""US03: índice Chroma persistente com metadados de arquivo, página e trecho."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ingest import get_vectorstore, search


def docs():
    return [
        Document(page_content="A prova é individual.", metadata={"source": "aula.pdf", "page": 1, "chunk_id": "aula.pdf#p1-0"}),
        Document(page_content="O projeto é em equipe.", metadata={"source": "aula.pdf", "page": 2, "chunk_id": "aula.pdf#p2-1"}),
    ]


def test_indice_sobrevive_a_reinicio(tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    store = get_vectorstore(emb, str(tmp_path))
    store.add_documents(docs(), ids=[d.metadata["chunk_id"] for d in docs()])
    del store

    reaberto = get_vectorstore(emb, str(tmp_path))  # simula reinício do app
    resultado = search("O projeto é em equipe.", reaberto, k=1)[0]
    assert resultado.metadata == {"source": "aula.pdf", "page": 2, "chunk_id": "aula.pdf#p2-1"}


def test_reindexar_mesmo_arquivo_nao_duplica(tmp_path):
    store = get_vectorstore(DeterministicFakeEmbedding(size=32), str(tmp_path))
    for _ in range(2):
        store.add_documents(docs(), ids=[d.metadata["chunk_id"] for d in docs()])
    assert len(store.get()["ids"]) == 2
