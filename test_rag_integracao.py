"""US04 + US05: recuperação top-4 com score e resposta citada, ponta a ponta com Chroma real."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
from langchain_chroma import Chroma
from rag import answer, ensure_citations, retrieve

TEXTOS = [
    "A live demo da II Unidade acontece em 08/12.",
    "O code freeze do projeto é em 05/12.",
    "O seminário Scrum acontece em 27/11.",
    "As reviews são feitas às terças em aula.",
    "O Product Owner do grupo é Lucas Leão.",
    "O índice vetorial usa o Chroma persistente.",
]


def indice(tmp_path):
    store = Chroma(collection_name="teste", embedding_function=DeterministicFakeEmbedding(size=64), persist_directory=str(tmp_path))
    docs = [Document(page_content=t, metadata={"source": "guia.pdf", "page": i + 1}) for i, t in enumerate(TEXTOS)]
    store.add_documents(docs, ids=[f"guia.pdf#p{i + 1}" for i in range(len(docs))])
    return store


def test_recupera_no_maximo_4_trechos_com_score(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MIN_RELEVANCE", float("-inf"))  # sem corte, para testar o top-k
    docs = retrieve("Quando é a live demo?", indice(tmp_path))
    assert len(docs) == 4
    scores = [d.metadata["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)


def test_trecho_identico_vem_em_primeiro(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MIN_RELEVANCE", float("-inf"))
    docs = retrieve(TEXTOS[1], indice(tmp_path))
    assert docs[0].metadata["page"] == 2


def test_ponta_a_ponta_prompt_so_com_trechos_e_resposta_citada(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MIN_RELEVANCE", float("-inf"))
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt.to_string())
        return AIMessage(content="O code freeze é em 05/12 [guia.pdf, p. 2].")

    result = answer(TEXTOS[1], indice(tmp_path), llm=RunnableLambda(fake_llm))
    enviado = prompts[0]
    assert "SOMENTE com base nos trechos" in enviado
    assert enviado.count("[guia.pdf, p.") == 4  # contexto delimitado: só os 4 trechos recuperados
    assert "[guia.pdf, p. 2]" in result.text
    assert result.found


def test_acrescenta_fontes_quando_modelo_nao_cita():
    doc = Document(page_content="x", metadata={"source": "aula.pdf", "page": 3})
    assert ensure_citations("Resposta sem fonte.", [doc]).endswith("Fontes: [aula.pdf, p. 3]")
    assert ensure_citations("Com fonte [aula.pdf, p. 3].", [doc]) == "Com fonte [aula.pdf, p. 3]."
