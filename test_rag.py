from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
from rag import answer


class FakeStore:
    def __init__(self, results):
        self.results = results

    def similarity_search_with_relevance_scores(self, question, k):
        return self.results[:k]


def test_sem_evidencia_responde_nao_encontrei():
    store = FakeStore([(Document(page_content="x", metadata={"source": "a.pdf", "page": 1}), 0.05)])
    result = answer("pergunta qualquer", store)
    assert result.text == config.NOT_FOUND_MESSAGE
    assert not result.found


def test_resposta_usa_trechos_relevantes():
    doc = Document(page_content="A avaliação é em equipe.", metadata={"source": "aula.pdf", "page": 2})
    fake_llm = RunnableLambda(lambda _: AIMessage(content="Em equipe [aula.pdf, p. 2]."))
    result = answer("Como é a avaliação?", FakeStore([(doc, 0.9)]), llm=fake_llm)
    assert result.found
    assert "[aula.pdf, p. 2]" in result.text
