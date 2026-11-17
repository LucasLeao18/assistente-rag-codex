"""US09: tempo, tokens e custo por resposta."""
import csv

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
from rag import Answer, answer, log_answer
from test_rag import FakeStore


def test_resposta_traz_tempo_tokens_e_custo(monkeypatch):
    monkeypatch.setattr(config, "PRICE_INPUT_PER_1M", 0.15)
    monkeypatch.setattr(config, "PRICE_OUTPUT_PER_1M", 0.60)
    doc = Document(page_content="A prova é dia 10.", metadata={"source": "aula.pdf", "page": 1})
    reply = AIMessage(
        content="Dia 10 [aula.pdf, p. 1].",
        usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
    )
    result = answer("Quando é a prova?", FakeStore([(doc, 0.9)]), llm=RunnableLambda(lambda _: reply))
    assert result.seconds >= 0
    assert (result.input_tokens, result.output_tokens) == (1000, 200)
    assert abs(result.cost_usd - 0.00027) < 1e-9


def test_log_em_csv(tmp_path):
    arquivo = tmp_path / "respostas.csv"
    log_answer("pergunta 1", Answer(text="ok", seconds=1.2, input_tokens=10, output_tokens=5), str(arquivo))
    log_answer("pergunta 2", Answer(text="ok", seconds=0.8), str(arquivo))
    linhas = list(csv.DictReader(arquivo.open(encoding="utf-8")))
    assert [l["pergunta"] for l in linhas] == ["pergunta 1", "pergunta 2"]
    assert linhas[0]["segundos"] == "1.2" and linhas[0]["tokens_entrada"] == "10"
