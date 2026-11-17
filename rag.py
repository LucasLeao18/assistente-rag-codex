"""Épico B — Consulta: pergunta -> top-k trechos -> prompt -> resposta citada (US04, US05, US06)."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import config

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um assistente que responde SOMENTE com base nos trechos fornecidos. "
            "Cite a fonte no formato [arquivo, p. N] após cada afirmação. "
            f"Se os trechos não bastarem, responda exatamente: \"{config.NOT_FOUND_MESSAGE}\"",
        ),
        ("human", "Trechos:\n{context}\n\nPergunta: {question}"),
    ]
)


@dataclass
class Answer:
    text: str
    sources: list[Document] = field(default_factory=list)
    seconds: float = 0.0  # US09: tempo total da resposta
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        """US09: custo estimado da chamada ao LLM."""
        return (
            self.input_tokens * config.PRICE_INPUT_PER_1M + self.output_tokens * config.PRICE_OUTPUT_PER_1M
        ) / 1_000_000

    @property
    def found(self) -> bool:
        return bool(self.sources)


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"{citation(d)}\n{d.page_content}" for d in docs)


def retrieve(question: str, store, k: int | None = None) -> list[Document]:
    """US04: busca os top-k (padrão 4) trechos, guarda o score e descarta os pouco relevantes (US06)."""
    results = store.similarity_search_with_relevance_scores(question, k=k or config.TOP_K)
    docs = []
    for doc, score in sorted(results, key=lambda r: r[1], reverse=True):
        if score >= config.MIN_RELEVANCE:
            doc.metadata["score"] = round(float(score), 3)
            docs.append(doc)
    return docs


def citation(doc: Document) -> str:
    """US05: formato padrão de citação."""
    return f"[{doc.metadata['source']}, p. {doc.metadata['page']}]"


def ensure_citations(text: str, docs: list[Document]) -> str:
    """US05: se o modelo não citou nenhuma fonte recuperada, acrescenta as fontes usadas."""
    if any(citation(d) in text for d in docs):
        return text
    fontes = ", ".join(dict.fromkeys(citation(d) for d in docs))
    return f"{text}\n\nFontes: {fontes}"


def answer(question: str, store, llm=None, log: bool = False) -> Answer:
    """US05 + US06 + US09: gera a resposta citada (ou admite que não encontrou), medindo tempo e tokens."""
    start = time.perf_counter()
    result = _answer(question, store, llm)
    result.seconds = round(time.perf_counter() - start, 3)
    if log:
        log_answer(question, result)
    return result


def _answer(question: str, store, llm=None) -> Answer:
    docs = retrieve(question, store)
    if not docs:
        return Answer(text=config.NOT_FOUND_MESSAGE)

    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)

    reply = (PROMPT | llm).invoke({"context": format_context(docs), "question": question})
    usage = getattr(reply, "usage_metadata", None) or {}
    tokens = {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}
    text = reply.content.strip()
    if text.startswith(config.NOT_FOUND_MESSAGE):  # US06: o modelo julgou que os trechos não bastam
        return Answer(text=config.NOT_FOUND_MESSAGE, **tokens)
    return Answer(text=ensure_citations(text, docs), sources=docs, **tokens)


def log_answer(question: str, result: Answer, path: str | None = None) -> None:
    """US09: registra tempo, tokens e custo de cada resposta em CSV."""
    file = Path(path or config.LOG_FILE)
    file.parent.mkdir(parents=True, exist_ok=True)
    new = not file.exists()
    with file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["data_hora", "pergunta", "encontrou", "segundos", "tokens_entrada", "tokens_saida", "custo_usd"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"), question, result.found, result.seconds,
            result.input_tokens, result.output_tokens, f"{result.cost_usd:.6f}",
        ])
