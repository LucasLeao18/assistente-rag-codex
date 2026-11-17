"""Épico B — Consulta: pergunta -> top-k trechos -> prompt -> resposta citada (US04, US05, US06)."""
from __future__ import annotations

from dataclasses import dataclass, field

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


def answer(question: str, store, llm=None) -> Answer:
    """US05 + US06: gera a resposta citada ou admite que não encontrou."""
    docs = retrieve(question, store)
    if not docs:
        return Answer(text=config.NOT_FOUND_MESSAGE)

    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)

    reply = (PROMPT | llm).invoke({"context": format_context(docs), "question": question})
    return Answer(text=ensure_citations(reply.content, docs), sources=docs)
