"""US08: avaliação do assistente com 20 perguntas e gabarito.

Uso: python avaliar.py
Indexa os PDFs de eval/docs/ num índice temporário, responde as perguntas de eval/perguntas.csv
e grava as métricas em eval/resultado.md e na seção "Avaliação" do README.
"""
from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

import config

EVAL_DIR = Path(__file__).parent / "eval"
SEM_RESPOSTA = "SEM_RESPOSTA"
README_INICIO = "<!-- avaliacao:inicio -->"
README_FIM = "<!-- avaliacao:fim -->"


@dataclass
class Caso:
    id: str
    pergunta: str
    arquivo: str
    pagina: int | None
    esperado: str

    @property
    def tem_resposta(self) -> bool:
        return self.esperado != SEM_RESPOSTA


@dataclass
class Resultado:
    caso: Caso
    texto: str
    fontes: list[tuple[str, int]]
    segundos: float = 0.0
    custo_usd: float = 0.0

    @property
    def recuperou(self) -> bool:
        """Recuperação: a página do gabarito está entre os trechos recuperados."""
        return (self.caso.arquivo, self.caso.pagina) in self.fontes

    @property
    def citou_certo(self) -> bool:
        """Citação correta: a resposta cita [arquivo, p. N] do gabarito."""
        return f"[{self.caso.arquivo}, p. {self.caso.pagina}]" in self.texto

    @property
    def fiel(self) -> bool:
        """Fidelidade: responde quando há evidência e diz que não encontrou quando não há."""
        nao_encontrou = self.texto.startswith(config.NOT_FOUND_MESSAGE)
        return not nao_encontrou if self.caso.tem_resposta else nao_encontrou


def carregar_casos(path: Path = EVAL_DIR / "perguntas.csv") -> list[Caso]:
    with path.open(encoding="utf-8") as f:
        return [
            Caso(r["id"], r["pergunta"], r["arquivo"], int(r["pagina"]) if r["pagina"] else None, r["resposta_esperada"])
            for r in csv.DictReader(f)
        ]


def metricas(resultados: list[Resultado]) -> dict[str, float]:
    com = [r for r in resultados if r.caso.tem_resposta]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else 0.0
    return {
        "perguntas": len(resultados),
        "recuperacao_pct": pct([r.recuperou for r in com]),
        "citacao_correta_pct": pct([r.citou_certo for r in com]),
        "fidelidade_pct": pct([r.fiel for r in resultados]),
        "tempo_medio_s": round(sum(r.segundos for r in resultados) / len(resultados), 2) if resultados else 0.0,
        "custo_total_usd": round(sum(r.custo_usd for r in resultados), 4),
    }


def relatorio(resultados: list[Resultado]) -> str:
    m = metricas(resultados)
    ok = lambda b: "✅" if b else "❌"
    linhas = [
        f"**{m['perguntas']} perguntas** · recuperação {m['recuperacao_pct']}% · citação correta "
        f"{m['citacao_correta_pct']}% · fidelidade {m['fidelidade_pct']}% · tempo médio {m['tempo_medio_s']} s "
        f"· custo total US$ {m['custo_total_usd']}",
        "",
        "| # | Pergunta | Recuperação | Citação | Fidelidade |",
        "|---|---|---|---|---|",
    ]
    for r in resultados:
        rec, cit = (ok(r.recuperou), ok(r.citou_certo)) if r.caso.tem_resposta else ("—", "—")
        linhas.append(f"| {r.caso.id} | {r.caso.pergunta} | {rec} | {cit} | {ok(r.fiel)} |")
    return "\n".join(linhas) + "\n"


def avaliar(store, llm=None, casos: list[Caso] | None = None) -> list[Resultado]:
    from rag import answer

    resultados = []
    for caso in casos or carregar_casos():
        a = answer(caso.pergunta, store, llm=llm)
        fontes = [(d.metadata["source"], d.metadata["page"]) for d in a.sources]
        resultados.append(Resultado(caso, a.text, fontes, a.seconds, a.cost_usd))
    return resultados


def registrar_no_readme(texto: str, readme: Path = Path(__file__).parent / "README.md") -> None:
    conteudo = readme.read_text(encoding="utf-8")
    ini, fim = conteudo.index(README_INICIO) + len(README_INICIO), conteudo.index(README_FIM)
    readme.write_text(conteudo[:ini] + "\n" + texto + conteudo[fim:], encoding="utf-8")


def main() -> None:
    from ingest import get_vectorstore, ingest_pdf

    with tempfile.TemporaryDirectory() as tmp:
        config.CHROMA_DIR = tmp
        store = get_vectorstore()
        for pdf in sorted((EVAL_DIR / "docs").glob("*.pdf")):
            ingest_pdf(pdf.read_bytes(), pdf.name, store)
        texto = relatorio(avaliar(store))
    (EVAL_DIR / "resultado.md").write_text(texto, encoding="utf-8")
    registrar_no_readme(texto)
    print(texto)


if __name__ == "__main__":
    main()
