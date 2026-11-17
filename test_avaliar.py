"""US08: métricas da avaliação com 20 perguntas."""
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
from avaliar import Caso, Resultado, avaliar, carregar_casos, metricas, registrar_no_readme


def test_gabarito_tem_20_perguntas_com_e_sem_resposta():
    casos = carregar_casos()
    assert len(casos) == 20
    assert sum(not c.tem_resposta for c in casos) >= 3
    assert all(c.arquivo and c.pagina for c in casos if c.tem_resposta)


def test_metricas():
    com = Caso("1", "Quando é a demo?", "guia.pdf", 3, "08/12")
    sem = Caso("2", "Quem é o professor?", "", None, "SEM_RESPOSTA")
    resultados = [
        Resultado(com, "Em 08/12 [guia.pdf, p. 3].", [("guia.pdf", 3)], 1.0, 0.001),
        Resultado(sem, config.NOT_FOUND_MESSAGE, [], 0.5, 0.0),
    ]
    m = metricas(resultados)
    assert (m["recuperacao_pct"], m["citacao_correta_pct"], m["fidelidade_pct"]) == (100.0, 100.0, 100.0)
    assert m["tempo_medio_s"] == 0.75


def test_inventar_resposta_sem_evidencia_derruba_fidelidade():
    sem = Caso("2", "Quem é o professor?", "", None, "SEM_RESPOSTA")
    assert not Resultado(sem, "É o professor X.", [("guia.pdf", 1)]).fiel


def test_avaliar_ponta_a_ponta_com_store_falso():
    class Store:
        def similarity_search_with_relevance_scores(self, q, k):
            doc = Document(page_content="Demo em 08/12.", metadata={"source": "guia.pdf", "page": 3})
            return [(doc, 0.9)]

    llm = RunnableLambda(lambda _: AIMessage(content="Em 08/12 [guia.pdf, p. 3]."))
    casos = [Caso("1", "Quando é a demo?", "guia.pdf", 3, "08/12")]
    [r] = avaliar(Store(), llm=llm, casos=casos)
    assert r.recuperou and r.citou_certo and r.fiel


def test_registra_no_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("antes\n<!-- avaliacao:inicio -->\nvelho\n<!-- avaliacao:fim -->\ndepois\n", encoding="utf-8")
    registrar_no_readme("novo resultado\n", readme)
    assert readme.read_text(encoding="utf-8") == "antes\n<!-- avaliacao:inicio -->\nnovo resultado\n<!-- avaliacao:fim -->\ndepois\n"
