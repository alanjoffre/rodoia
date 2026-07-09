"""Testes da avaliação de geração (LLM-as-judge) com componentes falsos."""

from __future__ import annotations

from rodoia.rag.avaliacao_geracao import _parse_notas, avaliar_geracao, julgar


class JuizFalso:
    def __init__(self, saida):
        self.saida = saida

    def gerar(self, prompt, sistema=None):
        return self.saida


class GeradorFalso:
    def gerar(self, prompt, sistema=None):
        return "Resposta citando a Resolução 6024/2023."


class RecFalso:
    reranker = None

    def buscar(self, consulta, k=5, modo="hibrido", rerank=False):
        return [
            {"chunk_id": "A::0", "numero": "6024/2023", "vigente": True, "texto": "vale-pedágio"}
        ]


def test_parse_notas_extrai_json() -> None:
    saida = 'Aqui está: {"faithfulness": 0.9, "relevancy": 0.8} — fim.'
    d = _parse_notas(saida)
    assert d["faithfulness"] == 0.9 and d["relevancy"] == 0.8


def test_parse_notas_tolerante_a_lixo() -> None:
    assert _parse_notas("sem json aqui") == {"faithfulness": 0.0, "relevancy": 0.0}


def test_julgar_usa_o_juiz() -> None:
    juiz = JuizFalso('{"faithfulness": 1.0, "relevancy": 0.75}')
    d = julgar("pergunta", "resposta", "contexto", juiz)
    assert d["faithfulness"] == 1.0 and d["relevancy"] == 0.75


def test_avaliar_geracao_agrega_medias() -> None:
    juiz = JuizFalso('{"faithfulness": 0.8, "relevancy": 1.0}')
    res = avaliar_geracao(RecFalso(), GeradorFalso(), juiz, ["p1", "p2"], k=1)
    assert res["n"] == 2
    assert res["faithfulness_media"] == 0.8
    assert res["relevancy_media"] == 1.0
    assert res["casos"][0]["fontes"] == ["6024/2023"]
