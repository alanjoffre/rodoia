"""Testes do rerank sobre o CUAD (reranker FALSO — sem GPU, sem modelo).

O que se protege: o rerank reordena **só os finalistas** e **preserva a cauda**.
Truncar a cauda não quebraria nada visivelmente — só faria Recall@k para
k > candidatos cair em silêncio.
"""

from __future__ import annotations

import numpy as np

from rodoia.rag.avaliacao_cuad import Chunk
from rodoia.rag.avaliacao_cuad_rerank import avaliar_rerank, rerankear
from rodoia.rag.cuad import Contrato, Pergunta, Span

_ENUN = 'related to "Exclusivity" that should. Details: {d}'


class RerankerFake:
    """Pontua por uma tabela texto→escore. Sem modelo, sem GPU."""

    def __init__(self, tabela: dict[str, float]):
        self._t = tabela

    def pontuar(self, consulta: str, textos: list[str]) -> list[float]:
        return [self._t.get(t, 0.0) for t in textos]


class EmbedderFake:
    dim = 2

    def __init__(self, regra: dict[str, list[float]] | None = None):
        self._r = regra or {}

    def _v(self, t: str) -> list[float]:
        return self._r.get(t, [0.0, 0.0])

    def encode_passages(self, textos: list[str]) -> np.ndarray:
        return np.asarray([self._v(t) for t in textos], dtype=float)

    def encode_queries(self, textos: list[str]) -> np.ndarray:
        return np.asarray([self._v(t) for t in textos], dtype=float)


def _chunks(n: int) -> list[Chunk]:
    return [Chunk("T", i, i * 10, (i + 1) * 10, f"texto{i}") for i in range(n)]


def test_rerank_reordena_os_finalistas() -> None:
    cs = _chunks(3)
    ranking = ["T::0", "T::1", "T::2"]
    # o cross-encoder discorda da ordem base: prefere o 2, depois o 0
    rk = RerankerFake({"texto2": 9.0, "texto0": 5.0, "texto1": 1.0})
    novo, top1 = rerankear(rk, "q", ranking, cs, candidatos=3)
    assert novo == ["T::2", "T::0", "T::1"]
    assert top1 == 9.0


def test_rerank_preserva_a_cauda() -> None:
    """Só os `candidatos` primeiros vão ao cross-encoder; o resto mantém a ordem
    original e CONTINUA no ranking — senão Recall@k para k>candidatos truncaria."""
    cs = _chunks(5)
    ranking = ["T::0", "T::1", "T::2", "T::3", "T::4"]
    rk = RerankerFake({"texto1": 9.0, "texto0": 1.0})  # só os 2 primeiros são finalistas
    novo, _ = rerankear(rk, "q", ranking, cs, candidatos=2)
    assert novo == ["T::1", "T::0", "T::2", "T::3", "T::4"]
    assert len(novo) == 5


def test_rerank_ranking_vazio() -> None:
    assert rerankear(RerankerFake({}), "q", [], _chunks(2), candidatos=5) == ([], 0.0)


def test_rerank_candidatos_maior_que_ranking() -> None:
    """Pedir mais finalistas do que há candidatos não pode estourar."""
    cs = _chunks(2)
    rk = RerankerFake({"texto0": 1.0, "texto1": 2.0})
    novo, top1 = rerankear(rk, "q", ["T::0", "T::1"], cs, candidatos=99)
    assert novo == ["T::1", "T::0"]
    assert top1 == 2.0


def test_avaliar_rerank_ponta_a_ponta() -> None:
    """O rerank promove o chunk de gold que a base tinha enterrado."""
    texto = "z" * 30
    contrato = Contrato(
        titulo="T",
        texto=texto,
        perguntas=(
            Pergunta(
                "q1", "T", _ENUN.format(d="excl"), "Exclusivity", False,
                (Span(texto="z" * 8, inicio=0),),  # gold no 1º chunk
            ),
        ),
    )
    # o cross-encoder prefere o primeiro chunk (que contém o gold)
    rk = RerankerFake({texto[0:20]: 9.0, texto[15:30]: 0.1})
    rel = avaliar_rerank(
        [contrato], EmbedderFake(), rk, max_chars=20, overlap=5, candidatos=5
    )
    assert rel["n_respondiveis"] == 1
    assert rel["metricas"]["recall_at_1"]["media"] == 1.0
    assert rel["config"]["recuperador"] == "hibrido_rrf_rerank"
    assert rel["config"]["candidatos"] == 5


def test_avaliar_rerank_conta_impossiveis() -> None:
    contrato = Contrato(
        titulo="T",
        texto="w" * 30,
        perguntas=(Pergunta("q", "T", _ENUN.format(d="a"), "Parties", True, ()),),
    )
    rel = avaliar_rerank(
        [contrato], EmbedderFake(), RerankerFake({}), max_chars=20, overlap=5
    )
    assert rel["n_impossiveis"] == 1
    assert rel["n_respondiveis"] == 0
