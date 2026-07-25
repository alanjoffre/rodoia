"""Testes da fusão híbrida (embedder FALSO — sem GPU, sem rede).

O foco é a fusão RRF sobre IDs: que ela reuse a lógica da Fase 1 corretamente e
que o híbrido herde o melhor de cada ranker quando eles discordam.
"""

from __future__ import annotations

import numpy as np

from rodoia.rag.avaliacao_cuad_hibrido import avaliar_hibrido, fundir
from rodoia.rag.cuad import Contrato, Pergunta, Span

_ENUN = 'related to "Exclusivity" that should. Details: {d}'


class EmbedderFake:
    dim = 2

    def __init__(self, regra: dict[str, list[float]]):
        self._regra = regra

    def _vec(self, t: str) -> list[float]:
        return self._regra.get(t, [0.0, 0.0])

    def encode_passages(self, textos: list[str]) -> np.ndarray:
        return np.asarray([self._vec(t) for t in textos], dtype=float)

    def encode_queries(self, textos: list[str]) -> np.ndarray:
        return np.asarray([self._vec(t) for t in textos], dtype=float)


def test_fundir_concordancia_total() -> None:
    """Rankings idênticos: a ordem se mantém e o top-1 soma as duas contribuições."""
    rk = ["a", "b", "c"]
    fundido, top1 = fundir(rk, rk)
    assert fundido == ["a", "b", "c"]
    # top-1 na posição 0 dos dois: 2 * 1/(60+1)
    assert abs(top1 - 2.0 / 61) < 1e-9


def test_fundir_consenso_bate_lider_exclusivo() -> None:
    """Um item que aparece nos DOIS rankings bate itens que aparecem em só um —
    é a robustez do RRF: presença nas duas listas soma contribuições."""
    bm25 = ["a", "consenso"]  # 'a' é líder exclusivo do bm25
    denso = ["b", "consenso"]  # 'b' é líder exclusivo do denso
    fundido, _ = fundir(bm25, denso)
    # a: 1/61 (só bm25). b: 1/61 (só denso). consenso: 1/62 + 1/62. consenso vence.
    assert fundido[0] == "consenso"


def test_fundir_vazio() -> None:
    assert fundir([], []) == ([], 0.0)


def test_fundir_ranking_completo_preservado() -> None:
    """A fusão devolve TODOS os IDs, não só o top-k — Recall@k para todo k precisa
    do ranking inteiro."""
    fundido, _ = fundir(["a", "b", "c", "d"], ["d", "c", "b", "a"])
    assert set(fundido) == {"a", "b", "c", "d"}
    assert len(fundido) == 4


def test_hibrido_herda_o_acerto_de_um_ranker() -> None:
    """Se o denso põe o chunk de gold em 1º e o BM25 o enterra, o híbrido ainda o
    recupera bem — o ponto inteiro da fusão."""
    texto = "z" * 30
    contrato = Contrato(
        titulo="T",
        texto=texto,
        perguntas=(
            Pergunta(
                "q1",
                "T",
                _ENUN.format(d="exclusividade"),
                "Exclusivity",
                False,
                (Span(texto="z" * 8, inicio=0),),  # gold no 1º chunk (0..20)
            ),
        ),
    )
    # embedder faz a query apontar para o 1º chunk (gold). BM25 sobre "z"*30 é
    # degenerado (tokens iguais), então o denso é quem carrega o acerto.
    embedder = EmbedderFake(
        {texto[0:20]: [1.0, 0.0], texto[15:30]: [0.0, 1.0], "Exclusivity exclusividade": [1.0, 0.0]}
    )
    rel = avaliar_hibrido([contrato], embedder, max_chars=20, overlap=5)
    assert rel["n_respondiveis"] == 1
    assert rel["metricas"]["recall_at_5"]["media"] == 1.0
    assert rel["config"]["recuperador"] == "hibrido_rrf"


def test_hibrido_conta_impossiveis() -> None:
    contrato = Contrato(
        titulo="T",
        texto="w" * 30,
        perguntas=(Pergunta("q", "T", _ENUN.format(d="a"), "Exclusivity", True, ()),),
    )
    rel = avaliar_hibrido([contrato], EmbedderFake({}), max_chars=20, overlap=5)
    assert rel["n_impossiveis"] == 1
    assert rel["n_respondiveis"] == 0
