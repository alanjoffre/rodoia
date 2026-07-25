"""Testes da avaliação densa (embedder FALSO — sem GPU, sem rede, sem modelo).

O padrão do embedder injetável vem da Fase 1: a interface `Embedder` permite um
duplo determinístico que faz o ranqueamento previsível, então dá para afirmar
que a máquina de métricas densa produz exatamente o que se espera.
"""

from __future__ import annotations

import numpy as np

from rodoia.rag.avaliacao_cuad import Chunk, _corpus_info
from rodoia.rag.avaliacao_cuad_denso import _ranquear_denso, avaliar_denso
from rodoia.rag.cuad import Contrato, Pergunta, Span

_ENUN = 'related to "Exclusivity" that should. Details: {d}'


class EmbedderFake:
    """Embedder determinístico: mapeia texto -> vetor por uma regra simples, para
    o ranqueamento ser previsível no teste. `dim=2`."""

    dim = 2

    def __init__(self, regra: dict[str, list[float]]):
        self._regra = regra

    def _vec(self, t: str) -> list[float]:
        return self._regra.get(t, [0.0, 0.0])

    def encode_passages(self, textos: list[str]) -> np.ndarray:
        return np.asarray([self._vec(t) for t in textos], dtype=float)

    def encode_queries(self, textos: list[str]) -> np.ndarray:
        return self._encode_q(textos)

    def _encode_q(self, textos: list[str]) -> np.ndarray:
        return np.asarray([self._vec(t) for t in textos], dtype=float)


def test_ranquear_denso_ordena_por_cosseno() -> None:
    chunks = [Chunk("T", 0, 0, 10, "a"), Chunk("T", 1, 10, 20, "b"), Chunk("T", 2, 20, 30, "c")]
    vecs = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    q = np.asarray([1.0, 0.0])  # mais perto do chunk 0, depois 2, depois 1
    ranking, top1 = _ranquear_denso(vecs, chunks, q)
    assert ranking == ["T::0", "T::2", "T::1"]
    assert top1 == 1.0


def test_avaliar_denso_recupera_o_chunk_certo() -> None:
    """Query cujo vetor aponta para o chunk que contém o span -> recall 1,0."""
    texto = "x" * 30
    # dois chunks (janela 20, overlap 5): [0..20], [15..30]. O span está em 0..10,
    # dentro do primeiro chunk apenas.
    contrato = Contrato(
        titulo="T",
        texto=texto,
        perguntas=(
            Pergunta(
                "q1",
                "T",
                _ENUN.format(d="tem exclusividade?"),
                "Exclusivity",
                False,
                (Span(texto="x" * 8, inicio=0),),
            ),
        ),
    )
    # texto dos chunks é fatia do contrato: primeiro chunk começa em 0.
    primeiro_chunk_texto = texto[0:20]
    query_texto = "Exclusivity tem exclusividade?"
    embedder = EmbedderFake(
        {
            primeiro_chunk_texto: [1.0, 0.0],
            texto[15:30]: [0.0, 1.0],
            query_texto: [1.0, 0.0],  # aponta para o primeiro chunk
        }
    )
    rel = avaliar_denso([contrato], embedder, max_chars=20, overlap=5)
    assert rel["n_respondiveis"] == 1
    assert rel["metricas"]["recall_at_1"]["media"] == 1.0
    assert rel["config"]["recuperador"] == "denso"


def test_avaliar_denso_conta_impossiveis() -> None:
    contrato = Contrato(
        titulo="T",
        texto="y" * 30,
        perguntas=(
            Pergunta("q1", "T", _ENUN.format(d="a"), "Exclusivity", True, ()),
            Pergunta("q2", "T", _ENUN.format(d="b"), "Parties", True, ()),
        ),
    )
    embedder = EmbedderFake({})
    rel = avaliar_denso([contrato], embedder, max_chars=20, overlap=5)
    assert rel["n_impossiveis"] == 2
    assert rel["n_respondiveis"] == 0


def test_corpus_info_mediana() -> None:
    contratos = [Contrato("A", "x", ()), Contrato("B", "y", ())]
    assert _corpus_info(contratos, 80)["chunks_por_contrato_mediana"] == 40.0
    assert _corpus_info([], 0)["chunks_por_contrato_mediana"] == 0.0
