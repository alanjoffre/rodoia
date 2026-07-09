"""Testes da recuperação híbrida e da avaliação (sem modelo/servidor)."""

from __future__ import annotations

import numpy as np

from rodoia.rag.avaliacao_retrieval import _rank_da_fonte, avaliar_modo
from rodoia.rag.indice import criar_cliente, indexar
from rodoia.rag.recuperador import RecuperadorHibrido, fundir_rrf, tokenizar


class EmbedderFalso:
    dim = 24

    def _vec(self, t: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for ch in t:
            v[ord(ch) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, textos):
        return np.array([self._vec(t) for t in textos])

    encode_queries = encode_passages


def _chunks():
    return [
        {
            "chunk_id": "A::0",
            "numero": "6024/2023",
            "vigente": True,
            "texto": "vale-pedagio obrigatorio no transporte rodoviario de cargas",
        },
        {
            "chunk_id": "B::0",
            "numero": "5232/2016",
            "vigente": True,
            "texto": "instrucoes complementares ao transporte de produtos perigosos",
        },
        {
            "chunk_id": "C::0",
            "numero": "5867/2020",
            "vigente": True,
            "texto": "pisos minimos de frete por eixo carregado coeficientes",
        },
    ]


def _recuperador():
    chunks = _chunks()
    cli = criar_cliente()
    indexar(chunks, EmbedderFalso(), cli)
    return RecuperadorHibrido(chunks, EmbedderFalso(), cli, reranker=None)


def test_tokenizar() -> None:
    assert tokenizar("Vale-Pedágio, obrigatório!") == ["vale", "pedágio", "obrigatório"]


def test_rrf_prioriza_consenso() -> None:
    lista1 = [{"chunk_id": "X"}, {"chunk_id": "Y"}]
    lista2 = [{"chunk_id": "Y"}, {"chunk_id": "Z"}]
    fundido = fundir_rrf([lista1, lista2], k=3)
    assert fundido[0]["chunk_id"] == "Y"  # aparece bem nas duas listas


def test_bm25_acha_termo_exato() -> None:
    rec = _recuperador()
    res = rec.buscar("produtos perigosos", k=1, modo="bm25")
    assert res[0]["numero"] == "5232/2016"


def test_hibrido_retorna_resultados() -> None:
    rec = _recuperador()
    res = rec.buscar("pisos minimos de frete por eixo", k=2, modo="hibrido")
    assert len(res) >= 1
    assert res[0]["numero"] == "5867/2020"


def test_rank_da_fonte() -> None:
    res = [{"numero": "1/2020"}, {"numero": "5867/2020"}, {"numero": "3/2020"}]
    assert _rank_da_fonte(res, ["5867/2020"]) == 2
    assert _rank_da_fonte(res, ["9/9999"]) is None


def test_avaliar_modo_com_recuperador_real() -> None:
    """Avalia o modo bm25 sobre um mini conjunto — deve achar as fontes."""
    rec = _recuperador()
    import rodoia.rag.avaliacao_retrieval as av

    original = av.CONJUNTO_DOURADO
    av.CONJUNTO_DOURADO = [
        {"consulta": "produtos perigosos", "fontes": ["5232/2016"]},
        {"consulta": "vale pedagio cargas", "fontes": ["6024/2023"]},
    ]
    try:
        m = avaliar_modo(rec, modo="bm25", rerank=False, k=3)
        assert m["recall_at_k"] == 1.0
        assert m["mrr"] > 0
    finally:
        av.CONJUNTO_DOURADO = original
