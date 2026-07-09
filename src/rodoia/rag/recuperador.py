"""Recuperação híbrida: combina busca densa (semântica) com BM25 (léxica) e um
reranker cross-encoder.

Por que híbrido: embeddings captam *significado* mas erram termos exatos (número
de resolução, siglas como RNTRC); BM25 capta *palavras exatas* mas ignora
sinônimos. Fundimos os dois por **Reciprocal Rank Fusion (RRF)** — robusto porque
usa a POSIÇÃO no ranking, não a escala (incomparável) dos scores. Um **reranker**
cross-encoder reordena os finalistas lendo query+trecho juntos (mais preciso, mais
caro — por isso só nos candidatos).
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from rodoia.rag.indice import COLECAO
from rodoia.rag.indice import buscar as buscar_denso

_RE_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenizar(texto: str) -> list[str]:
    return _RE_TOKEN.findall(texto.lower())


def fundir_rrf(listas: list[list[dict]], k: int, k_rrf: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: soma 1/(k_rrf + posição) de cada lista por chunk."""
    escores: dict[str, float] = defaultdict(float)
    registro: dict[str, dict] = {}
    for lista in listas:
        for pos, chunk in enumerate(lista):
            cid = chunk["chunk_id"]
            escores[cid] += 1.0 / (k_rrf + pos + 1)
            registro[cid] = chunk
    ordenados = sorted(escores, key=lambda c: escores[c], reverse=True)[:k]
    return [registro[cid] for cid in ordenados]


class Reranker:
    """Cross-encoder que pontua (consulta, trecho) diretamente."""

    def __init__(self, modelo: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        from sentence_transformers import CrossEncoder

        self._m = CrossEncoder(modelo)

    def reordenar(self, consulta: str, chunks: list[dict], k: int) -> list[dict]:
        if not chunks:
            return []
        escores = self._m.predict([[consulta, c["texto"]] for c in chunks])
        ordem = np.argsort(escores)[::-1][:k]
        return [chunks[i] for i in ordem]


class RecuperadorHibrido:
    """Orquestra denso + BM25 + (opcional) rerank sobre um corpus de chunks."""

    def __init__(
        self, chunks, embedder, cliente, colecao: str = COLECAO, reranker: Reranker | None = None
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.cliente = cliente
        self.colecao = colecao
        self.reranker = reranker
        self._bm25 = BM25Okapi([tokenizar(c["texto"]) for c in chunks])

    def _denso(self, consulta: str, n: int) -> list[dict]:
        return buscar_denso(consulta, self.embedder, self.cliente, self.colecao, k=n)

    def _bm25_buscar(self, consulta: str, n: int) -> list[dict]:
        escores = self._bm25.get_scores(tokenizar(consulta))
        idx = np.argsort(escores)[::-1][:n]
        return [self.chunks[i] for i in idx]

    def buscar(
        self,
        consulta: str,
        k: int = 5,
        candidatos: int = 20,
        modo: str = "hibrido",
        rerank: bool = False,
    ) -> list[dict]:
        """modo: 'denso' | 'bm25' | 'hibrido'. rerank: aplica o cross-encoder."""
        if modo == "denso":
            res = self._denso(consulta, candidatos)
        elif modo == "bm25":
            res = self._bm25_buscar(consulta, candidatos)
        elif modo == "hibrido":
            res = fundir_rrf(
                [self._denso(consulta, candidatos), self._bm25_buscar(consulta, candidatos)],
                k=candidatos,
            )
        else:
            raise ValueError(f"modo inválido: {modo!r}")

        if rerank:
            if self.reranker is None:
                raise ValueError("rerank=True exige um reranker")
            res = self.reranker.reordenar(consulta, res, k)
        return res[:k]
