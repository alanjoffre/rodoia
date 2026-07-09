"""Embeddings locais para o RAG (local-first, custo zero).

Modelo padrão: `intfloat/multilingual-e5-small` — multilíngue (bom em português),
384 dimensões, roda em CPU/MPS. Detalhe que muita gente erra: a família E5 exige
**prefixos** `query:` e `passage:` no texto; sem eles, a qualidade cai bastante.
Encapsulamos isso para o resto do pipeline não precisar saber.

Interface `Embedder` (dim, encode_passages, encode_queries) permite injetar um
embedder falso nos testes e trocar por API depois, sem tocar no índice.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode_passages(self, textos: list[str]) -> np.ndarray: ...
    def encode_queries(self, textos: list[str]) -> np.ndarray: ...


class E5Embedder:
    """Embedder local baseado em sentence-transformers (família E5)."""

    def __init__(self, modelo: str = "intfloat/multilingual-e5-small", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(modelo, device=device)
        self.dim = self._m.get_sentence_embedding_dimension()

    def _encode(self, textos: list[str], prefixo: str) -> np.ndarray:
        marcados = [f"{prefixo}: {t}" for t in textos]
        # normalize=True → vetores unitários, para usar distância de cosseno.
        return np.asarray(self._m.encode(marcados, normalize_embeddings=True))

    def encode_passages(self, textos: list[str]) -> np.ndarray:
        return self._encode(textos, "passage")

    def encode_queries(self, textos: list[str]) -> np.ndarray:
        return self._encode(textos, "query")
