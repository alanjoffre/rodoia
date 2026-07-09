"""Constrói o índice vetorial do RAG a partir do corpus de normas.

Lê `normas.jsonl`, aplica o chunking, gera embeddings locais (E5) e popula um
Qdrant local persistente. O índice é um **artefato regenerável** (não vai para o
Git nem para o DVC — reconstrói-se a partir do corpus + código).

Uso:
    python -m rodoia.rag.construir_indice
"""

from __future__ import annotations

import json
from pathlib import Path

from rodoia.config import settings
from rodoia.rag.chunking import chunk_norma
from rodoia.rag.embeddings import E5Embedder
from rodoia.rag.indice import criar_cliente, indexar


def carregar_chunks(jsonl: Path | None = None) -> list[dict]:
    jsonl = jsonl or settings.normas_jsonl
    if not jsonl.exists():
        raise FileNotFoundError(f"{jsonl} não existe — rode baixar_normas primeiro.")
    normas = [json.loads(linha) for linha in jsonl.open(encoding="utf-8")]
    return [chunk for norma in normas for chunk in chunk_norma(norma)]


def construir(qdrant_path: Path | None = None) -> int:
    qdrant_path = qdrant_path or settings.qdrant_path
    chunks = carregar_chunks()
    print(f"chunks: {len(chunks)} | modelo: {settings.embedding_model}")
    embedder = E5Embedder(settings.embedding_model)
    print(f"embeddings dim={embedder.dim} — indexando (pode baixar o modelo na 1ª vez)...")
    cliente = criar_cliente(path=str(qdrant_path))
    n = indexar(chunks, embedder, cliente)
    print(f"OK: {n} chunks indexados em {qdrant_path}")
    return n


if __name__ == "__main__":
    construir()
