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

from rodoia.config import REPO_ROOT, settings
from rodoia.proveniencia import carimbar
from rodoia.rag.chunking import chunk_norma
from rodoia.rag.embeddings import E5Embedder
from rodoia.rag.indice import criar_cliente, indexar


def _carregar_normas(jsonl: Path | None = None) -> list[dict]:
    jsonl = jsonl or settings.normas_jsonl
    if not jsonl.exists():
        raise FileNotFoundError(f"{jsonl} não existe — rode baixar_normas primeiro.")
    return [json.loads(linha) for linha in jsonl.open(encoding="utf-8")]


def carregar_chunks(jsonl: Path | None = None) -> list[dict]:
    return [chunk for norma in _carregar_normas(jsonl) for chunk in chunk_norma(norma)]


def escrever_stats_corpus(normas: list[dict], chunks: list[dict]) -> dict:
    """Emite a COMPOSIÇÃO do corpus num report versionado (a contagem vira evidência
    carimbada, não afirmação em prosa)."""
    stats = carimbar({
        "n_normas": len(normas),
        "n_vigentes": sum(1 for x in normas if x.get("vigente")),
        "n_chunks": len(chunks),
        "modelo_embedding": settings.embedding_model,
    })
    saida = REPO_ROOT / "reports" / "fase1_rag" / "corpus.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def construir(qdrant_path: Path | None = None) -> int:
    qdrant_path = qdrant_path or settings.qdrant_path
    normas = _carregar_normas()
    chunks = [c for n in normas for c in chunk_norma(n)]
    stats = escrever_stats_corpus(normas, chunks)
    print(f"corpus: {stats['n_normas']} normas ({stats['n_vigentes']} vigentes) | "
          f"chunks: {len(chunks)} | modelo: {settings.embedding_model}")
    embedder = E5Embedder(settings.embedding_model)
    print(f"embeddings dim={embedder.dim} — indexando (pode baixar o modelo na 1ª vez)...")
    cliente = criar_cliente(path=str(qdrant_path))
    n = indexar(chunks, embedder, cliente)
    print(f"OK: {n} chunks indexados em {qdrant_path}")
    return n


if __name__ == "__main__":
    construir()
