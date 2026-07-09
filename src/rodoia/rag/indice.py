"""Índice vetorial no Qdrant (modo local, sem servidor) para o RAG.

Local-first: `QdrantClient(path=...)` usa armazenamento em disco local e
`QdrantClient(':memory:')` roda tudo em memória (usado nos testes) — nenhum
container ou serviço externo necessário. Distância de cosseno (vetores já
normalizados pelo embedder).

Cada ponto guarda no payload os metadados do chunk (número, ano, órgão, vigência,
texto), o que permite **filtrar por vigência** e **citar a fonte** no retrieval.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from rodoia.rag.embeddings import Embedder

COLECAO = "normas_antt"


def criar_cliente(path: str | None = None) -> QdrantClient:
    """path=None → em memória (testes/efêmero); caminho → persistente em disco."""
    return QdrantClient(path=path) if path else QdrantClient(":memory:")


def indexar(
    chunks: list[dict],
    embedder: Embedder,
    cliente: QdrantClient,
    colecao: str = COLECAO,
    batch: int = 128,
) -> int:
    """Cria a coleção (recriando se já existir) e insere os chunks."""
    if cliente.collection_exists(colecao):
        cliente.delete_collection(colecao)
    cliente.create_collection(
        colecao,
        vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
    )
    for inicio in range(0, len(chunks), batch):
        lote = chunks[inicio : inicio + batch]
        vetores = embedder.encode_passages([c["texto"] for c in lote])
        pontos = [
            models.PointStruct(id=inicio + i, vector=vetor.tolist(), payload=chunk)
            for i, (vetor, chunk) in enumerate(zip(vetores, lote, strict=False))
        ]
        cliente.upsert(colecao, points=pontos)
    return len(chunks)


def buscar(
    consulta: str,
    embedder: Embedder,
    cliente: QdrantClient,
    colecao: str = COLECAO,
    k: int = 5,
    apenas_vigentes: bool = False,
) -> list[dict]:
    """Recupera os k chunks mais similares. Opcionalmente filtra só vigentes."""
    vetor = embedder.encode_queries([consulta])[0]
    filtro = None
    if apenas_vigentes:
        filtro = models.Filter(
            must=[models.FieldCondition(key="vigente", match=models.MatchValue(value=True))]
        )
    resposta = cliente.query_points(
        colecao, query=vetor.tolist(), limit=k, query_filter=filtro, with_payload=True
    )
    return [{"score": float(p.score), **p.payload} for p in resposta.points]
