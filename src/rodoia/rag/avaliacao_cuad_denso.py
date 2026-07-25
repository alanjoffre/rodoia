"""Recuperação DENSA no CUAD contra o baseline BM25 (Fase 6, sem LLM).

A §9.2 do docs/17 deixou uma hipótese específica, não uma esperança: o baseline
BM25 falha nas categorias de **lacuna lexical** (Document Name, Parties, Volume
Restriction — metadados cujo texto não repete os termos da query). Recuperação
densa casa por significado, não por sobreposição de tokens, então o ganho — se
houver — deve se concentrar exatamente ali. Este módulo testa isso.

**Sem Qdrant.** A recuperação é dentro do contrato (~41 chunks); um índice
vetorial para 41 pontos seria peso morto. Com vetores normalizados pelo
`E5Embedder`, similaridade de cosseno é um produto escalar em numpy.

**A máquina de métricas é a mesma do BM25** (`avaliacao_cuad.consolidar`): a
comparação precisa medir os recuperadores, não duas implementações de Recall@k.
Só o ranqueamento muda.

**Ressalva honesta sobre o modelo.** O `E5Embedder` padrão é o
`multilingual-e5-small` (384 dim) — validado na Fase 1 para português, e o CUAD é
inglês. É um modelo pequeno e multilíngue rodando num benchmark inglês: se ele já
bater o BM25 nas categorias de lacuna lexical, a hipótese se confirma mesmo com
modelo modesto; se perder, a conclusão NÃO é "denso não ajuda" e sim "o próximo
lever é um modelo inglês forte (bge-large-en)". A distinção fica registrada no
relatório e no doc, não escondida atrás de uma média.

Uso (roda na GPU — WSL/4050):
    python -m rodoia.rag.avaliacao_cuad_denso                # corpus inteiro
    python -m rodoia.rag.avaliacao_cuad_denso --limite 50    # dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rodoia.config import settings
from rodoia.proveniencia import carimbar
from rodoia.rag.avaliacao_cuad import (
    KS,
    MAX_CHARS,
    OVERLAP,
    Avaliada,
    Chunk,
    _corpus_info,
    _metricas_por_pergunta,
    chunkar,
    consolidar,
    gold_da_pergunta,
    montar_query,
)
from rodoia.rag.cuad import Contrato, carregar
from rodoia.rag.embeddings import Embedder


def _ranquear_denso(
    chunk_vecs: np.ndarray, chunks: list[Chunk], query_vec: np.ndarray
) -> tuple[list[str], float]:
    """(IDs por cosseno decrescente, escore do top-1).

    Vetores já normalizados → cosseno é produto escalar. `chunk_vecs` é
    (n_chunks, dim); `query_vec` é (dim,).
    """
    escores = chunk_vecs @ query_vec
    ordem = np.argsort(-escores)
    top1 = float(escores[ordem[0]]) if len(ordem) else 0.0
    return [chunks[i].id for i in ordem], top1


def avaliar_denso(
    contratos: list[Contrato],
    embedder: Embedder,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
) -> dict[str, Any]:
    """Avaliação densa completa: encoda em lote, ranqueia por cosseno, consolida.

    Encoda TODOS os chunks e TODAS as queries em dois lotes grandes (a GPU rende
    muito mais assim que uma chamada por pergunta), depois fatia por contrato.
    """
    # 1) Materializa chunks e queries de todos os contratos, guardando as
    # fronteiras para fatiar os vetores de volta por contrato.
    chunks_por_contrato: list[list[Chunk]] = []
    todos_chunks: list[Chunk] = []
    for contrato in contratos:
        cs = chunkar(contrato.texto, contrato.titulo, max_chars, overlap)
        chunks_por_contrato.append(cs)
        todos_chunks.extend(cs)

    queries = [montar_query(p) for c in contratos for p in c.perguntas]

    # 2) Dois encodes grandes — o caro, feito uma vez.
    chunk_vecs = (
        embedder.encode_passages([c.texto for c in todos_chunks])
        if todos_chunks
        else np.zeros((0, embedder.dim))
    )
    query_vecs = (
        embedder.encode_queries(queries) if queries else np.zeros((0, embedder.dim))
    )

    # 3) Ranqueia por contrato, reusando a fatia de vetores.
    avaliadas: list[Avaliada] = []
    off_chunk = 0
    off_query = 0
    for contrato, cs in zip(contratos, chunks_por_contrato, strict=True):
        vecs = chunk_vecs[off_chunk : off_chunk + len(cs)]
        off_chunk += len(cs)
        for pergunta in contrato.perguntas:
            qv = query_vecs[off_query]
            off_query += 1
            if not cs:
                continue
            ranking, top1 = _ranquear_denso(vecs, cs, qv)
            if pergunta.impossivel:
                avaliadas.append(Avaliada(pergunta.categoria, True, top1, False, {}, 0.0))
                continue
            gold = gold_da_pergunta(pergunta, cs)
            if not gold:
                avaliadas.append(Avaliada(pergunta.categoria, False, top1, False, {}, 0.0))
                continue
            reg = _metricas_por_pergunta(gold, ranking, top1)
            avaliadas.append(
                Avaliada(pergunta.categoria, False, top1, True, reg.recall_por_k, reg.rr)
            )

    config = {
        "recuperador": "denso",
        "modelo": settings.embedding_model,
        "max_chars": max_chars,
        "overlap": overlap,
        "ks": list(KS),
    }
    return consolidar(avaliadas, _corpus_info(contratos, len(todos_chunks)), config)


def _comparar_categorias(bm25_path: Path, denso: dict[str, Any]) -> dict[str, Any]:
    """Δ recall@5 por categoria (denso − bm25), se o relatório BM25 existir.

    É o teste direto da hipótese: o ganho deve concentrar nas categorias de
    lacuna lexical (as piores do BM25). Sem o relatório BM25, devolve {}.
    """
    if not bm25_path.exists():
        return {}
    bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
    cb = bm25.get("recall_at_5_por_categoria", {})
    cd = denso.get("recall_at_5_por_categoria", {})
    deltas = {
        cat: round(cd[cat] - cb[cat], 4)
        for cat in cd
        if cat in cb
    }
    return dict(sorted(deltas.items(), key=lambda kv: -kv[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia recuperação DENSA no CUAD (GPU, sem LLM).")
    parser.add_argument("--limite", type=int, default=None, help="usa só os N primeiros contratos")
    parser.add_argument("--zip", type=Path, default=None, help="caminho do cuad.zip")
    parser.add_argument("--modelo", type=str, default=None, help="override do modelo de embeddings")
    parser.add_argument("--device", type=str, default=None, help="cuda|cpu (default: auto)")
    args = parser.parse_args()

    from rodoia.rag.embeddings import E5Embedder

    modelo = args.modelo or settings.embedding_model
    embedder = E5Embedder(modelo=modelo, device=args.device)

    contratos = carregar(zip_path=args.zip)
    if args.limite:
        contratos = contratos[: args.limite]
    relatorio = avaliar_denso(contratos, embedder)

    destino = settings.data_processed.parent.parent / "reports" / "fase6_cuad"
    destino.mkdir(parents=True, exist_ok=True)
    relatorio["delta_vs_bm25_por_categoria"] = _comparar_categorias(
        destino / "retrieval_bm25.json", relatorio
    )
    caminho = destino / "retrieval_denso.json"
    caminho.write_text(json.dumps(carimbar(relatorio), ensure_ascii=False, indent=2))

    m = relatorio["metricas"]
    corpus = relatorio["corpus"]
    print(f"modelo: {modelo}")
    print(f"contratos: {corpus['n_contratos']} | chunks: {corpus['n_chunks']:,}")
    for k in KS:
        r = m[f"recall_at_{k}"]
        print(f"  recall@{k}: {r['media']:.3f} {r['ic95_bootstrap']}")
    print(f"  MRR: {relatorio['mrr']['media']:.3f} {relatorio['mrr']['ic95_bootstrap']}")
    delta = relatorio["delta_vs_bm25_por_categoria"]
    if delta:
        itens = list(delta.items())
        print("Δ recall@5 vs BM25 — 3 maiores ganhos:")
        for cat, d in itens[:3]:
            print(f"   {d:+.3f}  {cat}")
        print("Δ recall@5 vs BM25 — 3 maiores perdas:")
        for cat, d in itens[-3:]:
            print(f"   {d:+.3f}  {cat}")
    print(f"report: {caminho}")


if __name__ == "__main__":
    main()
