"""Fusão híbrida (RRF de BM25 + denso) no CUAD — o fecho do arco (Fase 6, sem LLM).

A §10 do docs/17 mostrou que BM25 e denso têm forças COMPLEMENTARES: BM25 ganha
onde o termo é raro/distintivo, o denso ganha na lacuna lexical (metadados). A
hipótese natural é que a fusão herde os ganhos dos dois e bata ambos isolados —
que é exatamente o argumento pelo qual a Fase 1 escolheu BM25+E5+RRF. Este módulo
testa a hipótese sobre um benchmark de TERCEIROS, categoria a categoria.

**Reusa a `fundir_rrf` da Fase 1** (`rag/recuperador.py`), não uma cópia: RRF é
robusto porque funde POSIÇÃO no ranking, não escala de escore (BM25 e cosseno são
incomparáveis). A função opera sobre dicts com `chunk_id`; envelopamos os IDs para
reusar a lógica testada em vez de reimplementá-la.

**Mesma máquina de métricas** dos outros dois (`avaliacao_cuad.consolidar`) — a
comparação de três recuperadores só é honesta se o Recall@k for calculado do mesmo
jeito para os três.

Uso (GPU — WSL/4050):
    python -m rodoia.rag.avaliacao_cuad_hibrido
    python -m rodoia.rag.avaliacao_cuad_hibrido --limite 50
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
    _corpus_info,
    _indice_bm25,
    _metricas_por_pergunta,
    _ranquear,
    chunkar,
    consolidar,
    gold_da_pergunta,
    montar_query,
)
from rodoia.rag.avaliacao_cuad_denso import _ranquear_denso
from rodoia.rag.cuad import Contrato, carregar
from rodoia.rag.embeddings import Embedder
from rodoia.rag.recuperador import fundir_rrf

K_RRF = 60  # mesma constante da Fase 1 (rag/recuperador.fundir_rrf)


def fundir(bm25_ranking: list[str], denso_ranking: list[str]) -> tuple[list[str], float]:
    """Funde dois rankings de IDs por RRF, reusando a fusão testada da Fase 1.

    Devolve (ranking fundido completo, escore RRF do top-1). O escore do top-1
    entra no diagnóstico de abstenção — mede a concordância entre os dois
    recuperadores no líder, não uma similaridade calibrada.
    """
    n = max(len(bm25_ranking), len(denso_ranking))
    fundido_dicts = fundir_rrf(
        [
            [{"chunk_id": cid} for cid in bm25_ranking],
            [{"chunk_id": cid} for cid in denso_ranking],
        ],
        k=n,
        k_rrf=K_RRF,
    )
    fundido = [d["chunk_id"] for d in fundido_dicts]
    if not fundido:
        return [], 0.0
    lider = fundido[0]
    top1 = 0.0
    for ranking in (bm25_ranking, denso_ranking):
        if lider in ranking:
            top1 += 1.0 / (K_RRF + ranking.index(lider) + 1)
    return fundido, top1


def avaliar_hibrido(
    contratos: list[Contrato],
    embedder: Embedder,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
) -> dict[str, Any]:
    """Avaliação híbrida: BM25 + denso por contrato, fusão RRF por pergunta."""
    # Encode grande de todos os chunks e queries — o caro, feito uma vez (mesma
    # estratégia do módulo denso).
    chunks_por_contrato = [chunkar(c.texto, c.titulo, max_chars, overlap) for c in contratos]
    todos_chunks = [ch for cs in chunks_por_contrato for ch in cs]
    queries = [montar_query(p) for c in contratos for p in c.perguntas]

    chunk_vecs = (
        embedder.encode_passages([c.texto for c in todos_chunks])
        if todos_chunks
        else np.zeros((0, embedder.dim))
    )
    query_vecs = embedder.encode_queries(queries) if queries else np.zeros((0, embedder.dim))

    avaliadas: list[Avaliada] = []
    off_chunk = off_query = 0
    for contrato, cs in zip(contratos, chunks_por_contrato, strict=True):
        vecs = chunk_vecs[off_chunk : off_chunk + len(cs)]
        off_chunk += len(cs)
        bm25 = _indice_bm25(cs) if cs else None
        for pergunta in contrato.perguntas:
            qv = query_vecs[off_query]
            off_query += 1
            if not cs or bm25 is None:
                continue
            bm25_rk, _ = _ranquear(bm25, cs, montar_query(pergunta))
            denso_rk, _ = _ranquear_denso(vecs, cs, qv)
            ranking, top1 = fundir(bm25_rk, denso_rk)
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
        "recuperador": "hibrido_rrf",
        "modelo": settings.embedding_model,
        "k_rrf": K_RRF,
        "max_chars": max_chars,
        "overlap": overlap,
        "ks": list(KS),
    }
    return consolidar(avaliadas, _corpus_info(contratos, len(todos_chunks)), config)


def _tabela_tres_vias(destino: Path, hibrido: dict[str, Any]) -> dict[str, Any]:
    """recall@5 por categoria nos três recuperadores + qual vence — se os
    relatórios BM25 e denso existirem. É o teste direto da hipótese: o híbrido
    deve ficar >= max(bm25, denso) na maioria das categorias."""
    b_path, d_path = destino / "retrieval_bm25.json", destino / "retrieval_denso.json"
    if not (b_path.exists() and d_path.exists()):
        return {}
    cb = json.loads(b_path.read_text(encoding="utf-8"))["recall_at_5_por_categoria"]
    cd = json.loads(d_path.read_text(encoding="utf-8"))["recall_at_5_por_categoria"]
    ch = hibrido["recall_at_5_por_categoria"]
    linhas = {}
    hibrido_vence_ou_empata = 0
    for cat in ch:
        if cat not in cb or cat not in cd:
            continue
        melhor_isolado = max(cb[cat], cd[cat])
        linhas[cat] = {
            "bm25": cb[cat],
            "denso": cd[cat],
            "hibrido": ch[cat],
            "hibrido_vs_melhor_isolado": round(ch[cat] - melhor_isolado, 4),
        }
        if ch[cat] >= melhor_isolado - 1e-9:
            hibrido_vence_ou_empata += 1
    return {
        "n_categorias": len(linhas),
        "hibrido_>=_melhor_isolado_em": hibrido_vence_ou_empata,
        "por_categoria": dict(
            sorted(linhas.items(), key=lambda kv: -kv[1]["hibrido_vs_melhor_isolado"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia fusão híbrida no CUAD (GPU, sem LLM).")
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
    relatorio = avaliar_hibrido(contratos, embedder)

    destino = settings.data_processed.parent.parent / "reports" / "fase6_cuad"
    destino.mkdir(parents=True, exist_ok=True)
    relatorio["tres_vias_por_categoria"] = _tabela_tres_vias(destino, relatorio)
    caminho = destino / "retrieval_hibrido.json"
    caminho.write_text(json.dumps(carimbar(relatorio), ensure_ascii=False, indent=2))

    m = relatorio["metricas"]
    corpus = relatorio["corpus"]
    print(f"modelo: {modelo} | RRF k={K_RRF}")
    print(f"contratos: {corpus['n_contratos']} | chunks: {corpus['n_chunks']:,}")
    for k in KS:
        r = m[f"recall_at_{k}"]
        print(f"  recall@{k}: {r['media']:.3f} {r['ic95_bootstrap']}")
    print(f"  MRR: {relatorio['mrr']['media']:.3f} {relatorio['mrr']['ic95_bootstrap']}")
    tv = relatorio["tres_vias_por_categoria"]
    if tv:
        print(
            f"híbrido >= melhor isolado em {tv['hibrido_>=_melhor_isolado_em']}/"
            f"{tv['n_categorias']} categorias"
        )
    print(f"report: {caminho}")


if __name__ == "__main__":
    main()
