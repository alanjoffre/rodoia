"""Rerank cross-encoder sobre o híbrido no CUAD — a arquitetura da Fase 1 COMPLETA
sobre um benchmark de terceiros (Fase 6).

A §11 do docs/17 fechou com um diagnóstico específico: o híbrido RRF é o melhor
estimador de ponto, mas o ganho **não é significativo** (ICs sobrepostos), porque
**RRF é consenso, não seletor** — onde os dois recuperadores discordam forte, a
fusão compromete em vez de escolher o vencedor. O lever apontado foi o estágio
seguinte, que a Fase 1 já tem: um **cross-encoder**, que lê query+trecho JUNTOS e
desempata **por query** — exatamente o que o RRF não consegue.

Este módulo testa isso: `denso + BM25 → RRF → rerank`, a pilha completa da Fase 1,
medida sobre gold que não é nosso.

**Por que cross-encoder é caro e por isso só nos finalistas.** O bi-encoder (denso)
codifica query e passagem SEPARADAMENTE — dá para pré-computar os vetores dos
chunks uma vez. O cross-encoder processa o PAR junto, então não há nada a
pré-computar: é uma inferência por (query, candidato). Daí o padrão: recupera
barato (top-N), reordena caro só nos N.

**Reusa o `Reranker` da Fase 1** (`rag/recuperador.py`), com modelo inglês — a
classe aceita o modelo por parâmetro. O default multilíngue da Fase 1 é para
português; num benchmark inglês seria a mesma armadilha de família documentada em
`rag/embeddings.py`.

Uso (GPU — WSL/4050):
    python -m rodoia.rag.avaliacao_cuad_rerank
    python -m rodoia.rag.avaliacao_cuad_rerank --limite 30 --candidatos 10
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
    CHUNKERS,
    KS,
    MAX_CHARS,
    OVERLAP,
    Avaliada,
    Chunk,
    _corpus_info,
    _indice_bm25,
    _ranquear,
    _registrar,
    consolidar,
    gold_da_pergunta,
    montar_query,
    obter_chunker,
)
from rodoia.rag.avaliacao_cuad_denso import _ranquear_denso
from rodoia.rag.avaliacao_cuad_hibrido import fundir
from rodoia.rag.cuad import Contrato, carregar
from rodoia.rag.embeddings import Embedder

# Cross-encoder INGLÊS. O default da Fase 1 (mmarco-mMiniLMv2) é multilíngue, para
# português — usá-lo aqui repetiria a armadilha de família dos embedders.
MODELO_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Finalistas que vão ao cross-encoder. 20 é o mesmo default da Fase 1
# (`RecuperadorHibrido.buscar(candidatos=20)`) — manter a paridade torna a
# comparação com a Fase 1 legítima em vez de um ajuste conveniente.
CANDIDATOS = 20


def rerankear(
    reranker: Any, query: str, ranking: list[str], chunks: list[Chunk], candidatos: int
) -> tuple[list[str], float]:
    """Reordena os `candidatos` primeiros do ranking com o cross-encoder.

    A cauda além dos finalistas é **preservada na ordem original** — o rerank
    reordena o topo, não descarta o resto. Sem isso, Recall@10 com 20 candidatos
    ficaria correto por acaso, mas Recall@k para k > candidatos seria silenciosamente
    truncado.
    """
    por_id = {c.id: c for c in chunks}
    finalistas = ranking[:candidatos]
    cauda = ranking[candidatos:]
    if not finalistas:
        return ranking, 0.0
    # Uma única passada do cross-encoder: pontua os finalistas e ordena aqui, em vez
    # de chamar `reordenar` e depois pontuar o líder de novo (2x inferência).
    escores = reranker.pontuar(query, [por_id[cid].texto for cid in finalistas])
    ordem = sorted(range(len(finalistas)), key=lambda i: -escores[i])
    novo = [finalistas[i] for i in ordem] + cauda
    return novo, escores[ordem[0]]


def avaliar_rerank(
    contratos: list[Contrato],
    embedder: Embedder,
    reranker: Any,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
    candidatos: int = CANDIDATOS,
    nome_modelo: str | None = None,
    chunker: str = "janela",
    escores: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Pilha completa: denso + BM25 → RRF → rerank cross-encoder.

    `escores`, se passado, recebe os escores do top-1 separados por população
    (`respondivel` / `impossivel`) — insumo de `calibracao_abstencao`. Sem isso o
    relatório só guarda percentis agregados, dos quais não se reconstrói a curva
    de limiar.
    """
    fatiar = obter_chunker(chunker)
    chunks_por_contrato = [fatiar(c.texto, c.titulo, max_chars, overlap) for c in contratos]
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
            query = montar_query(pergunta)
            bm25_rk, _ = _ranquear(bm25, cs, query)
            denso_rk, _ = _ranquear_denso(vecs, cs, qv)
            base, _ = fundir(bm25_rk, denso_rk)
            ranking, top1 = rerankear(reranker, query, base, cs, candidatos)

            if pergunta.impossivel:
                if escores is not None:
                    escores["impossivel"].append(top1)
                avaliadas.append(Avaliada(pergunta.categoria, True, top1, False, {}, 0.0))
                continue
            gold = gold_da_pergunta(pergunta, cs)
            if not gold:
                avaliadas.append(Avaliada(pergunta.categoria, False, top1, False, {}, 0.0))
                continue
            if escores is not None:
                escores["respondivel"].append(top1)
            avaliadas.append(_registrar(gold, ranking, top1, pergunta.categoria))

    config = {
        "recuperador": "hibrido_rrf_rerank",
        "modelo_embedding": nome_modelo or settings.embedding_model,
        "modelo_rerank": MODELO_RERANK,
        "chunker": chunker,
        "candidatos": candidatos,
        "max_chars": max_chars,
        "overlap": overlap,
        "ks": list(KS),
    }
    return consolidar(avaliadas, _corpus_info(contratos, len(todos_chunks)), config)


def _delta_vs_hibrido(destino: Path, rerank: dict[str, Any], sufixo: str = "") -> dict[str, Any]:
    """Δ recall@5 por categoria contra o híbrido sem rerank — testa a hipótese de
    que o cross-encoder ganha onde o RRF comprometia (rankers discordantes).

    `sufixo` é o do CHUNKER (não o da família): o delta só isola o efeito do
    cross-encoder se o híbrido de referência tiver sido fatiado do mesmo jeito.
    Ausente o par correto, devolve vazio em vez de um delta que mistura dois
    efeitos.
    """
    h_path = destino / f"retrieval_hibrido{sufixo}.json"
    if not h_path.exists():
        return {}
    ch = json.loads(h_path.read_text(encoding="utf-8"))["recall_at_5_por_categoria"]
    cr = rerank["recall_at_5_por_categoria"]
    deltas = {cat: round(cr[cat] - ch[cat], 4) for cat in cr if cat in ch}
    return dict(sorted(deltas.items(), key=lambda kv: -kv[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia RRF + rerank no CUAD (GPU, sem LLM).")
    parser.add_argument("--limite", type=int, default=None, help="usa só os N primeiros contratos")
    parser.add_argument("--zip", type=Path, default=None, help="caminho do cuad.zip")
    parser.add_argument("--familia", type=str, default="e5", choices=("e5", "bge"))
    parser.add_argument("--modelo", type=str, default=None, help="override do embedder")
    parser.add_argument("--rerank", type=str, default=MODELO_RERANK, help="modelo cross-encoder")
    parser.add_argument("--candidatos", type=int, default=CANDIDATOS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--chunker", type=str, default="janela", choices=tuple(CHUNKERS))
    parser.add_argument(
        "--dump-escores", action="store_true",
        help="grava os escores do top-1 por população (insumo de calibracao_abstencao)",
    )
    args = parser.parse_args()

    from rodoia.rag.embeddings import MODELO_PADRAO, construir_embedder
    from rodoia.rag.recuperador import Reranker

    modelo = args.modelo or (
        MODELO_PADRAO["bge"] if args.familia == "bge" else settings.embedding_model
    )
    embedder = construir_embedder(args.familia, modelo=modelo, device=args.device)
    reranker = Reranker(modelo=args.rerank)

    contratos = carregar(zip_path=args.zip)
    if args.limite:
        contratos = contratos[: args.limite]
    escores: dict[str, list[float]] | None = (
        {"respondivel": [], "impossivel": []} if args.dump_escores else None
    )
    relatorio = avaliar_rerank(
        contratos, embedder, reranker, candidatos=args.candidatos, nome_modelo=modelo,
        chunker=args.chunker, escores=escores,
    )

    destino = settings.data_processed.parent.parent / "reports" / "fase6_cuad"
    destino.mkdir(parents=True, exist_ok=True)
    suf_chunker = "" if args.chunker == "janela" else f"_{args.chunker}"
    relatorio["delta_vs_hibrido_por_categoria"] = _delta_vs_hibrido(
        destino, relatorio, suf_chunker
    )
    sufixo = ("" if args.familia == "e5" else f"_{args.familia}") + suf_chunker
    caminho = destino / f"retrieval_rerank{sufixo}.json"
    caminho.write_text(json.dumps(carimbar(relatorio), ensure_ascii=False, indent=2))
    if escores is not None:
        (destino / f"escores_rerank{sufixo}.json").write_text(
            json.dumps(escores, ensure_ascii=False)
        )

    m = relatorio["metricas"]
    print(f"embedder: {modelo} | rerank: {args.rerank} | candidatos: {args.candidatos}")
    for k in KS:
        r = m[f"recall_at_{k}"]
        print(f"  recall@{k}: {r['media']:.3f} {r['ic95_bootstrap']}")
    print(f"  MRR: {relatorio['mrr']['media']:.3f} {relatorio['mrr']['ic95_bootstrap']}")
    d = relatorio["diagnostico_abstencao"]
    print(
        f"abstenção — respondível {d['escore_top1_respondivel']['mediana']} | "
        f"impossível {d['escore_top1_impossivel']['mediana']}"
    )
    print(f"report: {caminho}")


if __name__ == "__main__":
    main()
