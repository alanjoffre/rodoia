"""Avaliação de recuperação sobre o **CUAD** — benchmark externo, sem LLM (Fase 6).

Com gold de terceiros e offsets conferidos (ver `rag/cuad.py`), Recall@k e MRR
saem **sem uma única chamada de modelo**. Este módulo faz o caminho inteiro:
chunking com offsets → mapeamento span→chunk → recuperação BM25 → métricas com IC.

**Recall@k aqui é Recall de verdade, não hit-rate.** `rag/avaliacao_retrieval.py`
documenta honestamente que a Fase 1 mede *hit-rate*: cada pergunta tem UMA
fonte-gold, então não dá para calcular "fração de todos os relevantes". No CUAD o
gold é exaustivo por pergunta (13.823 spans anotados por advogados), então a
fração de todos os chunks relevantes recuperados é computável — e é o que se
reporta.

**A recuperação é DENTRO do contrato, não global.** O enunciado é idêntico nos
510 contratos (`...related to "Exclusivity"...`), então buscar no corpus inteiro
seria rodar a mesma query contra 510 documentos — sem sentido e sem relação com
a tarefa. O CUAD pergunta: *dado este contrato, onde está esta cláusula?* O
candidato set de cada pergunta são os chunks do seu próprio contrato.

**A query descarta o boilerplate do enunciado.** `Highlight the parts (if any) of
this contract related to "X" that should be reviewed by a lawyer. Details: Y` é
~70% preâmbulo constante — idêntico em todas as 20.910 perguntas. Usá-lo cru dá
um baseline artificialmente ruim que não informa nada sobre a arquitetura. A
query é `X + Y` (categoria + detalhes), que é o mínimo de pré-processamento
honesto e está documentado aqui em vez de escondido.

**As perguntas `is_impossible` NÃO entram no Recall@k** — elas não têm gold a
recuperar. Ficam num diagnóstico separado: a distribuição do escore do top-1
para respondíveis vs impossíveis. Se as duas distribuições se sobrepõem, nenhum
limiar de abstenção vai funcionar, e é melhor descobrir isso agora do que depois
de construir a política. Métrica de abstenção com limiar é o passo seguinte.

Uso:
    python -m rodoia.rag.avaliacao_cuad                 # corpus inteiro
    python -m rodoia.rag.avaliacao_cuad --limite 50     # 50 contratos (dev)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rodoia.config import settings
from rodoia.estat import bootstrap_ic, percentil, wilson
from rodoia.proveniencia import carimbar
from rodoia.rag.cuad import Contrato, Pergunta, carregar

MAX_CHARS = 1500
OVERLAP = 200
KS = (1, 3, 5, 10)

_RE_DETALHES = re.compile(r"Details:\s*(.*)", re.S)
_RE_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    """Um trecho do contrato COM seu intervalo de caracteres no texto original.

    O intervalo é o que permite o mapeamento exato span→chunk. Guardar só o texto
    obrigaria a re-encontrar o span por busca de string, que é ambíguo quando o
    mesmo trecho se repete no contrato — e contrato é cheio de repetição.
    """

    contrato: str
    indice: int
    inicio: int
    fim: int
    texto: str

    @property
    def id(self) -> str:
        return f"{self.contrato}::{self.indice}"


def chunkar(
    texto: str, contrato: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP
) -> list[Chunk]:
    """Janela deslizante com sobreposição, preservando offsets.

    Deliberadamente NÃO reusa `rag.chunking.chunk_texto`: aquele divide em
    `Art. Nº` (estrutura jurídica brasileira), que não casa nada em contrato
    comercial americano — degeneraria para janela cega, só que sem offsets.
    Chunking consciente da estrutura do CUAD (cláusulas numeradas) é uma
    melhoria óbvia, e a comparação contra esta janela é justamente o experimento.
    """
    if not texto:
        return []
    passo = max(1, max_chars - overlap)
    chunks: list[Chunk] = []
    for i, inicio in enumerate(range(0, len(texto), passo)):
        fim = min(inicio + max_chars, len(texto))
        chunks.append(Chunk(contrato, i, inicio, fim, texto[inicio:fim]))
        if fim == len(texto):
            break
    return chunks


def gold_da_pergunta(pergunta: Pergunta, chunks: list[Chunk]) -> set[str]:
    """Chunks cujo intervalo INTERSECTA algum span de resposta.

    Interseção, não continência: com sobreposição de janela um span pode cair em
    dois chunks, e um span longo pode atravessar vários. Exigir continência
    total marcaria zero gold para spans maiores que a janela — silenciosamente.
    """
    ids: set[str] = set()
    for span in pergunta.spans:
        for c in chunks:
            if span.inicio < c.fim and span.fim > c.inicio:
                ids.add(c.id)
    return ids


def montar_query(pergunta: Pergunta) -> str:
    """Categoria + detalhes, sem o preâmbulo constante do enunciado."""
    m = _RE_DETALHES.search(pergunta.enunciado)
    detalhes = m.group(1).strip() if m else ""
    return f"{pergunta.categoria} {detalhes}".strip()


def _tokenizar(texto: str) -> list[str]:
    return _RE_TOKEN.findall(texto.lower())


def _indice_bm25(chunks: list[Chunk]) -> Any:
    """Índice BM25 dos chunks de UM contrato.

    Construído uma vez por contrato e reusado nas suas 41 perguntas — o índice
    não depende da query. Construir por pergunta re-tokenizaria o contrato
    inteiro 82 vezes (41 perguntas x ranking + escore), o que domina o tempo.

    Import local: `rank_bm25` é dependência da Fase 1 e não deve custar import a
    quem só usa o parser do CUAD.
    """
    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenizar(c.texto) for c in chunks])


def _ranquear(bm25: Any, chunks: list[Chunk], query: str) -> tuple[list[str], float]:
    """(IDs ordenados por BM25 decrescente, escore do top-1)."""
    escores = bm25.get_scores(_tokenizar(query))
    ordem = sorted(range(len(chunks)), key=lambda i: -escores[i])
    top1 = float(escores[ordem[0]]) if ordem else 0.0
    return [chunks[i].id for i in ordem], top1


def avaliar(
    contratos: list[Contrato], max_chars: int = MAX_CHARS, overlap: int = OVERLAP
) -> dict[str, Any]:
    """Roda a avaliação completa e devolve o dicionário do relatório."""
    recalls: dict[int, list[float]] = {k: [] for k in KS}
    reciprocos: list[float] = []
    escores_respondivel: list[float] = []
    escores_impossivel: list[float] = []
    por_categoria: dict[str, list[float]] = {}
    n_respondiveis = n_impossiveis = 0
    sem_gold = 0
    total_chunks = 0

    for contrato in contratos:
        chunks = chunkar(contrato.texto, contrato.titulo, max_chars, overlap)
        total_chunks += len(chunks)
        if not chunks:
            continue
        bm25 = _indice_bm25(chunks)  # uma vez por contrato, reusado nas 41 perguntas
        for pergunta in contrato.perguntas:
            query = montar_query(pergunta)
            if pergunta.impossivel:
                n_impossiveis += 1
                _, top1 = _ranquear(bm25, chunks, query)
                escores_impossivel.append(top1)
                continue

            gold = gold_da_pergunta(pergunta, chunks)
            if not gold:
                # Span existe mas nenhum chunk o intersecta — só possível se o
                # offset estiver fora do texto. Contado, não escondido.
                sem_gold += 1
                continue
            n_respondiveis += 1
            ranking, top1 = _ranquear(bm25, chunks, query)
            escores_respondivel.append(top1)

            for k in KS:
                topk = set(ranking[:k])
                recalls[k].append(len(gold & topk) / len(gold))
            posicao = next((i + 1 for i, cid in enumerate(ranking) if cid in gold), None)
            reciprocos.append(1.0 / posicao if posicao else 0.0)
            por_categoria.setdefault(pergunta.categoria, []).append(
                len(gold & set(ranking[:5])) / len(gold)
            )

    def _media(v: list[float]) -> float:
        return round(sum(v) / len(v), 4) if v else 0.0

    # Recall@k é média de frações (não proporção binária), então o IC vem de
    # bootstrap. Wilson entra na taxa de acerto-em-algum-lugar do top-k, que É
    # binária — as duas medem coisas diferentes e ambas são reportadas.
    metricas = {}
    for k in KS:
        acertou_algum = [1 if r > 0 else 0 for r in recalls[k]]
        metricas[f"recall_at_{k}"] = {
            "media": _media(recalls[k]),
            "ic95_bootstrap": bootstrap_ic(recalls[k]),
        }
        metricas[f"hit_at_{k}"] = {
            "taxa": _media([float(x) for x in acertou_algum]),
            "ic95_wilson": wilson(sum(acertou_algum), len(acertou_algum)),
        }

    return {
        "config": {"max_chars": max_chars, "overlap": overlap, "ks": list(KS)},
        "corpus": {
            "n_contratos": len(contratos),
            "n_chunks": total_chunks,
            "chunks_por_contrato_mediana": round(total_chunks / len(contratos), 1)
            if contratos
            else 0.0,
        },
        "n_respondiveis": n_respondiveis,
        "n_impossiveis": n_impossiveis,
        "n_sem_gold": sem_gold,
        "metricas": metricas,
        "mrr": {"media": _media(reciprocos), "ic95_bootstrap": bootstrap_ic(reciprocos)},
        # Diagnóstico de abstenção: se estas duas distribuições se sobrepõem,
        # nenhum limiar separa "tem cláusula" de "não tem" — melhor saber antes
        # de construir a política.
        "diagnostico_abstencao": {
            "escore_top1_respondivel": {
                "mediana": round(percentil(escores_respondivel, 0.5), 3),
                "p10": round(percentil(escores_respondivel, 0.10), 3),
                "p90": round(percentil(escores_respondivel, 0.90), 3),
            },
            "escore_top1_impossivel": {
                "mediana": round(percentil(escores_impossivel, 0.5), 3),
                "p10": round(percentil(escores_impossivel, 0.10), 3),
                "p90": round(percentil(escores_impossivel, 0.90), 3),
            },
        },
        "recall_at_5_por_categoria": {
            cat: round(sum(v) / len(v), 4)
            for cat, v in sorted(por_categoria.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia recuperação no CUAD (BM25, sem LLM).")
    parser.add_argument("--limite", type=int, default=None, help="usa só os N primeiros contratos")
    parser.add_argument("--zip", type=Path, default=None, help="caminho do cuad.zip")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    args = parser.parse_args()

    contratos = carregar(zip_path=args.zip)
    if args.limite:
        contratos = contratos[: args.limite]
    relatorio = avaliar(contratos, max_chars=args.max_chars, overlap=args.overlap)

    destino = settings.data_processed.parent.parent / "reports" / "fase6_cuad"
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / "retrieval_bm25.json"
    caminho.write_text(json.dumps(carimbar(relatorio), ensure_ascii=False, indent=2))

    m = relatorio["metricas"]
    corpus = relatorio["corpus"]
    print(f"contratos: {corpus['n_contratos']} | chunks: {corpus['n_chunks']:,}")
    print(
        f"respondíveis: {relatorio['n_respondiveis']:,} | "
        f"impossíveis: {relatorio['n_impossiveis']:,}"
    )
    if relatorio["n_sem_gold"]:
        print(f"  ATENÇÃO: {relatorio['n_sem_gold']} com span mas sem chunk gold")
    for k in KS:
        r = m[f"recall_at_{k}"]
        h = m[f"hit_at_{k}"]
        print(
            f"  recall@{k}: {r['media']:.3f} {r['ic95_bootstrap']}   "
            f"hit@{k}: {h['taxa']:.3f} {h['ic95_wilson']}"
        )
    print(f"  MRR: {relatorio['mrr']['media']:.3f} {relatorio['mrr']['ic95_bootstrap']}")
    d = relatorio["diagnostico_abstencao"]
    print(f"escore top-1 — respondível mediana {d['escore_top1_respondivel']['mediana']} | "
          f"impossível mediana {d['escore_top1_impossivel']['mediana']}")
    print(f"report: {caminho}")


if __name__ == "__main__":
    main()
