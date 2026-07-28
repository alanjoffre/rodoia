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
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
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


# Fronteiras de cláusula em contrato comercial americano. Medido nos 510 do CUAD:
# numeração `1.` / `2.3` domina (2.149 ocorrências), seguida de `(a)`/`(i)`, títulos em
# CAIXA ALTA, `SECTION` e `ARTICLE`. Só ~63% dos contratos têm 3+ fronteiras — por isso
# `chunkar_clausula` DEGRADA para janela quando não encontra estrutura (ver docstring).
_RE_FRONTEIRA = re.compile(
    r"(?m)^[ \t]{0,4}(?:"
    r"\d{1,2}\.\d{0,2}[ \t]+(?=[A-Z])"          # 1.  /  2.3  seguido de maiúscula
    r"|\((?:[a-z]{1,3}|[ivxlc]{1,4}|\d{1,2})\)[ \t]+(?=[A-Z])"  # (a) (iii) (12)
    r"|(?:SECTION|Section)[ \t]+\d+"
    r"|(?:ARTICLE|Article)[ \t]+[IVXLC0-9]+"
    r"|[A-Z][A-Z \-']{6,60}[ \t]*$"             # TÍTULO EM CAIXA ALTA
    r")"
)

# Abaixo disto, o texto não tem estrutura suficiente e a divisão por cláusula produziria
# chunks arbitrários — pior que a janela honesta. Medido sobre os 510 contratos
# (reports/fase6_cuad/fronteiras_clausula.json): 410 (80,4%) têm 3+ fronteiras, mediana 13;
# os outros 100 caem no fallback de janela.
_MIN_FRONTEIRAS = 3


def _fronteiras(texto: str) -> list[int]:
    """Posições de início de cláusula, sempre incluindo 0 (preâmbulo)."""
    pos = [m.start() for m in _RE_FRONTEIRA.finditer(texto)]
    return [0, *[p for p in pos if p > 0]]


def chunkar_clausula(
    texto: str, contrato: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP
) -> list[Chunk]:
    """Chunking consciente da estrutura de cláusula, **preservando offsets**.

    Mesma filosofia do `rag/chunking.py` da Fase 1 (que divide por `Art. Nº` na
    regulação brasileira): dividir na unidade semântica natural em vez de cortar
    no meio de uma regra. Aqui a unidade é a **cláusula contratual**.

    **Degrada para janela quando não há estrutura.** Medido: 410 dos 510 contratos
    (80,4%) têm 3+ fronteiras detectáveis. Forçar divisão por cláusula nos outros
    100 produziria chunks arbitrários — pior que a janela honesta. O fallback é
    explícito e medido, não escondido.

    Testa a hipótese (a) do docs/17 §13.1: *o gargalo do denso é o chunking, não o
    embedder?*
    """
    if not texto:
        return []
    marcos = _fronteiras(texto)
    if len(marcos) < _MIN_FRONTEIRAS:
        return chunkar(texto, contrato, max_chars, overlap)

    # Segmentos = intervalos entre fronteiras consecutivas.
    limites = [*marcos, len(texto)]
    segmentos = [(limites[i], limites[i + 1]) for i in range(len(limites) - 1)]

    chunks: list[Chunk] = []
    ini_atual: int | None = None
    fim_atual = 0
    for ini, fim in segmentos:
        if fim - ini > max_chars:
            # Cláusula grande demais: fecha o acumulado e fatia por janela, mantendo
            # os offsets absolutos (é o que o mapeamento span->chunk consome).
            if ini_atual is not None:
                chunks.append(
                    Chunk(contrato, len(chunks), ini_atual, fim_atual, texto[ini_atual:fim_atual])
                )
                ini_atual = None
            passo = max(1, max_chars - overlap)
            for off in range(ini, fim, passo):
                f = min(off + max_chars, fim)
                chunks.append(Chunk(contrato, len(chunks), off, f, texto[off:f]))
                if f == fim:
                    break
            continue
        if ini_atual is None:
            ini_atual, fim_atual = ini, fim
        elif fim - ini_atual <= max_chars:
            fim_atual = fim
        else:
            chunks.append(
                Chunk(contrato, len(chunks), ini_atual, fim_atual, texto[ini_atual:fim_atual])
            )
            ini_atual, fim_atual = ini, fim
    if ini_atual is not None:
        chunks.append(
            Chunk(contrato, len(chunks), ini_atual, fim_atual, texto[ini_atual:fim_atual])
        )
    return chunks


CHUNKERS = {"janela": chunkar, "clausula": chunkar_clausula}


def obter_chunker(nome: str) -> Callable[..., list[Chunk]]:
    """Seletor por nome — evita `if` espalhado nos 4 módulos de avaliação."""
    if nome not in CHUNKERS:
        raise ValueError(f"chunker inválido: {nome!r} (use {sorted(CHUNKERS)})")
    return CHUNKERS[nome]


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


@dataclass(frozen=True)
class Avaliada:
    """O resultado de UMA pergunta, no formato que a consolidação de métricas
    consome — independente do recuperador (BM25 ou denso) que a produziu.

    Este é o contrato que impede a comparação de medir a implementação em vez
    dos recuperadores: as métricas, os ICs e o corte por categoria são
    calculados por `consolidar` a partir DESTES registros, iguais para os dois.
    """

    categoria: str
    impossivel: bool
    top1: float
    tem_gold: bool
    recall_por_k: dict[int, float]  # vazio p/ impossível e sem-gold
    rr: float  # reciprocal rank; 0 quando não há gold no ranking
    # Tamanho do conjunto gold. Não é decoração: `recall@k = |gold ∩ top-k| / |gold|`
    # tem TETO `min(k, |gold|) / |gold|`, e |gold| depende do CHUNKER. Comparar
    # recall@1 entre dois chunkers sem olhar o teto compara réguas diferentes —
    # foi o que quase aconteceu com o chunking por cláusula (docs/17 §13.6).
    n_gold: int = 0


def _media(v: list[float]) -> float:
    return round(sum(v) / len(v), 4) if v else 0.0


def consolidar(
    avaliadas: list[Avaliada], corpus: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Agrega os registros por-pergunta no relatório final — MESMA máquina para
    BM25 e denso, para a comparação medir os recuperadores e não o código."""
    respondiveis = [a for a in avaliadas if not a.impossivel and a.tem_gold]
    recalls = {k: [a.recall_por_k[k] for a in respondiveis] for k in KS}
    reciprocos = [a.rr for a in respondiveis]
    escores_resp = [a.top1 for a in respondiveis]
    escores_imp = [a.top1 for a in avaliadas if a.impossivel]
    por_categoria: dict[str, list[float]] = {}
    for a in respondiveis:
        por_categoria.setdefault(a.categoria, []).append(a.recall_por_k[5])

    # Recall@k é média de frações (não proporção binária), então o IC vem de
    # bootstrap. Wilson entra na taxa de acerto-em-algum-lugar do top-k, que É
    # binária — as duas medem coisas diferentes e ambas são reportadas.
    # Teto do recall@k dado o gold DESTE chunking. Com |gold| > k o recall@k não
    # pode chegar a 1 — e como |gold| depende do chunker, o teto se move junto.
    # Reportá-lo ao lado do valor é o que torna a comparação entre chunkers
    # legítima; sem ele, uma mudança de régua se lê como ganho de qualidade.
    n_golds = [a.n_gold for a in respondiveis if a.n_gold > 0]
    metricas = {}
    for k in KS:
        acertou_algum = [1 if r > 0 else 0 for r in recalls[k]]
        teto = _media([min(k, g) / g for g in n_golds]) if n_golds else 0.0
        # Normalizado POR PERGUNTA: |gold ∩ top-k| / min(k, |gold|) — "do gold que
        # cabia em k posições, quanto foi recuperado". Cada pergunta contribui numa
        # escala 0–1 independente do seu |gold|, então a média (e o bootstrap) são
        # comparáveis entre chunkings. Normalizar só a média (razão de médias) não
        # daria IC.
        normalizados = [
            (a.recall_por_k[k] * a.n_gold) / min(k, a.n_gold)
            for a in respondiveis
            if a.n_gold > 0
        ]
        metricas[f"recall_at_{k}"] = {
            "media": _media(recalls[k]),
            "ic95_bootstrap": bootstrap_ic(recalls[k]),
            "teto": teto,
        }
        metricas[f"recall_norm_at_{k}"] = {
            "media": _media(normalizados),
            "ic95_bootstrap": bootstrap_ic(normalizados),
            "descricao": "|gold ∩ top-k| / min(k,|gold|) — comparável entre chunkings",
        }
        metricas[f"hit_at_{k}"] = {
            "taxa": _media([float(x) for x in acertou_algum]),
            "ic95_wilson": wilson(sum(acertou_algum), len(acertou_algum)),
        }

    return {
        "config": config,
        "corpus": corpus,
        "n_respondiveis": len(respondiveis),
        "n_impossiveis": sum(1 for a in avaliadas if a.impossivel),
        "n_sem_gold": sum(1 for a in avaliadas if not a.impossivel and not a.tem_gold),
        # |gold| médio: a régua do recall. Muda com o chunker (janela com overlap
        # duplica spans de fronteira; cláusula sem overlap não), então precisa
        # aparecer em todo relatório para as comparações serem auditáveis.
        "gold_medio_por_pergunta": round(sum(n_golds) / len(n_golds), 4) if n_golds else 0.0,
        "metricas": metricas,
        "mrr": {"media": _media(reciprocos), "ic95_bootstrap": bootstrap_ic(reciprocos)},
        # Diagnóstico de abstenção: se estas duas distribuições se sobrepõem,
        # nenhum limiar separa "tem cláusula" de "não tem" — melhor saber antes
        # de construir a política.
        "diagnostico_abstencao": {
            "escore_top1_respondivel": {
                "mediana": round(percentil(escores_resp, 0.5), 3),
                "p10": round(percentil(escores_resp, 0.10), 3),
                "p90": round(percentil(escores_resp, 0.90), 3),
            },
            "escore_top1_impossivel": {
                "mediana": round(percentil(escores_imp, 0.5), 3),
                "p10": round(percentil(escores_imp, 0.10), 3),
                "p90": round(percentil(escores_imp, 0.90), 3),
            },
        },
        "recall_at_5_por_categoria": {
            cat: round(sum(v) / len(v), 4)
            for cat, v in sorted(por_categoria.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
        },
    }


def _registrar(gold: set[str], ranking: list[str], top1: float, categoria: str) -> Avaliada:
    """`_metricas_por_pergunta` + a categoria, num passo só.

    Existe porque a versão anterior deixava cada um dos quatro módulos de avaliação
    reconstruir o `Avaliada` campo a campo — e, quando `n_gold` foi acrescentado,
    os quatro o descartaram em silêncio: os relatórios saíram com teto 0,0 e recall
    normalizado 0,0, sem erro nenhum. `replace` copia o que existir, então um campo
    novo passa a chegar sozinho.
    """
    return replace(_metricas_por_pergunta(gold, ranking, top1), categoria=categoria)


def _metricas_por_pergunta(gold: set[str], ranking: list[str], top1: float) -> Avaliada:
    """Constrói o registro de uma pergunta respondível a partir do ranking —
    partilhado por qualquer recuperador (recebe o ranking já pronto)."""
    recall_por_k = {k: len(gold & set(ranking[:k])) / len(gold) for k in KS}
    posicao = next((i + 1 for i, cid in enumerate(ranking) if cid in gold), None)
    return Avaliada(
        categoria="",  # preenchido pelo chamador
        impossivel=False,
        top1=top1,
        tem_gold=True,
        recall_por_k=recall_por_k,
        rr=1.0 / posicao if posicao else 0.0,
        n_gold=len(gold),
    )


def _corpus_info(contratos: list[Contrato], total_chunks: int) -> dict[str, Any]:
    return {
        "n_contratos": len(contratos),
        "n_chunks": total_chunks,
        "chunks_por_contrato_mediana": round(total_chunks / len(contratos), 1)
        if contratos
        else 0.0,
    }


def avaliar(
    contratos: list[Contrato],
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
    chunker: str = "janela",
) -> dict[str, Any]:
    """Avaliação BM25 completa: produz os registros e consolida."""
    fatiar = obter_chunker(chunker)
    avaliadas: list[Avaliada] = []
    total_chunks = 0
    for contrato in contratos:
        chunks = fatiar(contrato.texto, contrato.titulo, max_chars, overlap)
        total_chunks += len(chunks)
        if not chunks:
            continue
        bm25 = _indice_bm25(chunks)  # uma vez por contrato, reusado nas 41 perguntas
        for pergunta in contrato.perguntas:
            ranking, top1 = _ranquear(bm25, chunks, montar_query(pergunta))
            if pergunta.impossivel:
                avaliadas.append(
                    Avaliada(pergunta.categoria, True, top1, False, {}, 0.0)
                )
                continue
            gold = gold_da_pergunta(pergunta, chunks)
            if not gold:
                # Span existe mas nenhum chunk o intersecta — só possível se o
                # offset estiver fora do texto. Registrado (tem_gold=False), não escondido.
                avaliadas.append(Avaliada(pergunta.categoria, False, top1, False, {}, 0.0))
                continue
            avaliadas.append(_registrar(gold, ranking, top1, pergunta.categoria))

    config = {
        "recuperador": "bm25",
        "chunker": chunker,
        "max_chars": max_chars,
        "overlap": overlap,
        "ks": list(KS),
    }
    return consolidar(avaliadas, _corpus_info(contratos, total_chunks), config)


def diagnostico_chunker(contratos: list[Contrato]) -> dict[str, Any]:
    """Quantos contratos o detector de cláusula realmente cobre — e quantos caem no
    fallback de janela.

    A escolha de `_MIN_FRONTEIRAS` só é defensável se a cobertura for medida. Sem
    isto, "degrada para janela quando não há estrutura" é uma afirmação sobre um
    número que ninguém conferiu — e o fallback poderia estar engolindo a maioria
    do corpus sem aparecer em métrica nenhuma.
    """
    n_marcos = [len(_fronteiras(c.texto)) for c in contratos]
    com_estrutura = sum(1 for n in n_marcos if n >= _MIN_FRONTEIRAS)
    faixas = Counter(
        "0-2" if n < 3 else "3-9" if n < 10 else "10-49" if n < 50 else "50+" for n in n_marcos
    )
    return {
        "n_contratos": len(contratos),
        "min_fronteiras_exigidas": _MIN_FRONTEIRAS,
        "com_estrutura": com_estrutura,
        "fracao_com_estrutura": round(com_estrutura / len(contratos), 4) if contratos else 0.0,
        "fallback_para_janela": len(contratos) - com_estrutura,
        "fronteiras_por_contrato": {
            "mediana": sorted(n_marcos)[len(n_marcos) // 2] if n_marcos else 0,
            "min": min(n_marcos, default=0),
            "max": max(n_marcos, default=0),
        },
        "distribuicao": dict(sorted(faixas.items())),
        "n_chunks_janela": sum(len(chunkar(c.texto, c.titulo)) for c in contratos),
        "n_chunks_clausula": sum(len(chunkar_clausula(c.texto, c.titulo)) for c in contratos),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia recuperação no CUAD (BM25, sem LLM).")
    parser.add_argument(
        "--diagnostico-chunker", action="store_true",
        help="mede a cobertura do detector de cláusula e sai (não avalia recuperação)",
    )
    parser.add_argument("--limite", type=int, default=None, help="usa só os N primeiros contratos")
    parser.add_argument("--zip", type=Path, default=None, help="caminho do cuad.zip")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument(
        "--chunker", type=str, default="janela", choices=tuple(CHUNKERS),
        help="janela cega vs consciente de cláusula (hipótese (a) do docs/17 §13.1)",
    )
    args = parser.parse_args()

    contratos = carregar(zip_path=args.zip)
    if args.limite:
        contratos = contratos[: args.limite]

    destino_base = settings.data_processed.parent.parent / "reports" / "fase6_cuad"
    if args.diagnostico_chunker:
        diag = diagnostico_chunker(contratos)
        destino_base.mkdir(parents=True, exist_ok=True)
        saida = destino_base / "fronteiras_clausula.json"
        saida.write_text(json.dumps(carimbar(diag), ensure_ascii=False, indent=2))
        print(
            f"com estrutura: {diag['com_estrutura']}/{diag['n_contratos']} "
            f"({diag['fracao_com_estrutura']:.1%}) | fallback: {diag['fallback_para_janela']}"
        )
        print(
            f"chunks: janela {diag['n_chunks_janela']:,} → "
            f"cláusula {diag['n_chunks_clausula']:,}"
        )
        print(f"report: {saida}")
        return

    relatorio = avaliar(
        contratos, max_chars=args.max_chars, overlap=args.overlap, chunker=args.chunker
    )

    destino = destino_base
    destino.mkdir(parents=True, exist_ok=True)
    sufixo = "" if args.chunker == "janela" else f"_{args.chunker}"
    caminho = destino / f"retrieval_bm25{sufixo}.json"
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
