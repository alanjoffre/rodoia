"""Geração ancorada no CUAD — **a métrica de alucinação** (Fase 6, LLM local).

Todas as métricas anteriores da Fase 6 medem RECUPERAÇÃO. Esta mede o que o
gerador faz com o contexto recuperado — e o CUAD viabiliza a pergunta que quase
nenhum portfólio responde:

> Quando a cláusula **não existe** no contrato, o sistema diz "não consta" — ou
> inventa uma resposta plausível?

**67,9% das perguntas do CUAD são `is_impossible`**, com gold de advogados. Isso
dá as duas taxas que juntas descrevem a política de abstenção:

| população | comportamento correto | o que a taxa mede |
|---|---|---|
| `is_impossible` (14.208) | **abster** | **não alucinar** |
| respondível (6.702) | **responder** | não recusar indevidamente |

Reportar só a primeira seria maquiagem: um modelo que responde "não consta" a
TUDO teria 100% de não-alucinação e seria inútil. As duas andam juntas, com IC de
Wilson, e a média balanceada é o número honesto.

**Custo de API: zero.** Roda no LLM local (Ollama) que a Fase 1 já usa via
`rag/llm.py` — mesma interface, sem tocar no pipeline.

**Amostragem estratificada por categoria**, com seed fixa: as 41 categorias têm
dificuldade muito diferente (recall@5 de 0,17 a 0,92, docs/17 §9.2), então
amostra simples poderia cair num bolsão fácil e inflar o resultado.

**O prompt é em inglês** — os contratos e as perguntas são. Instruir em português
sobre documento inglês introduziria uma variável que não é a que se quer medir.

Uso (Ollama local, GPU):
    python -m rodoia.rag.avaliacao_cuad_geracao --amostra 150
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rodoia.config import settings
from rodoia.estat import mcnemar, wilson
from rodoia.proveniencia import carimbar
from rodoia.rag.avaliacao_cuad import (
    MAX_CHARS,
    OVERLAP,
    Chunk,
    _indice_bm25,
    _ranquear,
    chunkar,
    montar_query,
)
from rodoia.rag.avaliacao_cuad_denso import _ranquear_denso
from rodoia.rag.avaliacao_cuad_hibrido import fundir
from rodoia.rag.cuad import Contrato, Pergunta, carregar
from rodoia.rag.embeddings import Embedder
from rodoia.rag.llm import LLM

TOP_K = 5

# Marcador de abstenção distintivo: improvável de aparecer por acaso numa resposta
# real, o que torna a detecção robusta sem heurística frágil de linguagem natural.
MARCADOR = "NOT_FOUND"

# DUAS variantes de prompt, porque a conclusão depende disso. Se só o `estrito`
# fosse medido, "o modelo abstém demais" seria indistinguível de "o prompt do
# autor induziu abstenção" — um confundidor introduzido por quem mede. A ablação
# separa os dois (docs/17 §13.4).
PROMPTS: dict[str, dict[str, str]] = {
    # Conservador: proíbe explicitamente adivinhar. É o prompt que um sistema
    # jurídico usaria — o custo de inventar uma cláusula é alto.
    "estrito": {
        "sistema": (
            "You are a contract analyst. Answer ONLY from the provided excerpts. "
            f"If the excerpts do not contain the answer, reply with exactly {MARCADOR} "
            "and nothing else. Never guess, never use outside knowledge."
        ),
        "template": (
            "Contract excerpts:\n---\n{contexto}\n---\n\n"
            "Question: {pergunta}\n\n"
            f"Answer from the excerpts above, or reply exactly {MARCADOR} if they do "
            "not contain the answer."
        ),
    },
    # Equilibrado: mantém a opção de abster, mas instrui a RESPONDER quando a
    # informação estiver presente, mesmo que parcial. Sem a proibição enfática.
    "equilibrado": {
        "sistema": (
            "You are a contract analyst. Use the provided excerpts to answer. "
            "If the excerpts contain the answer — even partially — give it. "
            f"Only if the information is genuinely absent, reply {MARCADOR}."
        ),
        "template": (
            "Contract excerpts:\n---\n{contexto}\n---\n\n"
            "Question: {pergunta}\n\n"
            "Answer using the excerpts above. If the information is genuinely "
            f"absent from them, reply {MARCADOR}."
        ),
    },
}

# Detecta o marcador OU as recusas em linguagem natural que o modelo produz mesmo
# instruído (medido: modelos pequenos parafraseiam a instrução).
_RE_ABSTEVE = re.compile(
    rf"\b{MARCADOR}\b|\bnot (?:found|present|mentioned|specified|stated|contain)"
    r"|\bno (?:mention|reference|information)\b|\bdoes not (?:contain|specify|mention)\b",
    re.I,
)


def absteve(resposta: str) -> bool:
    """True se a resposta é uma recusa (marcador explícito ou paráfrase)."""
    return bool(_RE_ABSTEVE.search(resposta))


@dataclass(frozen=True)
class Julgada:
    categoria: str
    impossivel: bool
    absteve: bool
    correta: bool  # absteve quando devia OU respondeu quando devia
    # Identidade estável da pergunta DENTRO da amostra. O desenho da ablação é
    # pareado (mesma seed, mesmas perguntas), e o McNemar só vale se o pareamento
    # for por identidade — casar por posição na lista quebraria em silêncio se um
    # contrato fosse pulado numa das rodadas e não na outra.
    chave: str = ""


def amostrar(
    contratos: list[Contrato], n_por_populacao: int, seed: int
) -> list[tuple[Contrato, Pergunta]]:
    """Amostra estratificada por categoria, equilibrada entre as duas populações.

    Sem estratificar, um sorteio simples pode cair num bolsão de categorias fáceis
    e inflar o resultado — as 41 categorias variam 5,3× em recall (docs/17 §9.2).
    """
    rng = random.Random(seed)
    por_cat: dict[tuple[str, bool], list[tuple[Contrato, Pergunta]]] = {}
    for c in contratos:
        for p in c.perguntas:
            por_cat.setdefault((p.categoria, p.impossivel), []).append((c, p))

    escolhidas: list[tuple[Contrato, Pergunta]] = []
    for impossivel in (False, True):
        cats = sorted({k[0] for k in por_cat if k[1] == impossivel})
        if not cats:
            continue
        # Round-robin: uma por categoria a cada volta, até completar n ou esgotar.
        # Cota fixa (n // n_categorias) truncava — com 41 categorias e n=150 dava
        # 3 por categoria = 123, e `--amostra 150` devolvia 123 sem avisar.
        baldes = []
        for cat in cats:
            itens = list(por_cat.get((cat, impossivel), []))
            rng.shuffle(itens)
            baldes.append(itens)
        pool: list[tuple[Contrato, Pergunta]] = []
        volta = 0
        while len(pool) < n_por_populacao and any(len(b) > volta for b in baldes):
            for b in baldes:
                if len(b) > volta:
                    pool.append(b[volta])
                    if len(pool) >= n_por_populacao:
                        break
            volta += 1
        escolhidas.extend(pool[:n_por_populacao])
    return escolhidas


def montar_contexto(chunks: list[Chunk], ranking: list[str], k: int) -> str:
    por_id = {c.id: c for c in chunks}
    trechos = [por_id[cid].texto for cid in ranking[:k] if cid in por_id]
    return "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(trechos))


def avaliar_geracao(
    contratos: list[Contrato],
    embedder: Embedder,
    llm: LLM,
    n_por_populacao: int = 150,
    seed: int = 42,
    top_k: int = TOP_K,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
    prompt: str = "estrito",
) -> dict[str, Any]:
    """Gera respostas ancoradas para a amostra e mede a política de abstenção."""
    if prompt not in PROMPTS:
        raise ValueError(f"prompt inválido: {prompt!r} (use {sorted(PROMPTS)})")
    sistema = PROMPTS[prompt]["sistema"]
    template = PROMPTS[prompt]["template"]
    amostra = amostrar(contratos, n_por_populacao, seed)
    titulos = {c.titulo for c, _ in amostra}
    usados = [c for c in contratos if c.titulo in titulos]

    # Só os contratos amostrados são chunked/encodados — não faz sentido pagar o
    # encode dos 510 para gerar sobre uma fração.
    chunks_por: dict[str, list[Chunk]] = {
        c.titulo: chunkar(c.texto, c.titulo, max_chars, overlap) for c in usados
    }
    todos = [ch for cs in chunks_por.values() for ch in cs]
    vecs_all = (
        embedder.encode_passages([c.texto for c in todos])
        if todos
        else np.zeros((0, embedder.dim))
    )
    offs: dict[str, tuple[int, int]] = {}
    pos = 0
    for titulo, cs in chunks_por.items():
        offs[titulo] = (pos, pos + len(cs))
        pos += len(cs)

    julgadas: list[Julgada] = []
    for i, (contrato, pergunta) in enumerate(amostra):
        cs = chunks_por[contrato.titulo]
        if not cs:
            continue
        ini, fim = offs[contrato.titulo]
        query = montar_query(pergunta)
        bm25_rk, _ = _ranquear(_indice_bm25(cs), cs, query)
        qv = embedder.encode_queries([query])[0]
        denso_rk, _ = _ranquear_denso(vecs_all[ini:fim], cs, qv)
        ranking, _ = fundir(bm25_rk, denso_rk)

        texto_prompt = template.format(
            contexto=montar_contexto(cs, ranking, top_k), pergunta=query
        )
        resposta = llm.gerar(texto_prompt, sistema=sistema)
        a = absteve(resposta)
        julgadas.append(
            Julgada(
                categoria=pergunta.categoria,
                impossivel=pergunta.impossivel,
                absteve=a,
                # correto = absteve na impossível, respondeu na respondível
                correta=(a if pergunta.impossivel else not a),
                chave=f"{i:05d}|{contrato.titulo}|{pergunta.categoria}",
            )
        )

    return consolidar_geracao(
        julgadas,
        {
            "top_k": top_k, "seed": seed, "modelo": llm_nome(llm), "prompt": prompt,
            "seed_llm": getattr(llm, "seed", None),
        },
    )


def llm_nome(llm: LLM) -> str:
    return str(getattr(llm, "modelo", "desconhecido"))


def consolidar_geracao(julgadas: list[Julgada], config: dict[str, Any]) -> dict[str, Any]:
    """Agrega as duas taxas + a média balanceada. Função pura (testável sem LLM)."""
    imp = [j for j in julgadas if j.impossivel]
    res = [j for j in julgadas if not j.impossivel]
    n_abst_imp = sum(1 for j in imp if j.absteve)
    n_resp_res = sum(1 for j in res if not j.absteve)

    taxa_imp = n_abst_imp / len(imp) if imp else 0.0
    taxa_res = n_resp_res / len(res) if res else 0.0
    return {
        "config": config,
        "n_impossiveis": len(imp),
        "n_respondiveis": len(res),
        # A métrica-manchete: quando não há resposta no contrato, o sistema se cala?
        "nao_alucinacao": {
            "taxa": round(taxa_imp, 4),
            "ic95_wilson": wilson(n_abst_imp, len(imp)),
            "descricao": "abstenção correta nas perguntas is_impossible",
        },
        # O contrapeso obrigatório: um modelo que abstém de tudo zeraria isto.
        "cobertura": {
            "taxa": round(taxa_res, 4),
            "ic95_wilson": wilson(n_resp_res, len(res)),
            "descricao": "resposta dada nas perguntas com gold (não recusou indevidamente)",
        },
        "media_balanceada": round((taxa_imp + taxa_res) / 2, 4),
        "abstencao_por_categoria": {
            cat: round(
                sum(1 for j in imp if j.categoria == cat and j.absteve)
                / max(1, sum(1 for j in imp if j.categoria == cat)),
                4,
            )
            for cat in sorted({j.categoria for j in imp})
        },
        # Resultado POR PERGUNTA — o insumo do McNemar. Sem isto, comparar duas
        # variantes só é possível pelos ICs, que ignoram o pareamento e por isso
        # são o teste conservador (docs/17 §13.4).
        "julgamentos": [
            {"chave": j.chave, "impossivel": j.impossivel, "correta": j.correta}
            for j in julgadas
        ],
    }


def comparar_pareado(rel_a: dict[str, Any], rel_b: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
    """McNemar entre duas rodadas sobre as MESMAS perguntas (ablação de prompt/modelo).

    Casa por `chave` e testa três recortes, porque as duas taxas podem se mover em
    direções opostas: um prompt que abstém mais ganha em não-alucinação e perde em
    cobertura. Um McNemar só do agregado esconderia justamente esse trade-off.
    """
    ja = {j["chave"]: j for j in rel_a.get("julgamentos", [])}
    jb = {j["chave"]: j for j in rel_b.get("julgamentos", [])}
    comuns = sorted(set(ja) & set(jb))
    if not comuns:
        return {"erro": "sem perguntas em comum — os relatórios não são pareáveis"}

    def _recorte(filtro: Any) -> dict[str, float | int | str]:
        chaves = [c for c in comuns if filtro(ja[c])]
        return mcnemar([ja[c]["correta"] for c in chaves], [jb[c]["correta"] for c in chaves])

    return {
        "a": rel_a["config"], "b": rel_b["config"],
        "n_pareadas": len(comuns),
        "n_descartadas_a": len(ja) - len(comuns),
        "n_descartadas_b": len(jb) - len(comuns),
        "geral": _recorte(lambda j: True),
        "nao_alucinacao": _recorte(lambda j: j["impossivel"]),
        "cobertura": _recorte(lambda j: not j["impossivel"]),
        "leitura": (
            "b01 = só B acerta, b10 = só A acerta. p < 0,05 rejeita 'as duas variantes "
            "erram igual'. `metodo` diz qual teste produziu o p: `binomial_exato` "
            "abaixo de 25 discordantes (onde a aproximação χ² não vale), `qui2_yates` "
            "acima."
        ),
    }


def _sanitizar(modelo: str) -> str:
    """`gemma2:9b` → `gemma2-9b` — nome de modelo vira sufixo de arquivo."""
    return re.sub(r"[^a-z0-9]+", "-", modelo.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Geração ancorada no CUAD (LLM local, sem API).")
    parser.add_argument(
        "--comparar", type=Path, nargs=2, default=None, metavar=("A", "B"),
        help="McNemar pareado entre dois relatórios já gerados (não roda o LLM)",
    )
    parser.add_argument("--amostra", type=int, default=150, help="perguntas por população")
    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--familia", type=str, default="e5", choices=("e5", "bge"))
    parser.add_argument("--modelo-llm", type=str, default=None, help="modelo do Ollama")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt", type=str, default="estrito", choices=tuple(PROMPTS),
        help="variante do prompt (a ablação que separa 'modelo abstém' de 'prompt induziu')",
    )
    args = parser.parse_args()

    destino = settings.data_processed.parent.parent / "reports" / "fase6_cuad"
    if args.comparar:
        a, b = (json.loads(p.read_text(encoding="utf-8")) for p in args.comparar)
        cmp = comparar_pareado(a, b)
        destino.mkdir(parents=True, exist_ok=True)
        saida = destino / "mcnemar_geracao.json"
        saida.write_text(json.dumps(carimbar(cmp), ensure_ascii=False, indent=2))
        print(f"pareadas: {cmp.get('n_pareadas')}")
        for recorte in ("geral", "nao_alucinacao", "cobertura"):
            r = cmp.get(recorte, {})
            print(
                f"  {recorte:16s} b10={r.get('b10')} b01={r.get('b01')} "
                f"n_disc={r.get('n_discordantes')} p={r.get('p_valor')} "
                f"({r.get('metodo')})"
            )
        print(f"report: {saida}")
        return

    from rodoia.rag.embeddings import MODELO_PADRAO, construir_embedder
    from rodoia.rag.llm import OllamaLLM

    modelo_emb = MODELO_PADRAO["bge"] if args.familia == "bge" else settings.embedding_model
    embedder = construir_embedder(args.familia, modelo=modelo_emb, device=args.device)
    # Seed fixa no LLM: sem ela a ablação de prompt compararia duas rodadas que
    # diferem por amostragem, e o McNemar contaria ruído como discordância.
    llm = OllamaLLM(modelo=args.modelo_llm, seed=args.seed)

    contratos = carregar(zip_path=args.zip)
    rel = avaliar_geracao(
        contratos, embedder, llm,
        n_por_populacao=args.amostra, seed=args.seed, prompt=args.prompt,
    )

    destino.mkdir(parents=True, exist_ok=True)
    # Sufixo por variante: a ablação precisa dos DOIS relatórios lado a lado.
    sufixo = "" if args.prompt == "estrito" else f"_{args.prompt}"
    # ...e por modelo, senão um gerador maior sobrescreveria o baseline e a
    # comparação que motivou a rodada desapareceria.
    if args.modelo_llm:
        sufixo += f"_{_sanitizar(args.modelo_llm)}"
    caminho = destino / f"geracao_ancorada{sufixo}.json"
    caminho.write_text(json.dumps(carimbar(rel), ensure_ascii=False, indent=2))

    na, cob = rel["nao_alucinacao"], rel["cobertura"]
    print(
        f"modelo: {rel['config']['modelo']} | "
        f"amostra: {rel['n_impossiveis']} imposs + {rel['n_respondiveis']} resp"
    )
    print(f"  NÃO-ALUCINAÇÃO (abstém quando deve): {na['taxa']:.3f} {na['ic95_wilson']}")
    print(f"  COBERTURA (responde quando deve):    {cob['taxa']:.3f} {cob['ic95_wilson']}")
    print(f"  média balanceada: {rel['media_balanceada']:.3f}")
    print(f"report: {caminho}")


if __name__ == "__main__":
    main()
