"""Política de abstenção calibrada sobre o escore do cross-encoder (Fase 6).

A §13.3 do docs/17 mediu: o escore do rerank é o **primeiro sinal com separação
real** entre perguntas respondíveis e `is_impossible` (+3,758 nas medianas, ~9× a
do BM25) — mas as distribuições **ainda se sobrepõem**, então um limiar erraria.
"Erraria quanto" era a pergunta em aberto. Este módulo responde varrendo o limiar
e reportando a curva inteira, em vez de escolher um ponto e omitir o resto.

**Por que isso importa mais que a métrica agregada.** A abstenção do gerador
(§13.4) custa uma chamada de LLM por pergunta. Um limiar sobre o escore de
recuperação decide **antes de gerar** — é grátis. A pergunta de engenharia é se
ele é bom o bastante para valer, e a resposta honesta é a curva de trade-off, não
um número.

**As duas taxas de novo, agora do recuperador.** Mesmo par da §13.4, para os
números serem comparáveis:
- **não-alucinação**: fração das `is_impossible` que o limiar corretamente abstém;
- **cobertura**: fração das respondíveis que passam do limiar (não recusadas).

Um limiar altíssimo abstém de tudo: não-alucinação 1,0 e cobertura 0,0. A curva
mostra o custo de cada ponto — e o **J de Youden** (sens+esp−1) marca o ponto de
melhor equilíbrio, sem esconder que existe um trade-off.

Uso:
    python -m rodoia.rag.calibracao_abstencao          # lê o dump do rerank
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rodoia.config import settings
from rodoia.estat import wilson
from rodoia.proveniencia import carimbar

# Níveis de não-alucinação em que o trade-off é reportado. Incluem 0,99 de
# propósito: é a faixa em que o gerador opera (0,987 medido na §13.4), e portanto
# a única em que a comparação "limiar grátis vs. chamada de LLM" é honesta.
ALVOS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class Ponto:
    """Um ponto da curva: limiar e as duas taxas nele."""

    limiar: float
    nao_alucinacao: float
    cobertura: float
    j_youden: float


def varrer_limiar(
    escores_respondivel: list[float], escores_impossivel: list[float], n_pontos: int = 60
) -> list[Ponto]:
    """Varre limiares no intervalo observado e devolve a curva completa.

    Regra: escore **>= limiar** → responde; abaixo → abstém. Assim um limiar muito
    baixo responde a tudo (cobertura 1,0, não-alucinação 0,0) e um muito alto
    abstém de tudo — os dois extremos degenerados aparecem na curva, de propósito.
    """
    todos = escores_respondivel + escores_impossivel
    if not todos:
        return []
    lo, hi = min(todos), max(todos)
    if hi == lo:
        return []
    passo = (hi - lo) / max(1, n_pontos - 1)
    curva: list[Ponto] = []
    for i in range(n_pontos):
        t = lo + i * passo
        # cobertura = sensibilidade (respondíveis corretamente respondidas)
        cob = sum(1 for e in escores_respondivel if e >= t) / len(escores_respondivel)
        # não-alucinação = especificidade (impossíveis corretamente abstidas)
        na = sum(1 for e in escores_impossivel if e < t) / len(escores_impossivel)
        curva.append(Ponto(round(t, 4), round(na, 4), round(cob, 4), round(cob + na - 1, 4)))
    return curva


def economia_cascata(
    curva: list[Ponto], n_respondivel: int, n_impossivel: int, alvos: tuple[float, ...] = ALVOS
) -> list[dict[str, Any]]:
    """Traduz a curva na decisão de engenharia: **quanto se economiza e o que se perde**.

    A curva sozinha ainda não responde "vale a pena?". O uso real do limiar é uma
    cascata: abaixo dele, nem se chama o LLM. Então o que importa são dois números
    por ponto — a fração de chamadas de LLM evitadas e a fração de perguntas
    respondíveis que nunca chegam ao gerador (perda irrecuperável, porque o
    estágio seguinte nunca as vê).

    O prior do corpus entra aqui de propósito: 67,9% das perguntas do CUAD são
    `is_impossible`, e a economia depende dessa proporção. Reportar só as taxas
    condicionais esconderia isso.
    """
    total = n_respondivel + n_impossivel
    if total == 0:
        return []
    p_imp = n_impossivel / total
    p_resp = n_respondivel / total
    linhas: list[dict[str, Any]] = []
    for alvo in alvos:
        cand = [p for p in curva if p.nao_alucinacao >= alvo]
        if not cand:
            continue
        p = max(cand, key=lambda p: p.cobertura)
        # Chamadas evitadas = abstidas nas duas populações, ponderadas pelo prior.
        evitadas = p_imp * p.nao_alucinacao + p_resp * (1 - p.cobertura)
        linhas.append(
            {
                "alvo_nao_alucinacao": alvo,
                "limiar": p.limiar,
                "nao_alucinacao": p.nao_alucinacao,
                "cobertura": p.cobertura,
                "chamadas_llm_evitadas": round(evitadas, 4),
                # A perda: respondíveis barradas antes de o gerador poder acertá-las.
                "respondiveis_perdidas": round(1 - p.cobertura, 4),
            }
        )
    return linhas


def melhor_ponto(curva: list[Ponto]) -> Ponto | None:
    """Ponto de maior J de Youden (equilíbrio entre as duas taxas)."""
    return max(curva, key=lambda p: p.j_youden) if curva else None


def auc_roc(escores_respondivel: list[float], escores_impossivel: list[float]) -> float:
    """AUC-ROC pela estatística de Mann–Whitney U — a probabilidade de um
    respondível sorteado ter escore maior que um impossível sorteado.

    0,5 = o escore não separa nada (moeda). Resume a curva inteira num número
    **independente de limiar**, que é o que responde "esse sinal presta?".
    """
    n_r, n_i = len(escores_respondivel), len(escores_impossivel)
    if n_r == 0 or n_i == 0:
        return 0.5
    marcados = [(e, 1) for e in escores_respondivel] + [(e, 0) for e in escores_impossivel]
    marcados.sort(key=lambda x: x[0])
    # postos médios para empates
    postos: dict[int, float] = {}
    i = 0
    while i < len(marcados):
        j = i
        while j + 1 < len(marcados) and marcados[j + 1][0] == marcados[i][0]:
            j += 1
        posto_medio = (i + j) / 2 + 1
        for k in range(i, j + 1):
            postos[k] = posto_medio
        i = j + 1
    soma_r = sum(postos[k] for k, (_, rot) in enumerate(marcados) if rot == 1)
    u = soma_r - n_r * (n_r + 1) / 2
    return round(u / (n_r * n_i), 4)


def analisar(dump: dict[str, list[float]]) -> dict[str, Any]:
    """Relatório de calibração a partir dos escores por pergunta."""
    resp = dump["respondivel"]
    imp = dump["impossivel"]
    curva = varrer_limiar(resp, imp)
    melhor = melhor_ponto(curva)
    auc = auc_roc(resp, imp)
    return {
        "n_respondiveis": len(resp),
        "n_impossiveis": len(imp),
        # O número que responde "esse sinal presta?", sem depender de limiar.
        "auc_roc": auc,
        "interpretacao_auc": (
            "0,5 = não separa (moeda); 1,0 = separação perfeita. "
            "É a probabilidade de um respondível sorteado ter escore maior que um impossível."
        ),
        "melhor_ponto_youden": (
            {
                "limiar": melhor.limiar,
                "nao_alucinacao": melhor.nao_alucinacao,
                "ic95_nao_alucinacao": wilson(round(melhor.nao_alucinacao * len(imp)), len(imp)),
                "cobertura": melhor.cobertura,
                "ic95_cobertura": wilson(round(melhor.cobertura * len(resp)), len(resp)),
                "j_youden": melhor.j_youden,
            }
            if melhor
            else None
        ),
        # A tradução para a decisão: quanto de LLM se evita e o que isso custa.
        "cascata": economia_cascata(curva, len(resp), len(imp)),
        "curva": [
            {
                "limiar": p.limiar,
                "nao_alucinacao": p.nao_alucinacao,
                "cobertura": p.cobertura,
                "j": p.j_youden,
            }
            for p in curva
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibra o limiar de abstenção sobre o escore do cross-encoder."
    )
    parser.add_argument("--dump", type=Path, default=None, help="escores_rerank.json")
    args = parser.parse_args()

    destino = settings.data_processed.parent.parent / "reports" / "fase6_cuad"
    caminho_dump = args.dump or (destino / "escores_rerank.json")
    if not caminho_dump.exists():
        raise FileNotFoundError(
            f"{caminho_dump} ausente — rode `avaliacao_cuad_rerank --dump-escores` primeiro."
        )
    dump = json.loads(caminho_dump.read_text(encoding="utf-8"))
    rel = analisar(dump)

    destino.mkdir(parents=True, exist_ok=True)
    saida = destino / "calibracao_abstencao.json"
    saida.write_text(json.dumps(carimbar(rel), ensure_ascii=False, indent=2))

    print(f"n: {rel['n_respondiveis']} respondíveis + {rel['n_impossiveis']} impossíveis")
    print(f"AUC-ROC: {rel['auc_roc']}  (0,5 = sinal inútil)")
    m = rel["melhor_ponto_youden"]
    if m:
        print(f"melhor limiar (Youden J={m['j_youden']}): {m['limiar']}")
        print(f"  não-alucinação {m['nao_alucinacao']:.3f} {m['ic95_nao_alucinacao']}")
        print(f"  cobertura      {m['cobertura']:.3f} {m['ic95_cobertura']}")
    print("\ncascata (o que o limiar economiza e o que custa):")
    for linha in rel["cascata"]:
        print(
            f"  não-aluc>={linha['alvo_nao_alucinacao']:.2f} → limiar {linha['limiar']:8.3f} | "
            f"evita {linha['chamadas_llm_evitadas']:.1%} das chamadas | "
            f"perde {linha['respondiveis_perdidas']:.1%} das respondíveis"
        )
    print(f"report: {saida}")


if __name__ == "__main__":
    main()
