"""Intervalos de confiança compartilhados (avaliação com n pequeno).

Centraliza o IC de Wilson (proporções) e o bootstrap (médias) usados nas avaliações
das Fases 1 e 2 — evita reimplementar em cada módulo.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """IC de Wilson para uma proporção (robusto a n pequeno, ao contrário do normal)."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / denom
    margem = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centro - margem), 3), round(min(1.0, centro + margem), 3)]


def bootstrap_ic(valores: list[float], n_boot: int = 2000, seed: int = 42) -> list[float]:
    """IC 95% por bootstrap percentílico da média."""
    arr = np.asarray(valores, dtype=float)
    if arr.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    medias = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def percentil(valores: list[float], p: float) -> float:
    """Percentil nearest-rank (ceil), puro e testável. `p` em [0,1]; lista vazia → 0.0.

    Sem arredondamento — quem chama arredonda se quiser (contrato único usado por
    `ft.benchmark_vllm.percentil` e `mlops.carga._percentil`).
    """
    if not valores:
        return 0.0
    ordenado = sorted(valores)
    idx = max(0, min(len(ordenado) - 1, math.ceil(p * len(ordenado)) - 1))
    return float(ordenado[idx])


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """κ de Cohen — concordância entre DOIS anotadores (labels categóricos), corrigida pelo acaso.
    κ=1 concordância perfeita, 0 = ao acaso, <0 pior que o acaso."""
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    cats = set(a) | set(b)
    p_obs = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    p_esp = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return round((p_obs - p_esp) / (1 - p_esp), 4) if p_esp < 1 else 1.0


def cohen_kappa_ic95(
    a: Sequence[Hashable], b: Sequence[Hashable], n_boot: int = 10_000, seed: int = 42
) -> list[float]:
    """IC 95% do κ de Cohen por bootstrap percentílico dos PARES (reamostra os n pares com
    reposição e recomputa κ). Honra a régua do projeto: nenhum número sem incerteza — mesmo o
    κ, que tem n pequeno. Reamostragens degeneradas (κ indefinido) são descartadas."""
    n = len(a)
    if n == 0 or len(b) != n:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    kappas = [cohen_kappa([a[i] for i in linha], [b[i] for i in linha]) for linha in idx]
    lo, hi = np.percentile(kappas, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float | int | str]:
    """Teste de McNemar para duas condições sobre os **MESMOS itens** (desenho pareado).

    Comparar dois ICs de proporção é o teste **conservador**: ele ignora que os itens
    são os mesmos e por isso perde potência — dois ICs sobrepostos podem esconder uma
    diferença real. O McNemar olha só os **discordantes** (onde uma condição acerta e
    a outra erra), que é onde a informação está.

    `a` e `b` são vetores de acerto/erro alinhados por item. Devolve as contagens
    discordantes (b01, b10), a estatística com correção de continuidade de Yates e o
    p-valor bicaudal.

    **Dois regimes, e o teste muda entre eles.** A aproximação χ² degrada quando
    b01+b10 < 25 — e foi exatamente o caso da ablação de prompt do CUAD (23
    discordantes). Nesse regime o p-valor vem do **binomial exato**: sob H₀ cada
    discordante é cara-ou-coroa, então `p = 2·P(X ≤ min(b01,b10) | n, ½)`. Não há
    aproximação envolvida e não precisa de SciPy. Qual dos dois foi usado sai em
    `metodo`, porque reportar um p-valor sem dizer de onde ele veio esconde a
    decisão mais importante do teste.
    """
    vazio: dict[str, float | int | str] = {
        "b01": 0, "b10": 0, "n_discordantes": 0,
        "estatistica": 0.0, "p_valor": 1.0, "metodo": "sem_discordantes",
    }
    if len(a) != len(b) or not a:
        return vazio
    b01 = sum(1 for x, y in zip(a, b, strict=True) if not x and y)   # só b acerta
    b10 = sum(1 for x, y in zip(a, b, strict=True) if x and not y)   # só a acerta
    n_disc = b01 + b10
    if n_disc == 0:
        return vazio
    estat = (abs(b01 - b10) - 1) ** 2 / n_disc          # Yates
    saida: dict[str, float | int | str] = {
        "b01": b01, "b10": b10, "n_discordantes": n_disc, "estatistica": round(estat, 4),
    }
    if n_disc < 25:
        saida["p_valor"] = _arredondar_p(_p_binomial_exato(min(b01, b10), n_disc))
        saida["metodo"] = "binomial_exato"
    else:
        saida["p_valor"] = _arredondar_p(_p_qui2_1gl(estat))
        saida["metodo"] = "qui2_yates"
    return saida


def _arredondar_p(p: float) -> float:
    """3 algarismos significativos, não 6 casas decimais.

    `round(p, 6)` transforma 2,2e-08 em **0,0** — e um p-valor nunca é zero. O leitor
    perde a ordem de grandeza justamente quando a evidência é mais forte, e "p = 0,0"
    é uma afirmação falsa impressa num relatório. Significativos preservam 2,2e-08 e
    2,2e-14 como coisas diferentes.
    """
    return float(f"{p:.3g}")


def _p_binomial_exato(k: int, n: int) -> float:
    """p bicaudal de `k` sucessos em `n` sob p=½: `2·P(X ≤ k)`, truncado em 1,0.

    Soma exata das combinações — sem aproximação e sem SciPy. É o teste correto
    quando o χ² não vale, e como `k = min(b01, b10)` a cauda somada é sempre a
    menor, que é o que a versão bicaudal simétrica pede.
    """
    if n <= 0:
        return 1.0
    acumulado = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(1.0, 2.0 * acumulado / (2**n)))


def _p_qui2_1gl(x: float) -> float:
    """P(χ²₁ > x) — cauda superior. Com 1 gl é `erfc(sqrt(x/2))`, sem SciPy."""
    return float(math.erfc(math.sqrt(max(0.0, x) / 2)))


def fleiss_kappa(
    avaliacoes: Sequence[Sequence[Hashable]], categorias: Sequence[Hashable] = (0, 1, 2)
) -> float:
    """κ de Fleiss — concordância entre MÚLTIPLOS avaliadores (banca de juízes).

    `avaliacoes`: uma lista de itens; cada item é a lista de rótulos dados pelos avaliadores
    (ex.: [[2,2,1],[0,0,0],...] para 3 juízes numa escala 0/1/2). κ=1 concordância perfeita,
    0 = ao acaso, <0 pior que o acaso. Interpreta a força da concordância inter-anotador.
    """
    n_itens = len(avaliacoes)
    if n_itens == 0:
        return 0.0
    n = len(avaliacoes[0])                       # avaliadores por item (assume constante)
    if n < 2:
        return 1.0
    idx = {c: j for j, c in enumerate(categorias)}
    cont = [[0] * len(categorias) for _ in range(n_itens)]
    for i, item in enumerate(avaliacoes):
        for rotulo in item:
            cont[i][idx[rotulo]] += 1
    # concordância observada por item e média
    p_i = [(sum(c * c for c in cont[i]) - n) / (n * (n - 1)) for i in range(n_itens)]
    p_obs = sum(p_i) / n_itens
    # concordância esperada ao acaso
    p_j = [sum(cont[i][j] for i in range(n_itens)) / (n_itens * n) for j in range(len(categorias))]
    p_esp = sum(p * p for p in p_j)
    return round((p_obs - p_esp) / (1 - p_esp), 4) if p_esp < 1 else 1.0
