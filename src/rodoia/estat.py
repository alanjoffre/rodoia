"""Intervalos de confiança compartilhados (avaliação com n pequeno).

Centraliza o IC de Wilson (proporções) e o bootstrap (médias) usados nas avaliações
das Fases 1 e 2 — evita reimplementar em cada módulo.
"""
from __future__ import annotations

import math

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


def cohen_kappa(a: list, b: list) -> float:
    """κ de Cohen — concordância entre DOIS anotadores (labels categóricos), corrigida pelo acaso.
    κ=1 concordância perfeita, 0 = ao acaso, <0 pior que o acaso."""
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    cats = set(a) | set(b)
    p_obs = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    p_esp = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return round((p_obs - p_esp) / (1 - p_esp), 4) if p_esp < 1 else 1.0


def fleiss_kappa(avaliacoes: list[list], categorias=(0, 1, 2)) -> float:
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
