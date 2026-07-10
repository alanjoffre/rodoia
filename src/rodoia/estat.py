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
