"""Observabilidade e cache do serving (Fase 5 — latência).

Duas peças que atacam o p95 alto da geração e dão visibilidade de produção:
- **CacheLRU**: cache de respostas por consulta (a geração é o gargalo, ~p95 30 s; uma consulta
  repetida passa a ser instantânea). LRU simples, sem dependência externa.
- **registrar_metrica**: emite uma linha JSON por requisição (latência, cache_hit, taxa de hit,
  nº de fontes, bloqueado) — observabilidade estruturada, além da trilha de auditoria.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


# K = chave ((consulta, k) na API, str nos testes) · V = valor (a resposta cacheada).
# Sintaxe de type parameters do PEP 695 (nativa do 3.12, que é o piso do projeto).
class CacheLRU[K, V]:
    """Cache least-recently-used mínimo. `get` devolve None no miss e conta hits/misses."""

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self._d: OrderedDict[K, V] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, chave: K) -> V | None:
        if chave in self._d:
            self._d.move_to_end(chave)
            self.hits += 1
            return self._d[chave]
        self.misses += 1
        return None

    def set(self, chave: K, valor: V) -> None:
        self._d[chave] = valor
        self._d.move_to_end(chave)
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)      # remove o menos usado recentemente

    @property
    def taxa_hit(self) -> float:
        tot = self.hits + self.misses
        return round(self.hits / tot, 3) if tot else 0.0


def registrar_metrica(evento: dict[str, Any], caminho: Path) -> None:
    """Anexa uma métrica estruturada (dict) a um JSONL de observabilidade."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(evento, ensure_ascii=False) + "\n")
