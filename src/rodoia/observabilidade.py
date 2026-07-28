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

from rodoia.config import settings


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


def emitir_evento(evento: dict[str, Any], caminho: Path, fluxo: str) -> None:
    """Sink único das trilhas estruturadas (auditoria e métricas).

    **Por que existe.** As duas trilhas faziam `open(caminho, "a")` num arquivo do
    repositório. Em contêiner com disco efêmero isso não falha — **desaparece**: cada
    instância escreve a sua cópia e a reciclagem apaga tudo, sem erro nem log. Para a
    métrica é perda de observabilidade; para a **auditoria** é um controle de
    conformidade (LGPD) que evapora em silêncio, que é bem pior (docs/16 §7.1).

    Com `log_destino="stdout"` o processo **não escreve arquivo**: emite uma linha
    JSON por evento e deixa a plataforma coletar (CloudWatch Logs no App Runner,
    `docker logs` local). É a regra de 12-factor — processo não gerencia log — e
    resolve efemeridade e multi-instância sem S3, sem EFS e sem custo.

    `fluxo` vai no campo `_fluxo` porque, em stdout, as duas trilhas caem no MESMO
    lugar: sem o marcador, separar auditoria de métrica no destino seria adivinhação.
    """
    linha = json.dumps({**evento, "_fluxo": fluxo}, ensure_ascii=False)
    if settings.log_destino == "stdout":
        # `flush` explícito: stdout em contêiner costuma ser block-buffered, e um
        # evento de auditoria preso no buffer some se o processo for morto.
        print(linha, flush=True)
        return
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as fh:
        fh.write(linha + "\n")


def registrar_metrica(evento: dict[str, Any], caminho: Path) -> None:
    """Anexa uma métrica estruturada (dict) à trilha de observabilidade."""
    emitir_evento(evento, caminho, fluxo="metrica")
