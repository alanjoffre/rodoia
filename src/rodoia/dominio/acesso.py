"""Camada de acesso ao modelo de dados (Fase 3) — a ferramenta do agente (Fase 4).

Funções tipadas e **parametrizadas** (placeholders `?`, nunca concatenação de string —
anti-SQL-injection) sobre o esquema estrela. Retornos tipados (dicts simples), pensados
para serem chamados como ferramentas pelo agente da Fase 4. Testável com um DuckDB de
fixture (parâmetro `db`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rodoia.dominio.estrela import consultar_ro


def _consultar(sql: str, params: list[Any], db: Path | None = None) -> list[dict[str, Any]]:
    return consultar_ro(sql, params, db)


def ranking_pracas(top: int = 10, db: Path | None = None) -> list[dict[str, Any]]:
    """Top-N praças por volume acumulado (posição, praça, concessionária, volume)."""
    return _consultar(
        """SELECT rank() OVER (ORDER BY sum(f.volume_total) DESC) AS posicao,
                  p.praca, p.concessionaria, sum(f.volume_total) AS volume
           FROM fato_volume f JOIN dim_praca p USING (praca_id)
           GROUP BY p.praca, p.concessionaria ORDER BY volume DESC LIMIT ?""",
        [int(top)], db)


def volume_praca(praca: str, db: Path | None = None) -> float:
    """Volume total acumulado de uma praça (busca exata, parametrizada)."""
    r = _consultar(
        """SELECT COALESCE(sum(f.volume_total), 0) AS volume
           FROM fato_volume f JOIN dim_praca p USING (praca_id) WHERE p.praca = ?""",
        [praca], db)
    return float(r[0]["volume"]) if r else 0.0


def serie_mensal(praca: str, db: Path | None = None) -> list[dict[str, Any]]:
    """Série mensal (data, volume) de uma praça — insumo da previsão."""
    return _consultar(
        """SELECT f.data, sum(f.volume_total) AS volume
           FROM fato_volume f JOIN dim_praca p USING (praca_id)
           WHERE p.praca = ? GROUP BY f.data ORDER BY f.data""",
        [praca], db)


def volume_por_ano(db: Path | None = None) -> list[dict[str, Any]]:
    """Volume total da rede por ano (tendência agregada)."""
    return _consultar(
        """SELECT t.ano, sum(f.volume_total) AS volume
           FROM fato_volume f JOIN dim_tempo t USING (data)
           GROUP BY t.ano ORDER BY t.ano""",
        [], db)
