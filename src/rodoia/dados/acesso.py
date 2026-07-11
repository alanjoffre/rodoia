"""Camada de acesso ao modelo de dados (Fase 3) — a ferramenta do agente (Fase 4).

Funções tipadas e **parametrizadas** (placeholders `?`, nunca concatenação de string —
anti-SQL-injection) sobre o esquema estrela. Retornos tipados (dicts simples), pensados
para serem chamados como ferramentas pelo agente da Fase 4. Testável com um DuckDB de
fixture (parâmetro `db`).
"""
from __future__ import annotations

from pathlib import Path

from rodoia.dados.estrela import DB


def _consultar(sql: str, params: list, db: Path | None = None) -> list[dict]:
    import duckdb

    con = duckdb.connect(str(db or DB), read_only=True)
    try:
        return con.execute(sql, params).df().to_dict(orient="records")
    finally:
        con.close()


def ranking_pracas(top: int = 10, db: Path | None = None) -> list[dict]:
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


def serie_mensal(praca: str, db: Path | None = None) -> list[dict]:
    """Série mensal (data, volume) de uma praça — insumo da previsão."""
    return _consultar(
        """SELECT f.data, sum(f.volume_total) AS volume
           FROM fato_volume f JOIN dim_praca p USING (praca_id)
           WHERE p.praca = ? GROUP BY f.data ORDER BY f.data""",
        [praca], db)


def volume_por_ano(db: Path | None = None) -> list[dict]:
    """Volume total da rede por ano (tendência agregada)."""
    return _consultar(
        """SELECT t.ano, sum(f.volume_total) AS volume
           FROM fato_volume f JOIN dim_tempo t USING (data)
           GROUP BY t.ano ORDER BY t.ano""",
        [], db)
