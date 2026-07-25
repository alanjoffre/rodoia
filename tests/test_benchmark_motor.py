"""Testes do benchmark de motor (só DuckDB e a comparação pura — SEM Spark).

pyspark/JVM não estão no CI (extra `benchmark-motor`, fora do lock). Então:
- `comparar` é testada com dicionários falsos (pura, sem motor);
- `rodar_duckdb` é testada contra um Parquet minúsculo de fixture (duckdb ESTÁ no
  CI, via extra `estruturados`);
- `rodar_spark` não é exercitada aqui — é validada na execução real (docs/17 §12).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rodoia.mlops.benchmark_motor import _normalizar, comparar, rodar_duckdb


def _parquet_fixture(tmp_path: Path) -> Path:
    """Escreve um CFPB minúsculo particionado por ano, com o schema real reduzido."""
    raiz = tmp_path / "cfpb"
    dados = {
        "2024": [("Mortgage", "reclamo", "Acme"), ("Mortgage", "", "Acme"), ("Card", "x", "Beta")],
        "2025": [("Card", "y", "Beta")],
    }
    for ano, linhas in dados.items():
        d = raiz / f"ano={ano}"
        d.mkdir(parents=True)
        tab = pa.table(
            {
                "product": [x[0] for x in linhas],
                "consumer_complaint_narrative": [x[1] for x in linhas],
                "company": [x[2] for x in linhas],
            }
        )
        pq.write_table(tab, d / "parte.parquet")
    return raiz


def test_normalizar_arredonda_e_ordena() -> None:
    # todo float é formatado com 4 casas (aqui as duas colunas são float); a
    # ordenação torna a comparação entre motores robusta a ordem de linha.
    bruto = [(2.0, 0.123456), (1.0, 0.9)]
    assert _normalizar(bruto) == [("1.0000", "0.9000"), ("2.0000", "0.1235")]


def test_normalizar_int_fica_string_simples() -> None:
    """count(*) volta int nos dois motores — vira str direta, sem casas decimais."""
    assert _normalizar([("2025", 1), ("2024", 3)]) == [("2024", "3"), ("2025", "1")]


def test_rodar_duckdb_conta_por_ano(tmp_path: Path) -> None:
    raiz = _parquet_fixture(tmp_path)
    sql = "SELECT ano, count(*) n FROM cfpb GROUP BY ano ORDER BY ano"
    r = rodar_duckdb(raiz, {"por_ano": sql}, 1)
    assert r["por_ano"]["resultado"] == [("2024", "3"), ("2025", "1")]
    assert r["por_ano"]["tempo_s"] >= 0


def test_rodar_duckdb_taxa_narrativa(tmp_path: Path) -> None:
    raiz = _parquet_fixture(tmp_path)
    sql = (
        "SELECT ano, round(avg(CASE WHEN consumer_complaint_narrative <> '' "
        "THEN 1.0 ELSE 0.0 END), 4) taxa FROM cfpb GROUP BY ano ORDER BY ano"
    )
    r = rodar_duckdb(raiz, {"taxa": sql}, 1)
    # 2024: 2 de 3 têm narrativa -> 0.6667; 2025: 1 de 1 -> 1.0
    assert dict(r["taxa"]["resultado"]) == {"2024": "0.6667", "2025": "1.0000"}


def test_comparar_concordancia_e_speedup() -> None:
    duck = {
        "q1": {"tempo_s": 0.5, "resultado": [("a", "1")]},
        "q2": {"tempo_s": 1.0, "resultado": [("b", "2")]},
    }
    spark = {
        "q1": {"tempo_s": 2.0, "resultado": [("a", "1")]},  # concorda, 4x
        "q2": {"tempo_s": 3.0, "resultado": [("b", "2")]},  # concorda, 3x
    }
    rel = comparar(duck, spark)
    assert rel["divergencias"] == 0
    assert rel["por_query"]["q1"]["speedup_duckdb"] == 4.0
    assert rel["speedup_total_duckdb"] == round(5.0 / 1.5, 2)
    assert all(v["resultados_concordam"] for v in rel["por_query"].values())


def test_comparar_denuncia_divergencia() -> None:
    """Se os motores discordam, tem que aparecer — é o invariante gated."""
    duck = {"q": {"tempo_s": 0.1, "resultado": [("a", "1")]}}
    spark = {"q": {"tempo_s": 0.2, "resultado": [("a", "2")]}}  # resultado DIFERENTE
    rel = comparar(duck, spark)
    assert rel["divergencias"] == 1
    assert rel["por_query"]["q"]["resultados_concordam"] is False
