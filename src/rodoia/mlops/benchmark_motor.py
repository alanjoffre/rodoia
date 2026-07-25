"""Benchmark de motor analítico: **DuckDB vs Spark** sobre os 17,2 M do CFPB (Fase 6).

O honesto aqui não é "usei Spark" — é MEDIR os dois no mesmo conjunto de queries e
escolher com número. A hipótese, antes de rodar: numa única máquina com RAM
suficiente, o DuckDB ganha folgado — o overhead de JVM/shuffle do Spark só se paga
em cluster, acima da escala que cabe num nó. O benchmark testa isso em vez de
afirmar, e o resultado — qualquer que seja — vira a justificativa da escolha de
motor da ingestão (`ingestao/ingestao_cfpb.py`).

**Cross-check de correção é a parte que importa mais que o tempo.** Um motor rápido
que dá a resposta errada é inútil. Cada query roda nos dois e os resultados são
comparados célula a célula — a igualdade é o invariante gated, não o tempo (que é
dependente de hardware e não é regressão de métrica).

**pyspark fica fora do lock do CI** (extra `benchmark-motor`, stack JVM pesado que
brigaria com o torch/vLLM da Fase 2). Por isso o import de pyspark é LOCAL, dentro
de `rodar_spark`: o módulo importa sem pyspark, e o CI testa a comparação e o
caminho DuckDB sem tocar em Spark.

Uso (venv isolado com pyspark+duckdb+JRE — ver docs/17):
    python -m rodoia.mlops.benchmark_motor
    python -m rodoia.mlops.benchmark_motor --repeticoes 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rodoia.config import settings
from rodoia.estat import percentil
from rodoia.proveniencia import carimbar

# SQL IDÊNTICO para os dois motores (dialeto comum a DuckDB e Spark SQL): ambos rodam
# sobre uma VIEW `cfpb`. Queries escolhidas para exercer scan, group-by, agregação
# condicional, distinct e partition pruning — o trabalho real de um motor analítico.
QUERIES: dict[str, str] = {
    "linhas_por_ano": (
        "SELECT ano, count(*) AS n FROM cfpb GROUP BY ano ORDER BY ano"
    ),
    "top10_produtos": (
        "SELECT product, count(*) AS n FROM cfpb GROUP BY product ORDER BY n DESC, product LIMIT 10"
    ),
    "taxa_narrativa_por_ano": (
        "SELECT ano, "
        "round(avg(CASE WHEN consumer_complaint_narrative <> '' THEN 1.0 ELSE 0.0 END), 4) AS taxa "
        "FROM cfpb GROUP BY ano ORDER BY ano"
    ),
    "top10_empresas": (
        "SELECT company, count(*) AS n FROM cfpb GROUP BY company ORDER BY n DESC, company LIMIT 10"
    ),
    "empresas_distintas": (
        "SELECT count(DISTINCT company) AS n FROM cfpb"
    ),
    "pruning_um_ano": (
        "SELECT count(*) AS n FROM cfpb WHERE ano = '2024'"
    ),
}


def _normalizar(linhas: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Normaliza um resultado para comparação célula a célula entre motores:
    floats arredondados (evita ruído de ponto flutuante), tudo para string,
    ordenado (a ordem entre motores pode diferir onde o SQL não a fixa)."""
    def _cel(v: Any) -> str:
        if isinstance(v, float):
            return f"{round(v, 4):.4f}"
        return str(v)

    return sorted(tuple(_cel(v) for v in linha) for linha in linhas)


def rodar_duckdb(
    parquet_dir: Path, queries: dict[str, str], repeticoes: int
) -> dict[str, dict[str, Any]]:
    """Roda todas as queries no DuckDB, medindo o tempo (mediana de N execuções)."""
    import duckdb

    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW cfpb AS SELECT * FROM "
        f"read_parquet('{parquet_dir.as_posix()}/**/*.parquet', hive_partitioning=true)"
    )
    resultados: dict[str, dict[str, Any]] = {}
    for nome, sql in queries.items():
        tempos = []
        linhas: list[tuple[Any, ...]] = []
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            linhas = con.execute(sql).fetchall()
            tempos.append(time.perf_counter() - t0)
        resultados[nome] = {"tempo_s": percentil(tempos, 0.5), "resultado": _normalizar(linhas)}
    con.close()
    return resultados


def rodar_spark(
    parquet_dir: Path, queries: dict[str, str], repeticoes: int
) -> dict[str, dict[str, Any]]:
    """Roda todas as queries no Spark local. Import LOCAL — pyspark não está no CI."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("rodoia-benchmark")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    spark.read.parquet(parquet_dir.as_posix()).createOrReplaceTempView("cfpb")

    resultados: dict[str, dict[str, Any]] = {}
    for nome, sql in queries.items():
        tempos = []
        linhas: list[tuple[Any, ...]] = []
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            linhas = [tuple(r) for r in spark.sql(sql).collect()]
            tempos.append(time.perf_counter() - t0)
        resultados[nome] = {"tempo_s": percentil(tempos, 0.5), "resultado": _normalizar(linhas)}
    spark.stop()
    return resultados


def comparar(
    duck: dict[str, dict[str, Any]], spark: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Consolida o benchmark: por query, tempos + speedup + CONCORDÂNCIA de resultado.

    Função pura (testável sem motor): recebe os dois dicionários e produz o relatório.
    """
    por_query = {}
    divergencias = 0
    total_duck = total_spark = 0.0
    for nome in duck:
        td = duck[nome]["tempo_s"]
        ts = spark[nome]["tempo_s"]
        concorda = duck[nome]["resultado"] == spark[nome]["resultado"]
        if not concorda:
            divergencias += 1
        total_duck += td
        total_spark += ts
        por_query[nome] = {
            "duckdb_s": round(td, 4),
            "spark_s": round(ts, 4),
            "speedup_duckdb": round(ts / td, 2) if td > 0 else None,
            "resultados_concordam": concorda,
            "linhas": len(duck[nome]["resultado"]),
        }
    return {
        "n_queries": len(duck),
        "divergencias": divergencias,
        "tempo_total_duckdb_s": round(total_duck, 3),
        "tempo_total_spark_s": round(total_spark, 3),
        "speedup_total_duckdb": round(total_spark / total_duck, 2) if total_duck > 0 else None,
        "por_query": por_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DuckDB vs Spark sobre o CFPB.")
    parser.add_argument("--parquet", type=Path, default=None, help="raiz do Parquet particionado")
    parser.add_argument("--repeticoes", type=int, default=3, help="execuções por query (mediana)")
    args = parser.parse_args()

    parquet_dir = args.parquet or (settings.data_processed / "cfpb")
    if not parquet_dir.exists():
        raise FileNotFoundError(f"{parquet_dir} ausente — rode ingestao_cfpb primeiro.")

    print(f"DuckDB sobre {parquet_dir} ...")
    duck = rodar_duckdb(parquet_dir, QUERIES, args.repeticoes)
    print("Spark (local[*]) ...")
    spark = rodar_spark(parquet_dir, QUERIES, args.repeticoes)

    relatorio = comparar(duck, spark)
    destino = settings.data_processed.parent.parent / "reports" / "fase6_escala"
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / "benchmark_motor.json"
    caminho.write_text(json.dumps(carimbar(relatorio), ensure_ascii=False, indent=2))

    print(f"\n{'query':26} {'DuckDB':>9} {'Spark':>9} {'speedup':>8}  ok")
    for nome, v in relatorio["por_query"].items():
        print(
            f"{nome:26} {v['duckdb_s']:>8.3f}s {v['spark_s']:>8.3f}s "
            f"{v['speedup_duckdb']:>7}x  {'✓' if v['resultados_concordam'] else '✗'}"
        )
    print(
        f"\ntotal: DuckDB {relatorio['tempo_total_duckdb_s']}s vs "
        f"Spark {relatorio['tempo_total_spark_s']}s "
        f"({relatorio['speedup_total_duckdb']}x) | divergências: {relatorio['divergencias']}"
    )
    print(f"report: {caminho}")


if __name__ == "__main__":
    main()
