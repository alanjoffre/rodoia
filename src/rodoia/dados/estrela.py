"""Modelo dimensional (esquema ESTRELA) do Volume de Tráfego de Pedágio — DuckDB (Fase 3).

Escolha: **esquema estrela** (fato + dimensões) sobre a tabela achatada — porque as
análises são agregações por praça/tempo/categoria, e a estrela dá JOINs baratos e
consultas legíveis. DuckDB (embutido, colunar, SQL) casa com o local-first do projeto.

Grão do fato (`fato_volume`): 1 linha = **praça × mês × sentido × tipo de cobrança ×
categoria × tipo de veículo** → `volume_total`. É o grão mais fino da fonte; não
agregamos na ingestão para preservar todas as fatias (o agregado sai via SQL).

Dimensões:
- `dim_praca` (praca_id → praça, concessionária)
- `dim_tempo` (data → ano, mês, trimestre, nome_mes)
- `dim_categoria` (categoria → tipo_de_veiculo)

Uso:  python -m rodoia.dados.estrela
"""
from __future__ import annotations

from pathlib import Path

from rodoia.config import settings

DB = settings.data_processed / "volume.duckdb"
_PARQUET = settings.data_processed / "volume_pedagio.parquet"

_MESES = [
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
]


def construir(db: Path | None = None) -> dict:
    import duckdb

    db = db or DB
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    con = duckdb.connect(str(db))
    con.execute(f"CREATE TABLE bruto AS SELECT * FROM read_parquet('{_PARQUET.as_posix()}')")

    # Dimensão praça (chave surrogate por praça+concessionária).
    con.execute("""
        CREATE TABLE dim_praca AS
        SELECT row_number() OVER (ORDER BY concessionaria, praca) AS praca_id,
               praca, concessionaria
        FROM (SELECT DISTINCT praca, concessionaria FROM bruto)
    """)
    # Dimensão tempo (uma linha por mês presente).
    con.execute(f"""
        CREATE TABLE dim_tempo AS
        SELECT data, year(data) AS ano, month(data) AS mes,
               quarter(data) AS trimestre,
               (['{"','".join(_MESES)}'])[month(data)] AS nome_mes
        FROM (SELECT DISTINCT data FROM bruto)
    """)
    # Dimensão categoria.
    con.execute("""
        CREATE TABLE dim_categoria AS
        SELECT DISTINCT categoria, tipo_de_veiculo FROM bruto
    """)
    # Fato com FKs (praca_id + data), no grão da fonte.
    con.execute("""
        CREATE TABLE fato_volume AS
        SELECT p.praca_id, b.data, b.sentido, b.tipo_cobranca,
               b.categoria, b.tipo_de_veiculo, b.volume_total
        FROM bruto b JOIN dim_praca p USING (praca, concessionaria)
    """)
    con.execute("DROP TABLE bruto")

    tabelas = ["fato_volume", "dim_praca", "dim_tempo", "dim_categoria"]
    stats = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tabelas}
    con.close()
    print("esquema estrela em", db)
    for t, n in stats.items():
        print(f"  {t:16} {n:>10,} linhas")
    return stats


if __name__ == "__main__":
    construir()
