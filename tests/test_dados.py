"""Testes da Fase 3 (dados estruturados) — camada de acesso (fixture DuckDB) + funções puras."""
import duckdb
import pandas as pd
import pytest

from rodoia.dados.acesso import ranking_pracas, serie_mensal, volume_praca
from rodoia.dados.previsao import _mape


@pytest.fixture
def db_fixture(tmp_path):
    """DuckDB mínimo com o esquema estrela p/ testar a camada de acesso."""
    p = tmp_path / "t.duckdb"
    con = duckdb.connect(str(p))
    con.execute("CREATE TABLE dim_praca(praca_id INT, praca VARCHAR, concessionaria VARCHAR)")
    con.execute("INSERT INTO dim_praca VALUES (1,'PA','C1'),(2,'PB','C2')")
    con.execute("CREATE TABLE dim_tempo(data DATE, ano INT)")
    con.execute("INSERT INTO dim_tempo VALUES ('2020-01-01',2020),('2020-02-01',2020)")
    con.execute("CREATE TABLE fato_volume(praca_id INT, data DATE, volume_total DOUBLE)")
    con.execute("""INSERT INTO fato_volume VALUES
        (1,'2020-01-01',100),(1,'2020-02-01',150),(2,'2020-01-01',30)""")
    con.close()
    return p


def test_ranking_pracas(db_fixture):
    r = ranking_pracas(top=10, db=db_fixture)
    assert r[0]["praca"] == "PA" and r[0]["posicao"] == 1 and r[0]["volume"] == 250
    assert r[1]["praca"] == "PB"


def test_volume_praca(db_fixture):
    assert volume_praca("PA", db=db_fixture) == 250.0
    assert volume_praca("PB", db=db_fixture) == 30.0


def test_volume_praca_anti_injection(db_fixture):
    # entrada maliciosa é tratada como valor (parametrizada) — não executa, retorna 0
    assert volume_praca("PA'; DROP TABLE fato_volume; --", db=db_fixture) == 0.0
    # a tabela continua lá
    assert volume_praca("PA", db=db_fixture) == 250.0


def test_serie_mensal_ordenada(db_fixture):
    s = serie_mensal("PA", db=db_fixture)
    assert [r["volume"] for r in s] == [100, 150]


def test_mape():
    # erros 10/100 e 20/200 -> MAPE = (10% + 10%)/2 = 10%
    assert _mape([100, 200], [110, 180]) == 10.0


def test_previsao_multistep_justa():
    """Numa série com tendência+sazonalidade, prevendo 12 meses à frente (sem ver o teste),
    o Holt-Winters deve BATER o naïve (random walk). Trava a avaliação multi-step honesta:
    se algum modelo 'espiar' o teste (1-passo alimentado com o real), o teste denuncia."""
    import numpy as np
    from rodoia.dados.previsao import _prever_praca

    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    t = np.arange(120)
    y = 1000 + 8 * t + 200 * np.sin(2 * np.pi * t / 12)  # tendência + sazonalidade anual
    r = _prever_praca(pd.Series(y, index=idx))
    assert all(v is not None and v >= 0 for v in r.values())
    assert r["holt_winters"] < r["naive"]  # modelo sazonal vence o random walk a 12 meses


def test_para_data_formatos_mistos():
    from rodoia.data.ingestao_volume import _para_data
    d = _para_data(pd.Series(["01/01/2010", "03/2024"]))
    assert str(d.iloc[0].date()) == "2010-01-01"
    assert str(d.iloc[1].date()) == "2024-03-01"
