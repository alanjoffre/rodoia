"""Testes da Fase 3 (dados estruturados) — camada de acesso (fixture DuckDB) + funções puras."""
import duckdb
import pandas as pd
import pytest

from rodoia.dominio.acesso import ranking_pracas, serie_mensal, volume_praca
from rodoia.dominio.previsao import _mape


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

    from rodoia.dominio.previsao import _prever_praca

    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    t = np.arange(120)
    y = 1000 + 8 * t + 200 * np.sin(2 * np.pi * t / 12)  # tendência + sazonalidade anual
    r = _prever_praca(pd.Series(y, index=idx))
    assert all(v is not None and v >= 0 for v in r.values())
    assert r["holt_winters"] < r["naive"]  # modelo sazonal vence o random walk a 12 meses


def test_para_data_formatos_mistos():
    from rodoia.ingestao.ingestao_volume import _para_data
    d = _para_data(pd.Series(["01/01/2010", "03/2024"]))
    assert str(d.iloc[0].date()) == "2010-01-01"
    assert str(d.iloc[1].date()) == "2024-03-01"


def _series_sinteticas():
    import numpy as np
    idx = pd.date_range("2015-01-01", periods=110, freq="MS")
    t = np.arange(110)
    return [("PA", pd.Series(1000 + 5 * t + 100 * np.sin(2 * np.pi * t / 12), index=idx)),
            ("PB", pd.Series(500.5 + 3 * t[:100], index=idx[:100]))]


def test_recorte_ida_e_volta_e_deterministico(tmp_path, monkeypatch):
    """O recorte congelado devolve as MESMAS séries e o mesmo sha256 a cada gravação —
    é isso que permite reproduzir a âncora de previsão sem depender do dado ao vivo."""
    from rodoia.dominio import previsao as P
    monkeypatch.setattr(P, "REPO_ROOT", tmp_path)
    series = _series_sinteticas()
    a = P.salvar_recorte(series, tmp_path / "r1.json.gz")
    b = P.salvar_recorte(series, tmp_path / "r2.json.gz")
    assert a["sha256"] == b["sha256"]
    assert a["n_pracas"] == 2 and a["mes_final"] == "2024-02"
    for (n1, s1), (n2, s2) in zip(series, P.carregar_recorte(tmp_path / "r1.json.gz"), strict=True):
        assert n1 == n2
        pd.testing.assert_series_equal(s1, s2, check_names=False, check_freq=False)


def _preparar_reproducao(tmp_path, monkeypatch, sha_do_report=None):
    import json

    from rodoia.dominio import estrela
    from rodoia.dominio import previsao as P
    from rodoia.mlops import reproduzir as R
    monkeypatch.setattr(P, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(estrela, "DB", tmp_path / "nao_existe.duckdb")
    recorte = tmp_path / "recorte.json.gz"
    monkeypatch.setattr(P, "RECORTE", recorte)
    info = P.salvar_recorte(_series_sinteticas(), recorte)
    mape = R._mape_hw(P.carregar_recorte(recorte))
    rel = tmp_path / "reports" / "fase3_dados" / "previsao.json"
    rel.parent.mkdir(parents=True)
    rel.write_text(json.dumps({"modelos": {"holt_winters": {"mape_medio": mape}},
                               "recorte": {"sha256": sha_do_report or info["sha256"]}}))
    return R


def test_reproduzir_previsao_usa_o_recorte(tmp_path, monkeypatch):
    R = _preparar_reproducao(tmp_path, monkeypatch)
    ok, info = R.reproduzir_previsao()
    assert ok and info["delta"] == 0 and info["recorte_igual_ao_report"]
    assert "deriva_ao_vivo" not in info   # sem DuckDB, não há deriva a medir (e não pula)


def test_reproduzir_previsao_reprova_recorte_trocado(tmp_path, monkeypatch):
    """Mesmo com o MAPE igual, recorte diferente do report reprova: o número commitado
    não veio daquele dado."""
    R = _preparar_reproducao(tmp_path, monkeypatch, sha_do_report="0" * 64)
    ok, info = R.reproduzir_previsao()
    assert not ok and info["recorte_igual_ao_report"] is False
