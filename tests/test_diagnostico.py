"""Testes do diagnóstico em dado sintético (sem rede, saída em tmp)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rodoia.ml import diagnostico
from rodoia.ml.classico import FEATURES_CATEGORICAS, features


@pytest.fixture(autouse=True)
def _report_em_tmp(tmp_path, monkeypatch):
    """Redireciona a saída de figuras/CSV para um diretório temporário."""
    monkeypatch.setattr(diagnostico, "_REPORT_DIR", tmp_path)
    tmp_path.mkdir(exist_ok=True)


def _frame(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {c: rng.integers(0, 5, n) for c in features() if c not in FEATURES_CATEGORICAS}
    )
    X["concessionaria"] = rng.choice(["afd", "arb"], n)
    X["uf"] = rng.choice(["SP", "MG"], n)
    X["sentido"] = rng.choice(["norte", "sul"], n)
    X["tipo_de_acidente"] = rng.choice(["colisão", "capotamento"], n)
    X = X[features()]
    y = ((X["moto"] + rng.normal(0, 1, n)) > 2).astype(int)
    return X, y


def test_clustering_retorna_k_e_perfil() -> None:
    X, y = _frame()
    res = diagnostico.clustering_exploratorio(X, y, ks=(2, 3))
    assert res["melhor_k"] in (2, 3)
    assert set(res["silhuetas"]) == {2, 3}
    # cada cluster tem taxa de vítima entre 0 e 1
    for perfil in res["perfil_clusters"].values():
        assert 0.0 <= perfil["taxa_vitima"] <= 1.0


def test_curva_validacao_acha_profundidade() -> None:
    X, y = _frame()
    res = diagnostico.curva_validacao(X, y)
    assert res["profundidade_otima"] in [1, 2, 3, 5, 8, 12, 16, 20, 30]
    # a árvore mais profunda deve ter score de treino >= ao de validação (overfit)
    assert res["roc_auc_treino_na_prof_maxima"] >= res["roc_auc_validacao_na_prof_maxima"]
