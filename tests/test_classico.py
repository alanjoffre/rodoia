"""Testes do baseline de classificação: garante ausência de leakage e que o
pipeline realmente treina/pontua (sem tocar no dataset de 1M — dado sintético)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rodoia.ml.classico import (
    COLUNAS_PROIBIDAS,
    FEATURES_CATEGORICAS,
    construir_modelos,
    features,
)


def test_nenhuma_feature_e_proibida() -> None:
    """As colunas que definem o alvo não podem estar entre as features."""
    assert COLUNAS_PROIBIDAS.isdisjoint(set(features()))


def _frame_sintetico(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {c: rng.integers(0, 5, n) for c in features() if c not in FEATURES_CATEGORICAS}
    )
    X["concessionaria"] = rng.choice(["afd", "arb"], n)
    X["uf"] = rng.choice(["SP", "MG"], n)
    X["sentido"] = rng.choice(["norte", "sul"], n)
    X["tipo_de_acidente"] = rng.choice(["colisão", "capotamento"], n)
    X = X[features()]
    # Alvo com sinal (depende de total_veiculos) para o modelo ter o que aprender.
    y = (X["total_veiculos"] + rng.normal(0, 1, n) > 2).astype(int)
    return X, y


def test_pipeline_treina_e_pontua() -> None:
    X, y = _frame_sintetico()
    modelo = construir_modelos()["regressao_logistica"]
    modelo.fit(X, y)
    prob = modelo.predict_proba(X)[:, 1]
    assert prob.shape == (len(X),)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_todos_os_modelos_constroem() -> None:
    modelos = construir_modelos()
    assert set(modelos) == {
        "regressao_logistica",
        "arvore_decisao",
        "random_forest",
        "hist_gradient_boosting",
    }
