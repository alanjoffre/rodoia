"""Testes do CAMINHO DE TREINO da Fase 0 (crítico, antes só validado por execução):
`mlp_torch.treinar` e `classico.avaliar`. Usam um frame sintético (sem baixar dados)
e escrevem os artefatos em `tmp_path`."""
import numpy as np
import pandas as pd
import pytest
import torch

from rodoia.ml import classico, mlp_torch
from rodoia.ml.classico import ALVO, FEATURES_CATEGORICAS, FEATURES_NUMERICAS


def _frame_sintetico(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    dados = {c: rng.normal(size=n) for c in FEATURES_NUMERICAS}
    dados["concessionaria"] = rng.choice(["A", "B", "C"], size=n)
    dados["uf"] = rng.choice(["SP", "MG", "PR"], size=n)
    dados["sentido"] = rng.choice(["N", "S"], size=n)
    dados["tipo_de_acidente"] = rng.choice(["colisao", "capotagem", "atropelamento"], size=n)
    X = pd.DataFrame(dados)[FEATURES_NUMERICAS + FEATURES_CATEGORICAS]
    # alvo com sinal fraco em 'moto' → modelo aprende algo, sem ser trivial
    logito = 0.7 * dados["moto"] + rng.normal(scale=1.0, size=n)
    y = pd.Series((logito > np.quantile(logito, 0.64)).astype(int), name=ALVO)
    return X, y


@pytest.fixture
def _dados_fake(monkeypatch, tmp_path):
    X, y = _frame_sintetico()
    monkeypatch.setattr(mlp_torch, "carregar_dados", lambda amostra=None: (X, y))
    monkeypatch.setattr(classico, "carregar_dados", lambda amostra=None: (X, y))
    monkeypatch.setattr(mlp_torch, "_REPORT_DIR", tmp_path / "mlp")
    monkeypatch.setattr(classico, "_REPORT_DIR", tmp_path / "baseline")
    monkeypatch.setattr(mlp_torch, "dispositivo", lambda: torch.device("cpu"))  # CI-friendly
    return tmp_path


def test_treinar_mlp_reduz_perda_e_grava(_dados_fake):
    res = mlp_torch.treinar(epocas=3, batch=256)
    # o laço de otimização de fato reduz a perda de treino
    import json
    mlp_json = json.loads((_dados_fake / "mlp" / "mlp.json").read_text(encoding="utf-8"))
    assert mlp_json["loss_treino"][-1] < mlp_json["loss_treino"][0]
    # entrega as métricas completas (não só ROC-AUC) e o limiar sintonizado
    for chave in ("roc_auc", "pr_auc", "f1", "balanced_accuracy", "limiar"):
        assert chave in res
    assert 0.0 <= res["roc_auc"] <= 1.0
    # artefatos + proveniência carimbada
    assert (_dados_fake / "mlp" / "curva_treino_mlp.png").exists()
    assert "_proveniencia" in mlp_json and "git_sha" in mlp_json["_proveniencia"]


def test_avaliar_baseline_grava_metrics(_dados_fake):
    res = classico.avaliar(cv_amostra=300)
    assert "melhor" in res and "modelos" in res
    import json
    metrics = json.loads((_dados_fake / "baseline" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["melhor"] in metrics["modelos"]
    assert "_proveniencia" in metrics
