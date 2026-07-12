"""Testes da Fase 5 (MLOps) — funções puras de rastreio e drift (sem mlflow/duckdb)."""
import numpy as np

from rodoia.mlops.drift import classificar, psi
from rodoia.mlops.rastreio import coletar


def test_psi_identico_e_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert psi(x, x) < 1e-6                 # mesma distribuição → PSI ~ 0


def test_psi_detecta_deslocamento():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, size=2000)
    recente = rng.normal(3, 1, size=2000)  # média deslocada 3σ
    valor = psi(base, recente)
    assert valor > 0.25 and classificar(valor) == "relevante"


def test_classificar_faixas():
    assert classificar(0.05) == "estável"
    assert classificar(0.15) == "moderado"
    assert classificar(0.40) == "relevante"


def test_coletar_le_todas_as_fases():
    runs = coletar()
    fases = {r["fase"] for r in runs}
    assert {"fase0_mlp", "fase1_rag", "fase2_ner", "fase3_previsao", "fase4_agente"} <= fases
    # cada run tem métricas numéricas
    for r in runs:
        assert r["metrics"] and all(isinstance(v, (int, float)) for v in r["metrics"].values())
