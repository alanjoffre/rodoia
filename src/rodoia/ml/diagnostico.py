"""Diagnóstico do modelo de severidade (Fase 0, item 2).

Quatro análises que provam entendimento — não só uso — de ML:

1. **Curva de aprendizado**: score de treino vs. validação conforme cresce o
   volume de dados. Diagnostica bias (ambos baixos e juntos → simples demais) vs.
   variância (fenda grande → decorou o treino).
2. **Curva de validação**: varia um hiperparâmetro (profundidade da árvore) e
   mostra o *mecanismo* do overfitting — treino sobe, validação desce.
3. **Calibração**: as probabilidades previstas batem com a frequência real?
   `class_weight='balanced'` melhora o ranking (ROC-AUC) mas costuma *piorar* a
   calibração — mostramos isso e corrigimos com calibração isotônica.
4. **Clustering (KMeans)**: análise exploratória não supervisionada — encontra
   arquétipos de acidente e mede a severidade de cada um.

Tudo roda numa subamostra (declarada, não silenciosa) para custo tratável.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.metrics import brier_score_loss, silhouette_score
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from rodoia.config import settings
from rodoia.ml.classico import (
    FEATURES_NUMERICAS,
    carregar_dados,
    construir_modelos,
    construir_preprocessador,
)
from rodoia.proveniencia import carimbar

_REPORT_DIR = settings.data_processed.parent.parent / "reports" / "fase0_diagnostico"


def curva_aprendizado(X, y) -> dict:
    """Score de treino vs. CV para tamanhos crescentes de treino (modelo campeão)."""
    modelo = construir_modelos()["hist_gradient_boosting"]
    tamanhos = np.linspace(0.1, 1.0, 6)
    n, tr, val = learning_curve(
        modelo,
        X,
        y,
        train_sizes=tamanhos,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=settings.seed,
        shuffle=True,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n, tr.mean(axis=1), "o-", label="treino")
    ax.plot(n, val.mean(axis=1), "o-", label="validação (CV)")
    ax.fill_between(n, val.mean(1) - val.std(1), val.mean(1) + val.std(1), alpha=0.15)
    ax.set(
        xlabel="nº de exemplos de treino", ylabel="ROC-AUC", title="Curva de aprendizado (HistGB)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(_REPORT_DIR / "curva_aprendizado.png", dpi=110)
    plt.close(fig)
    return {
        "roc_auc_treino_final": float(tr.mean(1)[-1]),
        "roc_auc_validacao_final": float(val.mean(1)[-1]),
        "fenda": float(tr.mean(1)[-1] - val.mean(1)[-1]),
    }


def curva_validacao(X, y) -> dict:
    """Overfitting em ação: varia a profundidade de uma árvore de decisão."""
    prep = construir_preprocessador()
    pipe = Pipeline(
        [("prep", prep), ("modelo", DecisionTreeClassifier(random_state=settings.seed))]
    )
    profundidades = [1, 2, 3, 5, 8, 12, 16, 20, 30]
    tr, val = validation_curve(
        pipe,
        X,
        y,
        param_name="modelo__max_depth",
        param_range=profundidades,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(profundidades, tr.mean(1), "o-", label="treino")
    ax.plot(profundidades, val.mean(1), "o-", label="validação (CV)")
    ax.set(
        xlabel="profundidade máxima da árvore",
        ylabel="ROC-AUC",
        title="Curva de validação (árvore de decisão)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(_REPORT_DIR / "curva_validacao.png", dpi=110)
    plt.close(fig)
    melhor_idx = int(np.argmax(val.mean(1)))
    return {
        "profundidade_otima": profundidades[melhor_idx],
        "roc_auc_validacao_max": float(val.mean(1)[melhor_idx]),
        "roc_auc_treino_na_prof_maxima": float(tr.mean(1)[-1]),
        "roc_auc_validacao_na_prof_maxima": float(val.mean(1)[-1]),
    }


def diagrama_calibracao(X, y) -> dict:
    """Confiabilidade das probabilidades: balanced (bom ranking, calibração ruim)
    vs. o mesmo modelo com calibração isotônica."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=settings.seed
    )
    base = construir_modelos()["hist_gradient_boosting"]
    base.fit(X_tr, y_tr)
    p_base = base.predict_proba(X_te)[:, 1]

    calibrado = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrado.fit(X_tr, y_tr)
    p_cal = calibrado.predict_proba(X_te)[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfeito")
    for nome, p in [("balanced (bruto)", p_base), ("isotônico (corrigido)", p_cal)]:
        fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fr, "o-", label=f"{nome} — Brier={brier_score_loss(y_te, p):.3f}")
    ax.set(
        xlabel="probabilidade prevista",
        ylabel="frequência observada",
        title="Diagrama de calibração (HistGB)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(_REPORT_DIR / "calibracao.png", dpi=110)
    plt.close(fig)
    return {
        "brier_balanced": float(brier_score_loss(y_te, p_base)),
        "brier_isotonico": float(brier_score_loss(y_te, p_cal)),
    }


def clustering_exploratorio(X, y, ks=(2, 3, 4, 5, 6)) -> dict:
    """KMeans sobre as features numéricas padronizadas; escolhe k por silhueta e
    perfila os clusters (inclusive a taxa de severidade de cada arquétipo)."""
    Xnum = X[FEATURES_NUMERICAS].fillna(X[FEATURES_NUMERICAS].median())
    Z = StandardScaler().fit_transform(Xnum)

    # Silhueta numa subamostra (métrica é O(n^2)).
    rng = np.random.default_rng(settings.seed)
    idx = rng.choice(len(Z), size=min(8000, len(Z)), replace=False)
    silhuetas = {}
    for k in ks:
        rotulos = KMeans(n_clusters=k, random_state=settings.seed, n_init=10).fit_predict(Z[idx])
        silhuetas[k] = float(silhouette_score(Z[idx], rotulos))
    melhor_k = max(silhuetas, key=silhuetas.get)

    km = KMeans(n_clusters=melhor_k, random_state=settings.seed, n_init=10).fit(Z)
    perfil = Xnum.copy()
    perfil["cluster"] = km.labels_
    perfil["houve_vitima"] = y.to_numpy()
    resumo = perfil.groupby("cluster").agg(
        n=("moto", "size"),
        taxa_vitima=("houve_vitima", "mean"),
        moto_medio=("moto", "mean"),
        total_veiculos_medio=("total_veiculos", "mean"),
        km_medio=("km", "mean"),
        hora_media=("hora", "mean"),
    )
    resumo.to_csv(_REPORT_DIR / "clusters_perfil.csv")
    return {
        "silhuetas": silhuetas,
        "melhor_k": int(melhor_k),
        "perfil_clusters": json.loads(resumo.round(3).to_json(orient="index")),
    }


def diagnosticar(amostra: int = 150_000) -> dict:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = carregar_dados(amostra=amostra)
    print(f"diagnóstico sobre {len(X):,} exemplos (subamostra declarada)")

    resultado = {
        "n_amostra": len(X),
        "curva_aprendizado": curva_aprendizado(X, y),
        "curva_validacao": curva_validacao(X, y),
        "calibracao": diagrama_calibracao(X, y),
        "clustering": clustering_exploratorio(X, y),
    }
    (_REPORT_DIR / "diagnostico.json").write_text(
        json.dumps(carimbar(resultado), ensure_ascii=False, indent=2)
    )

    la = resultado["curva_aprendizado"]
    cv = resultado["curva_validacao"]
    cal = resultado["calibracao"]
    cl = resultado["clustering"]
    print(
        f"  aprendizado: treino={la['roc_auc_treino_final']:.3f} "
        f"validação={la['roc_auc_validacao_final']:.3f} fenda={la['fenda']:.3f}"
    )
    print(
        f"  validação: profundidade ótima={cv['profundidade_otima']} "
        f"(prof.30 treino={cv['roc_auc_treino_na_prof_maxima']:.3f} "
        f"val={cv['roc_auc_validacao_na_prof_maxima']:.3f} → overfitting)"
    )
    print(
        f"  calibração: Brier balanced={cal['brier_balanced']:.3f} "
        f"isotônico={cal['brier_isotonico']:.3f}"
    )
    print(f"  clustering: melhor k={cl['melhor_k']} silhuetas={cl['silhuetas']}")
    print(f"relatório e figuras em: {_REPORT_DIR}")
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico do modelo (Fase 0).")
    parser.add_argument("--amostra", type=int, default=150_000, help="tamanho da subamostra")
    args = parser.parse_args()
    diagnosticar(amostra=args.amostra)


if __name__ == "__main__":
    main()
