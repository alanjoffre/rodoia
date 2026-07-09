"""Baseline de ML clássico: prever a severidade de um acidente (`houve_vitima`)
a partir de condições conhecidas no local — hora, km, tipo de acidente, veículos
envolvidos, concessionária, UF. Compara uma família de modelos com métricas
apropriadas a dado desbalanceado.

Anti-leakage (crítico): as contagens de vítimas e o rótulo textual definem o
alvo, então NUNCA entram como feature. `COLUNAS_PROIBIDAS` fixa isso e um teste
garante que nenhuma delas está na lista de features.

Métricas: como só ~36% dos acidentes têm vítima (e ~2% são fatais), acurácia
engana. Reportamos ROC-AUC, PR-AUC (average precision), F1 e balanced accuracy,
com `class_weight='balanced'` em todos os modelos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")  # backend sem display (roda em CI/servidor)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from rodoia.config import settings

ALVO = "houve_vitima"

FEATURES_NUMERICAS = [
    "km",
    "hora",
    "mes",
    "dia_semana",
    "ano",
    "total_veiculos",
    "automovel",
    "bicicleta",
    "caminhao",
    "moto",
    "onibus",
    "outros",
    "tracao_animal",
    "transporte_de_cargas_especiais",
    "trator_maquinas",
    "utilitarios",
]
FEATURES_CATEGORICAS = ["concessionaria", "uf", "sentido", "tipo_de_acidente"]

# Colunas que revelam o alvo — proibidas como feature (guardadas por teste).
COLUNAS_PROIBIDAS = frozenset(
    {
        "houve_vitima",
        "houve_fatal",
        "n_feridos",
        "tipo_de_ocorrencia",
        "ilesos",
        "levemente_feridos",
        "moderadamente_feridos",
        "gravemente_feridos",
        "mortos",
    }
)

_REPORT_DIR = settings.data_processed.parent.parent / "reports" / "fase0_baseline"


def features() -> list[str]:
    return FEATURES_NUMERICAS + FEATURES_CATEGORICAS


def construir_preprocessador() -> ColumnTransformer:
    """Imputa e escala numéricas; one-hot nas categóricas com corte de raras
    (`min_frequency`) para não explodir a dimensionalidade do `tipo_de_acidente`."""
    num = Pipeline([("imput", SimpleImputer(strategy="median")), ("escala", StandardScaler())])
    cat = Pipeline(
        [
            ("imput", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    min_frequency=0.005,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer([("num", num, FEATURES_NUMERICAS), ("cat", cat, FEATURES_CATEGORICAS)])


def construir_modelos() -> dict[str, Pipeline]:
    """Uma família de modelos: linear, árvore, ensemble bagging e boosting."""
    seed = settings.seed
    especificacoes = {
        "regressao_logistica": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "arvore_decisao": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed
        ),
    }
    return {
        nome: Pipeline([("prep", construir_preprocessador()), ("modelo", est)])
        for nome, est in especificacoes.items()
    }


def carregar_dados(parquet: Path | None = None, amostra: int | None = None):
    """Carrega X (features) e y (alvo) do parquet consolidado."""
    parquet = parquet or (settings.data_processed / "acidentes.parquet")
    df = pd.read_parquet(parquet, columns=[*features(), ALVO])
    if amostra is not None and amostra < len(df):
        df = df.sample(n=amostra, random_state=settings.seed)
    return df[features()], df[ALVO].astype(int)


def _metricas(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "matriz_confusao": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def avaliar(amostra: int | None = None, cv_amostra: int = 150_000) -> dict:
    """Treina e avalia todos os modelos (holdout estratificado 80/20 + CV de
    ROC-AUC numa subamostra para bounded runtime). Retorna o dicionário de
    resultados e salva relatório + figuras."""
    X, y = carregar_dados(amostra=amostra)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=settings.seed
    )
    print(f"treino: {len(X_tr):,} | teste: {len(X_te):,} | prevalência y=1: {y.mean():.3f}")

    modelos = construir_modelos()
    resultados: dict[str, dict] = {}
    curvas = {}
    for nome, pipe in modelos.items():
        t0 = perf_counter()
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = _metricas(y_te, y_pred, y_prob)

        # CV de ROC-AUC numa subamostra do treino (estabilidade, sem custo de 1M).
        n_cv = min(cv_amostra, len(X_tr))
        Xc, yc = X_tr.iloc[:n_cv], y_tr.iloc[:n_cv]
        cv = cross_val_score(
            pipe,
            Xc,
            yc,
            cv=StratifiedKFold(5, shuffle=True, random_state=settings.seed),
            scoring="roc_auc",
            n_jobs=-1,
        )
        m["cv_roc_auc_media"] = float(cv.mean())
        m["cv_roc_auc_desvio"] = float(cv.std())
        m["cv_n"] = int(n_cv)
        m["segundos"] = round(perf_counter() - t0, 1)
        resultados[nome] = m
        curvas[nome] = (roc_curve(y_te, y_prob), precision_recall_curve(y_te, y_prob))
        print(
            f"  {nome:24} ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} "
            f"F1={m['f1']:.3f} bal_acc={m['balanced_accuracy']:.3f} ({m['segundos']}s)"
        )

    melhor = max(resultados, key=lambda k: resultados[k]["roc_auc"])
    _salvar_relatorio(resultados, melhor, len(X_tr), len(X_te), float(y.mean()))
    _plotar_curvas(curvas, y_te, resultados)
    _plotar_importancia(modelos["random_forest"])
    print(f"melhor modelo (ROC-AUC): {melhor}")
    print(f"relatório e figuras em: {_REPORT_DIR}")
    return {"melhor": melhor, "modelos": resultados}


def _salvar_relatorio(res, melhor, n_tr, n_te, prevalencia) -> None:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (_REPORT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "n_treino": n_tr,
                "n_teste": n_te,
                "prevalencia_y1": prevalencia,
                "melhor": melhor,
                "modelos": res,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    linhas = [
        "# Fase 0 — Baseline de classificação de severidade de acidentes",
        "",
        f"Alvo: `houve_vitima` (prevalência {prevalencia:.1%}). "
        f"Treino: {n_tr:,} · Teste: {n_te:,}. Métricas no conjunto de teste.",
        "",
        "| Modelo | ROC-AUC | PR-AUC | F1 | Bal.Acc | Precision | Recall | CV ROC-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for nome, m in sorted(res.items(), key=lambda kv: -kv[1]["roc_auc"]):
        estrela = " ⭐" if nome == melhor else ""
        linhas.append(
            f"| {nome}{estrela} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['f1']:.3f} | "
            f"{m['balanced_accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['cv_roc_auc_media']:.3f} ± {m['cv_roc_auc_desvio']:.3f} |"
        )
    linhas += [
        "",
        f"Melhor modelo por ROC-AUC: **{melhor}**.",
        "",
        "Gerado por `python -m rodoia.ml.classico`.",
    ]
    (_REPORT_DIR / "comparacao.md").write_text("\n".join(linhas))


def _plotar_curvas(curvas, y_te, res) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for nome, ((fpr, tpr, _), (prec, rec, _)) in curvas.items():
        ax1.plot(fpr, tpr, label=f"{nome} ({res[nome]['roc_auc']:.3f})")
        ax2.plot(rec, prec, label=f"{nome} ({res[nome]['pr_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(xlabel="FPR", ylabel="TPR", title="Curva ROC")
    ax2.axhline(float(y_te.mean()), ls="--", color="k", alpha=0.4, label="baseline")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_REPORT_DIR / "roc_pr_curvas.png", dpi=110)
    plt.close(fig)


def _plotar_importancia(pipe_rf: Pipeline, top: int = 15) -> None:
    nomes = pipe_rf.named_steps["prep"].get_feature_names_out()
    imp = pipe_rf.named_steps["modelo"].feature_importances_
    serie = pd.Series(imp, index=nomes).sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(serie.index, serie.values)
    ax.set_title("Importância de features (Random Forest, top 15)")
    fig.tight_layout()
    fig.savefig(_REPORT_DIR / "importancia_features.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline de classificação (Fase 0).")
    parser.add_argument("--amostra", type=int, default=None, help="subamostra N linhas (dev)")
    args = parser.parse_args()
    avaliar(amostra=args.amostra)


if __name__ == "__main__":
    main()
