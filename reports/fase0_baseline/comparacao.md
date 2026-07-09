# Fase 0 — Baseline de classificação de severidade de acidentes

Alvo: `houve_vitima` (prevalência 35.9%). Treino: 824,870 · Teste: 206,218. Métricas no conjunto de teste.

| Modelo | ROC-AUC | PR-AUC | F1 | Bal.Acc | Precision | Recall | CV ROC-AUC |
|---|---|---|---|---|---|---|---|
| hist_gradient_boosting ⭐ | 0.813 | 0.736 | 0.672 | 0.744 | 0.670 | 0.674 | 0.810 ± 0.002 |
| random_forest | 0.813 | 0.735 | 0.668 | 0.743 | 0.703 | 0.636 | 0.808 ± 0.002 |
| regressao_logistica | 0.791 | 0.698 | 0.654 | 0.731 | 0.661 | 0.647 | 0.791 ± 0.002 |
| arvore_decisao | 0.782 | 0.688 | 0.657 | 0.735 | 0.701 | 0.619 | 0.769 ± 0.002 |

Melhor modelo por ROC-AUC: **hist_gradient_boosting**.

Gerado por `python -m rodoia.ml.classico`.
