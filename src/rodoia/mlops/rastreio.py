"""Rastreio de experimentos com MLflow (Fase 5).

Consolida as métricas já produzidas pelas Fases 0–4 (em `reports/`) num store MLflow
local (`mlruns/`), uma run por fase — versões, parâmetros e métricas ficam navegáveis
no `mlflow ui`. `coletar()` é puro/testável; `registrar()` faz o log (mlflow importado
sob demanda, para o módulo não exigir mlflow instalado em quem só usa o resto).

Uso:  python -m rodoia.mlops.rastreio    # popula ./mlruns ; depois `mlflow ui`
"""
from __future__ import annotations

import json

from rodoia.config import REPO_ROOT


def coletar(raiz=None) -> list[dict]:
    """Extrai (fase → métricas) dos relatórios versionados. Puro: sem I/O de MLflow."""
    raiz = raiz or REPO_ROOT

    def ler(rel):
        return json.loads((raiz / rel).read_text(encoding="utf-8"))

    runs = []
    m = ler("reports/fase0_mlp/mlp.json")
    runs.append({"fase": "fase0_mlp", "metrics": {
        "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "f1": m["f1"]}})

    r = ler("reports/fase1_retrieval/avaliacao_retrieval.json")["hibrido"]
    g = ler("reports/fase1_geracao/avaliacao_geracao.json")
    runs.append({"fase": "fase1_rag", "metrics": {
        "hit_at_5": r["hit_rate_at_k"], "mrr": r["mrr"],
        "faithfulness": g["faithfulness_media"], "precisao_citacao": g["precisao_citacao_media"]}})

    ner = ler("reports/fase2_ner/comparacao.json")["modelos"]
    runs.append({"fase": "fase2_ner", "metrics": {
        "f1_base": ner["base_zero_shot"]["f1_micro"], "f1_ft": ner["ft_qlora"]["f1_micro"],
        "f1_sota": ner["bertimbau_sota"]["f1_micro"]}})

    p = ler("reports/fase3_dados/previsao.json")
    runs.append({"fase": "fase3_previsao", "metrics": {
        "mape_holt_winters": p["modelos"]["holt_winters"]["mape_medio"],
        "mape_naive": p["modelos"]["naive"]["mape_medio"],
        "delta_pareado_pp": p["comparacao_pareada"]["delta_mape_medio"]}})

    a = ler("reports/fase4_agente/avaliacao.json")["resumo"]
    runs.append({"fase": "fase4_agente", "metrics": {
        "acerto_roteamento": a["acerto_roteamento"], "rota_ok": a.get("rota_ok_medio", 0),
        "resposta_ok": a.get("resposta_ok_medio", 0)}})
    return runs


def registrar(tracking_uri: str | None = None) -> int:
    """Loga cada fase como uma run MLflow. Retorna o nº de runs registradas.
    Usa backend SQLite (recomendado; o file store foi descontinuado no MLflow novo)."""
    import mlflow

    if tracking_uri is None:
        (REPO_ROOT / "mlruns").mkdir(exist_ok=True)
        tracking_uri = f"sqlite:///{REPO_ROOT / 'mlruns' / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("rodoia")
    runs = coletar()
    for r in runs:
        with mlflow.start_run(run_name=r["fase"]):
            mlflow.log_param("fase", r["fase"])
            mlflow.log_metrics(r["metrics"])
    print(f"{len(runs)} runs registradas em {mlflow.get_tracking_uri()} (rode `mlflow ui`)")
    return len(runs)


if __name__ == "__main__":
    registrar()
