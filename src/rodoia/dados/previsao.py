"""Previsão de demanda de tráfego (Fase 3) — o resultado objetivo (RMSE/MAPE).

Prevê o volume mensal de uma praça de pedágio com histórico longo e contínuo (série
univariada limpa — evita o viés de cobertura crescente do total da rede). Compara
**baselines** (naïve e sazonal-naïve) com um **ML** (Gradient Boosting sobre features de
lag/sazonais), num **split temporal** (últimos 12 meses = teste, sem vazamento). Métrica
dura: **RMSE e MAPE**.

Uso:  python -m rodoia.dados.previsao
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rodoia.config import REPO_ROOT, settings
from rodoia.dados.estrela import DB
from rodoia.proveniencia import carimbar

N_TESTE = 12  # horizonte de avaliação (meses)


def serie_praca_mais_longa() -> tuple[str, pd.Series]:
    """Volume mensal total da praça com o histórico mais longo e contínuo."""
    import duckdb

    con = duckdb.connect(str(DB), read_only=True)
    try:
        alvo = con.execute("""
            SELECT p.praca_id, p.praca, count(*) n
            FROM (SELECT praca_id, data, sum(volume_total) v FROM fato_volume GROUP BY praca_id, data) s
            JOIN dim_praca p USING (praca_id)
            GROUP BY p.praca_id, p.praca ORDER BY n DESC LIMIT 1
        """).fetchone()
        df = con.execute("""
            SELECT data, sum(volume_total) AS volume
            FROM fato_volume WHERE praca_id = ? GROUP BY data ORDER BY data
        """, [alvo[0]]).df()
    finally:
        con.close()
    s = df.set_index("data")["volume"].asfreq("MS")  # mensal, marca meses faltantes
    return alvo[1], s.interpolate(limit_direction="both")


def _features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    for lag in (1, 2, 3, 12):
        df[f"lag{lag}"] = s.shift(lag)
    df["media3"] = s.shift(1).rolling(3).mean()
    df["media12"] = s.shift(1).rolling(12).mean()
    df["mes"] = df.index.month
    df["t"] = np.arange(len(df))
    return df.dropna()


def _metricas(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"rmse": round(rmse), "mape_pct": round(mape, 2)}


def avaliar() -> dict:
    from sklearn.ensemble import GradientBoostingRegressor

    praca, s = serie_praca_mais_longa()
    df = _features(s)
    treino, teste = df.iloc[:-N_TESTE], df.iloc[-N_TESTE:]
    cols = [c for c in df.columns if c != "y"]

    modelo = GradientBoostingRegressor(random_state=settings.seed)
    modelo.fit(treino[cols], treino["y"])
    pred_ml = modelo.predict(teste[cols])

    # baselines no mesmo horizonte de teste
    naive = s.shift(1).reindex(teste.index)      # último mês
    sazonal = s.shift(12).reindex(teste.index)   # mesmo mês do ano anterior

    res = carimbar({
        "praca": praca, "n_meses": len(s), "n_teste": N_TESTE,
        "periodo": [str(s.index.min().date()), str(s.index.max().date())],
        "modelos": {
            "naive": _metricas(teste["y"], naive),
            "sazonal_naive": _metricas(teste["y"], sazonal),
            "gradient_boosting": _metricas(teste["y"], pred_ml),
        },
    })
    _plotar(s, teste.index, pred_ml, praca)
    saida = REPO_ROOT / "reports" / "fase3_dados"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "previsao.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    m = res["modelos"]
    print(f"praça: {praca} ({len(s)} meses)")
    for nome, v in m.items():
        print(f"  {nome:18} RMSE={v['rmse']:>12,} | MAPE={v['mape_pct']}%")
    print(f"relatório: {saida / 'previsao.json'}")
    return res


def _plotar(s, idx_teste, pred_ml, praca) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(s.index, s.values, label="real", lw=1)
    ax.plot(idx_teste, pred_ml, "o-", label="previsão (GB)", color="crimson")
    ax.set(title=f"Previsão de volume mensal — {praca[:40]}", xlabel="mês", ylabel="volume")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "reports" / "fase3_dados" / "previsao.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    avaliar()
