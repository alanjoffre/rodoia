"""Previsão de demanda de tráfego (Fase 3) — o resultado objetivo, com IC.

Avaliação ROBUSTA (não uma praça só): backtest em TODAS as praças com histórico mensal
completo e longo — para cada uma, holdout dos últimos 12 meses e MAPE de cada modelo;
depois agrega com **IC por bootstrap** sobre as praças. Compara baselines (naïve,
sazonal-naïve), o clássico **Holt-Winters** (statsmodels) e um **Gradient Boosting** (lags +
médias móveis + mês). Split temporal, sem vazamento. Métrica: **MAPE** (escala-livre, agregável).

Uso:  python -m rodoia.dados.previsao
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from rodoia.config import REPO_ROOT, settings
from rodoia.dados.estrela import DB
from rodoia.estat import bootstrap_ic
from rodoia.proveniencia import carimbar

N_TESTE = 12
MIN_MESES = 100   # praças com histórico longo (≥ ~8 anos) e contíguo
MODELOS = ["naive", "sazonal_naive", "holt_winters", "gradient_boosting"]


def _series_completas(min_meses: int = MIN_MESES) -> list[tuple[str, pd.Series]]:
    """Praças com série mensal CONTÍGUA (sem buracos) e >= min_meses — sem interpolação."""
    import duckdb

    con = duckdb.connect(str(DB), read_only=True)
    try:
        cand = con.execute("""
            SELECT praca_id, count(DISTINCT data) n,
                   datediff('month', min(data), max(data)) + 1 AS span
            FROM fato_volume GROUP BY praca_id
        """).df()
        ids = cand[(cand.n == cand.span) & (cand.n >= min_meses)]["praca_id"].tolist()
        out = []
        for pid in ids:
            df = con.execute("""SELECT data, sum(volume_total) v FROM fato_volume
                                WHERE praca_id=? GROUP BY data ORDER BY data""", [pid]).df()
            nome = con.execute("SELECT praca FROM dim_praca WHERE praca_id=?", [pid]).fetchone()[0]
            out.append((nome, df.set_index("data")["v"].asfreq("MS")))
        return out
    finally:
        con.close()


def _features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    for lag in (1, 2, 3, 12):
        df[f"lag{lag}"] = s.shift(lag)
    df["media3"] = s.shift(1).rolling(3).mean()
    df["media12"] = s.shift(1).rolling(12).mean()
    df["mes"] = df.index.month
    return df.dropna()


def _mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


_COLS = ["lag1", "lag2", "lag3", "lag12", "media3", "media12", "mes"]


def _gb_recursivo(gb, hist: list[float], meses: list[int]) -> list[float]:
    """Previsão MULTI-STEP recursiva: cada mês usa só o histórico de treino + as próprias
    previsões anteriores (nunca o valor real do teste)."""
    h, preds = list(hist), []
    for mes in meses:
        feat = [[h[-1], h[-2], h[-3], h[-12],
                 float(np.mean(h[-3:])), float(np.mean(h[-12:])), mes]]
        p = float(gb.predict(pd.DataFrame(feat, columns=_COLS))[0])
        preds.append(p)
        h.append(p)
    return preds


def _prever_praca(s: pd.Series) -> dict:
    """Todos os modelos preveem 12 MESES À FRENTE a partir do fim do treino (sem ver o
    teste) — comparação justa e realista (planejamento de demanda)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    treino, y = s.iloc[:-N_TESTE], s.iloc[-N_TESTE:].values
    meses = list(s.index[-N_TESTE:].month)

    r = {
        # random walk multi-step: repete o último valor do treino
        "naive": _mape(y, np.full(N_TESTE, treino.iloc[-1])),
        # sazonal: mesmo mês do ano anterior (t-12, ainda no treino)
        "sazonal_naive": _mape(y, s.shift(12).iloc[-N_TESTE:].values),
    }
    df = _features(treino)  # features só do treino → sem vazamento
    gb = GradientBoostingRegressor(random_state=settings.seed).fit(df[_COLS], df["y"])
    r["gradient_boosting"] = _mape(y, _gb_recursivo(gb, list(treino.values), meses))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hw = ExponentialSmoothing(treino, trend="add", seasonal="add",
                                      seasonal_periods=12).fit()
        r["holt_winters"] = _mape(y, hw.forecast(N_TESTE).values)
    except Exception:
        r["holt_winters"] = None
    return r


def _melhor(agg: dict) -> str:
    return min(agg, key=lambda m: agg[m]["mape_medio"])


def avaliar() -> dict:
    series = _series_completas()
    por_modelo: dict[str, list[float]] = {m: [] for m in MODELOS}
    por_praca: list[dict] = []          # guarda o pareamento (mesma praça, todos os modelos)
    for _, s in series:
        d = _prever_praca(s)
        por_praca.append(d)
        for m, v in d.items():
            if v is not None and np.isfinite(v):
                por_modelo[m].append(v)

    agg = {m: {"mape_medio": round(float(np.mean(v)), 2),
               "mape_mediano": round(float(np.median(v)), 2),
               "ic95": bootstrap_ic(v), "n_pracas": len(v)}
           for m, v in por_modelo.items()}

    # Comparação PAREADA melhor-modelo vs naïve (mesma praça) — teste correto de significância:
    # diferença naïve−melhor por praça; se o IC95 do bootstrap não cruza 0, o ganho é significativo.
    melhor = _melhor(agg)
    deltas = [d["naive"] - d[melhor] for d in por_praca
              if d.get("naive") is not None and d.get(melhor) is not None
              and np.isfinite(d["naive"]) and np.isfinite(d[melhor])]
    ic_delta = bootstrap_ic(deltas)
    pareado = {
        "modelo": melhor, "vs": "naive",
        "delta_mape_medio": round(float(np.mean(deltas)), 2),  # pp de MAPE que o naïve perde
        "ic95_delta": ic_delta, "n_pracas": len(deltas),
        "vence_naive_em_pct_pracas": round(100 * np.mean([x > 0 for x in deltas]), 1),
        "significativo": ic_delta[0] > 0,   # IC não cruza 0 → melhor bate naïve de forma significativa
    }

    res = carimbar({
        "tarefa": "previsão de volume mensal 12 meses à frente (multi-step, backtest em múltiplas praças)",
        "n_pracas": len(series), "n_teste": N_TESTE, "min_meses": MIN_MESES,
        "metrica": "MAPE (%) — média entre praças com IC95 por bootstrap",
        "modelos": agg, "comparacao_pareada": pareado,
    })
    _plotar(max(series, key=lambda x: len(x[1])))
    saida = REPO_ROOT / "reports" / "fase3_dados"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "previsao.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"backtest 12m à frente em {len(series)} praças:")
    for m in sorted(MODELOS, key=lambda x: agg[x]["mape_medio"]):
        a = agg[m]
        print(f"  {m:18} MAPE médio={a['mape_medio']}% IC95={a['ic95']} (n={a['n_pracas']})")
    p = pareado
    print(f"pareado {p['modelo']} vs naïve: Δ={p['delta_mape_medio']}pp IC95={p['ic95_delta']} "
          f"| vence em {p['vence_naive_em_pct_pracas']}% das praças "
          f"| significativo={p['significativo']}")
    print(f"relatório: {saida / 'previsao.json'}")
    return res


def _plotar(alvo) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import GradientBoostingRegressor

    nome, s = alvo
    df = _features(s)
    treino, teste = df.iloc[:-N_TESTE], df.iloc[-N_TESTE:]
    cols = [c for c in df.columns if c != "y"]
    gb = GradientBoostingRegressor(random_state=settings.seed).fit(treino[cols], treino["y"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(s.index, s.values, label="real", lw=1)
    ax.plot(teste.index, gb.predict(teste[cols]), "o-", color="crimson", label="previsão (GB)")
    ax.set(title=f"Previsão de volume mensal — {nome[:40]}", xlabel="mês", ylabel="volume")
    ax.legend(); fig.tight_layout()
    fig.savefig(REPO_ROOT / "reports" / "fase3_dados" / "previsao.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    avaliar()
