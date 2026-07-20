"""Monitoramento de drift (Fase 5).

Mede desvio de distribuição entre um período BASE e um período RECENTE via **PSI**
(Population Stability Index) — a métrica padrão de drift em produção. Regras usuais:
PSI < 0,1 estável · 0,1–0,25 desvio moderado · > 0,25 desvio relevante (revisar).

Aplicado ao volume de tráfego (dados da Fase 3): compara a distribuição do volume
mensal POR PRAÇA no último ano vs. o ano anterior, restrito à **mesma coorte** de praças
(presentes nos dois períodos). Comparar coortes iguais em anos adjacentes isola o *drift
de demanda* do crescimento da malha (novas praças) — que inflaria o PSI artificialmente.

`psi()` é puro/testável. `drift_volume()` lê o DuckDB da Fase 3.

Uso:  python -m rodoia.mlops.drift    # -> reports/fase5_mlops/drift.json
"""
from __future__ import annotations

import json

import numpy as np

from rodoia.config import REPO_ROOT


def psi(base, recente, n_bins: int = 10) -> float:
    """PSI entre duas amostras. Bins por quantis da BASE (com epsilon p/ evitar log(0))."""
    base = np.asarray(base, dtype=float)
    recente = np.asarray(recente, dtype=float)
    bordas = np.quantile(base, np.linspace(0, 1, n_bins + 1))
    bordas[0], bordas[-1] = -np.inf, np.inf
    b = np.histogram(base, bins=bordas)[0] / len(base)
    r = np.histogram(recente, bins=bordas)[0] / len(recente)
    eps = 1e-6
    b, r = np.clip(b, eps, None), np.clip(r, eps, None)
    return float(np.sum((r - b) * np.log(r / b)))


def classificar(valor: float) -> str:
    return "estável" if valor < 0.1 else "moderado" if valor < 0.25 else "relevante"


def drift_volume(db=None) -> dict:
    """PSI do volume mensal POR PRAÇA: últimos 12 meses vs. os 12 anteriores, na mesma coorte."""
    import duckdb

    from rodoia.dominio.estrela import DB
    from rodoia.proveniencia import carimbar

    con = duckdb.connect(str(db or DB), read_only=True)
    try:
        df = con.execute("""
            SELECT praca_id, data, sum(volume_total) AS v
            FROM fato_volume GROUP BY praca_id, data
        """).df()
    finally:
        con.close()

    meses = np.sort(df["data"].unique())
    if len(meses) < 24:
        raise SystemExit("histórico < 24 meses — sem janelas para comparar drift.")
    ini_rec, ini_base = meses[-12], meses[-24]              # janelas de 12 meses adjacentes
    rec = df[df["data"] >= ini_rec]
    base = df[(df["data"] >= ini_base) & (df["data"] < ini_rec)]
    coorte = set(rec["praca_id"]) & set(base["praca_id"])   # praças presentes nos dois anos
    b = base[base["praca_id"].isin(coorte)]["v"].to_numpy()
    r = rec[rec["praca_id"].isin(coorte)]["v"].to_numpy()
    valor = psi(b, r)
    res = carimbar({
        "metrica": "PSI do volume mensal por praça (coorte comum), 12m recentes vs. 12m anteriores",
        "n_pracas_coorte": len(coorte), "n_base": len(b), "n_recente": len(r),
        "psi": round(valor, 4), "classificacao": classificar(valor),
        "acao": "re-treinar previsão" if valor >= 0.25 else "monitorar",
    })
    saida = REPO_ROOT / "reports" / "fase5_mlops"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "drift.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"PSI={res['psi']} ({res['classificacao']}) -> {res['acao']}")
    return res


def main() -> None:
    drift_volume()


if __name__ == "__main__":
    main()
