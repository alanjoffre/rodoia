"""Ingestão do Volume de Tráfego de Pedágio (Fase 3) — CSVs anuais → parquet limpo.

Consolida os CSVs (ISO-8859-1, `;`, decimal `,`) num único parquet tipado, com a data
mensal derivada de `mes_ano`. Valida o schema e reporta linhas/rejeições (observabilidade
de ingestão). Grão da linha: concessionária × praça × mês × sentido × cobrança × categoria
× tipo de veículo → volume_total.

Uso:  python -m rodoia.data.ingestao_volume
"""
from __future__ import annotations

import pandas as pd

from rodoia.config import settings

_DIR = settings.data_raw / "volume_pedagio"
COLUNAS = [
    "concessionaria", "mes_ano", "sentido", "praca",
    "tipo_cobranca", "categoria", "tipo_de_veiculo", "volume_total",
]


def _ler_csv(caminho) -> pd.DataFrame:
    df = pd.read_csv(caminho, sep=";", encoding="latin-1", dtype=str, keep_default_na=False)
    # arquivos consolidados (2024+) usam 'categoria_eixo' no lugar de 'categoria'
    if "categoria_eixo" in df.columns and "categoria" not in df.columns:
        df = df.rename(columns={"categoria_eixo": "categoria"})
    return df


def _para_data(serie: pd.Series) -> pd.Series:
    """Normaliza mes_ano: 'DD/MM/AAAA' (anuais) e 'MM/AAAA' (consolidados) → 1º do mês."""
    s = serie.str.strip()
    curta = s.str.match(r"^\d{1,2}/\d{4}$")  # MM/AAAA
    norm = s.where(~curta, "01/" + s)         # vira DD/MM/AAAA
    return pd.to_datetime(norm, format="%d/%m/%Y", errors="coerce")


def consolidar(destino=None) -> dict:
    destino = destino or (settings.data_processed / "volume_pedagio.parquet")
    arquivos = sorted(_DIR.glob("*.csv"))
    if not arquivos:
        raise SystemExit(f"Sem CSVs em {_DIR} — rode `python -m rodoia.data.baixar_volume` antes.")

    partes, lidas, rejeitadas = [], 0, 0
    for arq in arquivos:
        df = _ler_csv(arq)
        lidas += len(df)
        faltando = set(COLUNAS) - set(df.columns)
        if faltando:
            print(f"  {arq.name}: colunas faltando {faltando} — pulado")
            rejeitadas += len(df)
            continue
        df = df[COLUNAS].copy()
        for c in ("concessionaria", "sentido", "praca", "tipo_cobranca",
                  "categoria", "tipo_de_veiculo"):
            df[c] = df[c].str.strip()
        # normaliza caixa das categóricas (a fonte mistura 'Passeio'/'PASSEIO' etc.)
        for c in ("sentido", "tipo_cobranca", "categoria", "tipo_de_veiculo"):
            df[c] = df[c].str.upper()
        df["data"] = _para_data(df["mes_ano"])
        df["volume_total"] = pd.to_numeric(df["volume_total"].str.replace(".", "", regex=False)
                                           .str.replace(",", ".", regex=False), errors="coerce")
        antes = len(df)
        df = df[df["data"].notna() & df["volume_total"].notna() & (df["volume_total"] >= 0)]
        rejeitadas += antes - len(df)
        df["ano"], df["mes"] = df["data"].dt.year, df["data"].dt.month
        partes.append(df.drop(columns=["mes_ano"]))

    tudo = pd.concat(partes, ignore_index=True)
    # Normaliza ao MÊS: alguns arquivos vêm diários — truncamos ao 1º do mês e somamos,
    # garantindo grão mensal consistente em toda a série (2010–2026).
    tudo["data"] = tudo["data"].values.astype("datetime64[M]")
    chaves = ["concessionaria", "praca", "sentido", "tipo_cobranca",
              "categoria", "tipo_de_veiculo", "data"]
    tudo = tudo.groupby(chaves, as_index=False)["volume_total"].sum()
    tudo["ano"], tudo["mes"] = tudo["data"].dt.year, tudo["data"].dt.month
    destino.parent.mkdir(parents=True, exist_ok=True)
    tudo.to_parquet(destino, index=False)
    stats = {
        "linhas": len(tudo), "lidas": lidas, "rejeitadas": rejeitadas,
        "periodo": [int(tudo["ano"].min()), int(tudo["ano"].max())],
        "n_concessionarias": int(tudo["concessionaria"].nunique()),
        "n_pracas": int(tudo["praca"].nunique()),
        "volume_total_bi": round(tudo["volume_total"].sum() / 1e9, 2),
    }
    print(f"consolidado: {stats['linhas']:,} linhas ({stats['rejeitadas']:,} rejeitadas) "
          f"| {stats['periodo'][0]}–{stats['periodo'][1]} | "
          f"{stats['n_concessionarias']} concessionárias, {stats['n_pracas']} praças")
    print(f"gravado em: {destino}")
    return stats


def main() -> None:
    consolidar()


if __name__ == "__main__":
    main()
