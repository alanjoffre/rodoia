"""Ingestão consolidada dos Acidentes da ANTT: junta os 39 CSVs (37 concessionárias),
deriva o alvo de severidade e faz engenharia de features, gravando um único
`data/processed/acidentes.parquet`.

Decisões de modelagem (ver docs/00 e docs/02):
- **Alvo derivado das colunas numéricas de vítimas**, não do rótulo textual
  `tipo_de_ocorrencia` (que é inconsistente entre as fontes).
- **Sem vazamento (leakage):** as colunas de vítimas (`ilesos`, `*_feridos`,
  `mortos`) e o `tipo_de_ocorrencia` DEFINEM o alvo → não podem virar feature.
  Ficam no parquet apenas para auditoria; a lista de features do modelo as exclui.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from rodoia.config import settings
from rodoia.data.esquema_acidentes import COLUNAS_VEICULOS, ler_csv_acidentes

# Alvo é derivado destas (não entram como feature).
_COLUNAS_FERIDOS = ("levemente_feridos", "moderadamente_feridos", "gravemente_feridos")
_RE_UF = re.compile(r"/([A-Z]{2})\b")


def derivar_alvo(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona `n_feridos`, `houve_vitima` e `houve_fatal` a partir das
    contagens numéricas de vítimas (consistentes entre fontes)."""
    feridos = df[list(_COLUNAS_FERIDOS)].fillna(0).sum(axis=1)
    mortos = df["mortos"].fillna(0)
    out = df.copy()
    out["n_feridos"] = feridos.astype("int64")
    out["houve_vitima"] = ((feridos + mortos) > 0).astype("int64")
    out["houve_fatal"] = (mortos > 0).astype("int64")
    return out


def _extrair_uf(trecho: pd.Series) -> pd.Series:
    """UF a partir do `trecho` (ex.: 'BR-381/MG' -> 'MG'); 'NA' se não casar."""
    return trecho.fillna("").str.extract(_RE_UF, expand=False).fillna("NA")


def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva features temporais/geográficas conhecidas no momento do acidente
    (sem usar nada que revele a severidade)."""
    out = df.copy()

    data = pd.to_datetime(out["data"], format="%d/%m/%Y", errors="coerce")
    out["ano"] = data.dt.year.astype("Int64")
    out["mes"] = data.dt.month.astype("Int64")
    out["dia_semana"] = data.dt.dayofweek.astype("Int64")  # 0=segunda

    hora = pd.to_datetime(out["horario"], format="%H:%M:%S", errors="coerce").dt.hour
    out["hora"] = hora.astype("Int64")

    out["uf"] = _extrair_uf(out["trecho"])
    out["total_veiculos"] = df[list(COLUNAS_VEICULOS)].fillna(0).sum(axis=1).astype("int64")

    # Normaliza caixa/espaços das categóricas de texto: o dado tem duplicatas por
    # capitalização (ex.: 'Colisão Traseira' vs 'colisão traseira') que inflam a
    # cardinalidade sem significado. Reduz ruído para o encoding do modelo.
    out["tipo_de_acidente"] = out["tipo_de_acidente"].str.lower().str.strip()
    out["sentido"] = out["sentido"].str.lower().str.strip()
    return out


def consolidar(
    raw_dir: Path | None = None,
    saida: Path | None = None,
) -> pd.DataFrame:
    """Lê todos os CSVs de acidentes, aplica alvo + features e grava o parquet.

    A concessionária vem do nome do arquivo (código estável, ex.: 'afd', 'arb').
    """
    raw_dir = raw_dir or (settings.data_raw / "acidentes")
    saida = saida or (settings.data_processed / "acidentes.parquet")

    arquivos = sorted(raw_dir.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV em {raw_dir} — rode baixar_acidentes primeiro.")

    frames = []
    for arq in arquivos:
        parte = ler_csv_acidentes(arq)
        # 'demostrativo_acidentes_afd__359e38d3' -> 'afd'
        codigo = arq.stem.split("__")[0].replace("demostrativo_acidentes_", "")
        parte["concessionaria"] = codigo
        frames.append(parte)

    df = pd.concat(frames, ignore_index=True)
    df = derivar_alvo(df)
    df = engenharia_features(df)

    saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(saida, index=False)
    return df


def main() -> None:
    df = consolidar()
    saida = settings.data_processed / "acidentes.parquet"
    n_conc = df["concessionaria"].nunique()
    com_vitima = df["houve_vitima"].mean() * 100
    fatal = df["houve_fatal"].mean() * 100
    print(f"consolidado: {len(df):,} acidentes de {n_conc} concessionárias")
    print(f"  com vítima: {com_vitima:.1f}% | fatal: {fatal:.2f}%")
    print(f"  período: {int(df['ano'].min())}–{int(df['ano'].max())}")
    print(f"gravado em: {saida}")


if __name__ == "__main__":
    main()
