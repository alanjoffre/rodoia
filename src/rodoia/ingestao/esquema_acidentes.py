"""Schema canônico e leitura robusta dos CSVs de Acidentes da ANTT.

Realidade do dado (confirmada no arquivo real — ver docs/00_validacao_fontes_antt.md):
- Encoding ISO-8859-1 (latin-1), separador ';', decimal com vírgula.
- 23 colunas, consistentes entre as 39 concessionárias.
- Aspas órfãs mal-formadas em `tipo_de_acidente` (ex.: `... ou submarino""`) que
  vazam para o valor — precisam de limpeza.
- 1 linha = 1 acidente (dado por instância, apto a ML de classificação).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Schema canônico — a ordem e os nomes exatos do cabeçalho oficial.
COLUNAS_ESPERADAS: tuple[str, ...] = (
    "data",
    "horario",
    "n_da_ocorrencia",
    "tipo_de_ocorrencia",
    "km",
    "trecho",
    "sentido",
    "tipo_de_acidente",
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
    "ilesos",
    "levemente_feridos",
    "moderadamente_feridos",
    "gravemente_feridos",
    "mortos",
)

# Contagens de veículos envolvidos (numéricas).
COLUNAS_VEICULOS: tuple[str, ...] = (
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
)

# Contagens de pessoas por gravidade (numéricas) — base para derivar o alvo.
COLUNAS_VITIMAS: tuple[str, ...] = (
    "ilesos",
    "levemente_feridos",
    "moderadamente_feridos",
    "gravemente_feridos",
    "mortos",
)

# Colunas de texto que podem conter aspas órfãs a limpar.
_COLUNAS_TEXTO = ("tipo_de_ocorrencia", "trecho", "sentido", "tipo_de_acidente")


class ErroDeEsquema(ValueError):
    """Levantado quando um CSV não bate com o schema canônico."""


def validar_esquema(df: pd.DataFrame) -> None:
    """Garante que o DataFrame tem exatamente as colunas canônicas, na ordem."""
    cols = tuple(df.columns)
    if cols != COLUNAS_ESPERADAS:
        faltando = set(COLUNAS_ESPERADAS) - set(cols)
        sobrando = set(cols) - set(COLUNAS_ESPERADAS)
        raise ErroDeEsquema(
            f"Schema divergente. Faltando={sorted(faltando)} Sobrando={sorted(sobrando)}"
        )


def _limpar_texto(serie: pd.Series) -> pd.Series:
    """Remove aspas órfãs e espaços das bordas (ex.: `submarino\"\"` -> `submarino`)."""
    return serie.str.replace('"', "", regex=False).str.strip()


def ler_csv_acidentes(caminho: str | Path, *, validar: bool = True) -> pd.DataFrame:
    """Lê um CSV de acidentes tratando encoding, decimal e aspas órfãs.

    Colunas numéricas são convertidas para inteiro; `km` para float; `data`/`horario`
    ficam como texto (a normalização temporal é feita na ingestão/EDA).
    """
    df = pd.read_csv(
        caminho,
        sep=";",
        encoding="latin-1",
        decimal=",",
        dtype=str,
        keep_default_na=False,
    )
    if validar:
        validar_esquema(df)

    for col in _COLUNAS_TEXTO:
        df[col] = _limpar_texto(df[col])

    for col in (*COLUNAS_VEICULOS, *COLUNAS_VITIMAS):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["km"] = pd.to_numeric(df["km"].str.replace(",", ".", regex=False), errors="coerce")

    return df
