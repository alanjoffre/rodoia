"""Testes da leitura/validação dos CSVs de Acidentes.

Sem rede: um fixture minúsculo reproduz as pegadinhas reais do dado da ANTT
(encoding latin-1, decimal com vírgula, aspas órfãs em `tipo_de_acidente`,
acento em `trecho`). Testa o caminho crítico de ingestão da Fase 0.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rodoia.data.esquema_acidentes import (
    COLUNAS_ESPERADAS,
    ErroDeEsquema,
    ler_csv_acidentes,
    validar_esquema,
)

# Cabeçalho canônico + 2 linhas: uma com aspa órfã (submarino) e km decimal,
# outra "com vítima" com mortos>0.
_CABECALHO = ";".join(COLUNAS_ESPERADAS)
_LINHAS = [
    '"01/01/2010";"03:02:00";"20";"sem vítima";"506,5";"BR-381/MG";"Sul";'
    '"Choque - Defensa, barreira ou "submarino"";1;0;0;0;0;0;0;0;0;0;1;0;0;0;0',
    '"02/01/2010";"06:16:00";"39";"com vítima";"767";"BR-381/MG";"Norte";'
    '"Saida de Pista";0;0;1;0;0;0;0;0;0;0;0;0;0;2;1',
]


@pytest.fixture
def csv_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "acidentes_fixture.csv"
    p.write_text("\n".join([_CABECALHO, *_LINHAS]) + "\n", encoding="latin-1")
    return p


def test_le_e_valida_schema(csv_fixture: Path) -> None:
    df = ler_csv_acidentes(csv_fixture)
    assert tuple(df.columns) == COLUNAS_ESPERADAS
    assert len(df) == 2


def test_limpa_aspas_orfas(csv_fixture: Path) -> None:
    df = ler_csv_acidentes(csv_fixture)
    valor = df.loc[0, "tipo_de_acidente"]
    assert '"' not in valor
    assert valor == "Choque - Defensa, barreira ou submarino"


def test_encoding_latin1_preserva_acentos(csv_fixture: Path) -> None:
    df = ler_csv_acidentes(csv_fixture)
    assert df.loc[0, "tipo_de_ocorrencia"] == "sem vítima"


def test_km_decimal_virgula_vira_float(csv_fixture: Path) -> None:
    df = ler_csv_acidentes(csv_fixture)
    assert df.loc[0, "km"] == pytest.approx(506.5)
    assert df.loc[1, "km"] == pytest.approx(767.0)


def test_colunas_de_vitimas_sao_inteiras(csv_fixture: Path) -> None:
    df = ler_csv_acidentes(csv_fixture)
    assert df.loc[1, "mortos"] == 1
    assert df.loc[1, "gravemente_feridos"] == 2
    assert str(df["mortos"].dtype) == "Int64"


def test_esquema_divergente_e_rejeitado() -> None:
    df_errado = pd.DataFrame({"data": ["x"], "coluna_estranha": ["y"]})
    with pytest.raises(ErroDeEsquema):
        validar_esquema(df_errado)
