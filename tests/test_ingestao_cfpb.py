"""Testes da ingestão em escala do CFPB (sem rede: o zip é sintetizado aqui).

Cobre os dois caminhos que quebram silenciosamente em produção:
- `_extrair_ano` com os DOIS formatos de data em circulação (bulk ISO vs
  snapshot MM/DD/YYYY do Kaggle);
- o inflater de streaming + escritor particionado, ponta a ponta num zip real.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from rodoia.ingestao.ingestao_cfpb import COLUNAS, _data_iso, _extrair_ano, ingerir_cfpb


def _zip_sintetico(destino: Path, linhas: list[list[str]]) -> Path:
    """Escreve um complaints.csv.zip com o schema real e as linhas dadas."""
    buffer = io.StringIO()
    escritor = csv.writer(buffer)
    escritor.writerow(
        [
            "Date received",
            "Product",
            "Sub-product",
            "Issue",
            "Sub-issue",
            "Consumer complaint narrative",
            "Company public response",
            "Company",
            "State",
            "ZIP code",
            "Tags",
            "Submitted via",
            "Date sent to company",
            "Company response to consumer",
            "Timely response?",
            "Complaint ID",
        ]
    )
    escritor.writerows(linhas)
    caminho = destino / "complaints.csv.zip"
    with zipfile.ZipFile(caminho, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("complaints.csv", buffer.getvalue())
    return caminho


def _linha(data: str, narrativa: str = "", ident: str = "1") -> list[str]:
    linha = [""] * len(COLUNAS)
    linha[0] = data
    linha[1] = "Mortgage"
    linha[5] = narrativa
    linha[15] = ident
    return linha


def test_extrair_ano_formato_iso() -> None:
    """Bulk vivo da CFPB usa YYYY-MM-DD."""
    assert _extrair_ano("2026-07-24") == "2026"


def test_extrair_ano_formato_kaggle() -> None:
    """Snapshot de 2018 no Kaggle usa MM/DD/YYYY — o ano está no fim."""
    assert _extrair_ano("12/31/2018") == "2018"


def test_extrair_ano_timestamp_iso() -> None:
    """O bulk mistura data pura com timestamp completo nas linhas recentes —
    medido no snapshot de 2026-07-24."""
    assert _extrair_ano("2026-07-24T09:08:01.000Z") == "2026"


def test_extrair_ano_sujo_nao_derruba() -> None:
    """Célula suja vira 'desconhecido' em vez de exceção: uma linha ruim não
    pode abortar uma ingestão de 17M."""
    assert _extrair_ano("") == "desconhecido"
    assert _extrair_ano("n/d") == "desconhecido"


def test_data_iso_normaliza_os_tres_formatos() -> None:
    """Sem normalizar, `min`/`max` compara representações diferentes por ordem
    lexicográfica e reporta um período que não existe."""
    assert _data_iso("2024-01-15") == "2024-01-15"
    assert _data_iso("2026-07-24T09:08:01.000Z") == "2026-07-24"
    assert _data_iso("12/31/2018") == "2018-12-31"
    assert _data_iso("") == ""


def test_periodo_nao_mistura_representacoes(tmp_path: Path) -> None:
    """Data pura e timestamp na mesma coluna precisam sair comparáveis."""
    zip_path = _zip_sintetico(
        tmp_path,
        [_linha("2024-01-15", "", "1"), _linha("2026-07-24T09:08:01.000Z", "", "2")],
    )
    estat = ingerir_cfpb(zip_path=zip_path, saida=tmp_path / "cfpb")
    assert estat["periodo"] == {"min": "2024-01-15", "max": "2026-07-24"}


def test_ingestao_conta_e_particiona(tmp_path: Path) -> None:
    zip_path = _zip_sintetico(
        tmp_path,
        [
            _linha("2024-01-15", "reclamo que...", "1"),
            _linha("2024-06-30", "", "2"),
            _linha("2025-03-01", "outra narrativa", "3"),
        ],
    )
    saida = tmp_path / "cfpb"
    estat = ingerir_cfpb(zip_path=zip_path, saida=saida)

    assert estat["linhas_total"] == 3
    assert estat["com_narrativa"] == 2
    assert estat["particoes"] == 2
    assert estat["por_ano"]["2024"] == {"linhas": 2, "com_narrativa": 1}
    assert estat["periodo"] == {"min": "2024-01-15", "max": "2025-03-01"}


def test_ingestao_grava_parquet_legivel(tmp_path: Path) -> None:
    """O Parquet particionado precisa ser relido com as 16 colunas declaradas —
    é o que DuckDB e Spark vão consumir no benchmark de motor.

    NÃO se afirma igualdade exata do schema: o arquivo contém só as 16 colunas,
    mas o *leitor* pode materializar a chave de partição `ano` a partir do
    layout Hive — pyarrow 24 faz isso, pyarrow 25 não. Fixar a lista exata
    tornaria o teste dependente da versão (mesmo padrão do numpy 2.3.5 vs 2.5.1
    em docs/16 §2.1). O contrato é: as 16 colunas existem e os valores batem.
    """
    zip_path = _zip_sintetico(tmp_path, [_linha("2024-01-15", "texto", "42")])
    saida = tmp_path / "cfpb"
    ingerir_cfpb(zip_path=zip_path, saida=saida)

    parquet = saida / "ano=2024" / "parte-0000.parquet"
    assert parquet.exists()
    tabela = pq.read_table(parquet)
    assert tabela.num_rows == 1
    assert set(COLUNAS) <= set(tabela.schema.names)
    assert tabela.column("complaint_id")[0].as_py() == "42"
    assert tabela.column("consumer_complaint_narrative")[0].as_py() == "texto"


def test_parquet_no_disco_tem_so_as_colunas_declaradas(tmp_path: Path) -> None:
    """O `ano` NÃO é gravado no arquivo — vive no nome do diretório. Ler só o
    schema (sem descoberta de partição) prova isso em qualquer versão."""
    zip_path = _zip_sintetico(tmp_path, [_linha("2024-01-15", "texto", "42")])
    saida = tmp_path / "cfpb"
    ingerir_cfpb(zip_path=zip_path, saida=saida)

    schema = pq.read_schema(saida / "ano=2024" / "parte-0000.parquet")
    assert schema.names == list(COLUNAS)


def test_ingestao_respeita_limite(tmp_path: Path) -> None:
    linhas = [_linha(f"2024-01-0{i}", "", str(i)) for i in range(1, 6)]
    zip_path = _zip_sintetico(tmp_path, linhas)
    estat = ingerir_cfpb(zip_path=zip_path, saida=tmp_path / "cfpb", limite=2)
    assert estat["linhas_total"] == 2
    assert estat["limite_aplicado"] == 2


def test_ingestao_sem_zip_falha_claro(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="baixar_cfpb"):
        ingerir_cfpb(zip_path=tmp_path / "ausente.zip", saida=tmp_path / "cfpb")


def test_schema_divergente_e_recusado(tmp_path: Path) -> None:
    """Se a CFPB mudar o número de colunas, a ingestão precisa parar alto —
    não gravar Parquet desalinhado em silêncio."""
    caminho = tmp_path / "complaints.csv.zip"
    with zipfile.ZipFile(caminho, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("complaints.csv", "a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="schema inesperado"):
        ingerir_cfpb(zip_path=caminho, saida=tmp_path / "cfpb")
