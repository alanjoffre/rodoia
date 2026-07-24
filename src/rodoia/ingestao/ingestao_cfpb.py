"""Ingestão em escala do bulk da CFPB: 1,43 GB de zip -> ~13,5 GB de CSV ->
Parquet particionado por ano, **sem materializar o CSV em disco**.

Por que streaming: o CSV descomprimido não cabe confortavelmente no VHDX do WSL
e não há motivo para gravá-lo — o zip é lido, inflado e convertido em lotes,
com memória limitada por `LINHAS_POR_LOTE`. Custo medido: ~20 min ponta a ponta
na Nitro (download + parse), contra ~13,5 GB de escrita evitada.

Por que particionar por ano (layout Hive `ano=YYYY/`): a Fase 6 compara motores
(DuckDB vs Spark) no mesmo conjunto de queries, e ambos fazem *partition
pruning* nesse layout. Também isola o corte de 2025 exigido pela análise de
texto (ver abaixo).

**Três armadilhas do dado, verificadas antes de escrever este módulo:**

1. **A narrativa só existe em ~22% das linhas** (a CFPB publica apenas com
   consentimento do consumidor). O ano corrente fica em ~2% porque a
   publicação depende da resposta da empresa — **não use o ano corrente para
   trabalho de texto**; corte em 2025.
2. **O formato de data difere por fonte.** O bulk vivo usa ISO `YYYY-MM-DD`; o
   snapshot de 2018 no Kaggle usa `MM/DD/YYYY`. `_extrair_ano` cobre os dois.
3. **A taxonomia de `product` tem variantes sobrepostas** ("Credit reporting",
   "Credit reporting, credit repair services...", "Credit reporting or other
   personal consumer reports") — legado de mudanças de formulário. Aqui NÃO
   harmonizamos: gravamos o valor cru e deixamos a harmonização explícita na
   camada de domínio, para não esconder o drift de rótulo dentro da ingestão.

Uso:
    python -m rodoia.ingestao.ingestao_cfpb              # ingere tudo
    python -m rodoia.ingestao.ingestao_cfpb --limite 500000   # amostra p/ dev
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq

from rodoia.config import settings
from rodoia.proveniencia import carimbar

# Colunas do bulk, na ordem, normalizadas para snake_case. Mantemos a semântica
# em inglês (fonte é americana) em vez de traduzir: o rename data/->ingestao/
# existiu justamente para acabar com a mistura inglês/português no código.
COLUNAS = (
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "consumer_complaint_narrative",
    "company_public_response",
    "company",
    "state",
    "zip_code",
    "tags",
    "submitted_via",
    "date_sent_to_company",
    "company_response_to_consumer",
    "timely_response",
    "complaint_id",
)

# Lote de escrita: limita o pico de memória. 50k linhas x 16 colunas de texto
# (narrativa média ~1 KB) fica na casa de dezenas de MB por ano aberto.
LINHAS_POR_LOTE = 50_000

_ESQUEMA = pa.schema([(nome, pa.string()) for nome in COLUNAS])

csv.field_size_limit(10_000_000)


class _FluxoZip(io.RawIOBase):
    """Lê o zip por streaming e expõe o CSV **inflado** como um binário.

    O bulk é ZIP64: os campos de tamanho do cabeçalho local vêm com o sentinela
    0xFFFFFFFF, então não dá para confiar neles — por isso inflamos com
    `zlib.decompressobj(-15)` até o fim do fluxo em vez de contar bytes.
    """

    def __init__(self, resposta: BinaryIO) -> None:
        self._resp = resposta
        self._buf = b""
        self._fim = False
        self.bytes_lidos = 0

        cabecalho = self._ler_cru(30)
        assinatura, _, _, metodo, _, _, _, _, _, n_nome, n_extra = struct.unpack(
            "<IHHHHHIIIHH", cabecalho
        )
        if assinatura != 0x04034B50:
            raise ValueError(f"não parece um zip (assinatura {assinatura:#x})")
        self.nome_interno = self._ler_cru(n_nome).decode("utf-8", "replace")
        if n_extra:
            self._ler_cru(n_extra)
        if metodo != 8:
            raise ValueError(f"método de compressão {metodo} não suportado (esperado deflate)")
        self._dec = zlib.decompressobj(-15)

    def _ler_cru(self, n: int) -> bytes:
        saida = b""
        while len(saida) < n:
            pedaco = self._resp.read(n - len(saida))
            if not pedaco:
                break
            self.bytes_lidos += len(pedaco)
            saida += pedaco
        return saida

    def readable(self) -> bool:
        return True

    def readinto(self, destino: Any) -> int:  # noqa: ANN401 - assinatura de RawIOBase
        while not self._buf and not self._fim:
            pedaco = self._resp.read(1 << 20)
            if not pedaco:
                self._fim = True
                self._buf += self._dec.flush()
                break
            self.bytes_lidos += len(pedaco)
            self._buf += self._dec.decompress(pedaco)
        n = min(len(destino), len(self._buf))
        destino[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def _data_iso(data: str) -> str:
    """Normaliza para `YYYY-MM-DD`, cobrindo os TRÊS formatos em circulação.

    Medido no snapshot de 2026-07-24: o bulk mistura `YYYY-MM-DD` com
    **timestamp ISO completo** (`YYYY-MM-DDTHH:MM:SS.000Z`) nas linhas mais
    recentes; o snapshot de 2018 no Kaggle usa `MM/DD/YYYY`. Sem normalizar,
    o `min`/`max` do report compara representações diferentes por ordem
    lexicográfica e reporta um período que não existe.

    Devolve "" quando não reconhece — nunca levanta, para não derrubar uma
    ingestão de 17M linhas por causa de uma célula suja.
    """
    if len(data) >= 10:
        if data[:4].isdigit() and data[4] == "-":
            return data[:10]
        if data[6:10].isdigit() and data[2] == "/":
            return f"{data[6:10]}-{data[0:2]}-{data[3:5]}"
    return ""


def _extrair_ano(data: str) -> str:
    """Ano a partir da data, ou "desconhecido" se o formato não for reconhecido."""
    iso = _data_iso(data)
    return iso[:4] if iso else "desconhecido"


class _EscritorParticionado:
    """Mantém um ParquetWriter aberto por ano, com flush por lote."""

    def __init__(self, raiz: Path) -> None:
        self._raiz = raiz
        self._escritores: dict[str, pq.ParquetWriter] = {}
        self._buffers: dict[str, list[list[str]]] = {}

    def adicionar(self, ano: str, linha: list[str]) -> None:
        buffer = self._buffers.setdefault(ano, [])
        buffer.append(linha)
        if len(buffer) >= LINHAS_POR_LOTE:
            self._descarregar(ano)

    def _descarregar(self, ano: str) -> None:
        buffer = self._buffers.get(ano)
        if not buffer:
            return
        colunas = [pa.array([linha[i] for linha in buffer], type=pa.string()) for i in range(16)]
        lote = pa.RecordBatch.from_arrays(colunas, schema=_ESQUEMA)
        escritor = self._escritores.get(ano)
        if escritor is None:
            destino = self._raiz / f"ano={ano}"
            destino.mkdir(parents=True, exist_ok=True)
            escritor = pq.ParquetWriter(
                destino / "parte-0000.parquet", _ESQUEMA, compression="zstd"
            )
            self._escritores[ano] = escritor
        escritor.write_batch(lote)
        buffer.clear()

    def fechar(self) -> None:
        for ano in list(self._buffers):
            self._descarregar(ano)
        for escritor in self._escritores.values():
            escritor.close()


def ingerir_cfpb(
    zip_path: Path | None = None,
    saida: Path | None = None,
    *,
    limite: int | None = None,
) -> dict[str, Any]:
    """Lê o zip, grava o Parquet particionado e devolve as estatísticas do corpus.

    O dicionário devolvido é o insumo do portão de contagem (ver `main`).
    """
    zip_path = zip_path or (settings.data_raw / "cfpb" / "complaints.csv.zip")
    saida = saida or (settings.data_processed / "cfpb")
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} ausente — rode baixar_cfpb primeiro.")
    saida.mkdir(parents=True, exist_ok=True)

    total = 0
    com_narrativa = 0
    caracteres_narrativa = 0
    por_ano: Counter[str] = Counter()
    narrativa_por_ano: Counter[str] = Counter()
    data_min: str | None = None
    data_max: str | None = None

    escritor = _EscritorParticionado(saida)
    with zip_path.open("rb") as fh:
        fluxo = _FluxoZip(fh)
        texto = io.TextIOWrapper(
            io.BufferedReader(fluxo, 1 << 22), encoding="utf-8", errors="replace", newline=""
        )
        leitor = csv.reader(texto)
        cabecalho = next(leitor)
        if len(cabecalho) != len(COLUNAS):
            raise ValueError(
                f"schema inesperado: {len(cabecalho)} colunas (esperado {len(COLUNAS)}). "
                f"Cabeçalho: {cabecalho}"
            )

        for linha in leitor:
            if len(linha) != len(COLUNAS):
                continue  # linha truncada/suja: descarta em vez de derrubar a ingestão
            total += 1
            data = linha[0]
            ano = _extrair_ano(data)
            por_ano[ano] += 1
            # min/max sobre a data NORMALIZADA: o campo cru mistura formatos.
            iso = _data_iso(data)
            if iso:
                if data_min is None or iso < data_min:
                    data_min = iso
                if data_max is None or iso > data_max:
                    data_max = iso
            narrativa = linha[5]
            if narrativa:
                com_narrativa += 1
                narrativa_por_ano[ano] += 1
                caracteres_narrativa += len(narrativa)
            escritor.adicionar(ano, linha)
            if limite is not None and total >= limite:
                break
    escritor.fechar()

    return {
        "linhas_total": total,
        "com_narrativa": com_narrativa,
        "pct_narrativa": round(com_narrativa / total * 100, 2) if total else 0.0,
        "caracteres_narrativa": caracteres_narrativa,
        "periodo": {"min": data_min, "max": data_max},
        "por_ano": {
            ano: {"linhas": por_ano[ano], "com_narrativa": narrativa_por_ano.get(ano, 0)}
            for ano in sorted(por_ano)
        },
        "particoes": len(por_ano),
        "limite_aplicado": limite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingere o bulk da CFPB em Parquet particionado.")
    parser.add_argument("--limite", type=int, default=None, help="para após N linhas (dev)")
    parser.add_argument("--zip", type=Path, default=None, help="caminho do complaints.csv.zip")
    parser.add_argument("--saida", type=Path, default=None, help="raiz do Parquet particionado")
    args = parser.parse_args()

    estat = ingerir_cfpb(zip_path=args.zip, saida=args.saida, limite=args.limite)

    # Amarra a contagem ao snapshot exato: o bulk muda todo dia, então o número
    # sozinho não é reproduzível — o par (sha256, contagem) é.
    dir_raw = args.zip.parent if args.zip else settings.data_raw / "cfpb"
    manifesto_path = dir_raw / "manifesto.json"
    if manifesto_path.exists():
        manifesto = json.loads(manifesto_path.read_text())
        estat["snapshot"] = {
            "sha256": manifesto.get("sha256"),
            "last_modified": manifesto.get("last_modified"),
            "bytes": manifesto.get("bytes"),
        }

    destino_report = settings.data_processed.parent.parent / "reports" / "fase6_escala"
    destino_report.mkdir(parents=True, exist_ok=True)
    caminho = destino_report / "contagem_cfpb.json"
    caminho.write_text(json.dumps(carimbar(estat), ensure_ascii=False, indent=2))

    print(f"linhas: {estat['linhas_total']:,}")
    print(f"com narrativa: {estat['com_narrativa']:,} ({estat['pct_narrativa']}%)")
    print(f"caracteres de narrativa: {estat['caracteres_narrativa']:,}")
    print(f"período: {estat['periodo']['min']} .. {estat['periodo']['max']}")
    print(f"partições: {estat['particoes']}")
    print(f"report: {caminho}")


if __name__ == "__main__":
    main()
