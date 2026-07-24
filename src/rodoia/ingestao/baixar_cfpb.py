"""Download reprodutível do **Consumer Complaint Database** da CFPB
(Consumer Financial Protection Bureau, EUA), domínio público.

Por que este dataset entra num repositório de regulação brasileira: é um
benchmark de ESCALA que o corpus da ANTT não consegue exercer (17,2M linhas
contra 3.647 chunks) e vem com narrativa livre do cidadão — algo que o SOU da
ANTT **não** publica (ver docs/17 §1). Serve para validar externamente a
metodologia de avaliação construída nas Fases 1 e 5.

Por que o bulk e não a API de busca: o endpoint
`/data-research/consumer-complaints/search/api/v1/` está atrás de WAF e devolve
403 (ou o HTML da SPA); só o arquivo bulk é acessível programaticamente.

**Armadilha de WAF (invertida, e não documentada em lugar nenhum):** o Akamai
da CFPB BLOQUEIA User-Agent de navegador (`Mozilla/...` -> 403) e LIBERA
`curl`/`Python-urllib`/nenhum UA. Tentar "parecer um navegador" quebra o
download. Por isso `_USER_AGENT` abaixo é deliberadamente `curl/...`.

**O arquivo é atualizado diariamente.** A contagem de linhas NÃO é constante —
cresce a cada dia. O manifesto carimba sha256 + `last_modified` + bytes do
snapshot exato usado, e é ele (não um literal no código) que torna a ingestão
reproduzível. Ver `ingestao_cfpb.py` para o portão de contagem.

Uso:
    python -m rodoia.ingestao.baixar_cfpb           # baixa o zip completo (~1,43 GB)
    python -m rodoia.ingestao.baixar_cfpb --verificar  # só consulta metadados (HEAD)

Saída: `data/raw/cfpb/complaints.csv.zip` + `manifesto.json` com proveniência.
Os dados NÃO entram no Git — vão para o DVC (ver data/README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.request
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path

import certifi

from rodoia.config import settings

URL_BULK = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
FONTE = "https://www.consumerfinance.gov/data-research/consumer-complaints/"
LICENCA = "Domínio público (U.S. Government Works) — fonte: CFPB"

# NÃO trocar por um UA de navegador: o WAF da CFPB responde 403 a `Mozilla/...`
# e 200 a `curl`. Regra invertida em relação ao habitual — verificado em 2026-07.
_USER_AGENT = "curl/8.4.0"


def _contexto_ssl() -> ssl.SSLContext:
    """Contexto TLS ancorado no CA bundle do certifi (mesmo padrão de
    `baixar_acidentes`) — não depende dos certificados do sistema."""
    return ssl.create_default_context(cafile=certifi.where())


def _abrir(url: str, *, metodo: str = "GET") -> HTTPResponse:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT}, method=metodo)
    resposta: HTTPResponse = urllib.request.urlopen(req, timeout=300, context=_contexto_ssl())
    return resposta


@dataclass(frozen=True)
class MetadadosRemotos:
    """O que o servidor declara sobre o arquivo, antes de baixar."""

    bytes_declarados: int
    last_modified: str


def consultar_metadados(url: str = URL_BULK) -> MetadadosRemotos:
    """HEAD no bulk: tamanho e data de publicação, sem baixar 1,43 GB.

    Útil para decidir se vale re-baixar (o arquivo muda todo dia) e para
    detectar cedo uma mudança de política do WAF.
    """
    with _abrir(url, metodo="HEAD") as resp:
        return MetadadosRemotos(
            bytes_declarados=int(resp.headers.get("Content-Length") or 0),
            last_modified=resp.headers.get("Last-Modified") or "desconhecido",
        )


def baixar_cfpb(destino_dir: Path | None = None, *, forcar: bool = False) -> Path:
    """Baixa o zip do CFPB em streaming, calculando sha256, e grava o manifesto.

    Idempotente: se o arquivo local já existe com o mesmo tamanho declarado pelo
    servidor, pula o download (use `forcar=True` para re-baixar mesmo assim).

    Retorna o caminho do manifesto.json.
    """
    destino_dir = destino_dir or (settings.data_raw / "cfpb")
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / "complaints.csv.zip"

    meta = consultar_metadados()
    if destino.exists() and not forcar and destino.stat().st_size == meta.bytes_declarados:
        print(f"já baixado ({destino.stat().st_size:,} bytes) — use --forcar para refazer")
        manifesto_existente = destino_dir / "manifesto.json"
        if manifesto_existente.exists():
            return manifesto_existente

    print(f"baixando {meta.bytes_declarados / 1e9:.2f} GB (publicado em {meta.last_modified})")
    sha = hashlib.sha256()
    tamanho = 0
    with _abrir(URL_BULK) as resp, destino.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            sha.update(chunk)
            tamanho += len(chunk)
            if tamanho % (100 << 20) < (1 << 20):
                print(f"  ... {tamanho / 1e9:.2f} GB")

    if meta.bytes_declarados and tamanho != meta.bytes_declarados:
        raise RuntimeError(
            f"download incompleto: {tamanho} bytes recebidos, "
            f"{meta.bytes_declarados} declarados pelo servidor"
        )

    manifesto = {
        "dataset": "cfpb-consumer-complaints",
        "fonte": FONTE,
        "url": URL_BULK,
        "licenca": LICENCA,
        "arquivo": destino.name,
        "bytes": tamanho,
        "sha256": sha.hexdigest(),
        "last_modified": meta.last_modified,
        "nota": (
            "Bulk atualizado diariamente pela CFPB; a contagem de linhas cresce "
            "com o tempo. Este sha256 identifica o snapshot exato usado."
        ),
    }
    caminho_manifesto = destino_dir / "manifesto.json"
    caminho_manifesto.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2))
    print(f"OK: {tamanho / 1e9:.2f} GB -> {destino}")
    print(f"sha256: {manifesto['sha256']}")
    print(f"manifesto: {caminho_manifesto}")
    return caminho_manifesto


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa o Consumer Complaint Database da CFPB.")
    parser.add_argument(
        "--verificar", action="store_true", help="só consulta metadados remotos (HEAD)"
    )
    parser.add_argument("--forcar", action="store_true", help="re-baixa mesmo se já existir")
    parser.add_argument("--destino", type=Path, default=None, help="diretório de saída")
    args = parser.parse_args()

    if args.verificar:
        meta = consultar_metadados()
        print(f"bytes: {meta.bytes_declarados:,} ({meta.bytes_declarados / 1e9:.2f} GB)")
        print(f"last-modified: {meta.last_modified}")
        return
    baixar_cfpb(destino_dir=args.destino, forcar=args.forcar)


if __name__ == "__main__":
    main()
