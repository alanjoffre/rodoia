"""Download reprodutível do **CUAD** (Contract Understanding Atticus Dataset),
licença Apache 2.0 — o benchmark EXTERNO da Fase 6.

Por que este dataset: a metodologia de avaliação das Fases 1 e 5 foi construída
sobre gold rotulado pelo próprio autor. A auditoria κ tratou isso por dentro
(docs/16), mas a objeção "ele rotulou o próprio teste" só morre com gold de
TERCEIROS. O CUAD traz 510 contratos comerciais com **20.910 perguntas e 13.823
spans de resposta anotados por advogados** — e, crucialmente, **67,9% das
perguntas são `is_impossible`**: a resposta correta é "não consta no contrato".

Isso permite medir duas coisas separadas, **sem uma única chamada de LLM**:
recuperação (Recall@k, MRR, nDCG contra os spans) e **abstenção** (o sistema
recupera lixo quando não há nada a recuperar?). Ver docs/17 §8.

**A API pública do Kaggle dispensa autenticação** — verificado em 2026-07:
`/api/v1/datasets/download/{owner}/{slug}` devolve 200 sem token, sem conta e
sem `kaggle.json`. Não é preciso o pacote `kaggle` nem credencial. Mas o
endpoint **responde 404 a HEAD** e 200 a GET, então a consulta de metadados
abre com GET e fecha antes do corpo (ver `consultar_metadados`).

**Ressalva sobre espelhos do Kaggle:** espelho costuma ser subconjunto
desatualizado da fonte (o dataset de tráfego da ANTT no Kaggle tem 3,5 MB
contra 302 MB/ano na fonte oficial). Aqui o risco é baixo — o CUAD é um dataset
acadêmico congelado, não uma série viva — mas o manifesto carimba o sha256 para
que a alegação de reprodutibilidade não dependa dessa suposição.

Uso:
    python -m rodoia.rag.baixar_cuad              # baixa o zip (~108 MB)
    python -m rodoia.rag.baixar_cuad --verificar  # só consulta metadados (HEAD)

Saída: `data/raw/cuad/cuad.zip` + `manifesto.json`. Os dados NÃO entram no Git.
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

SLUG = "ashyou09/contract-understanding-atticus-dataset-cuad"
URL_ZIP = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"
FONTE = "https://www.atticusprojectai.org/cuad"
LICENCA = "Apache 2.0 — The Atticus Project (via espelho no Kaggle)"

# UA neutro: a API do Kaggle não exige nada específico (ao contrário do WAF
# invertido da CFPB, ver ingestao/baixar_cfpb.py). **Sem acento**: cabeçalho HTTP
# não aceita não-ASCII e o servidor devolve 400 — custou um diagnóstico.
_USER_AGENT = "RodoIA/0.0 (projeto open-source; benchmark externo de retrieval)"


def _contexto_ssl() -> ssl.SSLContext:
    """Contexto TLS ancorado no CA bundle do certifi — mesmo padrão dos demais
    downloaders do repo."""
    return ssl.create_default_context(cafile=certifi.where())


def _abrir(url: str) -> HTTPResponse:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    resposta: HTTPResponse = urllib.request.urlopen(req, timeout=300, context=_contexto_ssl())
    return resposta


@dataclass(frozen=True)
class MetadadosRemotos:
    """O que o servidor declara sobre o arquivo, antes de baixar."""

    bytes_declarados: int


def consultar_metadados(url: str = URL_ZIP) -> MetadadosRemotos:
    """Tamanho declarado pelo servidor, sem baixar os 108 MB.

    **Não usa HEAD**: o endpoint de download do Kaggle responde **404 a HEAD** e
    200 a GET (verificado em 2026-07). Então abrimos com GET, lemos só os
    cabeçalhos e fechamos a conexão antes do corpo — o servidor devolve
    `Content-Length` no cabeçalho, que é tudo o que precisamos.
    """
    resp = _abrir(url)
    try:
        return MetadadosRemotos(bytes_declarados=int(resp.headers.get("Content-Length") or 0))
    finally:
        resp.close()


def baixar_cuad(destino_dir: Path | None = None, *, forcar: bool = False) -> Path:
    """Baixa o zip do CUAD em streaming, calculando sha256, e grava o manifesto.

    Idempotente: pula o download se o arquivo local já bate com o tamanho
    declarado pelo servidor (`forcar=True` re-baixa).

    Retorna o caminho do manifesto.json.
    """
    destino_dir = destino_dir or (settings.data_raw / "cuad")
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / "cuad.zip"
    manifesto_path = destino_dir / "manifesto.json"

    meta = consultar_metadados()
    if (
        destino.exists()
        and not forcar
        and meta.bytes_declarados
        and destino.stat().st_size == meta.bytes_declarados
        and manifesto_path.exists()
    ):
        print(f"já baixado ({destino.stat().st_size:,} bytes) — use --forcar para refazer")
        return manifesto_path

    print(f"baixando {meta.bytes_declarados / 1e6:.1f} MB")
    sha = hashlib.sha256()
    tamanho = 0
    with _abrir(URL_ZIP) as resp, destino.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            sha.update(chunk)
            tamanho += len(chunk)

    if meta.bytes_declarados and tamanho != meta.bytes_declarados:
        raise RuntimeError(
            f"download incompleto: {tamanho} bytes recebidos, "
            f"{meta.bytes_declarados} declarados pelo servidor"
        )

    manifesto = {
        "dataset": "cuad-v1",
        "fonte": FONTE,
        "espelho": f"https://www.kaggle.com/datasets/{SLUG}",
        "licenca": LICENCA,
        "arquivo": destino.name,
        "bytes": tamanho,
        "sha256": sha.hexdigest(),
        "nota": (
            "Dataset acadêmico congelado (não é série viva). O sha256 identifica "
            "o espelho exato usado — espelho do Kaggle pode divergir da fonte."
        ),
    }
    manifesto_path.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2))
    print(f"OK: {tamanho / 1e6:.1f} MB -> {destino}")
    print(f"sha256: {manifesto['sha256']}")
    print(f"manifesto: {manifesto_path}")
    return manifesto_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa o CUAD (benchmark externo de recuperação).")
    parser.add_argument(
        "--verificar", action="store_true", help="só consulta metadados remotos (HEAD)"
    )
    parser.add_argument("--forcar", action="store_true", help="re-baixa mesmo se já existir")
    parser.add_argument("--destino", type=Path, default=None, help="diretório de saída")
    args = parser.parse_args()

    if args.verificar:
        meta = consultar_metadados()
        print(f"bytes: {meta.bytes_declarados:,} ({meta.bytes_declarados / 1e6:.1f} MB)")
        print("acesso sem credencial: OK")
        return
    baixar_cuad(destino_dir=args.destino, forcar=args.forcar)


if __name__ == "__main__":
    main()
