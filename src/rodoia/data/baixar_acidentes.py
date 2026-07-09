"""Download reprodutível do dataset público **Acidentes em rodovias concedidas**
da ANTT (portal CKAN dados.antt.gov.br), licença CC-BY.

Por que via API CKAN (`package_show`) em vez de URLs fixas: os recursos são
descobertos em tempo de execução, então o pipeline não quebra se a ANTT
adicionar concessionárias ou trocar IDs de recurso. Cada arquivo é 1 CSV por
concessionária; o schema é consistente entre elas (validado em docs/00).

Uso:
    python -m rodoia.data.baixar_acidentes            # baixa todas as concessionárias
    python -m rodoia.data.baixar_acidentes --limite 2 # baixa só as 2 primeiras (dev)

Saída: arquivos em data/raw/acidentes/ + um manifesto.json com proveniência
(URL, id do recurso, tamanho, sha256, data de coleta) — rastreabilidade exigida
pela licença CC-BY e pela reprodutibilidade científica.

Os dados NÃO entram no Git — este script + o manifesto ficam versionados; os CSVs
vão para o DVC (ver data/README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi

from rodoia.config import settings

CKAN_BASE = "https://dados.antt.gov.br"
DATASET_ID = "acidentes-rodovias"
LICENCA = "CC-BY (Creative Commons Atribuição) — fonte: ANTT"
_USER_AGENT = "RodoIA/0.0 (projeto open-source; download de dados abertos ANTT)"


def _contexto_ssl() -> ssl.SSLContext:
    """Contexto TLS ancorado no CA bundle do certifi — funciona em qualquer
    máquina, sem depender dos certificados do sistema (o Python do venv no macOS
    não os encontra por padrão)."""
    return ssl.create_default_context(cafile=certifi.where())


def _abrir(url: str, *, range_bytes: str | None = None):
    headers = {"User-Agent": _USER_AGENT}
    if range_bytes:
        headers["Range"] = f"bytes={range_bytes}"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=120, context=_contexto_ssl())


@dataclass(frozen=True)
class Recurso:
    """Um recurso CSV do dataset (uma concessionária)."""

    nome: str
    url: str
    resource_id: str


def listar_recursos_csv(dataset_id: str = DATASET_ID) -> list[Recurso]:
    """Consulta a API CKAN e devolve os recursos CSV do dataset."""
    url = f"{CKAN_BASE}/api/3/action/package_show?{urllib.parse.urlencode({'id': dataset_id})}"
    with _abrir(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload.get("success"):
        raise RuntimeError(f"CKAN retornou success=false para o dataset {dataset_id!r}")
    recursos = [
        Recurso(nome=r["name"], url=r["url"], resource_id=r["id"])
        for r in payload["result"]["resources"]
        if (r.get("format") or "").upper() == "CSV"
    ]
    if not recursos:
        raise RuntimeError(f"Nenhum recurso CSV encontrado no dataset {dataset_id!r}")
    return recursos


def _nome_arquivo(recurso: Recurso) -> str:
    """Nome de arquivo estável e ÚNICO por recurso.

    O `resource_id` (curto) é sufixado para evitar colisão: vários recursos do
    dataset compartilham o mesmo basename de download (ex.: concessionárias sem
    sufixo no nome), o que sobrescreveria arquivos se usássemos só o basename.
    """
    caminho = urllib.parse.urlparse(recurso.url).path
    base = Path(caminho).name
    stem = base[:-4] if base.endswith(".csv") else (base or "acidentes")
    return f"{stem}__{recurso.resource_id[:8]}.csv"


def baixar_recurso(recurso: Recurso, destino_dir: Path) -> dict:
    """Baixa um recurso em streaming, calculando sha256. Retorna metadados."""
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / _nome_arquivo(recurso)
    sha = hashlib.sha256()
    tamanho = 0
    with _abrir(recurso.url) as resp, destino.open("wb") as fh:
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
            sha.update(chunk)
            tamanho += len(chunk)
    return {
        "nome": recurso.nome,
        "arquivo": destino.name,
        "url": recurso.url,
        "resource_id": recurso.resource_id,
        "bytes": tamanho,
        "sha256": sha.hexdigest(),
    }


def baixar_acidentes(destino_dir: Path | None = None, limite: int | None = None) -> Path:
    """Baixa o dataset de acidentes e escreve o manifesto de proveniência.

    Retorna o caminho do manifesto.json.
    """
    destino_dir = destino_dir or (settings.data_raw / "acidentes")
    recursos = listar_recursos_csv()
    if limite is not None:
        recursos = recursos[:limite]

    entradas = []
    for i, recurso in enumerate(recursos, 1):
        print(f"[{i}/{len(recursos)}] baixando: {recurso.nome}")
        entradas.append(baixar_recurso(recurso, destino_dir))

    manifesto = {
        "dataset": DATASET_ID,
        "fonte": f"{CKAN_BASE}/dataset/{DATASET_ID}",
        "licenca": LICENCA,
        "n_arquivos": len(entradas),
        "recursos": entradas,
    }
    caminho_manifesto = destino_dir / "manifesto.json"
    caminho_manifesto.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2))
    total_mb = sum(e["bytes"] for e in entradas) / 1e6
    print(f"OK: {len(entradas)} arquivos, {total_mb:.1f} MB -> {destino_dir}")
    print(f"manifesto: {caminho_manifesto}")
    return caminho_manifesto


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa o dataset de Acidentes da ANTT.")
    parser.add_argument("--limite", type=int, default=None, help="baixar só os N primeiros CSVs")
    parser.add_argument("--destino", type=Path, default=None, help="diretório de saída")
    args = parser.parse_args()
    baixar_acidentes(destino_dir=args.destino, limite=args.limite)


if __name__ == "__main__":
    main()
