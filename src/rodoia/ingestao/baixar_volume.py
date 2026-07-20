"""Download do Volume de Tráfego nas Praças de Pedágio da ANTT (Fase 3).

Dados abertos (dados.antt.gov.br), série 2010–2026 por praça/concessionária, granularidade
mensal. Baixa via API CKAN os CSVs anuais/mensais-consolidados (ignora os diários, grandes
e desnecessários p/ a previsão mensal). Reproduzível; dados brutos fora do Git (DVC).

Uso:  python -m rodoia.ingestao.baixar_volume [--limite N]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from rodoia.config import settings

_CKAN = "https://dados.antt.gov.br/api/3/action/package_show?id=volume-trafego-praca-pedagio"
_DIR = settings.data_raw / "volume_pedagio"


def _recursos() -> list[dict]:
    """CSVs mensais/anuais (exclui 'Diário') via API CKAN."""
    with urllib.request.urlopen(_CKAN, timeout=60) as resp:
        pacote = json.loads(resp.read().decode("utf-8"))["result"]
    csvs = [
        r for r in pacote["resources"]
        if r.get("format", "").upper() == "CSV" and "diár" not in (r["name"] + r["url"]).lower()
    ]
    # dedup por ano: prefere 'mensal_consolidado' quando houver
    por_ano: dict[str, dict] = {}
    for r in csvs:
        nome = r["url"].split("/")[-1]
        ano = "".join(c for c in nome if c.isdigit())[:4]
        if ano not in por_ano or "mensal_consolidado" in nome:
            por_ano[ano] = r
    return sorted(por_ano.values(), key=lambda r: r["url"])


def baixar_volume(destino: Path | None = None, limite: int | None = None) -> Path:
    destino = destino or _DIR
    destino.mkdir(parents=True, exist_ok=True)
    recursos = _recursos()[:limite] if limite else _recursos()
    for i, r in enumerate(recursos, 1):
        nome = r["url"].split("/")[-1]
        alvo = destino / nome
        if not alvo.exists():
            urllib.request.urlretrieve(r["url"], alvo)
        print(f"  [{i}/{len(recursos)}] {nome} ({alvo.stat().st_size // 1024} KB)")
    print(f"OK: {len(recursos)} arquivos -> {destino}")
    return destino


def main() -> None:
    p = argparse.ArgumentParser(description="Baixa Volume de Tráfego de Pedágio (ANTT).")
    p.add_argument("--limite", type=int, default=None, help="baixar só os N primeiros anos")
    baixar_volume(limite=p.parse_args().limite)


if __name__ == "__main__":
    main()
