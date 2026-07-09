"""Baixa o corpus de resoluções da ANTT (ANTTlegis) para a Fase 1 (RAG).

Enumera os atos das páginas temáticas de transporte rodoviário, baixa o texto de
cada um (educadamente, com delay — o servidor é lento) e grava um JSONL, uma
linha por norma, com texto + metadados (número, ano, órgão, título, vigência).

Uso:
    python -m rodoia.rag.baixar_normas --limite 120   # corpus inicial (recentes)
    python -m rodoia.rag.baixar_normas                # todos os atos enumerados

Saída: data/raw/normas/normas.jsonl + manifesto.json. Dados vão para o DVC; o
script fica no Git. Licença: atos oficiais são domínio público (Lei 9.610/98).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rodoia.config import settings
from rodoia.rag.fontes_antt import TEMAS_PADRAO, baixar_ato, listar_atos


def baixar_corpus(
    limite: int | None = None,
    saida_dir: Path | None = None,
    delay: float = 0.4,
    temas=TEMAS_PADRAO,
) -> Path:
    saida_dir = saida_dir or (settings.data_raw / "normas")
    saida_dir.mkdir(parents=True, exist_ok=True)

    atos = listar_atos(temas)
    # Prioriza as mais recentes (mais relevantes e mais prováveis de vigentes).
    atos = sorted(atos, key=lambda a: (int(a.ano), int(a.num)), reverse=True)
    if limite is not None:
        atos = atos[:limite]
    print(f"enumerados {len(atos)} atos para baixar (de {len(temas)} temas)")

    normas, cascas = [], 0
    for i, ato in enumerate(atos, 1):
        try:
            reg = baixar_ato(ato)
        except Exception as e:  # rede instável: registra e segue
            print(f"  [{i}/{len(atos)}] {ato.numero_legivel} ERRO: {str(e)[:60]}")
            reg = None
        if reg is None:
            cascas += 1
        else:
            normas.append(reg)
            if i % 20 == 0 or i == len(atos):
                print(f"  [{i}/{len(atos)}] ok={len(normas)} casca/erro={cascas}")
        time.sleep(delay)  # educado com o servidor lento

    caminho = saida_dir / "normas.jsonl"
    with caminho.open("w", encoding="utf-8") as fh:
        for reg in normas:
            fh.write(json.dumps(reg, ensure_ascii=False) + "\n")

    vigentes = sum(1 for r in normas if r["vigente"])
    manifesto = {
        "fonte": "ANTTlegis (UrlPublicasAction/abrirAtoPublico)",
        "licenca": "Atos oficiais — domínio público (Lei 9.610/98, art. 8º IV)",
        "temas": [t.nome for t in temas],
        "n_normas": len(normas),
        "n_vigentes": vigentes,
        "n_casca_ou_erro": cascas,
    }
    (saida_dir / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2))
    print(f"OK: {len(normas)} normas ({vigentes} vigentes) -> {caminho}")
    return caminho


def main() -> None:
    p = argparse.ArgumentParser(description="Baixa resoluções da ANTT (RAG/Fase 1).")
    p.add_argument("--limite", type=int, default=None, help="baixar só as N mais recentes")
    p.add_argument("--delay", type=float, default=0.4, help="segundos entre requisições")
    args = p.parse_args()
    baixar_corpus(limite=args.limite, delay=args.delay)


if __name__ == "__main__":
    main()
