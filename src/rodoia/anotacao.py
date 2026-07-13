"""Kit de anotação HUMANA de relevância (Fase 1) — para um κ inter-anotador REAL.

O ponto que faltava para a régua staff: a avaliação era anotador único (o autor). Aqui montamos
um kit para **≥2 humanos** julgarem, de forma independente, se um trecho recuperado é **relevante**
para a pergunta (0=não, 1=sim), e computamos o **κ de Cohen** entre eles — a concordância
inter-anotador humana, corrigida pelo acaso.

Fluxo:
  1. `python -m rodoia.anotacao gerar`      -> gera anotacao/relevancia.csv (pares a julgar)
  2. Dois avaliadores preenchem a coluna `relevante` (0/1), INDEPENDENTES, salvando cada um o seu
     (ex.: anotacao/anotador_A.csv e anotacao/anotador_B.csv).
  3. `python -m rodoia.anotacao kappa anotacao/anotador_A.csv anotacao/anotador_B.csv`
     -> κ de Cohen + % de concordância -> reports/fase1_rag/kappa_humano.json
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from rodoia.config import REPO_ROOT
from rodoia.estat import cohen_kappa
from rodoia.proveniencia import carimbar

DIR = REPO_ROOT / "anotacao"
CSV_KIT = DIR / "relevancia.csv"
COL = "relevante"          # 0 = não relevante, 1 = relevante (o avaliador preenche)


def gerar_kit(n_consultas: int = 15, seed: int = 42) -> Path:
    """Para uma amostra do dourado, junta 1 trecho provavelmente relevante (top-1) + 1 distrator
    (posição baixa) por pergunta — variedade que torna o κ informativo. Embaralha as linhas."""
    from rodoia.rag.avaliacao_retrieval import CONJUNTO_DOURADO, carregar_recuperador

    rng = random.Random(seed)
    rec = carregar_recuperador(com_reranker=False)
    linhas = []
    for caso in CONJUNTO_DOURADO[:n_consultas]:
        res = rec.buscar(caso["consulta"], k=10, modo="hibrido")
        if not res:
            continue
        linhas.append((caso["consulta"], res[0]["numero"], res[0]["texto"][:400]))   # top-1
        if len(res) > 6:                                                              # distrator
            d = res[rng.randint(5, len(res) - 1)]
            linhas.append((caso["consulta"], d["numero"], d["texto"][:400]))
    rng.shuffle(linhas)

    DIR.mkdir(exist_ok=True)
    with CSV_KIT.open("w", encoding="utf-8-sig", newline="") as fh:   # BOM p/ Excel PT-BR
        w = csv.writer(fh, delimiter=";")
        w.writerow(["id", "consulta", "resolucao", "trecho", COL])
        for i, (c, num, t) in enumerate(linhas, 1):
            w.writerow([i, c, num, t, ""])
    print(f"kit gerado: {len(linhas)} pares -> {CSV_KIT} (preencha a coluna '{COL}' com 0 ou 1)")
    return CSV_KIT


def _norm_id(x) -> str:
    return str(x).strip().split(".")[0]        # "1", "1.0", 1 → "1" (alinha csv e xlsx)


def _ler(caminho: str | Path) -> dict[str, int]:
    """Lê os rótulos {id: 0/1}. Aceita CSV (`;`) ou XLSX (Excel)."""
    caminho = str(caminho)
    if caminho.lower().endswith((".xlsx", ".xls")):
        import pandas as pd
        registros = pd.read_excel(caminho, dtype=str).fillna("").to_dict("records")
    else:
        with open(caminho, encoding="utf-8-sig", newline="") as fh:
            registros = list(csv.DictReader(fh, delimiter=";"))
    return {_norm_id(r["id"]): int(str(r.get(COL, "")).strip())
            for r in registros if str(r.get(COL, "")).strip() in ("0", "1")}


def computar_kappa(csv_a: str, csv_b: str) -> dict:
    a, b = _ler(csv_a), _ler(csv_b)
    ids = sorted(set(a) & set(b), key=int)
    if not ids:
        raise SystemExit("Nenhum id em comum preenchido (0/1) nos dois CSVs.")
    la, lb = [a[i] for i in ids], [b[i] for i in ids]
    concord = sum(1 for x, y in zip(la, lb, strict=True) if x == y) / len(ids)
    res = carimbar({
        "tarefa": "relevância de trecho recuperado (0/1) — 2 anotadores HUMANOS",
        "n_pares": len(ids),
        "concordancia_percentual": round(100 * concord, 1),
        "cohen_kappa": cohen_kappa(la, lb),
        "prevalencia_relevante": round(sum(la + lb) / (2 * len(ids)), 3),
    })
    saida = REPO_ROOT / "reports" / "fase1_rag" / "kappa_humano.json"
    saida.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"κ de Cohen (humano) = {res['cohen_kappa']} | "
          f"concordância {res['concordancia_percentual']}% (n={res['n_pares']}) -> {saida}")
    return res


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "gerar":
        gerar_kit()
    elif len(sys.argv) >= 4 and sys.argv[1] == "kappa":
        computar_kappa(sys.argv[2], sys.argv[3])
    else:
        print("uso: python -m rodoia.anotacao gerar")
        print("     python -m rodoia.anotacao kappa <anotador_A.csv> <anotador_B.csv>")


if __name__ == "__main__":
    main()
