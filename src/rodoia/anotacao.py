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
from rodoia.estat import cohen_kappa, cohen_kappa_ic95
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


CSV_KIT_GOLD = DIR / "gold_fonte.csv"


def _indice_normas() -> dict[str, dict]:
    """{numero -> {titulo, texto}} a partir de normas.jsonl (para exibir a norma-gold)."""
    from rodoia.rag.chunking import _cortar_ate_cabecalho
    caminho = REPO_ROOT / "data" / "raw" / "normas" / "normas.jsonl"
    idx = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        r = json.loads(linha)
        idx[r["numero"]] = {"titulo": r.get("titulo", ""),
                            "texto": _cortar_ate_cabecalho(r.get("texto", ""))}
    return idx


def gerar_kit_gold(n_consultas: int = 25, seed: int = 42) -> Path:
    """Kit para VALIDAR os rótulos-gold de FONTE do hit@5 (hoje anotador único) — o elo que falta
    para o κ humano tocar a métrica do gate. Para cada query do dourado emite (a) o par
    query↔fonte-GOLD (esperado relevante) e (b) um DISTRATOR: a query com a fonte-gold de OUTRA
    query (esperado não-relevante) — a variância que torna o κ informativo. 2 humanos julgam
    'esta resolução é uma fonte correta para esta pergunta?' (0/1), INDEPENDENTES."""
    from rodoia.rag.avaliacao_retrieval import CONJUNTO_DOURADO

    rng = random.Random(seed)
    normas = _indice_normas()
    casos = CONJUNTO_DOURADO[:n_consultas]
    linhas = []
    for caso in casos:
        gold = caso["fontes"][0]
        if gold not in normas:
            continue
        linhas.append((caso["consulta"], gold, normas[gold]))                    # par GOLD
        # distrator: fonte-gold de outra query que NÃO seja fonte desta
        candidatos = [f for c in casos for f in c["fontes"]
                      if f not in caso["fontes"] and f in normas]
        if candidatos:
            d = rng.choice(candidatos)
            linhas.append((caso["consulta"], d, normas[d]))                      # par DISTRATOR
    rng.shuffle(linhas)

    DIR.mkdir(exist_ok=True)
    with CSV_KIT_GOLD.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["id", "consulta", "resolucao", "titulo", "trecho", COL])
        for i, (c, num, n) in enumerate(linhas, 1):
            w.writerow([i, c, num, n["titulo"][:120], n["texto"][:600], ""])
    print(f"kit GOLD gerado: {len(linhas)} pares (gold+distrator) -> {CSV_KIT_GOLD} "
          f"(preencha '{COL}': 1 se a resolução responde à pergunta, 0 se não)")
    return CSV_KIT_GOLD


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


def computar_kappa(
    csv_a: str, csv_b: str, saida: str | Path | None = None,
    tarefa: str = "relevância de trecho recuperado (0/1) — 2 anotadores HUMANOS",
) -> dict:
    a, b = _ler(csv_a), _ler(csv_b)
    ids = sorted(set(a) & set(b), key=int)
    if not ids:
        raise SystemExit("Nenhum id em comum preenchido (0/1) nos dois CSVs.")
    la, lb = [a[i] for i in ids], [b[i] for i in ids]
    concord = sum(1 for x, y in zip(la, lb, strict=True) if x == y) / len(ids)
    res = carimbar({
        "tarefa": tarefa,
        "n_pares": len(ids),
        "concordancia_percentual": round(100 * concord, 1),
        "cohen_kappa": cohen_kappa(la, lb),
        "cohen_kappa_ic95": cohen_kappa_ic95(la, lb),      # incerteza — a régua do projeto
        "prevalencia_relevante": round(sum(la + lb) / (2 * len(ids)), 3),
        "anotadores": [Path(csv_a).stem, Path(csv_b).stem],
        "coleta": ("2 pessoas DIFERENTES, cada uma preencheu seu arquivo de forma INDEPENDENTE "
                   "e cega (sem ver o rótulo do outro) — não é o autor avaliando a si mesmo"),
    })
    saida = Path(saida) if saida else REPO_ROOT / "reports" / "fase1_rag" / "kappa_humano.json"
    saida.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"κ de Cohen (humano) = {res['cohen_kappa']} | "
          f"concordância {res['concordancia_percentual']}% (n={res['n_pares']}) -> {saida}")
    return res


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) >= 2 else ""
    if cmd == "gerar":
        gerar_kit()
    elif cmd == "gerar-gold":
        gerar_kit_gold()
    elif cmd == "kappa" and len(sys.argv) >= 4:
        computar_kappa(sys.argv[2], sys.argv[3])
    elif cmd == "kappa-gold" and len(sys.argv) >= 4:
        computar_kappa(
            sys.argv[2], sys.argv[3],
            saida=REPO_ROOT / "reports" / "fase1_rag" / "kappa_gold_fonte.json",
            tarefa="rótulo-gold de FONTE do hit@5 é correto? (0/1) — 2 anotadores HUMANOS")
    else:
        print("uso: python -m rodoia.anotacao gerar          # kit de relevância de trecho")
        print("     python -m rodoia.anotacao gerar-gold     # kit de validação dos rótulos-gold")
        print("     python -m rodoia.anotacao kappa <A> <B>       # κ da relevância de trecho")
        print("     python -m rodoia.anotacao kappa-gold <A> <B>  # κ dos rótulos-gold de fonte")


if __name__ == "__main__":
    main()
