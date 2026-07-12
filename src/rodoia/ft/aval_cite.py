"""Avaliação base vs. fine-tunado (Fase 2) — métrica objetiva de citação, com IC.

Métrica reproduzível sobre o CONJUNTO_DOURADO (perguntas de intenção real com a(s)
resolução(ões)-fonte conhecidas). Como o corpus de normas via DVC está ausente, é o
substituto objetivo do juiz-com-referência:

- acuracia_citacao: fração de respostas que citam ALGUMA das fontes esperadas (correta).
- taxa_citacao    : fração que cita ALGUMA resolução (estilo de citar, acertando ou não).

Com n pequeno e in-domain, reporta-se **IC de Wilson**. Sem RAG e com dataset pequeno,
espera-se estilo≠conhecimento — reportado honestamente (docs/11).
"""
import json
import re
import sys
from pathlib import Path

from rodoia.estat import wilson
from rodoia.proveniencia import carimbar

RE_RES = re.compile(r"(\d[\d\.]{2,7})\s*/\s*(\d{4})")


def _norm(num: str) -> str:
    return num.replace(".", "").replace(" ", "")


def cita_correta(resp: str, fontes: list[str]) -> bool:
    """True se a resposta cita QUALQUER uma das fontes esperadas (suporta multi-fonte)."""
    alvos = {(_norm(f.split("/")[0]), f.split("/")[1]) for f in fontes}
    return any((_norm(m.group(1)), m.group(2)) in alvos for m in RE_RES.finditer(resp))


def cita_alguma(resp: str) -> bool:
    return RE_RES.search(resp) is not None


def avaliar(caminho: str) -> dict:
    dados = json.load(open(caminho, encoding="utf-8"))
    casos = [{
        "consulta": d["consulta"],
        "fontes_esperadas": d["fontes"],
        "citou_correta": cita_correta(d["resposta"], d["fontes"]),
        "citou_alguma": cita_alguma(d["resposta"]),
    } for d in dados]
    n = len(casos)
    corretas = sum(c["citou_correta"] for c in casos)
    alguma = sum(c["citou_alguma"] for c in casos)
    return {
        "n": n,
        "acuracia_citacao": round(corretas / n, 3),
        "acuracia_ic95": wilson(corretas, n),
        "taxa_citacao": round(alguma / n, 3),
        "taxa_ic95": wilson(alguma, n),
        "casos": casos,
    }


if __name__ == "__main__":
    base = avaliar(sys.argv[1])
    ft = avaliar(sys.argv[2])
    res = carimbar({
        "metrica": "citacao da resolucao-fonte no CONJUNTO_DOURADO (com IC de Wilson)",
        "n": base["n"],
        "base": {k: base[k] for k in
                 ("acuracia_citacao", "acuracia_ic95", "taxa_citacao", "taxa_ic95")},
        "ft": {k: ft[k] for k in
               ("acuracia_citacao", "acuracia_ic95", "taxa_citacao", "taxa_ic95")},
        "ganho_acuracia": round(ft["acuracia_citacao"] - base["acuracia_citacao"], 3),
        "casos_base": base["casos"],
        "casos_ft": ft["casos"],
    })
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BASE acuracia={base['acuracia_citacao']} {base['acuracia_ic95']} "
          f"taxa={base['taxa_citacao']}")
    print(f"FT   acuracia={ft['acuracia_citacao']} {ft['acuracia_ic95']} taxa={ft['taxa_citacao']}")
    print("relatorio:", out)
