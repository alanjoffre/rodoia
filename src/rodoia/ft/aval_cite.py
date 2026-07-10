"""Avaliação base vs. fine-tunado (Fase 2) — métrica objetiva de citação.

Como o corpus de normas (data/raw/normas.jsonl, gerido por DVC) não está disponível
nesta máquina, substituímos o juiz-LLM-com-referência por uma métrica OBJETIVA e
reproduzível sobre o CONJUNTO_DOURADO (perguntas com a resolução-fonte conhecida):

- acuracia_citacao: fração de respostas que citam a RESOLUÇÃO CORRETA (a fonte esperada).
- taxa_citacao    : fração de respostas que citam ALGUMA resolução no formato "nº N/AAAA"
                    (mede adesão ao ESTILO de citar, independente de acertar).

Sem RAG e com 84 exemplos, espera-se que o fine-tuning mude o estilo mas não injete
conhecimento factual — reportado honestamente (ver docs/11).
"""
import json
import re
import sys
from pathlib import Path

RE_RES = re.compile(r"(\d[\d\.]{2,7})\s*/\s*(\d{4})")


def _norm(num: str) -> str:
    return num.replace(".", "").replace(" ", "")


def cita_correta(resp: str, fonte: str) -> bool:
    alvo_num, alvo_ano = fonte.split("/")
    alvo_num = _norm(alvo_num)
    for m in RE_RES.finditer(resp):
        if _norm(m.group(1)) == alvo_num and m.group(2) == alvo_ano:
            return True
    return False


def cita_alguma(resp: str) -> bool:
    return RE_RES.search(resp) is not None


def avaliar(caminho: str) -> dict:
    dados = json.load(open(caminho, encoding="utf-8"))
    casos = []
    for d in dados:
        fonte = d["fontes"][0]
        casos.append({
            "consulta": d["consulta"],
            "fonte_esperada": fonte,
            "citou_correta": cita_correta(d["resposta"], fonte),
            "citou_alguma": cita_alguma(d["resposta"]),
        })
    n = len(casos)
    return {
        "n": n,
        "acuracia_citacao": round(sum(c["citou_correta"] for c in casos) / n, 3),
        "taxa_citacao": round(sum(c["citou_alguma"] for c in casos) / n, 3),
        "casos": casos,
    }


if __name__ == "__main__":
    base = avaliar(sys.argv[1])
    ft = avaliar(sys.argv[2])
    res = {
        "metrica": "citacao da resolucao-fonte no CONJUNTO_DOURADO (10 perguntas)",
        "base": {"acuracia_citacao": base["acuracia_citacao"], "taxa_citacao": base["taxa_citacao"]},
        "ft": {"acuracia_citacao": ft["acuracia_citacao"], "taxa_citacao": ft["taxa_citacao"]},
        "ganho_acuracia": round(ft["acuracia_citacao"] - base["acuracia_citacao"], 3),
        "ganho_taxa_citacao": round(ft["taxa_citacao"] - base["taxa_citacao"], 3),
        "n": base["n"],
        "casos_base": base["casos"],
        "casos_ft": ft["casos"],
    }
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BASE  acuracia={base['acuracia_citacao']}  taxa_citacao={base['taxa_citacao']}")
    print(f"FT    acuracia={ft['acuracia_citacao']}  taxa_citacao={ft['taxa_citacao']}")
    print(f"ganho acuracia={res['ganho_acuracia']:+}  ganho taxa_citacao={res['ganho_taxa_citacao']:+}")
    print("relatorio:", out)
