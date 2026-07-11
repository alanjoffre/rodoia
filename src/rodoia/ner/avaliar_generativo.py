"""Avaliação do NER generativo por LLM (Fase 2) — F1 de entidade, base vs. FT.

Gera as entidades de cada sentença de teste (vLLM, fp8), parseia o JSON e calcula
**F1 de entidade** (match exato texto+tipo) contra o gold do LeNER-Br. A métrica é dura
e comparável ao BERTimbau (SOTA) — é a vitória de fine-tuning que faltava.

Uso:  python -m rodoia.ner.avaliar_generativo <modelo> <saida.json> [rotulo]
"""
from __future__ import annotations

import json
import re
import sys

from rodoia.config import REPO_ROOT, settings
from rodoia.ner.generativo import SISTEMA, TIPOS, como_conjunto
from rodoia.proveniencia import carimbar

_RE_ARR = re.compile(r"\[.*\]", re.S)


def parse_entidades(saida: str) -> set[tuple[str, str]]:
    """Extrai o 1º array JSON da resposta → conjunto (texto_norm, tipo). Tolerante a lixo."""
    m = _RE_ARR.search(saida or "")
    if not m:
        return set()
    try:
        itens = json.loads(m.group(0))
    except (ValueError, TypeError):
        return set()
    ents = []
    for it in itens:
        if isinstance(it, dict) and "texto" in it and it.get("tipo") in TIPOS:
            ents.append((str(it["texto"]), it["tipo"]))
    return como_conjunto(ents)


def metricas_ner(preds: list[set], golds: list[set]) -> dict:
    """P/R/F1 micro + F1 por entidade a partir de conjuntos (texto,tipo). Pura/testável."""
    tp = fp = fn = 0
    por_tipo = {t: [0, 0, 0] for t in TIPOS}  # tp, fp, fn
    for p, g in zip(preds, golds):
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
        for t in TIPOS:
            pt = {e for e in p if e[1] == t}
            gt = {e for e in g if e[1] == t}
            por_tipo[t][0] += len(pt & gt)
            por_tipo[t][1] += len(pt - gt)
            por_tipo[t][2] += len(gt - pt)

    def f1(a, b, c):
        prec = a / (a + b) if (a + b) else 0.0
        rec = a / (a + c) if (a + c) else 0.0
        return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0

    return {
        "f1_micro": f1(tp, fp, fn),
        "f1_por_entidade": {t: f1(*por_tipo[t]) for t in TIPOS},
    }


def avaliar(modelo: str, saida: str, rotulo: str = "") -> dict:
    from vllm import LLM, SamplingParams

    teste = [json.loads(l) for l in (settings.data_processed / "ner_test.jsonl").open(encoding="utf-8")]
    llm = LLM(model=modelo, quantization="fp8", max_model_len=2048,
              gpu_memory_utilization=0.80, enforce_eager=True)
    convs = [[{"role": "system", "content": SISTEMA}, {"role": "user", "content": t["texto"]}]
             for t in teste]
    outs = llm.chat(convs, SamplingParams(max_tokens=512, temperature=0.0))
    preds = [parse_entidades(o.outputs[0].text) for o in outs]
    golds = [como_conjunto([(t, tp) for t, tp in x["entidades"]]) for x in teste]

    res = carimbar({
        "modelo": modelo, "rotulo": rotulo, "abordagem": "LLM generativo (extração)",
        "n_teste": len(teste), **metricas_ner(preds, golds),
    })
    open(saida, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[{rotulo}] F1-micro (generativo) = {res['f1_micro']} | {res['f1_por_entidade']}")
    return res


if __name__ == "__main__":
    rot = sys.argv[3] if len(sys.argv) > 3 else ""
    avaliar(sys.argv[1], sys.argv[2], rot)
