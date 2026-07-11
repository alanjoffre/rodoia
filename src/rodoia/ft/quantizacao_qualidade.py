"""Custo de QUALIDADE da quantização (Fase 2) — o eixo que faltava no trade-off.

O serving mede a economia de MEMÓRIA (fp16 5.8 GB não cabe → fp8 cabe em 6 GB). Aqui
medimos a perda de QUALIDADE via perplexidade, comparando o modelo merged em **fp32
(sem quantização, referência)** vs. **4-bit NF4** (a quantização do próprio QLoRA), no
mesmo framework (transformers) e nos mesmos textos held-out. ΔPPL = custo da quantização.

Nota: servimos em fp8, que é MAIS preciso que NF4 — logo o custo de qualidade do fp8 é
**menor ou igual** ao ΔPPL(NF4) aqui medido (um limite superior defensável).

Uso:  python -m rodoia.ft.quantizacao_qualidade models/antt-merged-ho \
          data/processed/ft_dataset_holdout.jsonl reports/fase2_ft/quantizacao_qualidade.json
"""
from __future__ import annotations

import json
import math
import sys

from rodoia.proveniencia import carimbar


def _textos(caminho: str) -> list[str]:
    out = []
    for linha in open(caminho, encoding="utf-8"):
        ex = json.loads(linha)
        out += [m["content"] for m in ex["messages"] if m["role"] == "assistant"]
    return out


def ppl(modelo: str, textos: list[str], precisao: str) -> float:
    """PPL (exp da NLL média por token) do modelo nos textos, em fp32 (CPU) ou nf4 (GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(modelo)
    if precisao == "nf4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        modelo_hf = AutoModelForCausalLM.from_pretrained(
            modelo, quantization_config=bnb, device_map="auto")
    else:  # fp32 na CPU (referência sem quantização)
        modelo_hf = AutoModelForCausalLM.from_pretrained(
            modelo, dtype=torch.float32, device_map="cpu")
    modelo_hf.eval()

    nll_total, n_tok = 0.0, 0
    for t in textos:
        ids = tok(t, return_tensors="pt").input_ids.to(modelo_hf.device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            perda = modelo_hf(ids, labels=ids).loss.item()  # NLL média sobre n-1 tokens
        nll_total += perda * (ids.shape[1] - 1)
        n_tok += ids.shape[1] - 1
    return round(math.exp(nll_total / n_tok), 3)


def main() -> None:
    modelo, dataset, saida = sys.argv[1], sys.argv[2], sys.argv[3]
    textos = _textos(dataset)
    ppl_fp32 = ppl(modelo, textos, "fp32")
    ppl_nf4 = ppl(modelo, textos, "nf4")
    res = carimbar({
        "modelo": modelo, "n_textos": len(textos),
        "ppl_fp32_ref": ppl_fp32,
        "ppl_nf4": ppl_nf4,
        "delta_ppl_pct": round(100 * (ppl_nf4 - ppl_fp32) / ppl_fp32, 2),
        "nota": "NF4 = quantizacao do QLoRA; fp8 servido e mais preciso -> custo de qualidade <= este",
    })
    open(saida, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"PPL fp32(ref)={ppl_fp32} | nf4={ppl_nf4} | ΔPPL={res['delta_ppl_pct']}%  -> {saida}")


if __name__ == "__main__":
    main()
