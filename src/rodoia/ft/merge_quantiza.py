"""Merge do adaptador LoRA + quantização AWQ para servir no vLLM (Fase 2).

>>> RODA NA NITRO (CUDA). <<<

Fluxo:
1. **Merge** — funde o adaptador LoRA no modelo base, gerando um modelo completo
   (fp16). QLoRA treina em 4-bit, mas para servir e quantizar precisamos do modelo
   fundido.
2. **Quantização AWQ (4-bit)** — comprime o modelo fundido. Por quê: um 3B fp16
   ocupa ~6 GB (aperta na RTX 4050); em AWQ 4-bit cai para ~2-3 GB, sobrando VRAM
   para o KV cache do vLLM. O trade-off qualidade × memória × latência é MEDIDO na
   etapa de avaliação (docs/10).

Uso:
    python -m rodoia.ft.merge_quantiza --base Qwen/Qwen2.5-3B-Instruct \
        --adaptador models/qlora-antt --merged models/antt-merged --awq models/antt-awq
"""

from __future__ import annotations

import argparse


def merge(base: str, adaptador: str, saida_merged: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    modelo = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="auto"
    )
    modelo = PeftModel.from_pretrained(modelo, adaptador)
    modelo = modelo.merge_and_unload()  # funde LoRA nos pesos
    modelo.save_pretrained(saida_merged)
    AutoTokenizer.from_pretrained(base).save_pretrained(saida_merged)
    print(f"merge OK -> {saida_merged}")


def quantizar_awq(merged: str, saida_awq: str) -> None:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    modelo = AutoAWQForCausalLM.from_pretrained(merged)
    tok = AutoTokenizer.from_pretrained(merged)
    modelo.quantize(tok, quant_config={"w_bit": 4, "q_group_size": 128, "zero_point": True})
    modelo.save_quantized(saida_awq)
    tok.save_pretrained(saida_awq)
    print(f"AWQ OK -> {saida_awq} (servir com: vllm serve {saida_awq} --quantization awq)")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge LoRA + quantização AWQ (Fase 2, Nitro).")
    p.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adaptador", default="models/qlora-antt")
    p.add_argument("--merged", default="models/antt-merged")
    p.add_argument("--awq", default="models/antt-awq")
    p.add_argument("--pular-awq", action="store_true", help="só faz o merge")
    args = p.parse_args()
    merge(args.base, args.adaptador, args.merged)
    if not args.pular_awq:
        quantizar_awq(args.merged, args.awq)


if __name__ == "__main__":
    main()
