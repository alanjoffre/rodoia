"""Perplexidade de domínio (Fase 2) — fit ao registro ANTT e GENERALIZAÇÃO.

Mede a PPL de um modelo sobre as respostas de domínio de um dataset, via
`prompt_logprobs` do vLLM. Dois usos:
- **in-sample** (respostas do treino) → mede *fit* de domínio;
- **held-out** (respostas de normas NÃO vistas no treino — ver `split_dataset.py`) →
  mede *generalização* do registro.

O parâmetro `quantizacao` permite também medir o **custo de qualidade da quantização**
(PPL em bf16 vs fp8) — o eixo que faltava no trade-off (só a memória estava medida).

Uso:
    python -m rodoia.ft.perplexidade <modelo> <saida.json> [dataset.jsonl] [fp8|bfloat16]
"""
from __future__ import annotations

import json
import math
import sys

from rodoia.config import settings
from rodoia.proveniencia import carimbar


def _textos_dominio(caminho: str) -> list[str]:
    """Respostas do assistente (o alvo que o modelo deve modelar)."""
    textos = []
    for linha in open(caminho, encoding="utf-8"):
        ex = json.loads(linha)
        for m in ex["messages"]:
            if m["role"] == "assistant":
                textos.append(m["content"])
    return textos


def _agregar_ppl(prompt_logprobs_por_texto: list) -> dict:
    """Agrega NLL → perplexidade (função pura, testável sem GPU).
    Entrada: lista (por texto) de listas de posições; cada posição é None (1º token)
    ou um dict {token_id: obj_com_.logprob} com exatamente o token real."""
    soma_nll = 0.0
    n_tokens = 0
    ppls = []
    for pl in prompt_logprobs_por_texto:
        nll_seq, cnt = 0.0, 0
        for pos in pl:
            if pos is None:
                continue
            lp = next(iter(pos.values())).logprob
            nll_seq += -lp
            cnt += 1
        if cnt:
            ppls.append(math.exp(nll_seq / cnt))
            soma_nll += nll_seq
            n_tokens += cnt
    return {
        "n_tokens": n_tokens,
        "ppl_micro": round(math.exp(soma_nll / n_tokens), 3) if n_tokens else None,
        "ppl_macro": round(sum(ppls) / len(ppls), 3) if ppls else None,
    }


def perplexidade(modelo: str, saida: str, dataset: str | None = None,
                 quantizacao: str = "fp8") -> dict:
    from vllm import LLM, SamplingParams

    dataset = dataset or str(settings.data_processed / "ft_dataset.jsonl")
    textos = _textos_dominio(dataset)
    kwargs = {"max_model_len": 2048, "gpu_memory_utilization": 0.80, "enforce_eager": True}
    if quantizacao and quantizacao != "none":
        kwargs["quantization"] = quantizacao
    llm = LLM(model=modelo, **kwargs)
    outs = llm.generate(textos, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0))

    res = carimbar({
        "modelo": modelo,
        "dataset": dataset,
        "quantizacao": quantizacao,
        "n_textos": len(textos),
        **_agregar_ppl([o.prompt_logprobs for o in outs]),
    })
    json.dump(res, open(saida, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"PPL {modelo} [{quantizacao}] em {len(textos)} textos: "
          f"micro={res['ppl_micro']} macro={res['ppl_macro']}")
    return res


if __name__ == "__main__":
    ds = sys.argv[3] if len(sys.argv) > 3 else None
    quant = sys.argv[4] if len(sys.argv) > 4 else "fp8"
    perplexidade(sys.argv[1], sys.argv[2], ds, quant)
