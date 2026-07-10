"""Perplexidade de domínio (Fase 2) — sinal quantitativo de fit ao domínio ANTT.

Mede a perplexidade (PPL) de um modelo sobre as respostas de domínio do
`ft_dataset.jsonl`, via `prompt_logprobs` do vLLM (fp8). Comparar BASE vs.
FINE-TUNADO quantifica **quanto o QLoRA aproximou a distribuição do modelo ao
registro jurídico-regulatório da ANTT** — complementa a acurácia de citação
(que mede fato, não estilo). PPL menor no FT = aprendeu o domínio.

Nota metodológica: as respostas são as usadas no treino (in-sample), então a PPL
mede **fit de domínio**, não generalização factual — reportado honestamente em
`docs/11`. O sinal de generalização vem do win-rate no CONJUNTO_DOURADO (juiz).

Uso:
    python -m rodoia.ft.perplexidade Qwen/Qwen2.5-3B-Instruct /tmp/ppl_base.json
    python -m rodoia.ft.perplexidade models/antt-merged        /tmp/ppl_ft.json
"""
from __future__ import annotations

import json
import math
import sys


def _textos_dominio(caminho: str = "data/processed/ft_dataset.jsonl") -> list[str]:
    """Respostas do assistente (o alvo que o modelo deve modelar)."""
    textos = []
    for linha in open(caminho, encoding="utf-8"):
        ex = json.loads(linha)
        for m in ex["messages"]:
            if m["role"] == "assistant":
                textos.append(m["content"])
    return textos


def perplexidade(modelo: str, saida: str) -> dict:
    from vllm import LLM, SamplingParams

    textos = _textos_dominio()
    llm = LLM(model=modelo, quantization="fp8", max_model_len=2048,
              gpu_memory_utilization=0.80, enforce_eager=True)
    # prompt_logprobs=0 -> devolve o logprob do próprio token de cada posição do prompt.
    outs = llm.generate(textos, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0))

    soma_nll = 0.0
    n_tokens = 0
    ppls = []
    for o in outs:
        pl = o.prompt_logprobs  # lista; pl[0] é None (1º token não tem contexto)
        nll_seq = 0.0
        cnt = 0
        for pos in pl:
            if pos is None:
                continue
            # pos: {token_id: Logprob}. Há exatamente 1 (o token real) com prompt_logprobs=0.
            lp = next(iter(pos.values())).logprob
            nll_seq += -lp
            cnt += 1
        if cnt:
            ppls.append(math.exp(nll_seq / cnt))
            soma_nll += nll_seq
            n_tokens += cnt

    res = {
        "modelo": modelo,
        "n_textos": len(textos),
        "n_tokens": n_tokens,
        "ppl_micro": round(math.exp(soma_nll / n_tokens), 3),  # agregada por token
        "ppl_macro": round(sum(ppls) / len(ppls), 3),          # média das PPLs por texto
    }
    json.dump(res, open(saida, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"PPL {modelo}: micro={res['ppl_micro']}  macro={res['ppl_macro']}  (n={len(textos)} textos)")
    return res


if __name__ == "__main__":
    perplexidade(sys.argv[1], sys.argv[2])
