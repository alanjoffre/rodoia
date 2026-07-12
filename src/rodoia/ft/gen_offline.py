"""Gera respostas de um modelo (fp8, offline vLLM) para o golden de domínio.

Usa o MESMO `CONJUNTO_DOURADO` da avaliação de retrieval (perguntas de intenção real,
com a resolução-fonte conhecida) — fonte única de verdade, sem golden divergente
hard-coded. As respostas alimentam `aval_cite` (citação) e `juiz_winrate` (win-rate).
"""
from __future__ import annotations

import json
import sys

SISTEMA = "Responda sobre a regulação da ANTT, citando a resolução."


def montar_conversas(golden: list[dict], sistema: str = SISTEMA) -> list[list[dict]]:
    """Constrói as conversas (system+user) para cada pergunta do golden. Pura/testável."""
    return [
        [{"role": "system", "content": sistema}, {"role": "user", "content": g["consulta"]}]
        for g in golden
    ]


def montar_respostas(golden: list[dict], textos: list[str]) -> list[dict]:
    """Casa cada pergunta do golden com o texto gerado (preserva TODAS as fontes)."""
    return [
        {"consulta": g["consulta"], "fontes": g["fontes"], "resposta": t.strip()}
        for g, t in zip(golden, textos, strict=False)
    ]


def main() -> None:
    from vllm import LLM, SamplingParams

    from rodoia.rag.avaliacao_retrieval import CONJUNTO_DOURADO

    modelo, out = sys.argv[1], sys.argv[2]
    llm = LLM(model=modelo, quantization="fp8", max_model_len=2048,
              gpu_memory_utilization=0.80, enforce_eager=True)
    convs = montar_conversas(CONJUNTO_DOURADO)
    saidas = llm.chat(convs, SamplingParams(max_tokens=256, temperature=0.0))
    respostas = montar_respostas(CONJUNTO_DOURADO, [s.outputs[0].text for s in saidas])
    json.dump(respostas, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"SALVO {out} n={len(respostas)}")
    print("DONE_OFFLINE")


if __name__ == "__main__":
    main()
