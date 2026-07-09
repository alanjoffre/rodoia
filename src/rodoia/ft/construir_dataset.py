"""Construção do dataset de fine-tuning (Fase 2) a partir do domínio ANTT.

Qualidade de dado é metade do fine-tuning. Geramos pares instrução→resposta
ANCORADOS no texto das resoluções: para cada norma vigente, um LLM propõe
perguntas realistas e respostas objetivas que CITAM a resolução. O objetivo do
fine-tuning é ensinar o modelo pequeno a responder no estilo do domínio (direto,
citando a fonte), reduzindo a dependência de contexto longo.

Roda no Mac (usa Ollama, sem CUDA). Saída em formato de chat (messages), pronto
para o TRL/SFTTrainer na Nitro.

Uso:
    python -m rodoia.ft.construir_dataset --por-norma 3 --limite-normas 30
"""

from __future__ import annotations

import argparse
import json
import re

from rodoia.config import settings

SISTEMA = (
    "Você é um assistente jurídico especializado na regulação da ANTT (transporte "
    "rodoviário). Responda de forma objetiva e cite a resolução (número/ano) que "
    "embasa a resposta."
)

_PROMPT_GERADOR = (
    "A partir do texto da {numero} abaixo, gere {n} pares de pergunta e resposta que "
    "um profissional de transporte faria. Regras:\n"
    "- A pergunta deve ser natural e específica.\n"
    "- A resposta deve ser objetiva, baseada SÓ no texto, e citar a {numero}.\n"
    "- Não invente artigos ou números.\n\n"
    "Responda APENAS com um array JSON: "
    '[{{"pergunta": "...", "resposta": "..."}}]\n\n'
    "TEXTO DA {numero}:\n{texto}"
)

_RE_ARRAY = re.compile(r"\[.*\]", re.S)


def _parse_pares(saida: str) -> list[dict]:
    m = _RE_ARRAY.search(saida)
    if not m:
        return []
    try:
        dados = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    pares = []
    for d in dados if isinstance(dados, list) else []:
        p, r = (d.get("pergunta") or "").strip(), (d.get("resposta") or "").strip()
        if len(p) > 10 and len(r) > 20:  # filtro de qualidade mínima
            pares.append({"pergunta": p, "resposta": r})
    return pares


def gerar_pares_norma(norma: dict, llm, n: int = 3, max_chars: int = 6000) -> list[dict]:
    prompt = _PROMPT_GERADOR.format(
        numero=f"Resolução {norma['numero']}", n=n, texto=norma["texto"][:max_chars]
    )
    pares = _parse_pares(llm.gerar(prompt))
    for par in pares:
        par["fonte"] = norma["numero"]
    return pares


def para_chat(par: dict) -> dict:
    """Formato de treino (messages) + metadados de proveniência."""
    return {
        "messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": par["pergunta"]},
            {"role": "assistant", "content": par["resposta"]},
        ],
        "fonte": par["fonte"],
    }


def construir(llm, por_norma: int = 3, limite_normas: int | None = None) -> dict:
    normas = [
        json.loads(linha)
        for linha in settings.normas_jsonl.open(encoding="utf-8")
        if json.loads(linha)["vigente"]
    ]
    normas.sort(key=lambda r: -r["ano"])
    if limite_normas:
        normas = normas[:limite_normas]

    registros, vistos = [], set()
    for i, norma in enumerate(normas, 1):
        try:
            pares = gerar_pares_norma(norma, llm, n=por_norma)
        except Exception as e:
            print(f"  [{i}/{len(normas)}] {norma['numero']} ERRO: {str(e)[:50]}")
            continue
        for par in pares:
            chave = par["pergunta"].lower()[:80]
            if chave not in vistos:
                vistos.add(chave)
                registros.append(para_chat(par))
        if i % 5 == 0 or i == len(normas):
            print(f"  [{i}/{len(normas)}] pares acumulados: {len(registros)}")

    saida = settings.data_processed / "ft_dataset.jsonl"
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("w", encoding="utf-8") as fh:
        for reg in registros:
            fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
    print(f"OK: {len(registros)} exemplos -> {saida}")
    return {"n_exemplos": len(registros), "n_normas": len(normas), "saida": str(saida)}


def main() -> None:
    from rodoia.rag.llm import OllamaLLM

    p = argparse.ArgumentParser(description="Constrói o dataset de fine-tuning (Fase 2).")
    p.add_argument("--por-norma", type=int, default=3)
    p.add_argument("--limite-normas", type=int, default=None)
    args = p.parse_args()
    construir(OllamaLLM(), por_norma=args.por_norma, limite_normas=args.limite_normas)


if __name__ == "__main__":
    main()
