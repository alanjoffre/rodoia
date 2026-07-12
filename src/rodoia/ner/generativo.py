"""NER GENERATIVO (Fase 2) — o mesmo LeNER-Br resolvido como extração por LLM.

Converte o NER (rótulo por token) numa tarefa generativa: dada a sentença, o modelo
produz as entidades como JSON [{texto, tipo}]. Permite fine-tunar o Qwen (QLoRA + vLLM,
reusando a infra) e comparar **F1 de entidade** do FT vs. base zero-shot vs. o teto
BERTimbau — uma vitória de fine-tuning com métrica DURA.
"""
from __future__ import annotations

import json

from rodoia.config import settings
from rodoia.ner.lener import LABELS, carregar

TIPOS = ["ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL", "LEGISLACAO", "JURISPRUDENCIA"]
SISTEMA = (
    "Você é um extrator de entidades nomeadas em textos jurídicos brasileiros. Extraia as "
    "entidades da sentença e responda APENAS com JSON: uma lista de objetos "
    '{"texto": <trecho>, "tipo": <TIPO>}. Tipos válidos: '
    "ORGANIZACAO, PESSOA, TEMPO, LOCAL, LEGISLACAO, JURISPRUDENCIA. Se não houver, responda []."
)


def entidades_bio(tokens: list[str], tags: list[int]) -> list[tuple[str, str]]:
    """Extrai entidades (texto, tipo) de uma sentença anotada em BIO."""
    ents, atual, tipo = [], [], None
    for tok, tid in zip(tokens, tags, strict=False):
        rot = LABELS[tid]
        if rot.startswith("B-"):
            if atual:
                ents.append((" ".join(atual), tipo))
            atual, tipo = [tok], rot[2:]
        elif rot.startswith("I-") and tipo == rot[2:]:
            atual.append(tok)
        else:
            if atual:
                ents.append((" ".join(atual), tipo))
            atual, tipo = [], None
    if atual:
        ents.append((" ".join(atual), tipo))
    return ents


def _norm(texto: str) -> str:
    return " ".join(texto.lower().split())


def como_conjunto(ents: list[tuple[str, str]]) -> set[tuple[str, str]]:
    """Entidades normalizadas p/ comparação exata (texto minúsculo + tipo)."""
    return {(_norm(t), tp) for t, tp in ents}


def para_chat(sent: dict) -> dict:
    ents = entidades_bio(sent["tokens"], sent["ner_tags"])
    alvo = json.dumps([{"texto": t, "tipo": tp} for t, tp in ents], ensure_ascii=False)
    return {"messages": [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": " ".join(sent["tokens"])},
        {"role": "assistant", "content": alvo},
    ]}


def construir_dataset(max_treino: int | None = None) -> dict:
    """Gera ner_train.jsonl (formato chat p/ QLoRA) e ner_test.jsonl (com gold)."""
    treino = [para_chat(s) for s in carregar("train")[: max_treino or None]]
    teste = carregar("test")
    d = settings.data_processed
    with (d / "ner_train.jsonl").open("w", encoding="utf-8") as fh:
        for r in treino:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (d / "ner_test.jsonl").open("w", encoding="utf-8") as fh:
        for s in teste:
            fh.write(json.dumps({
                "texto": " ".join(s["tokens"]),
                "entidades": [[t, tp] for t, tp in entidades_bio(s["tokens"], s["ner_tags"])],
            }, ensure_ascii=False) + "\n")
    print(f"ner_train={len(treino)} ner_test={len(teste)} -> {d}")
    return {"n_treino": len(treino), "n_teste": len(teste)}


if __name__ == "__main__":
    construir_dataset()
