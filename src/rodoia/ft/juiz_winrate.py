"""LLM-as-judge win-rate base vs. fine-tunado (Fase 2) — sinal de generalização.

Um juiz **independente** (Ollama `qwen2.5:7b`, família/checkpoint distinto do base-3B
e do FT) compara, par a par, as respostas do base e do FT às perguntas do
CONJUNTO_DOURADO. Para mitigar viés de ordem, cada par é julgado **duas vezes com as
posições trocadas**; só conta vitória quando o juiz é consistente nas duas ordens
(caso contrário, empate).

Critério do juiz: **qualidade como resposta de assistente jurídico-regulatório da
ANTT** (clareza, objetividade, estrutura, uso apropriado do formato de citação) —
NÃO correção factual verificada (o corpus de normas via DVC está ausente nesta
máquina; ver docs/11). Portanto mede *qualidade de apresentação/generalização de
estilo*, não acerto factual.

Uso:  python -m rodoia.ft.juiz_winrate \
          reports/fase2_ft/respostas_base.json reports/fase2_ft/respostas_ft.json \
          reports/fase2_ft/winrate_ft.json
"""
from __future__ import annotations

import json
import re
import sys

import requests

OLLAMA = "http://127.0.0.1:11434/api/chat"
JUIZ = "qwen2.5:7b"
_RE_JSON = re.compile(r"\{[^{}]*\}", re.S)

_PROMPT = (
    "Você é um avaliador. Compare duas respostas à mesma PERGUNTA sobre a regulação "
    "da ANTT. Escolha a que é MELHOR como resposta de um assistente jurídico-"
    "regulatório: clareza, objetividade, estrutura e uso apropriado do formato de "
    "citação de resolução. Não premie inventar números; se ambas inventam, decida "
    "pela qualidade geral. Responda APENAS JSON: {{\"melhor\": \"A\"|\"B\"|\"empate\"}}.\n\n"
    "PERGUNTA: {pergunta}\n\nRESPOSTA A:\n{a}\n\nRESPOSTA B:\n{b}"
)


def _julgar(pergunta: str, a: str, b: str) -> str:
    msg = _PROMPT.format(pergunta=pergunta, a=a, b=b)
    r = requests.post(OLLAMA, json={
        "model": JUIZ, "format": "json", "stream": False,
        "options": {"temperature": 0.0},
        "messages": [{"role": "user", "content": msg}],
    }, timeout=180)
    r.raise_for_status()
    saida = r.json()["message"]["content"]
    m = _RE_JSON.search(saida)
    if not m:
        return "empate"
    try:
        v = json.loads(m.group(0)).get("melhor", "empate").strip().upper()
    except (ValueError, TypeError):
        return "empate"
    return {"A": "A", "B": "B"}.get(v, "empate")


def comparar(base_path: str, ft_path: str, saida: str) -> dict:
    base = json.load(open(base_path, encoding="utf-8"))
    ft = json.load(open(ft_path, encoding="utf-8"))
    casos = []
    ft_wins = base_wins = empates = 0
    for b, f in zip(base, ft):
        # ordem 1: A=base, B=ft   |   ordem 2: A=ft, B=base
        r1 = _julgar(b["consulta"], b["resposta"], f["resposta"])
        r2 = _julgar(b["consulta"], f["resposta"], b["resposta"])
        if r1 == "B" and r2 == "A":
            venc = "ft"; ft_wins += 1
        elif r1 == "A" and r2 == "B":
            venc = "base"; base_wins += 1
        else:
            venc = "empate"; empates += 1
        casos.append({"consulta": b["consulta"], "ordem1": r1, "ordem2": r2, "vencedor": venc})
    n = len(casos)
    res = {
        "juiz": JUIZ,
        "criterio": "qualidade de resposta regulatoria (nao correcao factual verificada)",
        "n": n,
        "ft_wins": ft_wins, "base_wins": base_wins, "empates": empates,
        "ft_win_rate": round(ft_wins / n, 3),
        "base_win_rate": round(base_wins / n, 3),
        "casos": casos,
    }
    json.dump(res, open(saida, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"FT wins={ft_wins}  base wins={base_wins}  empates={empates}  "
          f"(FT win-rate={res['ft_win_rate']})")
    print("relatorio:", saida)
    return res


if __name__ == "__main__":
    comparar(sys.argv[1], sys.argv[2], sys.argv[3])
