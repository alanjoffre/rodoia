"""Juiz factual com REFERÊNCIA (Fase 2) — a correção factual base vs. fine-tunado.

Agora que o corpus `normas.jsonl` existe (regenerado por `rag.baixar_normas`), medimos o
que faltava: a **correção factual** das respostas, não só se citam a norma. Para cada
pergunta do golden, um juiz **independente** (llama3.1:8b) pontua 0–1 quão fundamentada a
resposta está no TEXTO da(s) norma(s)-fonte esperada(s). Roda sobre as respostas já
geradas (`respostas_base.json`/`respostas_ft.json`) — não regenera.

Uso:  python -m rodoia.ft.juiz_factual reports/fase2_ft/respostas_base.json \
          reports/fase2_ft/respostas_ft.json reports/fase2_ft/avaliacao_factual.json
"""
from __future__ import annotations

import json
import sys

import requests

from rodoia.config import settings
from rodoia.estat import bootstrap_ic
from rodoia.juiz import extrair_json
from rodoia.proveniencia import carimbar

OLLAMA = "http://127.0.0.1:11434/api/chat"
JUIZ = "llama3.1:8b"

_PROMPT = (
    "Avalie a CORREÇÃO FACTUAL da RESPOSTA usando o TEXTO DE REFERÊNCIA (trecho da(s) "
    "resolução(ões) da ANTT) como verdade. Nota 0.0 a 1.0: 1.0 = correta e fundamentada "
    "na referência; 0.0 = errada, inventada ou sem base. Responda APENAS JSON: "
    '{{"nota": <0-1>}}\n\nPERGUNTA: {pergunta}\n\nREFERÊNCIA:\n{ref}\n\nRESPOSTA:\n{resp}'
)


def carregar_referencias(max_chars: int = 4000) -> dict[str, str]:
    """Mapa numero_da_norma -> trecho do texto (para o juiz usar como verdade)."""
    refs = {}
    for linha in settings.normas_jsonl.open(encoding="utf-8"):
        r = json.loads(linha)
        refs[r["numero"]] = r["texto"][:max_chars]
    return refs


def _nota(saida: str) -> float:
    try:
        return max(0.0, min(1.0, float(extrair_json(saida).get("nota", 0.0))))
    except (ValueError, TypeError):
        return 0.0


def _julgar_factual(pergunta: str, ref: str, resposta: str) -> float:
    r = requests.post(OLLAMA, json={
        "model": JUIZ, "format": "json", "stream": False, "options": {"temperature": 0.0},
        "messages": [{"role": "user",
                      "content": _PROMPT.format(pergunta=pergunta, ref=ref, resp=resposta)}],
    }, timeout=180)
    r.raise_for_status()
    return _nota(r.json()["message"]["content"])


def avaliar(respostas: list[dict], refs: dict[str, str]) -> tuple[float, list[float]]:
    notas = []
    for d in respostas:
        ref = "\n\n".join(refs.get(f, "") for f in d["fontes"])
        notas.append(_julgar_factual(d["consulta"], ref, d["resposta"]))
    return (sum(notas) / len(notas) if notas else 0.0), notas


def main() -> None:
    refs = carregar_referencias()
    base = json.load(open(sys.argv[1], encoding="utf-8"))
    ft = json.load(open(sys.argv[2], encoding="utf-8"))
    base_media, base_notas = avaliar(base, refs)
    ft_media, ft_notas = avaliar(ft, refs)
    res = carimbar({
        "metrica": "correcao factual (0-1) por juiz com referencia",
        "juiz": JUIZ, "n": len(base),
        "base_media": round(base_media, 3), "base_ic95": bootstrap_ic(base_notas),
        "ft_media": round(ft_media, 3), "ft_ic95": bootstrap_ic(ft_notas),
        "ganho": round(ft_media - base_media, 3),
    })
    open(sys.argv[3], "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"factual base={res['base_media']} {res['base_ic95']} | ft={res['ft_media']} "
          f"{res['ft_ic95']} | ganho={res['ganho']:+}")


if __name__ == "__main__":
    main()
