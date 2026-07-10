"""LLM-as-judge win-rate base vs. fine-tunado (Fase 2) — sinal de generalização.

Um juiz **independente** (Ollama `qwen2.5:7b`, família/checkpoint distinto do base-3B
e do FT) compara, par a par, as respostas do base e do FT às perguntas do
CONJUNTO_DOURADO. Para mitigar viés de ordem, cada par é julgado **duas vezes com as
posições trocadas**; só conta vitória quando o juiz é consistente nas duas ordens
(caso contrário, empate).

Critério do juiz: **qualidade como resposta de assistente jurídico-regulatório da
ANTT** (clareza, objetividade, estrutura, uso apropriado do formato de citação) —
NÃO correção factual verificada (o corpus de normas via DVC está ausente nesta
máquina; ver docs/11).

Modo `--controlar-comprimento`: como o base é ~4x mais longo que o FT, o juiz pode ter
**viés de comprimento**. Nesse modo, ambas as respostas são **truncadas ao mesmo
tamanho** (a menor do par) e o juiz é instruído a julgar só o conteúdo apresentado —
isolando qualidade de verbosidade. Rodar os dois modos e comparar blinda o achado.

Uso:  python -m rodoia.ft.juiz_winrate <base.json> <ft.json> <saida.json> [--controlar-comprimento]
"""
from __future__ import annotations

import json
import re
import sys

import requests

from rodoia.estat import wilson
from rodoia.proveniencia import carimbar

OLLAMA = "http://127.0.0.1:11434/api/chat"
JUIZ = "qwen2.5:7b"
_RE_JSON = re.compile(r"\{[^{}]*\}", re.S)

_BASE_PROMPT = (
    "Você é um avaliador. Compare duas respostas à mesma PERGUNTA sobre a regulação "
    "da ANTT. Escolha a que é MELHOR como resposta de um assistente jurídico-"
    "regulatório: clareza, objetividade, estrutura e uso apropriado do formato de "
    "citação de resolução. Não premie inventar números; se ambas inventam, decida "
    "pela qualidade geral.{extra} Responda APENAS JSON: "
    '{{"melhor": "A"|"B"|"empate"}}.\n\n'
    "PERGUNTA: {pergunta}\n\nRESPOSTA A:\n{a}\n\nRESPOSTA B:\n{b}"
)
_EXTRA_COMPRIMENTO = (
    " As duas respostas foram truncadas ao MESMO tamanho; julgue só o conteúdo "
    "apresentado e NÃO considere o comprimento."
)


# ---- funções puras (testáveis sem rede) ----

def _parse_veredito(saida: str) -> str:
    """Extrai 'A' | 'B' | 'empate' do JSON do juiz (robusto a lixo em volta)."""
    m = _RE_JSON.search(saida or "")
    if not m:
        return "empate"
    try:
        v = json.loads(m.group(0)).get("melhor", "empate")
    except (ValueError, TypeError):
        return "empate"
    v = str(v).strip().upper()
    return {"A": "A", "B": "B"}.get(v, "empate")


def _decidir(r1: str, r2: str) -> str:
    """r1 julga (A=base, B=ft); r2 julga (A=ft, B=base) — posições trocadas.
    Só há vencedor se o juiz for consistente nas duas ordens; senão, empate."""
    if r1 == "B" and r2 == "A":
        return "ft"
    if r1 == "A" and r2 == "B":
        return "base"
    return "empate"


def _truncar_par(a: str, b: str) -> tuple[str, str]:
    """Trunca ambas ao tamanho da menor (isola conteúdo de verbosidade)."""
    n = min(len(a), len(b))
    return a[:n], b[:n]


# ---- chamada ao juiz (rede) ----

def _julgar(pergunta: str, a: str, b: str, controlar: bool) -> str:
    prompt = _BASE_PROMPT.format(
        pergunta=pergunta, a=a, b=b,
        extra=_EXTRA_COMPRIMENTO if controlar else "",
    )
    r = requests.post(OLLAMA, json={
        "model": JUIZ, "format": "json", "stream": False,
        "options": {"temperature": 0.0},
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=180)
    r.raise_for_status()
    return _parse_veredito(r.json()["message"]["content"])


def comparar(base_path: str, ft_path: str, saida: str, controlar_comprimento: bool = False) -> dict:
    base = json.load(open(base_path, encoding="utf-8"))
    ft = json.load(open(ft_path, encoding="utf-8"))
    casos = []
    ft_wins = base_wins = empates = 0
    for b, f in zip(base, ft):
        rb, rf = b["resposta"], f["resposta"]
        if controlar_comprimento:
            rb, rf = _truncar_par(rb, rf)
        r1 = _julgar(b["consulta"], rb, rf, controlar_comprimento)  # A=base, B=ft
        r2 = _julgar(b["consulta"], rf, rb, controlar_comprimento)  # A=ft,  B=base
        venc = _decidir(r1, r2)
        ft_wins += venc == "ft"
        base_wins += venc == "base"
        empates += venc == "empate"
        casos.append({"consulta": b["consulta"], "ordem1": r1, "ordem2": r2, "vencedor": venc})
    n = len(casos)
    res = carimbar({
        "juiz": JUIZ,
        "criterio": "qualidade de resposta regulatoria (nao correcao factual verificada)",
        "controle_comprimento": controlar_comprimento,
        "n": n,
        "ft_wins": ft_wins, "base_wins": base_wins, "empates": empates,
        "ft_win_rate": round(ft_wins / n, 3),
        "ft_win_rate_ic95": wilson(ft_wins, n),
        "base_win_rate": round(base_wins / n, 3),
        "base_win_rate_ic95": wilson(base_wins, n),
        "casos": casos,
    })
    json.dump(res, open(saida, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    modo = "COM controle de comprimento" if controlar_comprimento else "sem controle"
    print(f"[{modo}] FT wins={ft_wins}  base wins={base_wins}  empates={empates}  "
          f"(FT win-rate={res['ft_win_rate']})")
    print("relatorio:", saida)
    return res


if __name__ == "__main__":
    controlar = "--controlar-comprimento" in sys.argv[4:]
    comparar(sys.argv[1], sys.argv[2], sys.argv[3], controlar_comprimento=controlar)
