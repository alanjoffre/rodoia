"""Roteador — a decisão condicional central do grafo.

Classifica a pergunta no subconjunto de ferramentas necessárias (pode ser mais de uma:
casos combinados) ou marca `fora_de_escopo`. Usa o LLM-cérebro com saída JSON estrita e
um **fallback heurístico** por palavras-chave — se o LLM falhar/alucinar, o agente ainda
roteia de forma sensata (robustez é requisito, não enfeite).
"""
from __future__ import annotations

import json
import re

from rodoia.agente.estado import ROTAS

_RE_OBJ = re.compile(r"\{.*\}", re.S)

_SISTEMA = (
    "Você é o roteador de um agente sobre a ANTT (transporte rodoviário). Decida quais "
    "FERRAMENTAS são necessárias para responder à pergunta. Ferramentas:\n"
    "- regulatorio: normas/resoluções/regras da ANTT (ex.: vale-pedágio, tarifa, RNTRC).\n"
    "- entidades: extrair entidades jurídicas (leis, órgãos, datas) de um TEXTO fornecido.\n"
    "- dados: números de volume de tráfego nas praças de pedágio (ranking, volume, série).\n"
    "Responda APENAS com JSON: {\"rotas\": [...], \"motivo\": \"...\"}. Use [] se a pergunta "
    "não tem relação com a ANTT (fora de escopo). Pode combinar ferramentas."
)

# Heurística de reserva: sinais textuais fortes de cada ferramenta.
_PISTAS = {
    "regulatorio": ("resolução", "resolucao", "norma", "regra", "vale-pedágio", "vale-pedagio",
                    "tarifa", "rntrc", "regula", "legislação", "legislacao", "lei", "obrigató"),
    "entidades": ("entidade", "extraia", "extrair", "identifique as", "quais órgãos",
                  "quais orgaos", "quais leis", "trecho", "ementa", "cite as"),
    "dados": ("volume", "praça", "praca", "ranking", "maior", "líder", "lider", "tráfego",
              "trafego", "crescimento", "acumulado", "quantos veículos", "quantos veiculos"),
}


def _heuristica(pergunta: str) -> list[str]:
    p = pergunta.lower()
    return [r for r in ROTAS if any(t in p for t in _PISTAS[r])]


def _parse(saida: str) -> list[str] | None:
    """Extrai as rotas do JSON do LLM; None se não parseável (aí cai na heurística)."""
    m = _RE_OBJ.search(saida or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    rotas = obj.get("rotas")
    if not isinstance(rotas, list):
        return None
    return [r for r in rotas if r in ROTAS]


def rotear(pergunta: str, llm) -> dict:
    """Devolve {'rotas': [...], 'motivo': str}. rotas vazio ⇒ fora de escopo."""
    saida = ""
    try:
        saida = llm.gerar(pergunta, sistema=_SISTEMA)
    except Exception as e:  # LLM indisponível → heurística pura
        rotas = _heuristica(pergunta)
        return {"rotas": rotas, "motivo": f"heurística (LLM indisponível: {type(e).__name__})"}

    rotas = _parse(saida)
    if rotas is None:                       # LLM não deu JSON válido → heurística
        rotas = _heuristica(pergunta)
        return {"rotas": rotas, "motivo": "heurística (JSON inválido do roteador)"}
    if not rotas:                           # LLM disse [] mas a heurística pode discordar
        rotas = _heuristica(pergunta)
        motivo = "fora de escopo" if not rotas else "heurística (LLM retornou vazio)"
        return {"rotas": rotas, "motivo": motivo}

    motivo = ""
    m = _RE_OBJ.search(saida)
    if m:
        try:
            motivo = str(json.loads(m.group(0)).get("motivo", ""))[:200]
        except (ValueError, TypeError):
            motivo = ""
    return {"rotas": rotas, "motivo": motivo or "roteado pelo LLM"}
