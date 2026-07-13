"""Ferramentas reais do agente — finas cascas sobre as Fases 1–3.

Cada construtor devolve um `Callable[[str], dict]` pronto para injetar em `DepsAgente`.
Mantê-las finas e injetáveis é o que permite testar o grafo sem GPU/Ollama/DuckDB.
"""
from __future__ import annotations

import re

from rodoia.agente.estado import DepsAgente


# ── Ferramenta 1: regulatório (RAG da Fase 1) ────────────────────────────────
def regulatorio_real(recuperador, llm):
    """Envelopa o RAG seguro da Fase 1 (retrieval híbrido + geração + guardrails)."""
    from rodoia.rag.gerar import responder_seguro

    def _f(pergunta: str) -> dict:
        r = responder_seguro(pergunta, recuperador, llm)
        return {"tipo": "regulatorio", "resposta": r["resposta"], "fontes": r["fontes"]}

    return _f


# ── Ferramenta 2: entidades (modelo fine-tunado da Fase 2) ───────────────────
def entidades_real(llm_ft):
    """Chama o modelo NER fine-tunado (via vLLM/OpenAI-compat) e parseia as entidades."""
    from rodoia.ner.avaliar_generativo import parse_entidades
    from rodoia.ner.generativo import SISTEMA

    def _f(texto: str) -> dict:
        saida = llm_ft.gerar(texto, sistema=SISTEMA)
        ents = [{"texto": t, "tipo": tp} for t, tp in sorted(parse_entidades(saida))]
        return {"tipo": "entidades", "entidades": ents}

    return _f


# ── Ferramenta 3: dados estruturados (camada de acesso da Fase 3 + cálculo) ──
_RE_PRACA = re.compile(r"pra[çc]a\s+([\w\s./-]{2,40}?)(?:\?|$|\.|,| no | em | teve | tem )", re.I)
# palavras que seguem "praça" mas NÃO são o nome de uma praça (ex.: "praça líder")
_NAO_NOME = {"lider", "líder", "de", "com", "mais", "maior", "campeã", "campea"}


def _crescimento_yoy(serie: list[dict]) -> float | None:
    """Cálculo determinístico: variação % dos últimos 12 meses vs. os 12 anteriores."""
    if len(serie) < 24:
        return None
    vols = [r["volume"] for r in serie]
    ult, ant = sum(vols[-12:]), sum(vols[-24:-12])
    return round((ult - ant) / ant * 100, 1) if ant else None


def dados_real(db=None):
    """Interpreta a pergunta (heurística explícita) e chama a camada de acesso parametrizada.
    Intenções cobertas: ranking de praças, volume/crescimento de uma praça citada."""
    from rodoia.dados.acesso import ranking_pracas, serie_mensal, volume_praca

    def _f(pergunta: str) -> dict:
        p = pergunta.lower()
        out: dict = {"tipo": "dados"}
        m = _RE_PRACA.search(pergunta)
        nome = m.group(1).strip() if m else ""
        tem_nome = bool(nome) and nome.lower() not in _NAO_NOME
        if any(t in p for t in ("maior", "ranking", "líder", "lider", "top")) or not tem_nome:
            ranking = ranking_pracas(top=5, db=db)
            out["ranking"] = ranking
            # encadeia: se pergunta pede crescimento do líder, calcula do 1º colocado
            if ranking and any(t in p for t in ("crescimento", "cresceu", "variação", "variacao")):
                lider = ranking[0]["praca"]
                out["lider"] = lider
                out["crescimento_lider_yoy_pct"] = _crescimento_yoy(serie_mensal(lider, db=db))
        if tem_nome:
            serie = serie_mensal(nome, db=db)
            out["praca"] = nome
            out["volume_total"] = volume_praca(nome, db=db)
            out["crescimento_yoy_pct"] = _crescimento_yoy(serie)
            out["meses"] = len(serie)
        return out

    return _f


# ── Montagem das dependências reais (demo) ───────────────────────────────────
def deps_reais(com_reranker: bool = False) -> DepsAgente:  # rerank não ajuda (docs/09) → off
    """Fia o agente com as ferramentas de produção: Ollama (cérebro), RAG (F1),
    vLLM do modelo FT (F2, se disponível) e DuckDB (F3)."""
    from rodoia.config import settings
    from rodoia.rag.avaliacao_retrieval import carregar_recuperador
    from rodoia.rag.llm import OllamaLLM, OpenAICompatLLM

    cerebro = OllamaLLM()
    rec = carregar_recuperador(com_reranker=com_reranker)
    # O modelo FT é servido pelo vLLM (OpenAI-compat). Se não estiver no ar, a ferramenta
    # de entidades falha e o grafo degrada — tratado no nó `executar`.
    llm_ft = OpenAICompatLLM(modelo=getattr(settings, "ft_model", "rodoia-ner-ft"),
                             base_url="http://localhost:8001/v1")
    return DepsAgente(
        llm_cerebro=cerebro,
        regulatorio=regulatorio_real(rec, cerebro),
        entidades=entidades_real(llm_ft),
        dados=dados_real(),
    )
