"""Grafo do agente (LangGraph) — estado, nós e arestas condicionais.

Fluxo:
    START → guardrail ─(injection)→ bloqueio → END
                 │
              roteador ─(sem rota)→ escopo → END
                 │
             executar (chama as ferramentas escolhidas; falha de uma degrada, não derruba)
                 │
             sintetizar → END

As arestas condicionais (guardrail e roteador) são a "decisão condicional real": o caminho
percorrido depende do conteúdo da pergunta, e casos combinados acionam 2+ ferramentas.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rodoia.agente.estado import DepsAgente, EstadoAgente
from rodoia.agente.roteador import rotear
from rodoia.rag.seguranca import detectar_injection, mascarar_pii

_SISTEMA_SINTESE = (
    "Você é um assistente sobre a ANTT (transporte rodoviário brasileiro). Responda à pergunta "
    "do usuário usando SOMENTE as evidências fornecidas pelas ferramentas. Cite as fontes "
    "regulatórias quando houver (número da resolução) e deixe claro quando um número vem dos "
    "dados de tráfego. Se as evidências forem insuficientes, diga isso. Seja direto, em português."
)


def _no_guardrail(estado: EstadoAgente, deps: DepsAgente) -> dict:
    inj, motivo = detectar_injection(estado["pergunta"])
    return {"bloqueado": bool(inj), "motivo_bloqueio": motivo, "trajetoria": [
        {"no": "guardrail", "bloqueado": bool(inj), "motivo": motivo}]}


def _no_roteador(estado: EstadoAgente, deps: DepsAgente) -> dict:
    r = rotear(estado["pergunta"], deps.llm_cerebro)
    return {
        "rotas": r["rotas"], "motivo_rota": r["motivo"],
        "fora_de_escopo": not r["rotas"],
        "trajetoria": [{"no": "roteador", "rotas": r["rotas"], "motivo": r["motivo"]}],
    }


def _no_executar(estado: EstadoAgente, deps: DepsAgente) -> dict:
    evidencias, passos = {}, []
    for rota in estado.get("rotas", []):
        try:
            evidencias[rota] = deps.ferramenta(rota)(estado["pergunta"])
            passos.append({"no": "ferramenta", "rota": rota, "ok": True})
        except Exception as e:  # ferramenta indisponível → degrada, registra, segue
            evidencias[rota] = {"tipo": rota, "erro": f"{type(e).__name__}: {e}"}
            passos.append({"no": "ferramenta", "rota": rota, "ok": False,
                           "erro": type(e).__name__})
    return {"evidencias": evidencias, "trajetoria": passos}


def _coletar_fontes(evidencias: dict) -> list[str]:
    fontes = []
    reg = evidencias.get("regulatorio", {})
    fontes += [f"Resolução {n}" for n in reg.get("fontes", [])]
    if "dados" in evidencias and "erro" not in evidencias["dados"]:
        fontes.append("Dados abertos ANTT — Volume de Pedágio")
    return list(dict.fromkeys(fontes))


def _no_sintetizar(estado: EstadoAgente, deps: DepsAgente) -> dict:
    evidencias = estado.get("evidencias", {})
    contexto = "\n".join(f"[{k}] {v}" for k, v in evidencias.items())
    prompt = (f"Pergunta: {estado['pergunta']}\n\n"
              f"Evidências das ferramentas:\n{contexto}\n\nResposta:")
    try:
        resposta = deps.llm_cerebro.gerar(prompt, sistema=_SISTEMA_SINTESE)
    except Exception as e:
        resposta = f"Não foi possível sintetizar a resposta (LLM indisponível: {type(e).__name__})."
    return {"resposta": mascarar_pii(resposta), "fontes": _coletar_fontes(evidencias),
            "trajetoria": [{"no": "sintetizar", "n_evidencias": len(evidencias)}]}


def _no_bloqueio(estado: EstadoAgente, deps: DepsAgente) -> dict:
    return {"resposta": "Solicitação bloqueada: padrão suspeito de manipulação de instruções. "
            "Reformule sua pergunta sobre a ANTT.", "fontes": []}


def _no_escopo(estado: EstadoAgente, deps: DepsAgente) -> dict:
    return {"resposta": "Esta pergunta está fora do escopo do agente, que cobre a regulação e os "
            "dados abertos de transporte rodoviário da ANTT.", "fontes": []}


def _apos_guardrail(estado: EstadoAgente) -> str:
    return "bloqueio" if estado.get("bloqueado") else "roteador"


def _apos_roteador(estado: EstadoAgente) -> str:
    return "escopo" if estado.get("fora_de_escopo") else "executar"


def construir_agente(deps: DepsAgente):
    """Compila o grafo com as dependências injetadas. Retorna um objeto invocável (.invoke)."""
    def _bind(fn):
        return lambda estado: fn(estado, deps)

    g = StateGraph(EstadoAgente)
    g.add_node("guardrail", _bind(_no_guardrail))
    g.add_node("roteador", _bind(_no_roteador))
    g.add_node("executar", _bind(_no_executar))
    g.add_node("sintetizar", _bind(_no_sintetizar))
    g.add_node("bloqueio", _bind(_no_bloqueio))
    g.add_node("escopo", _bind(_no_escopo))

    g.add_edge(START, "guardrail")
    g.add_conditional_edges("guardrail", _apos_guardrail,
                            {"bloqueio": "bloqueio", "roteador": "roteador"})
    g.add_conditional_edges("roteador", _apos_roteador,
                            {"escopo": "escopo", "executar": "executar"})
    g.add_edge("executar", "sintetizar")
    for fim in ("sintetizar", "bloqueio", "escopo"):
        g.add_edge(fim, END)
    return g.compile()


def responder(pergunta: str, deps: DepsAgente) -> dict:
    """Conveniência: roda o agente e devolve um dicionário limpo (resposta, fontes, rotas,
    trajetória) — usado pela API e pela avaliação de trajetória."""
    agente = construir_agente(deps)
    estado = agente.invoke({"pergunta": pergunta, "trajetoria": []})
    return {
        "pergunta": pergunta,
        "resposta": estado.get("resposta", ""),
        "fontes": estado.get("fontes", []),
        "rotas": estado.get("rotas", []),
        "bloqueado": estado.get("bloqueado", False),
        "fora_de_escopo": estado.get("fora_de_escopo", False),
        "trajetoria": estado.get("trajetoria", []),
    }
