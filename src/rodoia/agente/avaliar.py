"""Avaliação de TRAJETÓRIA do agente (Fase 4).

Duas medidas, complementares:
1. **Acerto de roteamento** (objetivo): o conjunto de ferramentas acionadas bate com o
   esperado por caso (Jaccard + acerto exato). Não depende de juiz — é determinístico.
2. **Qualidade da trajetória/resposta** (LLM-as-judge INDEPENDENTE): um segundo modelo
   (llama3.1, distinto do cérebro qwen) julga se a rota fez sentido e se a resposta se
   apoia nas evidências. Mesma disciplina de juiz independente das Fases 1–2.

Uso:  python -m rodoia.agente.avaliar
"""
from __future__ import annotations

import json

from rodoia.agente.casos import CASOS_TRAJETORIA
from rodoia.agente.grafo import responder
from rodoia.config import REPO_ROOT
from rodoia.proveniencia import carimbar

_SISTEMA_JUIZ = (
    "Você é um avaliador rigoroso de um agente sobre a ANTT. Dada a pergunta, as ferramentas "
    "acionadas e a resposta, julgue de 0 a 2: rota adequada? resposta fundamentada nas "
    "evidências (sem inventar)? Responda APENAS JSON: "
    '{"rota_ok": 0|1|2, "resposta_ok": 0|1|2, "justificativa": "..."}.'
)


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return 1.0 if not sa and not sb else len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def _julgar(caso: dict, res: dict, juiz) -> dict:
    prompt = (f"Pergunta: {caso['pergunta']}\nFerramentas acionadas: {res['rotas']}\n"
              f"Bloqueado: {res['bloqueado']} | Fora de escopo: {res['fora_de_escopo']}\n"
              f"Resposta: {res['resposta']}")
    try:
        saida = juiz.gerar(prompt, sistema=_SISTEMA_JUIZ)
        import re
        m = re.search(r"\{.*\}", saida, re.S)
        obj = json.loads(m.group(0)) if m else {}
        return {"rota_ok": int(obj.get("rota_ok", 0)),
                "resposta_ok": int(obj.get("resposta_ok", 0)),
                "justificativa": str(obj.get("justificativa", ""))[:200]}
    except Exception as e:
        return {"erro": f"{type(e).__name__}: {e}"}


def avaliar(deps, juiz=None) -> dict:
    """Roda os casos, mede roteamento (objetivo) e, se `juiz` for dado, julga a qualidade."""
    linhas = []
    for caso in CASOS_TRAJETORIA:
        res = responder(caso["pergunta"], deps)
        # rota "efetiva": vazia se bloqueado/fora de escopo (comparável ao esperado []).
        efetiva = [] if res["bloqueado"] or res["fora_de_escopo"] else res["rotas"]
        linha = {
            "id": caso["id"], "esperadas": caso["rotas_esperadas"], "obtidas": efetiva,
            "acerto_exato": sorted(efetiva) == sorted(caso["rotas_esperadas"]),
            "jaccard": round(_jaccard(efetiva, caso["rotas_esperadas"]), 3),
            "bloqueado": res["bloqueado"], "fora_de_escopo": res["fora_de_escopo"],
            "n_passos": len(res["trajetoria"]),
        }
        if juiz is not None:
            linha["juiz"] = _julgar(caso, res, juiz)
        linhas.append(linha)

    n = len(linhas)
    resumo = {
        "n_casos": n,
        # Roteamento OBJETIVO (não depende de juiz): inclui declinar corretamente injection/escopo.
        "acerto_roteamento": round(sum(x["acerto_exato"] for x in linhas) / n, 3),
        "jaccard_medio": round(sum(x["jaccard"] for x in linhas) / n, 3),
    }
    if juiz is not None:
        # O juiz só é bem-definido para casos IN-SCOPE (rota esperada não-vazia): nos casos
        # declinados o correto é "nenhuma rota", que a métrica objetiva acima já captura.
        in_scope = [x for x in linhas if x["esperadas"] and "juiz" in x and "erro" not in x["juiz"]]
        if in_scope:
            resumo["n_in_scope"] = len(in_scope)
            resumo["rota_ok_medio"] = round(
                sum(x["juiz"]["rota_ok"] for x in in_scope) / len(in_scope), 2)
            resumo["resposta_ok_medio"] = round(
                sum(x["juiz"]["resposta_ok"] for x in in_scope) / len(in_scope), 2)

    res = carimbar({"tarefa": "avaliação de trajetória do agente (Fase 4)",
                    "resumo": resumo, "casos": linhas})
    saida = REPO_ROOT / "reports" / "fase4_agente"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "avaliacao.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"roteamento: acerto={resumo['acerto_roteamento']} | Jaccard={resumo['jaccard_medio']}")
    if "resposta_ok_medio" in resumo:
        print(f"juiz: rota_ok={resumo['rota_ok_medio']}/2 "
              f"resposta_ok={resumo['resposta_ok_medio']}/2")
    print(f"relatório: {saida / 'avaliacao.json'}")
    return res


def _rota_efetiva(pergunta: str, llm) -> list[str]:
    """Rota que o agente tomaria: guardrail primeiro (injection → vazia), senão o roteador."""
    from rodoia.agente.roteador import rotear
    from rodoia.rag.seguranca import detectar_injection

    inj, _ = detectar_injection(pergunta)
    return [] if inj else rotear(pergunta, llm)["rotas"]


def avaliar_roteamento(llm, casos=None) -> dict:
    """Avaliação OBJETIVA e barata do roteamento (guardrail + roteador), sem executar as
    ferramentas nem juiz — permite medir o roteamento num n maior (responde a "1,0? são 6 casos").
    """
    from rodoia.agente.casos import CASOS

    casos = casos or CASOS
    linhas = []
    for c in casos:
        efetiva = _rota_efetiva(c["pergunta"], llm)
        linhas.append({"id": c["id"], "esperadas": c["rotas_esperadas"], "obtidas": efetiva,
                       "acerto_exato": sorted(efetiva) == sorted(c["rotas_esperadas"]),
                       "jaccard": round(_jaccard(efetiva, c["rotas_esperadas"]), 3)})
    n = len(linhas)
    resumo = {"n_casos": n,
              "acerto_roteamento": round(sum(x["acerto_exato"] for x in linhas) / n, 3),
              "jaccard_medio": round(sum(x["jaccard"] for x in linhas) / n, 3)}
    res = carimbar({"tarefa": "roteamento do agente — objetivo (guardrail+roteador), n ampliado",
                    "resumo": resumo, "casos": linhas})
    saida = REPO_ROOT / "reports" / "fase4_agente" / "roteamento.json"
    saida.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"roteamento (n={n}): acerto_exato={resumo['acerto_roteamento']} "
          f"jaccard={resumo['jaccard_medio']} -> {saida}")
    return res


def main() -> None:
    import sys

    from rodoia.rag.llm import OllamaLLM

    if len(sys.argv) > 1 and sys.argv[1] == "roteamento":
        avaliar_roteamento(OllamaLLM())          # barato: só o roteador (Ollama), sem vLLM/juiz
        return
    from rodoia.agente.ferramentas import deps_reais
    deps = deps_reais()
    juiz = OllamaLLM(modelo="llama3.1:8b")  # juiz independente do cérebro (qwen2.5)
    avaliar(deps, juiz)


if __name__ == "__main__":
    main()
