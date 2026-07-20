"""Avaliação da GERAÇÃO por LLM-as-judge (métricas estilo RAGAS).

Enquanto docs/07 mede o *retrieval* (achou a fonte?), aqui medimos a *resposta*:

- **Faithfulness** (fidelidade): as afirmações da resposta são sustentadas pelos
  trechos recuperados? Mede alucinação — o mais importante num RAG jurídico.
- **Answer relevancy** (relevância): a resposta de fato responde à pergunta?
- **Precisão de citação** (objetiva, sem juiz): das resoluções citadas na resposta,
  quantas estão ancoradas no contexto recuperado (a joia de um RAG jurídico) + se a
  fonte-gold esperada foi citada.
- **Observabilidade**: latência de geração (p50/p95) e tokens por resposta.

O LLM juiz é **INDEPENDENTE do gerador** (família/checkpoint distintos — gerador
qwen2.5:7b, juiz llama3.1:8b) para evitar viés de auto-avaliação. É o mesmo princípio
do RAGAS; implementá-lo à mão dá controle e transparência. `ragas` seria a de produção.
"""

from __future__ import annotations

import json
import re
import statistics

from rodoia.estat import bootstrap_ic
from rodoia.juiz import extrair_json
from rodoia.proveniencia import carimbar
from rodoia.rag.gerar import montar_contexto, responder
from rodoia.rag.recuperador import RecuperadorHibrido

_RE_RES = re.compile(r"(\d[\d.]{2,7})\s*/\s*(\d{4})")


def citacoes(texto: str) -> set[str]:
    """Extrai as resoluções citadas ('NNNN/AAAA', sem pontos) de uma resposta."""
    return {f"{m.group(1).replace('.', '')}/{m.group(2)}" for m in _RE_RES.finditer(texto)}

PROMPT_JUIZ = (
    "Você é um avaliador rigoroso de um sistema de perguntas e respostas sobre a "
    "regulação da ANTT. Avalie a RESPOSTA em relação à PERGUNTA e ao CONTEXTO.\n\n"
    "Dê duas notas de 0.0 a 1.0:\n"
    "- faithfulness: 1.0 se TODAS as afirmações da resposta são sustentadas pelo "
    "contexto; menor se há afirmações não sustentadas (alucinação).\n"
    "- relevancy: 1.0 se a resposta responde diretamente à pergunta; menor se foge "
    "ou é vaga.\n\n"
    "Responda APENAS com um JSON: "
    '{{"faithfulness": <0-1>, "relevancy": <0-1>}}\n\n'
    "PERGUNTA: {pergunta}\n\nCONTEXTO:\n{contexto}\n\nRESPOSTA:\n{resposta}"
)


def _parse_notas(texto: str) -> dict:
    """Extrai o primeiro objeto JSON da saída do juiz, de forma tolerante."""
    d = extrair_json(texto)
    try:
        return {
            "faithfulness": float(d.get("faithfulness", 0.0)),
            "relevancy": float(d.get("relevancy", 0.0)),
        }
    except (ValueError, TypeError):
        return {"faithfulness": 0.0, "relevancy": 0.0}


def julgar(consulta: str, resposta: str, contexto: str, juiz) -> dict:
    prompt = PROMPT_JUIZ.format(pergunta=consulta, contexto=contexto, resposta=resposta)
    return _parse_notas(juiz.gerar(prompt))


def ics_geracao(casos: list[dict]) -> dict:
    """IC 95% (bootstrap) das notas do juiz por caso. Com n=12 a régua do projeto exige a
    faixa: uma média de juiz sem IC engana. Pura sobre `casos` — dá backfill sem re-rodar o LLM."""
    prec = [c["precisao_citacao"] for c in casos if c["precisao_citacao"] is not None]
    return {
        "faithfulness_ic95": bootstrap_ic([c["faithfulness"] for c in casos]),
        "relevancy_ic95": bootstrap_ic([c["relevancy"] for c in casos]),
        "precisao_citacao_ic95": bootstrap_ic(prec) if prec else None,
    }


def avaliar_geracao(
    recuperador: RecuperadorHibrido, gerador, juiz, dourados: list[dict], k: int = 4
) -> dict:
    """Para cada caso dourado: gera a resposta, julga (juiz independente) e mede
    correção de citação + observabilidade. `dourados` = [{consulta, fontes}]."""
    casos, latencias, tokens_resp = [], [], []
    for d in dourados:
        r = responder(d["consulta"], recuperador, gerador, k=k)
        notas = julgar(d["consulta"], r["resposta"], montar_contexto(r["chunks"]), juiz)
        citadas = citacoes(r["resposta"])
        no_contexto = {c["numero"] for c in r["chunks"]}
        esperadas = set(d["fontes"])
        casos.append({
            "consulta": d["consulta"],
            "fontes_esperadas": d["fontes"],
            "citou_esperada": bool(citadas & esperadas),
            # das citações da resposta, quantas estão ancoradas no contexto recuperado:
            "precisao_citacao": (len(citadas & no_contexto) / len(citadas)) if citadas else None,
            **notas,
        })
        m = r.get("metricas", {})
        if m.get("geracao_s") is not None:
            latencias.append(m["geracao_s"])
        if m.get("tokens_resposta") is not None:
            tokens_resp.append(m["tokens_resposta"])
    n = len(casos) or 1
    prec = [c["precisao_citacao"] for c in casos if c["precisao_citacao"] is not None]
    obs = {}
    if latencias:
        latencias.sort()
        obs = {
            "geracao_p50_s": round(statistics.median(latencias), 3),
            "geracao_p95_s": round(latencias[max(0, int(0.95 * len(latencias)) - 1)], 3),
            "tokens_resposta_medio": round(statistics.mean(tokens_resp)) if tokens_resp else None,
        }
    return {
        "faithfulness_media": round(sum(c["faithfulness"] for c in casos) / n, 3),
        "relevancy_media": round(sum(c["relevancy"] for c in casos) / n, 3),
        "precisao_citacao_media": round(sum(prec) / len(prec), 3) if prec else None,
        "taxa_citou_esperada": round(sum(c["citou_esperada"] for c in casos) / n, 3),
        **ics_geracao(casos),
        "observabilidade": obs,
        "n": len(casos),
        "casos": casos,
    }


def _backfill_ic() -> None:
    """Recomputa os IC das notas do juiz a partir dos `casos` já versionados (sem LLM) e os
    injeta no JSON — para adicionar a faixa sem reexecutar a geração cara."""
    import sys

    from rodoia.config import REPO_ROOT
    p = REPO_ROOT / "reports" / "fase1_geracao" / "avaliacao_geracao.json"
    dados = json.loads(p.read_text(encoding="utf-8"))
    ics = ics_geracao(dados["casos"])
    reconstruido = {}
    for chave, valor in dados.items():                 # insere os IC logo após taxa_citou_esperada
        reconstruido[chave] = valor
        if chave == "taxa_citou_esperada":
            reconstruido.update(ics)
    p.write_text(json.dumps(reconstruido, ensure_ascii=False, indent=2))
    print(f"IC injetados: faithfulness {ics['faithfulness_ic95']} "
          f"relevancy {ics['relevancy_ic95']} -> {p}", file=sys.stderr)


def main() -> None:
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "ic":
        _backfill_ic()
        return

    from rodoia.config import REPO_ROOT
    from rodoia.rag.avaliacao_retrieval import CONJUNTO_DOURADO, carregar_recuperador
    from rodoia.rag.llm import OllamaLLM

    rec = carregar_recuperador(com_reranker=True)
    # Juiz INDEPENDENTE do gerador (família/checkpoint distintos) — evita auto-avaliação.
    gerador = OllamaLLM(modelo="qwen2.5:7b")
    juiz = OllamaLLM(modelo="llama3.1:8b")
    dourados = CONJUNTO_DOURADO[:12]  # amostra p/ tempo tratável
    print(f"avaliando geração em {len(dourados)} perguntas "
          f"(gerador=qwen2.5:7b, juiz INDEPENDENTE=llama3.1:8b)...")
    res = avaliar_geracao(rec, gerador, juiz, dourados)
    res["gerador"], res["juiz"] = "qwen2.5:7b", "llama3.1:8b"
    print(f"faithfulness={res['faithfulness_media']} relevancy={res['relevancy_media']} "
          f"precisao_citacao={res['precisao_citacao_media']} "
          f"citou_esperada={res['taxa_citou_esperada']}")
    print(f"observabilidade: {res['observabilidade']}")
    saida = REPO_ROOT / "reports" / "fase1_geracao"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "avaliacao_geracao.json").write_text(
        json.dumps(carimbar(res), ensure_ascii=False, indent=2))
    print(f"relatório: {saida / 'avaliacao_geracao.json'}")


if __name__ == "__main__":
    main()
