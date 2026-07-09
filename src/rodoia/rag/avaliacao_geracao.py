"""Avaliação da GERAÇÃO por LLM-as-judge (métricas estilo RAGAS).

Enquanto docs/07 mede o *retrieval* (achou a fonte?), aqui medimos a *resposta*:

- **Faithfulness** (fidelidade): as afirmações da resposta são sustentadas pelos
  trechos recuperados? Mede alucinação — o mais importante num RAG jurídico.
- **Answer relevancy** (relevância): a resposta de fato responde à pergunta?

Um LLM juiz (independente do gerador) pontua 0–1 cada métrica, lendo pergunta +
resposta + contexto. É o mesmo princípio do RAGAS; implementá-lo à mão dá controle
e transparência sobre o critério. A lib `ragas` seria a alternativa de produção.
"""

from __future__ import annotations

import json
import re

from rodoia.rag.gerar import montar_contexto, responder
from rodoia.rag.recuperador import RecuperadorHibrido

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

_RE_JSON = re.compile(r"\{[^{}]*\}", re.S)


def _parse_notas(texto: str) -> dict:
    """Extrai o primeiro objeto JSON da saída do juiz, de forma tolerante."""
    m = _RE_JSON.search(texto)
    if not m:
        return {"faithfulness": 0.0, "relevancy": 0.0}
    try:
        d = json.loads(m.group(0))
        return {
            "faithfulness": float(d.get("faithfulness", 0.0)),
            "relevancy": float(d.get("relevancy", 0.0)),
        }
    except (ValueError, TypeError):
        return {"faithfulness": 0.0, "relevancy": 0.0}


def julgar(consulta: str, resposta: str, contexto: str, juiz) -> dict:
    prompt = PROMPT_JUIZ.format(pergunta=consulta, contexto=contexto, resposta=resposta)
    return _parse_notas(juiz.gerar(prompt))


def avaliar_geracao(
    recuperador: RecuperadorHibrido, gerador, juiz, consultas: list[str], k: int = 4
) -> dict:
    """Para cada consulta: gera a resposta e a julga. Retorna médias + casos."""
    casos = []
    for consulta in consultas:
        r = responder(consulta, recuperador, gerador, k=k)
        notas = julgar(consulta, r["resposta"], montar_contexto(r["chunks"]), juiz)
        casos.append({"consulta": consulta, "fontes": r["fontes"], **notas})
    n = len(casos) or 1
    return {
        "faithfulness_media": sum(c["faithfulness"] for c in casos) / n,
        "relevancy_media": sum(c["relevancy"] for c in casos) / n,
        "n": len(casos),
        "casos": casos,
    }


def main() -> None:
    from rodoia.config import REPO_ROOT
    from rodoia.rag.avaliacao_retrieval import CONJUNTO_DOURADO, carregar_recuperador
    from rodoia.rag.llm import OllamaLLM

    rec = carregar_recuperador(com_reranker=True)
    llm = OllamaLLM()
    consultas = [c["consulta"] for c in CONJUNTO_DOURADO[:8]]  # amostra p/ tempo tratável
    print(f"avaliando geração em {len(consultas)} perguntas (gerador + juiz = qwen2.5:7b)...")
    res = avaliar_geracao(rec, llm, llm, consultas)
    print(f"faithfulness média: {res['faithfulness_media']:.2f}")
    print(f"relevancy   média: {res['relevancy_media']:.2f}")
    saida = REPO_ROOT / "reports" / "fase1_geracao"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "avaliacao_geracao.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"relatório: {saida / 'avaliacao_geracao.json'}")


if __name__ == "__main__":
    main()
