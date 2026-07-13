"""Banca de juízes INDEPENDENTE (Fase 1, Tier-1) — ataca o "você avaliando você".

O juiz único (llama3.1) já é independente do gerador (qwen2.5), mas continua sendo **um só
anotador**. Aqui a fidelidade (faithfulness) da resposta do RAG é julgada por uma **banca de 3
juízes de famílias distintas** (Meta/llama3.1, Google/gemma2, Mistral/mistral — todos ≠ do gerador
qwen), em escala **0/1/2**, e reportamos:
- **fidelidade média** (banca) e **voto majoritário** por caso;
- **κ de Fleiss** entre os 3 juízes — a concordância inter-anotador que faltava.

Honestidade: são **juízes automáticos**, não humanos — não substitui anotação humana com κ, mas é
um proxy rigoroso e mensurável de independência (3 famílias, concordância explícita). O caminho
para o κ humano segue registrado como backlog (docs/09).

Uso:  python -m rodoia.rag.painel_juizes
"""
from __future__ import annotations

import json
import re

from rodoia.config import REPO_ROOT
from rodoia.estat import fleiss_kappa
from rodoia.proveniencia import carimbar
from rodoia.rag.gerar import montar_contexto, responder

JUIZES = ("llama3.1:8b", "gemma2:9b", "mistral:7b")  # 3 famílias, todas != gerador (qwen)

_PROMPT = (
    "Você é um avaliador rigoroso de um sistema de perguntas e respostas sobre a regulação da "
    "ANTT. Dê UMA nota de FIDELIDADE da RESPOSTA em relação ao CONTEXTO:\n"
    "0 = há afirmações NÃO sustentadas pelo contexto (alucinação);\n"
    "1 = parcialmente sustentada;\n"
    "2 = TODAS as afirmações são sustentadas pelo contexto.\n"
    'Responda APENAS com JSON: {{"faithfulness": 0|1|2}}\n\n'
    "PERGUNTA: {pergunta}\n\nCONTEXTO:\n{contexto}\n\nRESPOSTA:\n{resposta}"
)
_RE_JSON = re.compile(r"\{[^{}]*\}", re.S)


def _nota_012(texto: str) -> int:
    """Extrai a nota 0/1/2 da saída do juiz (tolerante); clampa ao intervalo."""
    m = _RE_JSON.search(texto or "")
    if not m:
        return 0
    try:
        v = int(round(float(json.loads(m.group(0)).get("faithfulness", 0))))
    except (ValueError, TypeError):
        return 0
    return max(0, min(2, v))


def julgar(consulta: str, resposta: str, contexto: str, juiz) -> int:
    return _nota_012(juiz.gerar(_PROMPT.format(pergunta=consulta, contexto=contexto,
                                               resposta=resposta)))


def avaliar_painel(recuperador, gerador, juizes: dict, dourados: list[dict], k: int = 4) -> dict:
    """Gera a resposta e coleta a nota 0/1/2 de cada juiz da banca, por caso.

    Ordem escolhida para MINIMIZAR troca de modelo na GPU (6 GB): gera todas as respostas com o
    gerador (carregado 1x), depois cada juiz pontua todos os casos (cada juiz carregado 1x)."""
    gerados = []
    for d in dourados:
        r = responder(d["consulta"], recuperador, gerador, k=k)
        gerados.append((d["consulta"], r["resposta"], montar_contexto(r["chunks"])))
    notas_por_juiz = {
        nome: [julgar(c, resp, ctx, juiz) for c, resp, ctx in gerados]
        for nome, juiz in juizes.items()
    }
    linhas, avaliacoes = [], []
    for i, (consulta, _, _) in enumerate(gerados):
        vetor = [notas_por_juiz[nome][i] for nome in juizes]
        avaliacoes.append(vetor)
        maioria = max(set(vetor), key=vetor.count)
        notas = {nome: notas_por_juiz[nome][i] for nome in juizes}
        linhas.append({"consulta": consulta, "notas": notas, "voto_majoritario": maioria,
                       "media": round(sum(vetor) / len(vetor), 2)})
    n = len(linhas) or 1
    return {
        "juizes": list(juizes),
        "n": len(linhas),
        "fidelidade_media_banca": round(sum(x["media"] for x in linhas) / n, 3),   # 0..2
        "fidelidade_media_norm": round(sum(x["media"] for x in linhas) / n / 2, 3),  # 0..1
        "fleiss_kappa": fleiss_kappa(avaliacoes, categorias=(0, 1, 2)),
        "casos": linhas,
    }


def main() -> None:
    from rodoia.rag.avaliacao_retrieval import CONJUNTO_DOURADO, carregar_recuperador
    from rodoia.rag.llm import OllamaLLM

    rec = carregar_recuperador(com_reranker=False)
    gerador = OllamaLLM(modelo="qwen2.5:7b")
    juizes = {nome: OllamaLLM(modelo=nome) for nome in JUIZES}
    dourados = CONJUNTO_DOURADO[:12]
    print(f"banca de {len(juizes)} juízes ({', '.join(JUIZES)}) em {len(dourados)} casos...")
    res = carimbar(avaliar_painel(rec, gerador, juizes, dourados))
    saida = REPO_ROOT / "reports" / "fase1_geracao" / "painel_juizes.json"
    saida.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"fidelidade média (banca)={res['fidelidade_media_norm']} (0-1) | "
          f"κ de Fleiss={res['fleiss_kappa']} -> {saida}")


if __name__ == "__main__":
    main()
