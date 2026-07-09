"""Avaliação do retrieval sobre um conjunto dourado de perguntas com fonte
conhecida. Mede e COMPARA os modos (denso, BM25, híbrido, híbrido+rerank) —
o antes/depois numérico que justifica a arquitetura.

Métricas:
- **recall@k**: fração de perguntas em que algum chunk do top-k veio da resolução
  esperada (o RAG "achou a fonte certa").
- **MRR**: média do inverso da posição da 1ª fonte correta (recompensa achar cedo).
"""

from __future__ import annotations

import json

from rodoia.config import REPO_ROOT, settings
from rodoia.rag.recuperador import RecuperadorHibrido

# Perguntas naturais → resolução(ões) que as respondem (verificado nos títulos).
CONJUNTO_DOURADO = [
    {
        "consulta": "Como funciona o vale-pedágio obrigatório no transporte de cargas?",
        "fontes": ["6024/2023"],
    },
    {
        "consulta": "Regras para o transporte rodoviário internacional de cargas",
        "fontes": ["6038/2024"],
    },
    {
        "consulta": "O que é o Registro Nacional do Agente Transportador de cargas?",
        "fontes": ["5990/2022"],
    },
    {
        "consulta": "Quais documentos são exigidos no transporte de produtos perigosos?",
        "fontes": ["5232/2016"],
    },
    {
        "consulta": "Como são calculados os pisos mínimos de frete por eixo carregado?",
        "fontes": ["5867/2020"],
    },
    {
        "consulta": "Regulamento das concessões rodoviárias federais, segunda norma",
        "fontes": ["6000/2022"],
    },
    {"consulta": "Parcelamento de débitos não inscritos em dívida ativa", "fontes": ["5830/2018"]},
    {
        "consulta": "Regulamento do transporte rodoviário coletivo interestadual de passageiros",
        "fontes": ["5998/2022"],
    },
    {
        "consulta": "Delegação de competências da diretoria colegiada da ANTT",
        "fontes": ["5818/2018"],
    },
    {
        "consulta": "Programa de regularização de débitos não tributários PRD",
        "fontes": ["5386/2017"],
    },
]


def _rank_da_fonte(resultados: list[dict], fontes: list[str]) -> int | None:
    """Posição (1-based) do 1º chunk cuja resolução está entre as esperadas."""
    for pos, chunk in enumerate(resultados, 1):
        if chunk.get("numero") in fontes:
            return pos
    return None


def avaliar_modo(recuperador: RecuperadorHibrido, modo: str, rerank: bool, k: int = 5) -> dict:
    acertos, soma_rr = 0, 0.0
    for caso in CONJUNTO_DOURADO:
        res = recuperador.buscar(caso["consulta"], k=k, modo=modo, rerank=rerank)
        rank = _rank_da_fonte(res, caso["fontes"])
        if rank is not None:
            acertos += 1
            soma_rr += 1.0 / rank
    n = len(CONJUNTO_DOURADO)
    return {"recall_at_k": acertos / n, "mrr": soma_rr / n, "k": k, "n": n}


def comparar(recuperador: RecuperadorHibrido, k: int = 5) -> dict:
    """Compara os modos (rerank só se houver reranker)."""
    configs = [("denso", False), ("bm25", False), ("hibrido", False)]
    if recuperador.reranker is not None:
        configs.append(("hibrido", True))
    resultados = {}
    for modo, rerank in configs:
        nome = f"{modo}+rerank" if rerank else modo
        resultados[nome] = avaliar_modo(recuperador, modo, rerank, k)
    return resultados


def carregar_recuperador(com_reranker: bool = True) -> RecuperadorHibrido:
    from rodoia.rag.chunking import chunk_norma
    from rodoia.rag.embeddings import E5Embedder
    from rodoia.rag.indice import criar_cliente
    from rodoia.rag.recuperador import Reranker

    normas = [json.loads(linha) for linha in settings.normas_jsonl.open(encoding="utf-8")]
    chunks = [c for n in normas for c in chunk_norma(n)]
    embedder = E5Embedder(settings.embedding_model)
    cliente = criar_cliente(path=str(settings.qdrant_path))
    reranker = Reranker() if com_reranker else None
    return RecuperadorHibrido(chunks, embedder, cliente, reranker=reranker)


def main() -> None:
    rec = carregar_recuperador(com_reranker=True)
    resultados = comparar(rec)
    print(f"{'modo':16} {'recall@5':>9} {'MRR':>7}")
    for nome, m in resultados.items():
        print(f"{nome:16} {m['recall_at_k']:>9.2f} {m['mrr']:>7.3f}")
    saida = REPO_ROOT / "reports" / "fase1_retrieval"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "avaliacao_retrieval.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2)
    )
    print(f"\nrelatório: {saida / 'avaliacao_retrieval.json'}")


if __name__ == "__main__":
    main()
