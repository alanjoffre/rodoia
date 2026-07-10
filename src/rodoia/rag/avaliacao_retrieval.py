"""Avaliação do retrieval sobre um conjunto dourado de perguntas com fonte
conhecida. Mede e COMPARA os modos (denso, BM25, híbrido, híbrido+rerank) —
o antes/depois numérico que justifica a arquitetura.

Métricas (com intervalo de confiança 95%, dado o n pequeno):
- **hit_rate@k**: fração de perguntas em que algum chunk do top-k veio da resolução
  esperada. (Nome honesto: como cada pergunta tem uma fonte-gold, isto é *hit-rate*,
  não *recall* verdadeiro — que exigiria a fração de TODOS os relevantes recuperados.)
  IC de **Wilson** (apropriado p/ proporção com n pequeno).
- **MRR**: média do inverso da posição da 1ª fonte correta. IC por **bootstrap**.
"""

from __future__ import annotations

import json

from rodoia.config import REPO_ROOT, settings
from rodoia.estat import bootstrap_ic as _bootstrap_ic
from rodoia.estat import wilson as _wilson
from rodoia.rag.recuperador import RecuperadorHibrido

# Conjunto dourado: perguntas de INTENÇÃO REAL (como um usuário pergunta, não
# paráfrase do título da norma), cada uma mapeada à(s) resolução(ões) vigente(s) que a
# embasam. Fontes múltiplas quando o tema é coberto por mais de uma norma. Base para as
# métricas com IC (n≈25). Verificação: título + tema das normas do corpus.
CONJUNTO_DOURADO = [
    {"consulta": "Sou embarcador e contratei um caminhoneiro autônomo; preciso adiantar o "
     "pedágio da viagem?", "fontes": ["6024/2023", "673/2004"]},
    {"consulta": "Vou mandar uma carga de caminhão para a Argentina; que regras da ANTT "
     "valem para essa viagem internacional?", "fontes": ["6038/2024"]},
    {"consulta": "Como faço para tirar o registro que me habilita a transportar cargas pela "
     "ANTT?", "fontes": ["5990/2022"]},
    {"consulta": "Vou levar produtos químicos inflamáveis na estrada; que documentos e "
     "exigências preciso cumprir?", "fontes": ["5232/2016"]},
    {"consulta": "Existe um valor mínimo que devo receber por um frete? Como esse piso é "
     "calculado?", "fontes": ["5867/2020"]},
    {"consulta": "Tenho débitos com a ANTT que ainda não foram para a dívida ativa; consigo "
     "parcelar?", "fontes": ["5830/2018"]},
    {"consulta": "Quero abrir uma linha regular de ônibus entre dois estados; o que a "
     "regulação exige?", "fontes": ["5998/2022"]},
    {"consulta": "Quem dentro da ANTT tem competência delegada para decidir sobre esses "
     "assuntos?", "fontes": ["5818/2018"]},
    {"consulta": "Existe algum programa para eu regularizar débitos não tributários com a "
     "agência?", "fontes": ["5386/2017"]},
    {"consulta": "Qual regulamento rege as concessões de rodovias federais atualmente?",
     "fontes": ["6000/2022", "6032/2023", "6053/2024"]},
    {"consulta": "Que multas e penalidades a ANTT pode aplicar a quem transporta carga "
     "irregularmente?", "fontes": ["4071/2013"]},
    {"consulta": "O vale-pedágio precisa mesmo ser pago à parte do frete, ou pode ser "
     "descontado do motorista?", "fontes": ["6024/2023", "673/2004"]},
    {"consulta": "Meu RNTRC está vencendo; como funciona a renovação do registro de "
     "transportador?", "fontes": ["5990/2022"]},
    {"consulta": "Preciso transportar explosivos; a carga tem que ter sinalização e rótulo "
     "de risco?", "fontes": ["5232/2016"]},
    {"consulta": "Como a ANTT reajusta a tarifa de pedágio cobrada pelas concessionárias?",
     "fontes": ["5831/2018"]},
    {"consulta": "Quais são as regras para uma viagem internacional de ônibus com "
     "passageiros?", "fontes": ["5998/2022"]},
    {"consulta": "A tabela de piso mínimo muda conforme o número de eixos do caminhão?",
     "fontes": ["5867/2020"]},
    {"consulta": "Onde encontro a estrutura organizacional e as competências internas da "
     "ANTT?", "fontes": ["5976/2022", "5888/2020"]},
    {"consulta": "Sou transportador de carga própria; também sou obrigado a me registrar na "
     "ANTT?", "fontes": ["5990/2022"]},
    {"consulta": "Quero saber as condições para participar do programa de regularização de "
     "débitos da ANTT.", "fontes": ["5386/2017"]},
    {"consulta": "Que norma trata da revisão e do reajuste das tarifas nas concessões "
     "rodoviárias?", "fontes": ["5831/2018"]},
    {"consulta": "Transporto cargas para o Mercosul; preciso de alguma habilitação "
     "internacional específica?", "fontes": ["6038/2024"]},
    {"consulta": "O parcelamento de débito com a ANTT tem número máximo de parcelas?",
     "fontes": ["5830/2018"]},
    {"consulta": "Qual a metodologia oficial para definir o preço mínimo do frete "
     "rodoviário?", "fontes": ["5867/2020"]},
    {"consulta": "As novas normas do regulamento de concessões de 2023 e 2024 mudaram o "
     "quê?", "fontes": ["6032/2023", "6053/2024"]},
]


def _rank_da_fonte(resultados: list[dict], fontes: list[str]) -> int | None:
    """Posição (1-based) do 1º chunk cuja resolução está entre as esperadas."""
    for pos, chunk in enumerate(resultados, 1):
        if chunk.get("numero") in fontes:
            return pos
    return None


def avaliar_modo(recuperador: RecuperadorHibrido, modo: str, rerank: bool, k: int = 5) -> dict:
    rrs = []  # reciprocal rank por pergunta (0 se não achou)
    for caso in CONJUNTO_DOURADO:
        res = recuperador.buscar(caso["consulta"], k=k, modo=modo, rerank=rerank)
        rank = _rank_da_fonte(res, caso["fontes"])
        rrs.append(1.0 / rank if rank is not None else 0.0)
    n = len(rrs)
    hits = sum(1 for r in rrs if r > 0)
    return {
        "hit_rate_at_k": round(hits / n, 3),
        "hit_rate_ic95": _wilson(hits, n),
        "mrr": round(sum(rrs) / n, 3),
        "mrr_ic95": _bootstrap_ic(rrs, seed=settings.seed),
        "k": k,
        "n": n,
    }


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
    print(f"{'modo':16} {'hit@5':>6} {'IC95 hit':>16} {'MRR':>6} {'IC95 MRR':>16}")
    for nome, m in resultados.items():
        print(f"{nome:16} {m['hit_rate_at_k']:>6.2f} {str(m['hit_rate_ic95']):>16} "
              f"{m['mrr']:>6.3f} {str(m['mrr_ic95']):>16}")
    saida = REPO_ROOT / "reports" / "fase1_retrieval"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "avaliacao_retrieval.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2)
    )
    print(f"\nrelatório: {saida / 'avaliacao_retrieval.json'}")


if __name__ == "__main__":
    main()
