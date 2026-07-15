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
from rodoia.proveniencia import carimbar
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
    {"consulta": "Um caminhoneiro pode transportar carga sem estar inscrito na ANTT?",
     "fontes": ["5990/2022"]},
    {"consulta": "Como contesto uma multa aplicada pela ANTT no transporte de cargas?",
     "fontes": ["4071/2013"]},
    {"consulta": "Quais as regras para transportar produtos perigosos em área urbana?",
     "fontes": ["5232/2016"]},
    {"consulta": "A concessionária pode aumentar o pedágio quando quiser?",
     "fontes": ["5831/2018"]},
    {"consulta": "Como funciona a habilitação para o transporte internacional de "
     "passageiros?", "fontes": ["5998/2022"]},
    {"consulta": "Qual o prazo para pagar as parcelas de um débito parcelado com a ANTT?",
     "fontes": ["5830/2018"]},
    {"consulta": "Como é feita a fiscalização do vale-pedágio nas rodovias?",
     "fontes": ["6024/2023", "673/2004"]},
    {"consulta": "O RNTRC tem categorias diferentes para transportador autônomo e "
     "empresa?", "fontes": ["5990/2022"]},
    {"consulta": "Que documentos são de porte obrigatório no transporte de produtos "
     "perigosos?", "fontes": ["5232/2016"]},
    {"consulta": "A metodologia do piso mínimo de frete considera o custo do diesel?",
     "fontes": ["5867/2020"]},
    {"consulta": "O que é exigido para operar uma concessão de rodovia federal?",
     "fontes": ["6000/2022", "6032/2023", "6053/2024"]},
    {"consulta": "Como a ANTT tipifica as infrações no transporte rodoviário?",
     "fontes": ["4071/2013"]},
    {"consulta": "Preciso declarar o valor da carga para calcular o vale-pedágio?",
     "fontes": ["6024/2023"]},
    {"consulta": "Qual resolução trata da delegação de competências na diretoria da "
     "ANTT?", "fontes": ["5818/2018"]},
    {"consulta": "Como funciona o parcelamento de multas ainda não inscritas em dívida "
     "ativa?", "fontes": ["5830/2018"]},
    {"consulta": "Que regras valem para transportar cargas perigosas a granel?",
     "fontes": ["5232/2016"]},
    {"consulta": "Onde encontro o regimento interno e a estrutura da ANTT?",
     "fontes": ["5976/2022", "5888/2020"]},
    {"consulta": "Como é calculada a tarifa de pedágio nas concessões federais?",
     "fontes": ["5831/2018"]},
    {"consulta": "Preciso de habilitação específica para transporte internacional de "
     "cargas no Mercosul?", "fontes": ["6038/2024"]},
    {"consulta": "Quais as condições para aderir ao programa de regularização de débitos "
     "não tributários?", "fontes": ["5386/2017"]},
    {"consulta": "O transporte coletivo interestadual de passageiros precisa de "
     "autorização da ANTT?", "fontes": ["5998/2022"]},
    {"consulta": "Como o piso de frete varia com a distância e o tipo de carga?",
     "fontes": ["5867/2020"]},
    {"consulta": "Sou empresa de transporte de cargas; como me inscrevo no RNTRC?",
     "fontes": ["5990/2022"]},
    {"consulta": "Que penalidades existem para excesso de peso no transporte de cargas?",
     "fontes": ["4071/2013"]},
    {"consulta": "O vale-pedágio se aplica ao transporte internacional de cargas?",
     "fontes": ["6024/2023"]},
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


# Rótulos-gold REFUTADOS pela auditoria humana (κ inter-anotador, ver rodoia.anotacao) — verificado
# nos títulos do corpus, erro documental objetivo (resíduo de numeração antiga, corpus 45→125):
#   5998/2022 = "Transporte de Produtos Perigosos" — rotulava queries de ÔNIBUS de passageiros.
#     O corpus é temático de CARGAS: não há fonte de transporte de passageiros → FORA-DE-DOMÍNIO.
#   5831/2018 = "metas FERROVIÁRIAS" — rotulava queries de TARIFA de pedágio. Fonte correta EXISTE
#     no corpus: 675/2004 (revisões do equilíbrio econ.-fin. das concessões) + 6032/2023 (gestão
#     econ.-fin. dos contratos) → REROTULÁVEL.
GOLD_FORA_DOMINIO = "5998/2022"                       # ônibus: sem fonte correta no corpus
REROTULAR = {"5831/2018": ["675/2004", "6032/2023"]}  # tarifa: gold correto (rerotula)
GOLDS_REFUTADOS = (GOLD_FORA_DOMINIO, *REROTULAR)     # os 2 golds quebrados


def avaliar_auditado(recuperador: RecuperadorHibrido, k: int = 5) -> dict:
    """hit@5 (híbrido): dourado CANÔNICO (no gate) vs. DEFINITIVO após a auditoria. NÃO altera o
    dourado canônico nem o gate — só quantifica o efeito dos labels quebrados. Correção honesta,
    não "deleta os erros": rerotula a tarifa para a fonte correta e trata o ônibus como
    fora-de-domínio (2 enquadramentos: removido, ou contado como miss legítimo)."""
    def _hit(casos):
        hits = sum(1 for q, f in casos
                   if _rank_da_fonte(recuperador.buscar(q, k=k, modo="hibrido"), f) is not None)
        return {"hit_rate_at_k": round(hits / len(casos), 3),
                "hit_rate_ic95": _wilson(hits, len(casos)), "hits": hits, "n": len(casos)}

    def _rerotula(fontes):
        for ruim, corretas in REROTULAR.items():
            if ruim in fontes:
                return corretas
        return fontes

    base = [(c["consulta"], c["fontes"]) for c in CONJUNTO_DOURADO]
    fora = [c["consulta"] for c in CONJUNTO_DOURADO if GOLD_FORA_DOMINIO in c["fontes"]]
    rerot = [c["consulta"] for c in CONJUNTO_DOURADO
             if set(c["fontes"]) & set(REROTULAR)]
    # definitivo A: rerotula tarifa + REMOVE ônibus fora-de-domínio
    defin_a = [(q, _rerotula(f)) for q, f in base if GOLD_FORA_DOMINIO not in f]
    # definitivo B: rerotula tarifa + MANTÉM ônibus como miss legítimo (sistema acerta ao não achar)
    defin_b = [(q, _rerotula(f)) for q, f in base]
    return {
        "metodo": ("hit@5 híbrido: canônico (no gate) vs. definitivo pós-auditoria. Rerotula a "
                   "tarifa p/ a fonte correta; ônibus é fora-de-domínio. Gate NÃO mexido."),
        "golds_refutados": list(GOLDS_REFUTADOS),
        "gold_fora_dominio": GOLD_FORA_DOMINIO,
        "rerotulados": REROTULAR,
        "n_queries_refutadas": len(fora) + len(rerot),
        "n_confirmadas_por_humano": 4,
        "queries_fora_dominio": fora,
        "queries_rerotuladas": rerot,
        "hit5_canonico": _hit(base),
        "hit5_definitivo_dropa_onibus": _hit(defin_a),
        "hit5_definitivo_onibus_miss": _hit(defin_b),
    }


def main() -> None:
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "auditado":
        rec = carregar_recuperador(com_reranker=False)
        res = avaliar_auditado(rec)
        saida = REPO_ROOT / "reports" / "fase1_retrieval" / "hit5_auditado.json"
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text(json.dumps(carimbar(res), ensure_ascii=False, indent=2))
        c, a, b = (res["hit5_canonico"], res["hit5_definitivo_dropa_onibus"],
                   res["hit5_definitivo_onibus_miss"])
        print(f"canônico (gate) hit@5={c['hit_rate_at_k']} ({c['hits']}/{c['n']}) · "
              f"definitivo {b['hit_rate_at_k']} ({b['hits']}/{b['n']}, ônibus=miss) "
              f"a {a['hit_rate_at_k']} ({a['hits']}/{a['n']}, ônibus removido) -> {saida}")
        return

    rec = carregar_recuperador(com_reranker=True)
    resultados = comparar(rec)
    print(f"{'modo':16} {'hit@5':>6} {'IC95 hit':>16} {'MRR':>6} {'IC95 MRR':>16}")
    for nome, m in resultados.items():
        print(f"{nome:16} {m['hit_rate_at_k']:>6.2f} {str(m['hit_rate_ic95']):>16} "
              f"{m['mrr']:>6.3f} {str(m['mrr_ic95']):>16}")
    saida = REPO_ROOT / "reports" / "fase1_retrieval"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "avaliacao_retrieval.json").write_text(
        json.dumps(carimbar(resultados), ensure_ascii=False, indent=2)
    )
    print(f"\nrelatório: {saida / 'avaliacao_retrieval.json'}")


if __name__ == "__main__":
    main()
