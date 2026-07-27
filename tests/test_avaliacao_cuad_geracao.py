"""Testes da geração ancorada no CUAD (LLM e embedder FALSOS — sem GPU/rede).

Foco no que decide a métrica: a detecção de abstenção e a consolidação das DUAS
taxas. Um bug em `absteve` inverteria a manchete do relatório sem quebrar nada.
"""

from __future__ import annotations

import numpy as np

from rodoia.rag.avaliacao_cuad import Chunk
from rodoia.rag.avaliacao_cuad_geracao import (
    Julgada,
    absteve,
    amostrar,
    consolidar_geracao,
    montar_contexto,
)
from rodoia.rag.cuad import Contrato, Pergunta


def test_absteve_marcador_explicito() -> None:
    assert absteve("NOT_FOUND")
    assert absteve("  NOT_FOUND  ")


def test_absteve_parafrase_natural() -> None:
    """Modelos pequenos parafraseiam a instrução em vez de emitir o marcador —
    medir só o marcador subestimaria a abstenção."""
    assert absteve("The excerpts do not contain the answer.")
    assert absteve("This information is not present in the provided text.")
    assert absteve("There is no mention of exclusivity.")
    assert absteve("The contract does not specify a renewal term.")


def test_nao_absteve_em_resposta_real() -> None:
    assert not absteve("The agreement was signed on March 15, 2019.")
    assert not absteve("Party A is Acme Corp and Party B is Beta LLC.")


def test_nao_absteve_com_negacao_de_conteudo() -> None:
    """Uma resposta REAL que apenas contém a palavra 'not' não é abstenção."""
    assert not absteve("The license is not transferable, per section 4.2.")


def test_montar_contexto_numera_e_limita() -> None:
    chunks = [Chunk("T", i, 0, 10, f"trecho{i}") for i in range(4)]
    ctx = montar_contexto(chunks, ["T::2", "T::0", "T::3"], k=2)
    assert ctx == "[1] trecho2\n\n[2] trecho0"


def test_consolidar_duas_taxas() -> None:
    julgadas = [
        Julgada("A", True, True, True),      # impossível, absteve  -> correto
        Julgada("A", True, False, False),    # impossível, respondeu -> alucinou
        Julgada("B", False, False, True),    # respondível, respondeu -> correto
        Julgada("B", False, True, False),    # respondível, absteve  -> recusou demais
    ]
    r = consolidar_geracao(julgadas, {})
    assert r["nao_alucinacao"]["taxa"] == 0.5
    assert r["cobertura"]["taxa"] == 0.5
    assert r["media_balanceada"] == 0.5


def test_consolidar_denuncia_abstencao_universal() -> None:
    """Um modelo que abstém de TUDO tem não-alucinação 1,0 — e cobertura 0. A média
    balanceada é o que impede essa métrica de ser maquiada."""
    julgadas = [Julgada("A", True, True, True) for _ in range(5)] + [
        Julgada("B", False, True, False) for _ in range(5)
    ]
    r = consolidar_geracao(julgadas, {})
    assert r["nao_alucinacao"]["taxa"] == 1.0
    assert r["cobertura"]["taxa"] == 0.0
    assert r["media_balanceada"] == 0.5


def test_amostrar_equilibra_as_populacoes() -> None:
    contratos = [
        Contrato(
            titulo=f"C{i}",
            texto="x",
            perguntas=tuple(
                Pergunta(f"q{i}{j}", f"C{i}", "e", f"cat{j % 3}", j % 2 == 0, ())
                for j in range(6)
            ),
        )
        for i in range(10)
    ]
    amostra = amostrar(contratos, n_por_populacao=6, seed=42)
    imp = sum(1 for _, p in amostra if p.impossivel)
    res = sum(1 for _, p in amostra if not p.impossivel)
    assert imp == 6
    assert res == 6


def test_amostrar_completa_o_n_pedido() -> None:
    """`--amostra N` tem que devolver N. A cota fixa (n // n_categorias) truncava:
    com 41 categorias e n=150 dava 3 por categoria = 123, silenciosamente."""
    contratos = [
        Contrato(
            titulo=f"C{i}",
            texto="x",
            perguntas=tuple(
                Pergunta(f"q{i}{j}", f"C{i}", "e", f"cat{j % 41}", False, ())
                for j in range(41)
            ),
        )
        for i in range(10)
    ]
    amostra = amostrar(contratos, n_por_populacao=150, seed=1)
    assert sum(1 for _, p in amostra if not p.impossivel) == 150


def test_amostrar_nao_estoura_o_disponivel() -> None:
    """Pedir mais do que existe devolve tudo o que existe, sem repetir nem quebrar."""
    contratos = [
        Contrato("C0", "x", (Pergunta("q", "C0", "e", "cat", False, ()),)),
    ]
    amostra = amostrar(contratos, n_por_populacao=99, seed=1)
    assert len(amostra) == 1


def test_amostrar_e_deterministica() -> None:
    """Seed fixa: duas chamadas dão a MESMA amostra — senão o número não é
    reproduzível e o relatório vira ruído."""
    contratos = [
        Contrato(
            titulo=f"C{i}",
            texto="x",
            perguntas=tuple(
                Pergunta(f"q{i}{j}", f"C{i}", "e", f"cat{j}", j % 2 == 0, ()) for j in range(4)
            ),
        )
        for i in range(8)
    ]
    a = [p.id for _, p in amostrar(contratos, 4, seed=7)]
    b = [p.id for _, p in amostrar(contratos, 4, seed=7)]
    assert a == b


class _EmbFake:
    dim = 2

    def encode_passages(self, textos: list[str]) -> np.ndarray:
        return np.zeros((len(textos), 2))

    def encode_queries(self, textos: list[str]) -> np.ndarray:
        return np.zeros((len(textos), 2))


def test_avaliar_geracao_ponta_a_ponta() -> None:
    """LLM falso que sempre abstém: não-alucinação 1,0 e cobertura 0."""
    from rodoia.rag.avaliacao_cuad_geracao import avaliar_geracao

    class LLMFake:
        modelo = "fake"
        ultima_metrica: dict = {}

        def gerar(self, prompt: str, sistema: str | None = None) -> str:
            return "NOT_FOUND"

    contratos = [
        Contrato(
            titulo="C0",
            texto="y" * 60,
            perguntas=(
                Pergunta("q1", "C0", 'related to "A" that. Details: d', "A", True, ()),
                Pergunta("q2", "C0", 'related to "B" that. Details: d', "B", False, ()),
            ),
        )
    ]
    r = avaliar_geracao(
        contratos, _EmbFake(), LLMFake(), n_por_populacao=1, max_chars=30, overlap=5
    )
    assert r["nao_alucinacao"]["taxa"] == 1.0
    assert r["cobertura"]["taxa"] == 0.0
    assert r["config"]["modelo"] == "fake"
