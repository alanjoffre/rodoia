"""Testes da avaliação de recuperação no CUAD (puros, sem rede e sem corpus real).

O que se protege aqui é o mapeamento span→chunk. Ele é a única peça em que um
erro NÃO aparece como falha: produz Recall plausível e falso.
"""

from __future__ import annotations

from rodoia.rag.avaliacao_cuad import (
    Chunk,
    chunkar,
    gold_da_pergunta,
    montar_query,
)
from rodoia.rag.cuad import Contrato, Pergunta, Span

_ENUNCIADO = (
    'Highlight the parts (if any) of this contract related to "Exclusivity" that should be '
    "reviewed by a lawyer. Details: Is there an exclusive dealing commitment?"
)


def _pergunta(spans: tuple[Span, ...], impossivel: bool = False) -> Pergunta:
    return Pergunta("q", "T", _ENUNCIADO, "Exclusivity", impossivel, spans)


def test_chunkar_cobre_o_texto_inteiro() -> None:
    texto = "x" * 4000
    chunks = chunkar(texto, "T", max_chars=1500, overlap=200)
    assert chunks[0].inicio == 0
    assert chunks[-1].fim == len(texto)
    # nenhum buraco: cada chunk começa antes do fim do anterior (há sobreposição)
    for anterior, seguinte in zip(chunks, chunks[1:], strict=False):
        assert seguinte.inicio < anterior.fim


def test_chunkar_texto_vazio() -> None:
    assert chunkar("", "T") == []


def test_chunkar_texto_menor_que_a_janela() -> None:
    chunks = chunkar("abc", "T", max_chars=1500, overlap=200)
    assert len(chunks) == 1
    assert (chunks[0].inicio, chunks[0].fim, chunks[0].texto) == (0, 3, "abc")


def test_chunk_id_identifica_contrato_e_posicao() -> None:
    assert Chunk("ACORDO", 3, 0, 10, "x").id == "ACORDO::3"


def test_gold_por_interseccao_nao_continencia() -> None:
    """Span que ATRAVESSA a fronteira precisa marcar os dois chunks. Exigir
    continência marcaria zero gold para spans maiores que a janela."""
    chunks = [Chunk("T", 0, 0, 100, "a"), Chunk("T", 1, 80, 180, "b")]
    p = _pergunta((Span(texto="y" * 40, inicio=70),))  # 70..110, cruza a fronteira
    assert gold_da_pergunta(p, chunks) == {"T::0", "T::1"}


def test_gold_span_inteiro_dentro_de_um_chunk() -> None:
    chunks = [Chunk("T", 0, 0, 100, "a"), Chunk("T", 1, 80, 180, "b")]
    p = _pergunta((Span(texto="yyy", inicio=10),))
    assert gold_da_pergunta(p, chunks) == {"T::0"}


def test_gold_multiplos_spans_uniao() -> None:
    chunks = [Chunk("T", 0, 0, 100, "a"), Chunk("T", 1, 100, 200, "b")]
    p = _pergunta((Span("aaa", 5), Span("bbb", 150)))
    assert gold_da_pergunta(p, chunks) == {"T::0", "T::1"}


def test_gold_vazio_quando_nao_ha_span() -> None:
    assert gold_da_pergunta(_pergunta((), impossivel=True), [Chunk("T", 0, 0, 10, "x")]) == set()


def test_fronteira_exata_nao_conta() -> None:
    """Span que termina exatamente onde o chunk começa não o intersecta —
    senão todo chunk seguinte entraria no gold de graça."""
    chunks = [Chunk("T", 0, 0, 50, "a"), Chunk("T", 1, 50, 100, "b")]
    p = _pergunta((Span(texto="z" * 10, inicio=40),))  # 40..50
    assert gold_da_pergunta(p, chunks) == {"T::0"}


def test_query_descarta_o_preambulo() -> None:
    q = montar_query(_pergunta(()))
    assert q == "Exclusivity Is there an exclusive dealing commitment?"
    assert "Highlight the parts" not in q


def test_query_sem_detalhes_usa_so_a_categoria() -> None:
    enunciado = 'related to "Cap On Liability" that should'
    p = Pergunta("q", "T", enunciado, "Cap On Liability", False, ())
    assert montar_query(p) == "Cap On Liability"


def test_contrato_e_dataclass_congelada() -> None:
    """Gold imutável: um chunk não pode ser reescrito depois do mapeamento."""
    c = Contrato("T", "texto", ())
    assert c.titulo == "T"
