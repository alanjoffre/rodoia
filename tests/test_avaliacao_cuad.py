"""Testes da avaliação de recuperação no CUAD (puros, sem rede e sem corpus real).

O que se protege aqui é o mapeamento span→chunk. Ele é a única peça em que um
erro NÃO aparece como falha: produz Recall plausível e falso.
"""

from __future__ import annotations

import pytest

from rodoia.rag.avaliacao_cuad import (
    Chunk,
    _metricas_por_pergunta,
    _registrar,
    chunkar,
    chunkar_clausula,
    consolidar,
    gold_da_pergunta,
    montar_query,
    obter_chunker,
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


# --- chunking por cláusula -------------------------------------------------

_ESTRUTURADO = (
    "PREAMBLE text here.\n"
    "1. Definitions. As used herein the following terms apply.\n"
    "2. Exclusivity. Buyer shall purchase exclusively from Seller.\n"
    "3. Termination. Either party may terminate on notice.\n"
    "4. Governing Law. This agreement is governed by Delaware law.\n"
)


def test_clausula_corta_nas_fronteiras_numeradas() -> None:
    """Com max_chars pequeno, cada cláusula vira seu próprio chunk — e o corte cai
    NO marcador, não no meio da frase, que é o ponto do exercício."""
    chunks = chunkar_clausula(_ESTRUTURADO, "T", max_chars=60, overlap=0)
    assert len(chunks) > 1
    assert any(c.texto.lstrip().startswith("2. Exclusivity") for c in chunks)
    assert any(c.texto.lstrip().startswith("3. Termination") for c in chunks)


def test_clausula_preserva_offsets_absolutos() -> None:
    """O gold do CUAD é por offset de caractere. Se o chunk por cláusula reportar
    offset relativo, o Recall sai plausível e FALSO — o modo de falha que este
    módulo inteiro existe para evitar."""
    chunks = chunkar_clausula(_ESTRUTURADO, "T", max_chars=60, overlap=0)
    for c in chunks:
        assert _ESTRUTURADO[c.inicio : c.fim] == c.texto


def test_clausula_cobre_o_texto_inteiro() -> None:
    chunks = chunkar_clausula(_ESTRUTURADO, "T", max_chars=60, overlap=0)
    assert chunks[0].inicio == 0
    assert chunks[-1].fim == len(_ESTRUTURADO)


def test_clausula_degrada_para_janela_sem_estrutura() -> None:
    """100 dos 510 contratos do CUAD não têm 3+ fronteiras detectáveis. Nesses, o
    chunker por cláusula tem que virar o de janela em vez de devolver um chunk
    gigante — degradação explícita, não silenciosa."""
    texto = "x" * 4000
    assert chunkar_clausula(texto, "T", max_chars=1500, overlap=200) == chunkar(
        texto, "T", max_chars=1500, overlap=200
    )


def test_clausula_janela_clausula_grande_demais() -> None:
    """Uma cláusula maior que max_chars não pode virar um chunk gigante — é
    janelada por dentro, senão o embedder truncaria a cauda em silêncio."""
    texto = "1. Definitions. " + "y" * 5000 + "\n2. Term. Short.\n3. End. Short.\n"
    chunks = chunkar_clausula(texto, "T", max_chars=800, overlap=100)
    assert all(len(c.texto) <= 800 for c in chunks)
    assert texto[chunks[0].inicio : chunks[0].fim] == chunks[0].texto


def test_clausula_texto_vazio() -> None:
    assert chunkar_clausula("", "T") == []


def test_teto_do_recall_depende_do_tamanho_do_gold() -> None:
    """`recall@k = |gold ∩ top-k| / |gold|` tem teto `min(k,|gold|)/|gold|`. Com 4
    chunks de gold, recall@1 **não pode** passar de 0,25 — então comparar recall@1
    entre dois chunkings que produzem |gold| diferente compara réguas diferentes.
    """
    avaliadas = [
        _metricas_por_pergunta({"T::0"}, ["T::0"], 0.0),                       # |gold|=1
        _metricas_por_pergunta({f"T::{i}" for i in range(4)}, ["T::0"], 0.0),  # |gold|=4
    ]
    r = consolidar(avaliadas, {}, {})
    assert r["gold_medio_por_pergunta"] == 2.5
    # teto de recall@1 = média de 1/|gold| = (1/1 + 1/4)/2 = 0,625
    assert r["metricas"]["recall_at_1"]["teto"] == 0.625
    # bruto = (1,0 + 0,25)/2 = 0,625 — indistinguível do teto, e é essa colagem que
    # o normalizado desfaz: as duas perguntas acertaram tudo o que cabia em 1 slot.
    assert r["metricas"]["recall_at_1"]["media"] == 0.625
    assert r["metricas"]["recall_norm_at_1"]["media"] == 1.0


def test_recall_normalizado_penaliza_o_que_deve() -> None:
    """Normalizar não é inflar: uma pergunta com 4 golds que recupera 0 em 1 slot
    continua valendo 0, e uma que recupera 2 de 4 em k=2 vale 1,0 — porque 2 é
    tudo que cabia."""
    a = _metricas_por_pergunta({f"T::{i}" for i in range(4)}, ["T::9"], 0.0)
    b = _metricas_por_pergunta({f"T::{i}" for i in range(4)}, ["T::0", "T::1"], 0.0)
    r = consolidar([a], {}, {})
    assert r["metricas"]["recall_norm_at_1"]["media"] == 0.0
    r = consolidar([b], {}, {})
    assert r["metricas"]["recall_at_3"]["media"] == 0.5          # 2 de 4 golds
    assert r["metricas"]["recall_norm_at_3"]["media"] == 0.6667  # 2 de min(3,4)=3


def test_registrar_preserva_todo_campo_do_registro() -> None:
    """O bug que este teste existe para impedir: os quatro módulos de avaliação
    reconstruíam `Avaliada` campo a campo, e quando `n_gold` foi acrescentado os
    quatro o descartaram — os relatórios saíram com teto 0,0 e recall normalizado
    0,0, sem erro, sem falha de tipo e sem teste vermelho. `_registrar` copia o
    registro inteiro, então a asserção é sobre TODOS os campos, não sobre `n_gold`.
    """
    from dataclasses import fields

    gold = {"T::0", "T::1"}
    base = _metricas_por_pergunta(gold, ["T::0", "T::5"], 1.5)
    reg = _registrar(gold, ["T::0", "T::5"], 1.5, "Exclusivity")
    assert reg.categoria == "Exclusivity"
    for f in fields(base):
        if f.name != "categoria":
            assert getattr(reg, f.name) == getattr(base, f.name), f.name


def test_obter_chunker_rejeita_nome_invalido() -> None:
    assert obter_chunker("janela") is chunkar
    assert obter_chunker("clausula") is chunkar_clausula
    with pytest.raises(ValueError, match="chunker inválido"):
        obter_chunker("semantico")
