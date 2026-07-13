"""Testes da estratégia de chunking consciente da estrutura jurídica."""

from __future__ import annotations

from rodoia.rag.chunking import (
    _e_boilerplate,
    chunk_norma,
    chunk_texto,
    dividir_por_artigos,
    empacotar,
)


def test_filtra_boilerplate_do_portal():
    # navegação do portal (2+ sinais) é detectada; o corte até o cabeçalho a remove antes do chunk
    nav = "ANTTlegis Portal Gov.br Acesso rápido Órgãos do Governo Navegação Entrar com o Login"
    assert _e_boilerplate(nav) is True
    assert _e_boilerplate("Art. 1º O transportador deve pagar o vale-pedágio.") is False
    reg = {"id": "RES_6000_2022", "numero": "6000/2022", "ano": 2022, "orgao": "ANTT",
           "vigente": True, "titulo": "RESOLUÇÃO Nº 6.000",
           "texto": nav + " RESOLUÇÃO Nº 6.000, DE 1 DE JANEIRO DE 2022. "
                    "Art. 1º O transportador deve pagar o vale-pedágio à parte do frete."}
    chunks = chunk_norma(reg)
    # o conteúdo normativo sobrevive e nenhum chunk é navegação do portal
    assert chunks and all(not _e_boilerplate(c["texto"]) for c in chunks)
    assert any("vale-pedágio" in c["texto"] for c in chunks)


def test_divide_por_artigos() -> None:
    texto = "Ementa da norma. Art. 1º Primeira regra. Art. 2º Segunda regra."
    partes = dividir_por_artigos(texto)
    assert len(partes) == 3  # ementa + 2 artigos
    assert partes[1].startswith("Art. 1")
    assert partes[2].startswith("Art. 2")


def test_empacota_respeita_tamanho() -> None:
    blocos = ["a" * 400, "b" * 400, "c" * 400]
    chunks = empacotar(blocos, max_chars=900, overlap=0)
    assert all(len(c) <= 900 for c in chunks)
    assert len(chunks) == 2  # (400+400) e (400)


def test_artigo_gigante_e_fatiado_com_overlap() -> None:
    blocos = ["Art. 1º " + "x" * 3000]
    chunks = empacotar(blocos, max_chars=1000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    # sobreposição: o fim de um chunk reaparece no início do próximo
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_texto_cobre_todo_conteudo() -> None:
    texto = "Preâmbulo. " + " ".join(f"Art. {i}º regra {i}." for i in range(1, 10))
    chunks = chunk_texto(texto, max_chars=60, overlap=0)
    assert len(chunks) >= 3
    assert all(c.strip() for c in chunks)


def test_chunk_norma_carrega_metadados() -> None:
    reg = {
        "id": "RES_6000_2022",
        "numero": "6000/2022",
        "ano": 2022,
        "orgao": "DG/ANTT/MT",
        "vigente": True,
        "titulo": "RESOLUÇÃO Nº 6.000",
        "texto": "Ementa. Art. 1º Regra um. Art. 2º Regra dois.",
    }
    chunks = chunk_norma(reg, max_chars=30, overlap=0)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c["numero"] == "6000/2022"
        assert c["vigente"] is True
        assert c["chunk_id"] == f"RES_6000_2022::{i}"
