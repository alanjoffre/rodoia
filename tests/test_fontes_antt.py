"""Testes das funções puras do cliente ANTTlegis (parsing, limpeza, vigência)."""

from __future__ import annotations

from rodoia.rag.fontes_antt import (
    Ato,
    esta_vigente,
    extrair_atos,
    extrair_titulo,
    limpar_html,
    texto_valido,
)


def test_extrai_atos_de_linktexto() -> None:
    html = (
        "onclick=\"LinkTexto('RES','00000059','000','2002','DG/ANTT/MT')\" "
        "onclick=\"LinkTexto('POR','00000123','000','2010','DG/ANTT/MT')\" "
        "onclick=\"LinkTexto('RES','00006078','000','2026','DG/ANTT/MT')\""
    )
    atos = extrair_atos(html, apenas_tipo="RES")
    assert len(atos) == 2  # ignora a POR
    assert atos[0].numero_legivel == "59/2002"
    assert atos[1].id == "RES_6078_2026"


def test_limpar_html_remove_tags_e_desescapa() -> None:
    raw = "<script>x=1</script><b>RESOLU&Ccedil;&Atilde;O</b>  N&ordm; 10 <p>art.  1&ordm;</p>"
    texto = limpar_html(raw)
    assert "<" not in texto and "script" not in texto
    assert "RESOLUÇÃO" in texto
    assert "  " not in texto  # espaços normalizados


def test_extrair_titulo() -> None:
    texto = "RESOLUÇÃO Nº 5.849, DE 16 DE JULHO DE 2019 Estabelece as regras gerais..."
    assert extrair_titulo(texto).startswith("RESOLUÇÃO Nº 5.849")


def test_vigencia_detecta_revogacao_passiva() -> None:
    revogada = "RESOLUÇÃO Nº 5.849 ... Revogada pela Resolução 5867/2020/DG/ANTT/MI ..."
    ativa = "RESOLUÇÃO Nº 5.867 ... Revoga a Resolução 5849/2019. Redação dada pela ..."
    assert esta_vigente(revogada) is False
    assert esta_vigente(ativa) is True  # 'Revoga' (ativo) ≠ 'Revogada por'


def test_texto_valido_distingue_casca() -> None:
    corpo = "RESOLUÇÃO Nº 6.000 " + ("texto da norma " * 2000)  # > 18k chars
    casca = "Portal Gov.br Acesso rápido Menu " * 100  # sem título, curto
    assert texto_valido(corpo) is True
    assert texto_valido(casca) is False


def test_ato_id_e_url() -> None:
    from rodoia.rag.fontes_antt import url_ato

    ato = Ato("RES", "00005386", "000", "2017", "DG/ANTT/MTPA")
    assert ato.id == "RES_5386_2017"
    u = url_ato(ato)
    assert "num_ato=00005386" in u and "sgl_orgao=DG%2FANTT%2FMTPA" in u
