"""Testes do gate de avaliação (Fase 5) — o portão de qualidade do CI."""
import json

from rodoia.mlops.gate import _acessar, _passou, avaliar

# Espelha o número anunciado no badge do README ("gate de avaliação 12/12"). É um LITERAL de
# propósito: derivá-lo de `len(GATES)` tornaria o teste tautológico (passaria sempre) e não
# protegeria o badge, que é justamente o que ele existe para proteger.
N_PORTOES_ANUNCIADOS = 19


def test_reports_atuais_passam():
    """Com os relatórios versionados, o gate aprova (nenhuma regressão)."""
    tudo_ok, linhas = avaliar()
    reprovados = [x["nome"] for x in linhas if not x["ok"]]
    assert tudo_ok, f"portões reprovados: {reprovados}"


def test_numero_de_portoes_travado():
    """O README anuncia "gate 12/12": este teste é o que TRAVA esse número.

    Um `>=` deixaria alguém remover portões com o CI verde e o badge virando mentira em
    silêncio. Aqui a igualdade força a decisão a ser consciente: mexeu no nº de portões,
    o teste falha e o badge/README têm de ser atualizados no mesmo diff.
    """
    _, linhas = avaliar()
    assert len(linhas) == N_PORTOES_ANUNCIADOS, (
        f"o gate tem {len(linhas)} portões, mas o README/badge anuncia "
        f"{N_PORTOES_ANUNCIADOS}. Atualize os dois no mesmo commit."
    )


def test_comparadores():
    assert _passou(0.8, ">=", 0.78)
    assert not _passou(0.5, ">=", 0.78)
    assert _passou(13.2, "<=", 15.0)
    assert not _passou(16.0, "<=", 15.0)
    assert _passou(True, "==", True)


def test_acessar_caminho_pontilhado():
    obj = {"modelos": {"ft": {"f1": 0.77}}, "ic": [1.7, 4.4]}
    assert _acessar(obj, "modelos.ft.f1") == 0.77
    assert _acessar(obj, "ic.0") == 1.7


def test_relatorio_ausente_falha(tmp_path):
    """Relatório faltando ⇒ o portão FALHA (não estoura exceção) — CI reprova."""
    # aponta a raiz para um diretório vazio: todos os relatórios somem
    tudo_ok, linhas = avaliar(raiz=tmp_path)
    assert tudo_ok is False
    assert all(x["erro"] for x in linhas)


def test_regressao_reprova(tmp_path):
    """Simula uma métrica regredida e confirma que o portão correspondente reprova."""
    # cria só o relatório do RAG com hit@5 abaixo do piso
    (tmp_path / "reports" / "fase1_retrieval").mkdir(parents=True)
    (tmp_path / "reports" / "fase1_retrieval" / "avaliacao_retrieval.json").write_text(
        json.dumps({"hibrido": {"hit_rate_at_k": 0.30}}))
    _, linhas = avaliar(raiz=tmp_path)
    rag = [x for x in linhas if "hit@5" in x["nome"]][0]
    assert rag["ok"] is False and rag["valor"] == 0.30
