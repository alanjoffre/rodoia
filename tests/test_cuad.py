"""Testes do parser do CUAD (sem rede: o zip é sintetizado aqui).

Cobre o que quebra silenciosamente num benchmark externo:
- extração da categoria embutida no enunciado;
- **conferência de offset** — gold que aponta para o lugar errado é pior que
  gold ausente, porque a métrica sai plausível e errada;
- recusa alta quando o formato diverge do esperado.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from rodoia.rag.cuad import (
    Contrato,
    Pergunta,
    Span,
    _extrair_categoria,
    carregar,
    estatisticas,
    gravar_jsonl,
    validar_offsets,
)

_ENUNCIADO = (
    'Highlight the parts (if any) of this contract related to "{cat}" that should be '
    "reviewed by a lawyer. Details: {det}"
)


def _zip_cuad(destino: Path, contratos: list[dict]) -> Path:
    caminho = destino / "cuad.zip"
    payload = {"version": "aok_v1.0", "data": contratos}
    with zipfile.ZipFile(caminho, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("CUAD_v1/CUAD_v1.json", json.dumps(payload))
    return caminho


def _contrato(titulo: str, texto: str, qas: list[dict]) -> dict:
    return {"title": titulo, "paragraphs": [{"context": texto, "qas": qas}]}


def test_extrair_categoria() -> None:
    assert _extrair_categoria(_ENUNCIADO.format(cat="Exclusivity", det="x")) == "Exclusivity"
    cap = _extrair_categoria(_ENUNCIADO.format(cat="Cap On Liability", det="y"))
    assert cap == "Cap On Liability"


def test_categoria_fora_do_padrao_nao_derruba() -> None:
    """Enunciado fora do padrão vira 'desconhecida' — auditável no relatório,
    em vez de exceção que aborta 20 mil perguntas."""
    assert _extrair_categoria("pergunta sem aspas") == "desconhecida"


def test_carrega_e_conta(tmp_path: Path) -> None:
    texto = "O CONTRATO. A parte se obriga a nada."
    zip_path = _zip_cuad(
        tmp_path,
        [
            _contrato(
                "ACORDO_A",
                texto,
                [
                    {
                        "id": "q1",
                        "question": _ENUNCIADO.format(cat="Document Name", det="nome"),
                        "answers": [{"text": "O CONTRATO", "answer_start": 0}],
                        "is_impossible": False,
                    },
                    {
                        "id": "q2",
                        "question": _ENUNCIADO.format(cat="Exclusivity", det="excl"),
                        "answers": [],
                        "is_impossible": True,
                    },
                ],
            )
        ],
    )
    contratos = carregar(zip_path)
    assert len(contratos) == 1
    estat = estatisticas(contratos)
    assert estat["n_perguntas"] == 2
    assert estat["n_impossiveis"] == 1
    assert estat["pct_impossiveis"] == 50.0
    assert estat["n_spans"] == 1
    assert estat["categorias"] == {"Document Name": 1, "Exclusivity": 1}


def test_offsets_conferem() -> None:
    texto = "abcdefghij"
    c = Contrato(
        titulo="T",
        texto=texto,
        perguntas=(
            Pergunta("q", "T", "e", "cat", False, (Span(texto="cde", inicio=2),)),
        ),
    )
    assert validar_offsets([c]) == {"spans_conferidos": 1, "spans_divergentes": 0}


def test_offset_errado_e_denunciado() -> None:
    """Gold apontando para o lugar errado produz métrica plausível e falsa —
    por isso a conferência é explícita, não implícita."""
    c = Contrato(
        titulo="T",
        texto="abcdefghij",
        perguntas=(
            Pergunta("q", "T", "e", "cat", False, (Span(texto="cde", inicio=5),)),
        ),
    )
    assert validar_offsets([c]) == {"spans_conferidos": 0, "spans_divergentes": 1}


def test_multiplos_paragrafos_sao_recusados(tmp_path: Path) -> None:
    """Concatenar parágrafos invalidaria os offsets — melhor falhar alto."""
    caminho = tmp_path / "cuad.zip"
    dois = [{"context": "a", "qas": []}, {"context": "b", "qas": []}]
    payload = {"version": "x", "data": [{"title": "T", "paragraphs": dois}]}
    with zipfile.ZipFile(caminho, "w") as zf:
        zf.writestr("CUAD_v1.json", json.dumps(payload))
    with pytest.raises(ValueError, match="offsets"):
        carregar(caminho)


def test_zip_sem_json_e_recusado(tmp_path: Path) -> None:
    caminho = tmp_path / "cuad.zip"
    with zipfile.ZipFile(caminho, "w") as zf:
        zf.writestr("leiame.txt", "nada aqui")
    with pytest.raises(ValueError, match="não encontrado"):
        carregar(caminho)


def test_sem_zip_falha_claro(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="baixar_cuad"):
        carregar(tmp_path / "ausente.zip")


def test_gravar_jsonl(tmp_path: Path) -> None:
    c = Contrato(
        titulo="T",
        texto="abcdefghij",
        perguntas=(Pergunta("q1", "T", "enun", "Exclusivity", False, (Span("cde", 2),)),),
    )
    p_contratos, p_perguntas = gravar_jsonl([c], tmp_path / "cuad")
    linha_c = json.loads(p_contratos.read_text(encoding="utf-8").strip())
    linha_p = json.loads(p_perguntas.read_text(encoding="utf-8").strip())
    assert linha_c == {"titulo": "T", "texto": "abcdefghij"}
    assert linha_p["categoria"] == "Exclusivity"
    assert linha_p["spans"] == [{"texto": "cde", "inicio": 2, "fim": 5}]
