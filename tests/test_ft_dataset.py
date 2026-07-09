"""Testes do construtor de dataset de fine-tuning (sem rede/modelo)."""

from __future__ import annotations

from rodoia.ft.construir_dataset import _parse_pares, gerar_pares_norma, para_chat


class LLMFalso:
    def __init__(self, saida):
        self.saida = saida

    def gerar(self, prompt, sistema=None):
        return self.saida


def test_parse_pares_extrai_array() -> None:
    saida = 'Claro: [{"pergunta": "O que é o vale-pedágio obrigatório?", "resposta": "É um mecanismo previsto na Resolução 6024/2023."}] pronto.'  # noqa: E501
    pares = _parse_pares(saida)
    assert len(pares) == 1
    assert pares[0]["pergunta"].startswith("O que é")


def test_parse_pares_filtra_curtos() -> None:
    saida = '[{"pergunta": "curto", "resposta": "x"}]'
    assert _parse_pares(saida) == []  # abaixo do mínimo de qualidade


def test_parse_pares_tolerante_a_lixo() -> None:
    assert _parse_pares("sem json") == []


def test_gerar_pares_marca_fonte() -> None:
    llm = LLMFalso(
        '[{"pergunta": "Uma pergunta suficientemente longa?", "resposta": "Uma resposta objetiva e suficientemente longa citando a norma."}]'  # noqa: E501
    )
    norma = {"numero": "6024/2023", "texto": "texto da resolução " * 50}
    pares = gerar_pares_norma(norma, llm, n=1)
    assert pares[0]["fonte"] == "6024/2023"


def test_para_chat_formato_messages() -> None:
    par = {"pergunta": "P?", "resposta": "R.", "fonte": "6024/2023"}
    reg = para_chat(par)
    papeis = [m["role"] for m in reg["messages"]]
    assert papeis == ["system", "user", "assistant"]
    assert reg["messages"][1]["content"] == "P?"
    assert reg["fonte"] == "6024/2023"
