"""Testes da geração RAG com LLM e recuperador falsos (sem servidor/modelo)."""

from __future__ import annotations

from rodoia.rag.gerar import montar_contexto, montar_prompt, responder


class LLMFalso:
    """Captura o prompt recebido e devolve uma resposta previsível."""

    def __init__(self):
        self.ultimo_prompt = None
        self.ultimo_sistema = None

    def gerar(self, prompt: str, sistema: str | None = None) -> str:
        self.ultimo_prompt = prompt
        self.ultimo_sistema = sistema
        return "Conforme a Resolução 6024/2023, o vale-pedágio é obrigatório."


class RecuperadorFalso:
    reranker = None

    def __init__(self, chunks):
        self._chunks = chunks

    def buscar(self, consulta, k=5, modo="hibrido", rerank=False):
        return self._chunks[:k]


_CHUNKS = [
    {
        "chunk_id": "A::0",
        "numero": "6024/2023",
        "vigente": True,
        "texto": "vale-pedágio obrigatório.",
    },
    {
        "chunk_id": "B::0",
        "numero": "6024/2023",
        "vigente": True,
        "texto": "responsabilidade do embarcador.",
    },
    {
        "chunk_id": "C::0",
        "numero": "5000/2010",
        "vigente": False,
        "texto": "norma antiga revogada.",
    },
]


def test_montar_contexto_numera_e_marca_vigencia() -> None:
    ctx = montar_contexto(_CHUNKS[:1])
    assert "[Fonte 1 — Resolução 6024/2023 (vigente)]" in ctx
    assert "vale-pedágio" in ctx


def test_montar_prompt_inclui_pergunta_e_contexto() -> None:
    p = montar_prompt("O vale-pedágio é obrigatório?", _CHUNKS[:1])
    assert "O vale-pedágio é obrigatório?" in p
    assert "6024/2023" in p


def test_responder_retorna_fontes_unicas_e_chama_llm() -> None:
    llm = LLMFalso()
    rec = RecuperadorFalso(_CHUNKS)
    r = responder("O vale-pedágio é obrigatório?", rec, llm, k=3, apenas_vigentes=True)
    # revogada filtrada -> só 6024/2023, e única (aparece 2x nos chunks)
    assert r["fontes"] == ["6024/2023"]
    assert "6024/2023" in r["resposta"]
    # o LLM recebeu o contexto e o sistema de grounding
    assert "vale-pedágio" in llm.ultimo_prompt
    assert "EXCLUSIVAMENTE" in llm.ultimo_sistema


def test_responder_sem_contexto() -> None:
    r = responder("pergunta qualquer", RecuperadorFalso([]), LLMFalso())
    assert r["fontes"] == []
    assert "não encontrei" in r["resposta"].lower()
