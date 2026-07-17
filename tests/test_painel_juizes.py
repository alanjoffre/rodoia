"""Testes da banca de 3 juízes (Fase 1) — o caminho que produz o κ de Fleiss citado no README.

O módulo tinha 0% de cobertura apesar de gerar uma métrica de vitrine
(`reports/fase1_geracao/painel_juizes.json`). Aqui exercitamos o parse tolerante da nota e a
agregação (voto majoritário, médias, κ) com juízes falsos — sem Ollama, sem GPU.
"""

from __future__ import annotations

from rodoia.rag.painel_juizes import _nota_012, avaliar_painel, julgar


class JuizFalso:
    """Juiz com saída fixa (ou uma por chamada, em sequência)."""

    ultima_metrica: dict = {}        # exigido pelo Protocol LLM (ver rodoia.rag.llm)

    def __init__(self, *saidas: str):
        self._saidas = list(saidas)
        self._i = 0

    def gerar(self, prompt: str, sistema: str | None = None) -> str:
        s = self._saidas[min(self._i, len(self._saidas) - 1)]
        self._i += 1
        return s


class GeradorFalso:
    ultima_metrica: dict = {}

    def gerar(self, prompt: str, sistema: str | None = None) -> str:
        return "O vale-pedágio é obrigatório, conforme a Resolução 6024/2023."


class RecFalso:
    reranker = None

    def buscar(self, consulta, k=5, modo="hibrido", rerank=False):
        return [
            {"chunk_id": "A::0", "numero": "6024/2023", "vigente": True, "texto": "vale-pedágio"}
        ]


# --- parse da nota (tolerante por desenho: juiz é LLM e às vezes foge do formato) ---


def test_nota_extrai_json_valido() -> None:
    assert _nota_012('{"faithfulness": 2}') == 2
    assert _nota_012('Claro! {"faithfulness": 1} — espero ter ajudado.') == 1


def test_nota_sem_json_vira_zero() -> None:
    """Sem JSON, a nota é 0 (conservador: não inventa fidelidade que o juiz não deu)."""
    assert _nota_012("a resposta parece boa") == 0
    assert _nota_012("") == 0
    assert _nota_012(None) == 0


def test_nota_clampa_fora_da_faixa() -> None:
    """A escala é 0/1/2: um juiz alucinando '7' não pode inflar a média."""
    assert _nota_012('{"faithfulness": 7}') == 2
    assert _nota_012('{"faithfulness": -3}') == 0


def test_nota_aceita_float_e_rejeita_lixo() -> None:
    assert _nota_012('{"faithfulness": 1.6}') == 2      # arredonda
    assert _nota_012('{"faithfulness": "duas"}') == 0   # tipo errado → 0, sem estourar


def test_julgar_usa_a_saida_do_juiz() -> None:
    assert julgar("pergunta?", "resposta", "contexto", JuizFalso('{"faithfulness": 2}')) == 2


# --- agregação da banca ---


def test_painel_concordancia_total() -> None:
    """3 juízes dando 2: média 2.0 (norm 1.0), voto majoritário 2, κ de Fleiss = 1 (perfeito)."""
    juizes = {n: JuizFalso('{"faithfulness": 2}') for n in ("j1", "j2", "j3")}
    res = avaliar_painel(RecFalso(), GeradorFalso(), juizes, [{"consulta": "vale-pedágio?"}])
    assert res["n"] == 1
    assert res["fidelidade_media_banca"] == 2.0
    assert res["fidelidade_media_norm"] == 1.0
    assert res["casos"][0]["voto_majoritario"] == 2
    assert res["fleiss_kappa"] == 1.0
    assert res["juizes"] == ["j1", "j2", "j3"]


def test_painel_voto_majoritario_com_divergencia() -> None:
    """2 juízes dizem 2, um diz 0 → maioria 2, mas a média cai (a divergência aparece)."""
    juizes = {
        "j1": JuizFalso('{"faithfulness": 2}'),
        "j2": JuizFalso('{"faithfulness": 2}'),
        "j3": JuizFalso('{"faithfulness": 0}'),
    }
    res = avaliar_painel(RecFalso(), GeradorFalso(), juizes, [{"consulta": "vale-pedágio?"}])
    caso = res["casos"][0]
    assert caso["voto_majoritario"] == 2
    assert caso["media"] == round(4 / 3, 2)
    assert caso["notas"] == {"j1": 2, "j2": 2, "j3": 0}


def test_painel_agrega_varios_casos() -> None:
    """Cada juiz pontua todos os casos (1 carga de modelo por juiz — ver docstring do módulo)."""
    juizes = {
        "j1": JuizFalso('{"faithfulness": 2}', '{"faithfulness": 0}'),
        "j2": JuizFalso('{"faithfulness": 2}', '{"faithfulness": 0}'),
        "j3": JuizFalso('{"faithfulness": 2}', '{"faithfulness": 0}'),
    }
    dourados = [{"consulta": "a?"}, {"consulta": "b?"}]
    res = avaliar_painel(RecFalso(), GeradorFalso(), juizes, dourados)
    assert res["n"] == 2
    assert [c["voto_majoritario"] for c in res["casos"]] == [2, 0]
    assert res["fidelidade_media_banca"] == 1.0        # (2.0 + 0.0) / 2
    assert res["fleiss_kappa"] == 1.0                  # concordam em ambos
