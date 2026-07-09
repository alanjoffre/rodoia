"""Testes do índice vetorial com embedder falso + Qdrant em memória (sem modelo,
sem servidor). Valida indexação, recuperação e filtro de vigência."""

from __future__ import annotations

import numpy as np

from rodoia.rag.indice import buscar, criar_cliente, indexar


class EmbedderFalso:
    """Mapeia texto -> vetor determinístico (bag-of-chars normalizado). Texto
    idêntico gera vetor idêntico, então uma consulta igual a um passage o recupera."""

    dim = 24

    def _vec(self, t: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for ch in t:
            v[ord(ch) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, textos):
        return np.array([self._vec(t) for t in textos])

    encode_queries = encode_passages


def _chunks():
    return [
        {
            "chunk_id": "A::0",
            "numero": "6000/2022",
            "vigente": True,
            "texto": "transporte rodoviario de cargas perigosas",
        },
        {
            "chunk_id": "B::0",
            "numero": "5000/2018",
            "vigente": False,
            "texto": "tarifas de pedagio em rodovias concedidas",
        },
        {
            "chunk_id": "C::0",
            "numero": "6050/2024",
            "vigente": True,
            "texto": "jornada de trabalho do motorista profissional",
        },
    ]


def test_indexa_e_conta() -> None:
    cli = criar_cliente()
    n = indexar(_chunks(), EmbedderFalso(), cli)
    assert n == 3


def test_busca_recupera_o_chunk_certo() -> None:
    cli = criar_cliente()
    chunks = _chunks()
    indexar(chunks, EmbedderFalso(), cli)
    res = buscar("transporte rodoviario de cargas perigosas", EmbedderFalso(), cli, k=1)
    assert res[0]["numero"] == "6000/2022"
    assert res[0]["score"] > 0.99  # match quase perfeito


def test_filtro_apenas_vigentes() -> None:
    cli = criar_cliente()
    indexar(_chunks(), EmbedderFalso(), cli)
    res = buscar("tarifas de pedagio", EmbedderFalso(), cli, k=5, apenas_vigentes=True)
    assert all(r["vigente"] for r in res)
    assert "5000/2018" not in {r["numero"] for r in res}  # revogada foi filtrada
