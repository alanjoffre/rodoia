"""Teste-fumaça: garante que o pacote importa e o CI tem algo verde para rodar
desde o commit 1. Será substituído por testes reais dos caminhos críticos em
cada fase (ingestão, treino, retrieval, nós do agente, endpoints)."""

import rodoia


def test_versao_exposta() -> None:
    assert rodoia.__version__ == "0.0.0"
