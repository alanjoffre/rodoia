"""Estado do agente e contrato das ferramentas (injeção de dependência).

O estado é o que trafega entre os nós do grafo; `DepsAgente` isola o grafo dos
provedores concretos (RAG/vLLM/DuckDB/Ollama) — nos testes injetamos fakes
determinísticos, na demo injetamos as ferramentas reais das Fases 1–3.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import add
from typing import Annotated, Any, Protocol, TypedDict

# Rotas válidas = as três ferramentas (fases anteriores). "fora_de_escopo" é tratada à parte.
ROTAS = ("regulatorio", "entidades", "dados")


class EstadoAgente(TypedDict, total=False):
    pergunta: str
    rotas: list[str]              # ferramentas escolhidas pelo roteador
    motivo_rota: str              # justificativa (observabilidade + avaliação de trajetória)
    bloqueado: bool               # guardrail de injection
    motivo_bloqueio: str | None
    fora_de_escopo: bool
    evidencias: dict[str, dict[str, Any]]    # {ferramenta: resultado estruturado}
    # `add` acumula os passos escritos por cada nó (senão o último sobrescreveria os anteriores).
    trajetoria: Annotated[list[dict[str, Any]], add]   # passos executados (avaliação de trajetória)
    resposta: str
    fontes: list[str]


# Uma ferramenta recebe a pergunta (ou o texto) e devolve um dicionário estruturado.
Ferramenta = Callable[[str], dict[str, Any]]


class LLMCerebro(Protocol):
    def gerar(self, prompt: str, sistema: str | None = None) -> str: ...


@dataclass
class DepsAgente:
    """Dependências injetadas no grafo. `llm_cerebro` roteia e sintetiza; as três
    ferramentas encapsulam F1/F2/F3."""
    llm_cerebro: LLMCerebro             # objeto com .gerar(prompt, sistema) -> str
    regulatorio: Ferramenta             # RAG da Fase 1
    entidades: Ferramenta               # modelo fine-tunado (NER) da Fase 2
    dados: Ferramenta                   # camada de acesso + cálculo da Fase 3

    def ferramenta(self, rota: str) -> Ferramenta:
        return {"regulatorio": self.regulatorio, "entidades": self.entidades,
                "dados": self.dados}[rota]
