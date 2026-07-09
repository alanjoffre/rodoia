"""Interface de LLM para a geração (local-first via Ollama).

A interface `LLM` isola o resto do pipeline do provedor concreto — trocar Ollama
por uma API (OpenAI/Anthropic) é só implementar outra classe, sem tocar no RAG.
O backend Ollama fala HTTP direto (sem dependência extra) com o serviço local.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol

from rodoia.config import settings


class LLM(Protocol):
    def gerar(self, prompt: str, sistema: str | None = None) -> str: ...


class OllamaLLM:
    """Backend local via Ollama (`/api/chat`). Temperatura baixa para respostas
    factuais e ancoradas no contexto."""

    def __init__(
        self,
        modelo: str | None = None,
        base_url: str | None = None,
        temperatura: float = 0.1,
        timeout: int = 120,
    ):
        self.modelo = modelo or settings.llm_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.temperatura = temperatura
        self.timeout = timeout

    def gerar(self, prompt: str, sistema: str | None = None) -> str:
        mensagens = []
        if sistema:
            mensagens.append({"role": "system", "content": sistema})
        mensagens.append({"role": "user", "content": prompt})
        corpo = json.dumps(
            {
                "model": self.modelo,
                "messages": mensagens,
                "stream": False,
                "options": {"temperature": self.temperatura},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=corpo,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            dados = json.loads(resp.read().decode("utf-8"))
        return dados["message"]["content"].strip()
