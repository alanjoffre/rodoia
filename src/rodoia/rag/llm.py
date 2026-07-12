"""Interface de LLM para a geração (local-first via Ollama).

A interface `LLM` isola o resto do pipeline do provedor concreto — trocar Ollama
por uma API (OpenAI/Anthropic) é só implementar outra classe, sem tocar no RAG.
O backend Ollama fala HTTP direto (sem dependência extra) com o serviço local.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Protocol

from rodoia.config import settings


class LLM(Protocol):
    def gerar(self, prompt: str, sistema: str | None = None) -> str: ...

    # métricas da última chamada (observabilidade): tokens_prompt, tokens_resposta, latencia_s
    ultima_metrica: dict


class OllamaLLM:
    """Backend local via Ollama (`/api/chat`). Temperatura baixa para respostas
    factuais e ancoradas no contexto."""

    def __init__(
        self,
        modelo: str | None = None,
        base_url: str | None = None,
        temperatura: float = 0.1,
        timeout: int = 300,   # tolera inferência em CPU (brain fora da GPU) sem estourar
    ):
        self.modelo = modelo or settings.llm_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.temperatura = temperatura
        self.timeout = timeout
        self.ultima_metrica: dict = {}

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
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            dados = json.loads(resp.read().decode("utf-8"))
        # observabilidade: o Ollama já devolve os contadores de tokens — não descartar.
        self.ultima_metrica = {
            "tokens_prompt": dados.get("prompt_eval_count"),
            "tokens_resposta": dados.get("eval_count"),
            "latencia_s": round(time.perf_counter() - t0, 3),
        }
        return dados["message"]["content"].strip()


class OpenAICompatLLM:
    """Backend para qualquer endpoint compatível com a API OpenAI — inclusive o
    **vLLM** (que serve o modelo fine-tunado na Fase 2, `/v1/chat/completions`) e
    as APIs OpenAI/Together. Mesma interface `LLM`, troca por base_url."""

    def __init__(
        self,
        modelo: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "nao-usada",
        temperatura: float = 0.1,
        timeout: int = 120,
    ):
        self.modelo = modelo
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperatura = temperatura
        self.timeout = timeout
        self.ultima_metrica: dict = {}

    def gerar(self, prompt: str, sistema: str | None = None) -> str:
        mensagens = []
        if sistema:
            mensagens.append({"role": "system", "content": sistema})
        mensagens.append({"role": "user", "content": prompt})
        corpo = json.dumps(
            {"model": self.modelo, "messages": mensagens, "temperature": self.temperatura}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=corpo,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            dados = json.loads(resp.read().decode("utf-8"))
        uso = dados.get("usage", {})
        self.ultima_metrica = {
            "tokens_prompt": uso.get("prompt_tokens"),
            "tokens_resposta": uso.get("completion_tokens"),
            "latencia_s": round(time.perf_counter() - t0, 3),
        }
        return dados["choices"][0]["message"]["content"].strip()
