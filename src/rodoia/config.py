"""Configuração central do RodoIA.

Config sobre hard-code: caminhos e parâmetros vêm daqui (com override por variável
de ambiente), nunca espalhados no código. Seeds fixas para reprodutibilidade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do repositório = três níveis acima deste arquivo (src/rodoia/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Parâmetros globais. Override via .env ou variável de ambiente."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Diretórios de dados (versionados por DVC, fora do Git).
    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"

    # RAG (Fase 1): corpus, índice vetorial local e modelo de embeddings.
    normas_jsonl: Path = REPO_ROOT / "data" / "raw" / "normas" / "normas.jsonl"
    qdrant_path: Path = REPO_ROOT / "data" / "processed" / "qdrant_normas"
    embedding_model: str = "intfloat/multilingual-e5-small"

    # LLM de geração (local-first via Ollama; troca-se por API mudando o backend).
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"

    # Reprodutibilidade científica.
    seed: int = 42

    # Destino da trilha de auditoria e das métricas por requisição.
    # `arquivo` (default) mantém o comportamento local: JSONL em `logs/`.
    # `stdout` emite uma linha JSON por evento e deixa a PLATAFORMA coletar — é o
    # que torna a API deployável em contêiner efêmero. Com disco efêmero e mais de
    # uma instância, o append em arquivo perde a trilha SEM ERRO NENHUM: cada
    # instância escreve a sua e o reciclo apaga (docs/16 §7.1).
    log_destino: Literal["arquivo", "stdout"] = "arquivo"


settings = Settings()
