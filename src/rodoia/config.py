"""Configuração central do RodoIA.

Config sobre hard-code: caminhos e parâmetros vêm daqui (com override por variável
de ambiente), nunca espalhados no código. Seeds fixas para reprodutibilidade.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do repositório = três níveis acima deste arquivo (src/rodoia/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Parâmetros globais. Override via .env ou variável de ambiente."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Diretórios de dados (versionados por DVC, fora do Git).
    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"

    # Reprodutibilidade científica.
    seed: int = 42


settings = Settings()
