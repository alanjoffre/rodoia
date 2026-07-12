"""Proveniência de artefatos — reprodutibilidade científica (PROMPT_MESTRE §3).

Carimba cada JSON de `reports/` com o suficiente para amarrar um número ao código e
ao ambiente que o gerou: seed, commit git, versões das libs e timestamp. Assim um
resultado deixa de ser um número solto e passa a ser reproduzível/rastreável.
"""
from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from rodoia.config import settings

# Cobre o stack das Fases 0–2; libs ausentes no ambiente viram "n/d" (sem erro).
_LIBS = (
    "numpy", "scikit-learn", "torch", "pandas",
    "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets",
    "vllm", "sentence-transformers", "qdrant-client",
)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "desconhecido"


def _versoes() -> dict[str, str]:
    v = {}
    for lib in _LIBS:
        try:
            v[lib] = version(lib)
        except PackageNotFoundError:
            v[lib] = "n/d"
    return v


def proveniencia() -> dict:
    """Metadados de reprodutibilidade para carimbar num report."""
    return {
        "seed": settings.seed,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "versoes": _versoes(),
    }


def carimbar(report: dict) -> dict:
    """Adiciona `_proveniencia` ao dict do report (in-place) e o devolve."""
    report["_proveniencia"] = proveniencia()
    return report
