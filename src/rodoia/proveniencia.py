"""Proveniência de artefatos — reprodutibilidade científica (PROMPT_MESTRE §3).

Carimba cada JSON de `reports/` com o suficiente para amarrar um número ao código e
ao ambiente que o gerou: seed, commit git, versões das libs e timestamp. Assim um
resultado deixa de ser um número solto e passa a ser reproduzível/rastreável.
"""
from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

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


def _git_dirty() -> dict[str, Any]:
    """Denuncia árvore suja: se há mudança não commitada (tracked ou não), marca `git_dirty`
    e o hash do diff. Sem isso, um número pode sair de working tree modificado sem rastro."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {"git_dirty": None}
    if not status.strip():
        return {"git_dirty": False}
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"], capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        h = hashlib.sha1(diff.encode("utf-8", "ignore")).hexdigest()[:12]
    except (subprocess.SubprocessError, OSError):
        h = None
    return {"git_dirty": True, "git_diff_sha1": h}


def _versoes() -> dict[str, str]:
    v = {}
    for lib in _LIBS:
        try:
            v[lib] = version(lib)
        except PackageNotFoundError:
            v[lib] = "n/d"
    return v


def proveniencia() -> dict[str, Any]:
    """Metadados de reprodutibilidade para carimbar num report."""
    return {
        "seed": settings.seed,
        "git_sha": _git_sha(),
        **_git_dirty(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "versoes": _versoes(),
    }


def carimbar(report: dict[str, Any]) -> dict[str, Any]:
    """Adiciona `_proveniencia` ao dict do report (in-place) e o devolve."""
    report["_proveniencia"] = proveniencia()
    return report
