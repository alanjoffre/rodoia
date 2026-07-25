"""Testes da proveniência — foco no bug de ENCODING que só aparecia no Windows.

`git diff HEAD` pode devolver bytes inválidos na codepage ANSI do Windows (cp1252).
Com `text=True`, o subprocess devolvia stdout=None e o `.encode()` estourava —
mas só no Windows; Linux/UTF-8 (CI) nunca caía. Estes testes fixam o contrato
independente de plataforma, mockando o subprocess para devolver bytes crus.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from rodoia import proveniencia as prov


class _FakeCompleted:
    def __init__(self, stdout: bytes):
        self.stdout = stdout


def _fake_run_factory(por_comando: dict[str, bytes]):
    """Devolve um `subprocess.run` falso que responde por bytes conforme o comando."""

    def _run(cmd: list[str], **_: Any) -> _FakeCompleted:
        chave = " ".join(cmd[:3])
        if chave not in por_comando:
            raise subprocess.SubprocessError(f"comando inesperado: {chave}")
        return _FakeCompleted(por_comando[chave])

    return _run


def test_git_dirty_com_byte_invalido_em_cp1252(monkeypatch: pytest.MonkeyPatch) -> None:
    """O byte 0x90 é indecodificável em cp1252 — era exatamente o que derrubava o
    Windows. Hasheando bytes crus, não pode mais estourar."""
    diff = b"diff --git a/x b/x\n+conteudo com byte \x90 no meio\n"
    monkeypatch.setattr(
        prov.subprocess,
        "run",
        _fake_run_factory(
            {
                "git status --porcelain": b" M x\n",
                "git diff HEAD": diff,
            }
        ),
    )
    r = prov._git_dirty()
    assert r["git_dirty"] is True
    # sha1 dos bytes crus, estável e determinístico
    import hashlib

    assert r["git_diff_sha1"] == hashlib.sha1(diff).hexdigest()[:12]


def test_git_dirty_arvore_limpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prov.subprocess,
        "run",
        _fake_run_factory({"git status --porcelain": b""}),
    )
    assert prov._git_dirty() == {"git_dirty": False}


def test_git_dirty_sem_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_: Any, **__: Any) -> None:
        raise OSError("git ausente")

    monkeypatch.setattr(prov.subprocess, "run", _boom)
    assert prov._git_dirty() == {"git_dirty": None}


def test_git_sha_decodifica_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prov.subprocess,
        "run",
        _fake_run_factory({"git rev-parse --short": b"abc1234\n"}),
    )
    assert prov._git_sha() == "abc1234"


def test_carimbar_nunca_estoura(monkeypatch: pytest.MonkeyPatch) -> None:
    """O contrato de mais alto nível: carimbar um report não pode levantar,
    mesmo com diff sujo de bytes estranhos."""
    monkeypatch.setattr(
        prov.subprocess,
        "run",
        _fake_run_factory(
            {
                "git rev-parse --short": b"deadbee\n",
                "git status --porcelain": b" M y\n",
                "git diff HEAD": b"\x90\x91\x92 binario",
            }
        ),
    )
    report = prov.carimbar({"metrica": 1.0})
    assert report["metrica"] == 1.0
    assert report["_proveniencia"]["git_sha"] == "deadbee"
    assert report["_proveniencia"]["git_dirty"] is True
