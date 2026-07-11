"""LeNER-Br — NER jurídico brasileiro (Fase 2, tarefa de rótulo objetivo).

Dataset público (MIT, cite Luz de Araujo et al., PROPOR 2018) de reconhecimento de
entidades nomeadas em textos jurídicos: legislação, jurisprudência, pessoa, organização,
tempo e local. Escolhido porque tem **rótulo objetivo + baseline publicado (~F1 92%)** —
o fine-tuning é medido por F1 de entidade contra SOTA, não por LLM-juiz sobre dado
sintético. Domínio jurídico ⟂ regulação da ANTT (LEGISLACAO/JURISPRUDENCIA).

Baixa os CoNLL consolidados do repositório oficial e os parseia em sentenças.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from rodoia.config import settings

_BASE = "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br"
_DIR = settings.data_raw / "lener_br"

# 6 entidades × BIO + O (ordem canônica do dataset).
LABELS = [
    "O",
    "B-ORGANIZACAO", "I-ORGANIZACAO",
    "B-PESSOA", "I-PESSOA",
    "B-TEMPO", "I-TEMPO",
    "B-LOCAL", "I-LOCAL",
    "B-LEGISLACAO", "I-LEGISLACAO",
    "B-JURISPRUDENCIA", "I-JURISPRUDENCIA",
]
LABEL2ID = {t: i for i, t in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
ENTIDADES = ["ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL", "LEGISLACAO", "JURISPRUDENCIA"]


def baixar(destino: Path | None = None) -> Path:
    """Baixa train/dev/test.conll do repositório oficial (idempotente)."""
    destino = destino or _DIR
    destino.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        alvo = destino / f"{split}.conll"
        if not alvo.exists():
            urllib.request.urlretrieve(f"{_BASE}/{split}/{split}.conll", alvo)
    return destino


def _parse_conll(caminho: Path) -> list[dict]:
    """CoNLL (`token TAG`, sentenças separadas por linha vazia) → [{tokens, ner_tags}]."""
    sentencas, toks, tags = [], [], []
    for linha in caminho.open(encoding="utf-8"):
        linha = linha.rstrip("\n")
        if not linha.strip():
            if toks:
                sentencas.append({"tokens": toks, "ner_tags": tags})
                toks, tags = [], []
            continue
        partes = linha.split()
        toks.append(partes[0])
        tags.append(LABEL2ID.get(partes[-1], 0))
    if toks:
        sentencas.append({"tokens": toks, "ner_tags": tags})
    return sentencas


def carregar(split: str) -> list[dict]:
    """Sentenças de um split ('train'/'dev'/'test') como [{tokens, ner_tags}]."""
    baixar()
    return _parse_conll(_DIR / f"{split}.conll")
