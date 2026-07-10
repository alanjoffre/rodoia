"""Camada de segurança e governança do RAG (Fase 1).

Três defesas:
1. **Guardrail anti-prompt-injection** — detecta tentativas de sobrescrever as
   instruções ou exfiltrar o prompt do sistema, e faz o pipeline recusar.
2. **PII masking** — mascara dados pessoais (CPF, CNPJ, e-mail, telefone, CEP)
   nas respostas e nos logs, atendendo a LGPD.
3. **Trilha de auditoria** — registra cada consulta (com flags) em JSONL, para
   rastreabilidade.

Detecção por heurística/regex é imperfeita por natureza (defesa em profundidade,
não bala de prata) — por isso o prompt do sistema também ancora o modelo.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# --- 1. Prompt injection ---------------------------------------------------
# Padrões específicos o bastante para não disparar em perguntas jurídicas normais
# (ex.: "quais as regras de X" NÃO casa; "ignore as regras acima" casa).
_PADROES_INJECTION: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(r"ignore\s+(as\s+|todas\s+as\s+)?(instru[çc][õo]es|regras|orienta)", re.I),
        "ignorar-instrucoes",
    ),
    (re.compile(r"esque[çc]a\s+(as\s+|tudo)", re.I), "esquecer-instrucoes"),
    (
        re.compile(r"desconsidere\s+(as\s+|o\s+)?(instru|regras|acima|anterior)", re.I),
        "desconsiderar",
    ),
    (
        re.compile(r"ignore\s+(all\s+|previous\s+)?(instructions|rules|the\s+above)", re.I),
        "ignore-en",
    ),
    (re.compile(r"disregard\s+(all|previous|the)", re.I), "disregard-en"),
    (
        re.compile(
            r"(revele|mostre|exiba|repita|imprima)\s+(o\s+|seu\s+|as\s+)?(prompt|sistema|system|instru)",
            re.I,
        ),
        "exfiltrar-prompt",
    ),
    (re.compile(r"system\s*prompt", re.I), "system-prompt"),
    (re.compile(r"voc[êe]\s+(agora\s+)?[ée]\s+(um|uma|o|a)\b", re.I), "trocar-persona"),
    (re.compile(r"you\s+are\s+now\b|act\s+as\s+", re.I), "trocar-persona-en"),
    (re.compile(r"developer\s+mode|modo\s+desenvolvedor|jailbreak", re.I), "jailbreak"),
    (re.compile(r"\[/?INST\]|<\|[^|]*\|>|###\s*(system|instruction)", re.I), "marcadores-role"),
)


def _normalizar(texto: str) -> str:
    """Normaliza p/ frustrar evasões triviais: sem acentos, minúsculas, espaços/
    pontuação repetidos colapsados (ex.: 'IGNORE  as   instruções!!!' → 'ignore as
    instrucoes ')."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[\s\W]+", " ", sem_acento.lower())


def detectar_injection(texto: str) -> tuple[bool, str | None]:
    """Retorna (True, nome_do_padrão) se a entrada parecer uma tentativa de injeção;
    (False, None) caso contrário. Checa o texto cru E a forma normalizada — pega
    evasões por acento/caixa/espaçamento. Heurística: não cobre ofuscação forte
    (base64, letra-a-letra, outra língua) — daí o prompt de sistema também ancorar."""
    normal = _normalizar(texto)
    for padrao, nome in _PADROES_INJECTION:
        if padrao.search(texto) or padrao.search(normal):
            return True, nome
    return False, None


# --- 2. PII masking --------------------------------------------------------
# Ordem importa: CNPJ antes de CPF (o CNPJ é mais longo e específico).
_MASCARAS_PII: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"), "[CNPJ]"),
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\(?\d{2}\)?\s?9\d{4}-?\d{4}\b"), "[TELEFONE]"),
    (re.compile(r"\b\d{5}-\d{3}\b"), "[CEP]"),
    (re.compile(r"\b\d{11}\b"), "[CPF]"),  # CPF sem pontuação (CNPJ bare já cai no 1º padrão)
)


def mascarar_pii(texto: str) -> str:
    """Substitui dados pessoais por rótulos ([CPF], [EMAIL], ...)."""
    for padrao, rotulo in _MASCARAS_PII:
        texto = padrao.sub(rotulo, texto)
    return texto


# --- 3. Auditoria ----------------------------------------------------------
def registrar_auditoria(evento: dict, caminho: Path) -> None:
    """Anexa um evento (dict) à trilha de auditoria em JSONL."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(evento, ensure_ascii=False) + "\n")
