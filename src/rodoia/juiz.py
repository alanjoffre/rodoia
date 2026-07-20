"""Utilitário compartilhado de juiz-LLM: extração robusta do JSON da saída.

Os juízes (win-rate, factual, painel de 3, avaliação de geração) pedem ao modelo uma
resposta em JSON (`{"nota": ...}`, `{"melhor": ...}`, `{"faithfulness": ...}`) e precisam
lê-la tolerando lixo em volta. Este módulo centraliza a extração; cada juiz aplica a
própria leitura de chave/clamp (as regras variam e devem ficar no site de uso).
"""
from __future__ import annotations

import json
import re
from typing import Any

_RE_JSON = re.compile(r"\{[^{}]*\}", re.S)


def extrair_json(saida: str | None) -> dict[str, Any]:
    """Extrai o primeiro objeto ``{...}`` da saída de um juiz-LLM e o parseia.

    Tolerante a texto em volta e a saídas malformadas: retorna ``{}`` se não houver
    objeto JSON ou se o parse falhar. O chamador decide chave, tipo e clamp.
    """
    m = _RE_JSON.search(saida or "")
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}
