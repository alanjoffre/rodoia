"""Chunking da regulação da ANTT — estratégia consciente da estrutura jurídica.

Por que não janela cega de N caracteres: uma resolução tem estrutura semântica
(Art. 1º, Art. 2º, §, incisos). Quebrar no meio de um artigo separa a regra do seu
contexto e piora o retrieval. Estratégia adotada:

1. Divide o texto nos limites de **artigo** (`Art. Nº`) — a unidade natural.
2. **Empacota** artigos consecutivos num chunk até um alvo de tamanho
   (`max_chars`), preservando artigos inteiros juntos.
3. Um artigo grande demais para um chunk é fatiado por **janela deslizante com
   sobreposição** (`overlap`), para não perder contexto na fronteira.

Complexidade: O(n) no tamanho do texto (uma varredura para dividir, uma para
empacotar). Cada chunk carrega os metadados da norma para permitir **citação da
fonte** na resposta.
"""

from __future__ import annotations

import re

# Início de artigo: "Art. 1º", "Art. 2", "Artigo 3º" — âncora de divisão.
_RE_ARTIGO = re.compile(r"(?=\bArt(?:igo)?\.?\s*\d+)")


def dividir_por_artigos(texto: str) -> list[str]:
    """Divide o texto nos limites de artigo, preservando cada 'Art. N' com seu
    conteúdo. O trecho anterior ao 1º artigo (ementa/preâmbulo) vira o 1º bloco."""
    partes = [p.strip() for p in _RE_ARTIGO.split(texto) if p.strip()]
    return partes or [texto.strip()]


def _fatiar_janela(bloco: str, max_chars: int, overlap: int) -> list[str]:
    """Janela deslizante com sobreposição para um bloco maior que max_chars."""
    passo = max(1, max_chars - overlap)
    return [bloco[i : i + max_chars] for i in range(0, len(bloco), passo)]


def empacotar(blocos: list[str], max_chars: int, overlap: int) -> list[str]:
    """Agrupa blocos (artigos) em chunks até max_chars; fatia os grandes demais."""
    chunks: list[str] = []
    atual = ""
    for bloco in blocos:
        if len(bloco) > max_chars:
            if atual:
                chunks.append(atual.strip())
                atual = ""
            chunks.extend(_fatiar_janela(bloco, max_chars, overlap))
            continue
        if atual and len(atual) + len(bloco) + 1 > max_chars:
            chunks.append(atual.strip())
            atual = bloco
        else:
            atual = f"{atual} {bloco}".strip() if atual else bloco
    if atual.strip():
        chunks.append(atual.strip())
    return chunks


def chunk_texto(texto: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Pipeline de chunking: por artigos + empacotamento + janela p/ artigos longos."""
    return empacotar(dividir_por_artigos(texto), max_chars, overlap)


def chunk_norma(registro: dict, max_chars: int = 1500, overlap: int = 200) -> list[dict]:
    """Transforma uma norma (dict do JSONL) numa lista de chunks com metadados
    para citação (número, ano, órgão, vigência, título)."""
    pedacos = chunk_texto(registro["texto"], max_chars, overlap)
    meta = {k: registro[k] for k in ("id", "numero", "ano", "orgao", "vigente", "titulo")}
    return [
        {**meta, "chunk_id": f"{registro['id']}::{i}", "chunk_index": i, "texto": pedaco}
        for i, pedaco in enumerate(pedacos)
    ]
