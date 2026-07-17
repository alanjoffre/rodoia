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
from typing import Any

# Início de artigo: "Art. 1º", "Art. 2", "Artigo 3º" — âncora de divisão.
_RE_ARTIGO = re.compile(r"(?=\bArt(?:igo)?\.?\s*\d+)")

# Assinaturas do 'chrome' do portal ANTTlegis (menu/cabeçalho/rodapé de navegação) — NÃO é texto
# normativo e não deve virar chunk recuperável. Um trecho com 2+ sinais é navegação, não norma.
_BOILERPLATE = (
    "portal gov.br", "acesso rápido", "órgãos do governo", "acesso à informação",
    "entrar com o login", "navegação", "mapa do site", "fale conosco", "voltar ao topo",
)


def _e_boilerplate(texto: str) -> bool:
    """True se o trecho é navegação do portal (2+ sinais), não conteúdo da norma."""
    t = texto.lower()
    return sum(1 for s in _BOILERPLATE if s in t) >= 2


# Piso conservador: nada de normativo cabe em <5 caracteres, mas um resíduo de corte cabe — ex.: o
# "1" órfão que sobra de "Seção 1 Carregando... Voltar ao Topo" depois de tirar o rodapé. Chunk
# assim não é recuperável de forma útil, só ocupa uma linha do índice com um embedding sem sentido.
_MIN_CHARS_CHUNK = 5


def _e_degenerado(texto: str) -> bool:
    """True se o trecho é curto demais para carregar sentido (resíduo de corte/janela)."""
    return len(texto.strip()) < _MIN_CHARS_CHUNK


# O texto raspado traz o 'chrome' do portal ANTES do ato; o conteúdo real começa no cabeçalho.
_RE_CABECALHO = re.compile(r"RESOLU[ÇC][ÃA]O\s+N", re.I)


def _cortar_ate_cabecalho(texto: str) -> str:
    """Remove o menu/cabeçalho de navegação que precede o ato: corta até 'RESOLUÇÃO Nº'."""
    m = _RE_CABECALHO.search(texto)
    return texto[m.start():] if m else texto


# ...e traz o rodapé DEPOIS dele. `_e_boilerplate` não pega este: exige 2+ sinais e o rodapé só
# tem um ("voltar ao topo"), então 133 chunks (3,6%) carregavam "Carregando... Voltar ao Topo"
# grudado no fim — e 2 chunks eram SÓ isso, recuperáveis numa busca. Simétrico ao corte do
# cabeçalho: o conteúdo da norma termina onde a navegação do portal começa.
# Corta a partir de "Carregando...", sem tentar engolir um número antes dele: o texto real termina
# em "D.O.U., dd/mm/aaaa - Seção 1", e um `\d+\s+` opcional aqui levaria embora o "1" da Seção.
_RE_RODAPE = re.compile(r"\s*Carregando\.{3}\s*Voltar ao Topo.*$", re.I | re.S)


def _cortar_do_rodape(texto: str) -> str:
    """Remove o rodapé de navegação do portal a partir de 'Carregando... Voltar ao Topo'."""
    return _RE_RODAPE.sub("", texto)


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


def chunk_norma(
    registro: dict[str, Any], max_chars: int = 1500, overlap: int = 200
) -> list[dict[str, Any]]:
    """Transforma uma norma (dict do JSONL) numa lista de chunks com metadados
    para citação (número, ano, órgão, vigência, título)."""
    # 1) corta o menu/cabeçalho do portal ANTES do ato e o rodapé DEPOIS dele — nos dois casos no
    # texto inteiro, antes de fatiar, para o lixo não sobreviver picado num chunk;
    # 2) dropa trechos residuais de navegação que ainda escapem.
    texto = _cortar_do_rodape(_cortar_ate_cabecalho(registro["texto"]))
    pedacos = [
        p for p in chunk_texto(texto, max_chars, overlap)
        if not _e_boilerplate(p) and not _e_degenerado(p)
    ]
    meta = {k: registro[k] for k in ("id", "numero", "ano", "orgao", "vigente", "titulo")}
    return [
        {**meta, "chunk_id": f"{registro['id']}::{i}", "chunk_index": i, "texto": pedaco}
        for i, pedaco in enumerate(pedacos)
    ]
