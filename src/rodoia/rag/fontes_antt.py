"""Cliente do ANTTlegis para obter o texto das resoluções da ANTT.

Receita (validada em docs/06): o portal legado 'datalegis' não é buscável por
HTTP, MAS tem dois endpoints públicos e STATELESS:

- **Enumeração** — páginas temáticas `TematicaAction.php?acao=abrirVinculos&...`
  embutem chamadas JS `LinkTexto('RES','00000059','000','2002','DG/ANTT/MT')`,
  de onde extraímos a tupla exata de cada ato.
- **Texto** — `UrlPublicasAction.php?acao=abrirAtoPublico&sgl_tipo=...&num_ato=...`
  devolve o HTML com o corpo integral da norma.

Cuidados confirmados no teste:
- `sgl_orgao` (ministério supervisor) varia por ano → usar SEMPRE o que veio do
  `LinkTexto`, nunca deduzir.
- Se a tupla não bater exata, o servidor devolve a 'casca' do portal (sem o cabeçalho
  'RESOLUÇÃO Nº'). Detector de sucesso: cabeçalho presente E texto > `MIN_CHARS_NORMA`
  (2.500 — inclui resoluções curtas legítimas; um piso alto descartava metade do corpus).
- Servidor Apache lento → ser educado (delay, concorrência baixa).
"""

from __future__ import annotations

import html as htmllib
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass

import certifi

_BASE = "https://anttlegis.antt.gov.br/action"
_UA = {"User-Agent": "Mozilla/5.0 (RodoIA; projeto open-source de dados públicos ANTT)"}
_CTX = ssl.create_default_context(cafile=certifi.where())

# Regex que captura LinkTexto('TIPO','num','seq','ano','orgao').
_RE_LINK = re.compile(r"LinkTexto\('([^']+)','(\d+)','(\d+)','(\d+)','([^']*)'")
# Título próprio do ato: cabeçalho em MAIÚSCULAS (evita casar referências a
# outras resoluções, que vêm em minúsculas no corpo).
_RE_TITULO = re.compile(r"(RESOLU[ÇC][ÃA]O\s+N[º°]\s*[\d.]+[^\n.]{0,200})")
_RE_TITULO_FALLBACK = re.compile(r"(RESOLU[ÇC][ÃA]O\s+N[º°]\s*[\d.]+[^\n.]{0,200})", re.I)
# 'Revogada/Revogado ... pela/por' no cabeçalho = ato morto (≠ 'Revoga' ativo).
_RE_REVOGADA = re.compile(r"Revogad[ao]s?\s+(?:expressamente\s+)?(?:pel[oa]s?|por)\b", re.I)


@dataclass(frozen=True)
class Tema:
    cotematica: str
    nome: str
    cod_menu: str = "7221"
    cod_modulo: str = "392"


# Temas de transporte rodoviário confirmados (há mais sob cod_menu 7220/8719).
TEMAS_PADRAO = (
    Tema("16181785", "cargas_por_numero"),
    Tema("13956887", "produtos_perigosos"),
    Tema("15971753", "carga_lotacao_tabela_a"),
)


@dataclass(frozen=True)
class Ato:
    tipo: str
    num: str  # zero-padded, 8 dígitos (como o endpoint exige)
    seq: str
    ano: str
    orgao: str

    @property
    def id(self) -> str:
        return f"{self.tipo}_{int(self.num)}_{self.ano}"

    @property
    def numero_legivel(self) -> str:
        return f"{int(self.num)}/{self.ano}"


def _get(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
        return resp.read().decode("latin-1", "replace")


def url_tema(tema: Tema) -> str:
    q = urllib.parse.urlencode(
        {
            "acao": "abrirVinculos",
            "cotematica": tema.cotematica,
            "cod_menu": tema.cod_menu,
            "cod_modulo": tema.cod_modulo,
        }
    )
    return f"{_BASE}/TematicaAction.php?{q}"


def url_ato(ato: Ato) -> str:
    q = urllib.parse.urlencode(
        {
            "acao": "abrirAtoPublico",
            "sgl_tipo": ato.tipo,
            "num_ato": ato.num,
            "seq_ato": ato.seq,
            "vlr_ano": ato.ano,
            "sgl_orgao": ato.orgao,
        }
    )
    return f"{_BASE}/UrlPublicasAction.php?{q}"


def extrair_atos(html_tema: str, apenas_tipo: str = "RES") -> list[Ato]:
    """Extrai os atos (tuplas LinkTexto) de uma página temática."""
    atos = [Ato(*m) for m in _RE_LINK.findall(html_tema)]
    return [a for a in atos if a.tipo == apenas_tipo]


def listar_atos(temas=TEMAS_PADRAO) -> list[Ato]:
    """Enumera atos únicos das páginas temáticas (deduplicados por id)."""
    vistos: dict[str, Ato] = {}
    for tema in temas:
        for ato in extrair_atos(_get(url_tema(tema))):
            vistos.setdefault(ato.id, ato)
    return sorted(vistos.values(), key=lambda a: (a.ano, int(a.num)))


def limpar_html(raw: str) -> str:
    """Remove script/style/tags, desescapa entidades e normaliza espaços."""
    sem = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    texto = htmllib.unescape(re.sub(r"<[^>]+>", " ", sem))
    return re.sub(r"\s+", " ", texto).strip()


def extrair_titulo(texto: str) -> str:
    """Título próprio do ato — prefere o cabeçalho em maiúsculas; se não houver,
    aceita qualquer capitalização."""
    m = _RE_TITULO.search(texto) or _RE_TITULO_FALLBACK.search(texto)
    return m.group(1).strip() if m else ""


# Marcadores do início do CORPO do ato (após título/ementa/nota de revogação).
_MARCADORES_CORPO = (
    "A Diretoria Colegiada",
    "O Diretor-Geral",
    "O DIRETOR",
    "R E S O L V E",
    "RESOLVE:",
)


def esta_vigente(texto: str) -> bool:
    """Vigente se o CABEÇALHO (título + ementa, antes do corpo) não marca revogação
    passiva ('Revogada pela Resolução X'). Cortar no início do corpo evita tanto
    perder a marca (ementas longas) quanto falso-positivo de referências no corpo."""
    corte = min([texto.find(m) for m in _MARCADORES_CORPO if texto.find(m) != -1] or [2000])
    return not _RE_REVOGADA.search(texto[:corte])


# Mínimo de caracteres de uma norma real. A 'casca' do portal (tupla que não resolve) NÃO tem o
# cabeçalho oficial "RESOLUÇÃO Nº ..." — o `_RE_TITULO` já a filtra; o tamanho mínimo apenas evita
# stubs. 2.500 inclui resoluções curtas legítimas (ex.: as que só alteram outra), antes descartadas
# por um piso de 18k que subestimava o corpus pela metade.
MIN_CHARS_NORMA = 2_500


def texto_valido(texto: str) -> bool:
    """Distingue o corpo real da norma da 'casca' do portal: exige o cabeçalho oficial
    (RESOLUÇÃO Nº ...) + tamanho mínimo — sem descartar normas curtas mas válidas."""
    return bool(_RE_TITULO.search(texto)) and len(texto) > MIN_CHARS_NORMA


def baixar_ato(ato: Ato) -> dict | None:
    """Baixa e limpa o texto de um ato. Retorna dict com metadados+texto, ou None
    se o servidor devolveu a casca (tupla não resolveu)."""
    texto = limpar_html(_get(url_ato(ato)))
    if not texto_valido(texto):
        return None
    return {
        "id": ato.id,
        "tipo": ato.tipo,
        "numero": ato.numero_legivel,
        "ano": int(ato.ano),
        "orgao": ato.orgao,
        "titulo": extrair_titulo(texto),
        "vigente": esta_vigente(texto),
        "n_chars": len(texto),
        "url": url_ato(ato),
        "texto": texto,
    }
