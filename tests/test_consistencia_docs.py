"""Consistência entre a DOCUMENTAÇÃO e o repositório — no CI, não numa auditoria manual.

**Por que este arquivo existe.** Três vezes nesta fase a mesma falha apareceu: os
documentos de fase acompanhavam cada medição e os **transversais** (README, diário,
arquitetura) ficavam para trás. Não é descuido pontual — é estrutural, porque cada
medição nova toca meia dúzia de arquivos e é fácil parar em quatro. O badge do gate
chegou a anunciar 27 com 30 portões no código; o diário visual chegou a parar uma
FASE inteira atrás do markdown.

O gate de avaliação existe porque métrica regride em silêncio. Documentação
desatualiza em silêncio pelo mesmo motivo, e a resposta é a mesma: **um portão que
reprova**, em vez de alguém lembrar de conferir.

Estes testes NÃO julgam redação. Checam o que é mecanicamente verificável e que,
quando quebra, transforma o README numa afirmação falsa.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
README = (RAIZ / "README.md").read_text(encoding="utf-8")
DIARIO = (RAIZ / "docs/DIARIO.md").read_text(encoding="utf-8")
DIARIO_HTML = (RAIZ / "docs/diario.html").read_text(encoding="utf-8")

DOCS = [
    *sorted(RAIZ.glob("docs/*.md")),
    RAIZ / "README.md",
    RAIZ / "PROMPT_MESTRE.md",
    RAIZ / "data/README.md",
]


def test_links_relativos_resolvem() -> None:
    """Link quebrado num README público é erro visível para quem avalia o projeto."""
    quebrados = []
    for doc in DOCS:
        for rotulo, alvo in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if alvo.startswith(("http://", "https://", "#", "mailto:")):
                continue
            caminho = alvo.split("#")[0]
            if caminho and not (doc.parent / caminho).resolve().exists():
                quebrados.append(f"{doc.relative_to(RAIZ)} -> {alvo} ({rotulo[:30]})")
    assert not quebrados, "links quebrados:\n" + "\n".join(quebrados)


def test_badge_do_gate_bate_com_o_gate() -> None:
    """O badge anunciou 27/27 enquanto o gate tinha 30 portões. Um número no README
    que ninguém verifica é um número que eventualmente mente."""
    badge = re.search(r"gate%20de%20avalia\S*?-(\d+)%2F(\d+)", README)
    assert badge, "badge do gate não encontrado no README"
    metas = (RAIZ / "src/rodoia/mlops/gate.py").read_text(encoding="utf-8").count("    Meta(")
    assert int(badge.group(1)) == int(badge.group(2)) == metas


def test_badge_de_testes_bate_com_a_coleta() -> None:
    """Conta pela COLETA do pytest, não por `def test_` no arquivo: helpers e
    parametrização fazem a contagem estática divergir (295 estáticos x 280 coletados
    na escrita deste teste). `--collect-only` não executa nada — não há recursão."""
    badge = re.search(r"testes-(\d+)%20passando", README)
    assert badge, "badge de testes não encontrado no README"
    saida = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=RAIZ, timeout=300,
    ).stdout
    m = re.search(r"(\d+) tests? collected", saida)
    if not m:                       # ambiente sem coleta completa: não inventar veredito
        pytest.skip("pytest --collect-only não reportou total")
    assert int(badge.group(1)) == int(m.group(1)), (
        f"README anuncia {badge.group(1)} testes, pytest coleta {m.group(1)}"
    )


def _passos_markdown() -> list[int]:
    return [int(x) for x in re.findall(r"^(\d+)\. \*\*", DIARIO, re.M)]


def _passos_html() -> list[int]:
    return sorted(int(x) for x in re.findall(r"\{n:(\d+),", DIARIO_HTML))


def test_contagem_de_passos_bate_nos_quatro_lugares() -> None:
    """O número de passos vive em quatro lugares: badge do README, texto do DIARIO,
    itens do DIARIO e a timeline HTML. Já esteve com três valores diferentes ao
    mesmo tempo (81 / 92 / 98)."""
    valores = {
        "badge README": int(re.search(r"timeline_dos_(\d+)_passos", README).group(1)),
        "DIARIO declara": int(re.search(r"São \*\*(\d+) passos\*\*", DIARIO).group(1)),
        "DIARIO itens": len(_passos_markdown()),
        "html declara": int(re.search(r"RodoIA — (\d+) passos", DIARIO_HTML).group(1)),
        "html itens": len(_passos_html()),
    }
    assert len(set(valores.values())) == 1, f"contagens divergentes: {valores}"


def test_numeracao_dos_passos_sem_lacuna() -> None:
    md, html = _passos_markdown(), _passos_html()
    assert md == list(range(1, len(md) + 1)), "DIARIO.md com lacuna ou repetição"
    assert html == list(range(1, len(html) + 1)), "diario.html com lacuna ou repetição"


def test_diario_markdown_e_html_tem_os_mesmos_passos() -> None:
    """A timeline visual é gerada à mão a partir do markdown — já divergiu uma FASE
    inteira (o HTML parou no passo 81 com o markdown em 92)."""
    assert _passos_markdown() == _passos_html()


def test_referencias_de_secao_existem() -> None:
    """`docs/17 §13.8` só ajuda quem lê se a §13.8 existir."""
    faltando = []
    for doc in DOCS:
        txt = doc.read_text(encoding="utf-8")
        for numdoc, sec in re.findall(r"docs/(\d+)[_\w]*\.md\)?\s*§\s*([\d.]+)", txt):
            alvos = list(RAIZ.glob(f"docs/{numdoc}_*.md"))
            if not alvos:
                faltando.append(f"{doc.name}: docs/{numdoc} inexistente (§{sec})")
            elif not re.search(
                rf"^#+\s*{re.escape(sec)}[ .]", alvos[0].read_text(encoding="utf-8"), re.M
            ):
                faltando.append(f"{doc.name}: §{sec} ausente em {alvos[0].name}")
    assert not faltando, "referências quebradas:\n" + "\n".join(faltando)


def test_reports_citados_existem() -> None:
    """Evidência citada e ausente é pior que evidência não citada."""
    faltando = []
    for doc in DOCS:
        for rel in set(re.findall(r"`?(reports/[\w/]+\.json)`?", doc.read_text(encoding="utf-8"))):
            if not (RAIZ / rel).exists():
                faltando.append(f"{doc.name}: {rel}")
    assert not faltando, "reports citados e ausentes:\n" + "\n".join(faltando)


def test_contagem_de_momentos_de_rigor() -> None:
    """O README anuncia por extenso quantas vezes a evidência corrigiu a narrativa,
    e a lista logo abaixo é a fonte. Já ficou em 'Onze' com catorze itens."""
    numeros = {
        "Onze": 11, "Doze": 12, "Treze": 13, "Catorze": 14, "Quinze": 15,
        "Dezesseis": 16, "Dezessete": 17, "Dezoito": 18, "Dezenove": 19, "Vinte": 20,
    }
    m = re.search(r"(\w+) momentos em que a avaliação honesta mudou a conclusão", README)
    assert m, "frase dos momentos de rigor não encontrada"
    anunciado = numeros.get(m.group(1))
    assert anunciado, f"numeral por extenso não reconhecido: {m.group(1)!r}"
    bloco = re.search(r"(?s)momentos em que.*?\*\*Restrições assumidas", README).group(0)
    assert len(re.findall(r"(?m)^- \*\*", bloco)) == anunciado
