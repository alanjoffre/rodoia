"""Parser e normalização do **CUAD** — o benchmark externo de recuperação (Fase 6).

Lê `CUAD_v1.json` de dentro do zip (formato **SQuAD 2.0**) e normaliza para dois
artefatos JSONL consumíveis pela avaliação, mais um relatório de integridade.

**Por que `is_impossible` é o ativo, não o defeito.** Em SQuAD 2.0 esse campo
marca perguntas sem resposta no contexto. No CUAD elas são **67,9% do total** —
e isso é proposital: um contrato que não tem cláusula de exclusividade *deve*
produzir "não consta". Elas são o conjunto que mede **abstenção**, a métrica que
quase nenhum portfólio mostra. Descartá-las (o reflexo comum ao ver "impossible")
jogaria fora dois terços do benchmark e justamente a parte difícil.

**Char offsets são preservados.** `answer_start` é um deslocamento de caractere
dentro do `context` do contrato inteiro. A avaliação de recuperação precisa
mapear span -> chunk, e esse mapeamento só é possível com o offset original;
guardar apenas o texto do span obrigaria a re-encontrá-lo por busca de string,
que é ambíguo quando o mesmo trecho aparece mais de uma vez.

**A categoria da cláusula vive dentro do enunciado.** As perguntas têm a forma
`Highlight the parts (if any) of this contract related to "Document Name" that
should be reviewed by a lawyer. Details: ...` — o rótulo entre aspas é uma das
41 categorias. Extraído para `categoria`, que permite quebrar as métricas por
tipo de cláusula em vez de reportar só uma média que esconde tudo.

Uso:
    python -m rodoia.rag.cuad          # normaliza + grava relatório de integridade
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rodoia.config import settings
from rodoia.proveniencia import carimbar

ARQUIVO_INTERNO = "CUAD_v1.json"

# `... related to "Exclusivity" that should be reviewed ...` -> Exclusivity
_RE_CATEGORIA = re.compile(r'related to "([^"]+)"')


@dataclass(frozen=True)
class Span:
    """Um trecho de resposta, com offset de caractere no texto do contrato."""

    texto: str
    inicio: int

    @property
    def fim(self) -> int:
        return self.inicio + len(self.texto)


@dataclass(frozen=True)
class Pergunta:
    id: str
    contrato: str
    enunciado: str
    categoria: str
    impossivel: bool
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class Contrato:
    titulo: str
    texto: str
    perguntas: tuple[Pergunta, ...]


def _extrair_categoria(enunciado: str) -> str:
    """Categoria da cláusula a partir do enunciado; "desconhecida" se não casar.

    Não levanta: um enunciado fora do padrão não pode derrubar o parse de 20 mil
    perguntas — ele aparece no relatório como categoria "desconhecida", que é
    auditável, em vez de virar exceção silenciada.
    """
    m = _RE_CATEGORIA.search(enunciado)
    return m.group(1) if m else "desconhecida"


def carregar(zip_path: Path | None = None) -> list[Contrato]:
    """Lê o `CUAD_v1.json` de dentro do zip e devolve os contratos normalizados."""
    zip_path = zip_path or (settings.data_raw / "cuad" / "cuad.zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} ausente — rode baixar_cuad primeiro.")

    with zipfile.ZipFile(zip_path) as zf:
        nomes = [n for n in zf.namelist() if n.endswith(ARQUIVO_INTERNO)]
        if not nomes:
            raise ValueError(
                f"{ARQUIVO_INTERNO} não encontrado no zip ({len(zf.namelist())} itens)"
            )
        bruto: dict[str, Any] = json.loads(zf.read(nomes[0]).decode("utf-8"))

    contratos: list[Contrato] = []
    for doc in bruto["data"]:
        titulo = doc["title"]
        # No CUAD cada contrato tem exatamente 1 parágrafo (o texto inteiro).
        # Concatenar mais de um invalidaria os offsets, então é erro se houver.
        paragrafos = doc["paragraphs"]
        if len(paragrafos) != 1:
            raise ValueError(
                f"contrato {titulo!r} tem {len(paragrafos)} parágrafos (esperado 1) — "
                "os offsets de resposta seriam ambíguos"
            )
        contexto = paragrafos[0]["context"]
        perguntas = tuple(
            Pergunta(
                id=qa["id"],
                contrato=titulo,
                enunciado=qa["question"],
                categoria=_extrair_categoria(qa["question"]),
                impossivel=bool(qa.get("is_impossible")),
                spans=tuple(
                    Span(texto=a["text"], inicio=int(a["answer_start"]))
                    for a in (qa.get("answers") or [])
                ),
            )
            for qa in paragrafos[0]["qas"]
        )
        contratos.append(Contrato(titulo=titulo, texto=contexto, perguntas=perguntas))
    return contratos


def validar_offsets(contratos: list[Contrato]) -> dict[str, int]:
    """Confere que cada span REALMENTE casa com o texto no offset declarado.

    Um benchmark é inútil se o gold aponta para o lugar errado — e isso não se
    descobre olhando o JSON, só conferindo. Devolve as contagens de conferidos
    e divergentes.
    """
    ok = divergentes = 0
    for c in contratos:
        for p in c.perguntas:
            for s in p.spans:
                if c.texto[s.inicio : s.fim] == s.texto:
                    ok += 1
                else:
                    divergentes += 1
    return {"spans_conferidos": ok, "spans_divergentes": divergentes}


def estatisticas(contratos: list[Contrato]) -> dict[str, Any]:
    """Números de integridade do benchmark (insumo do portão)."""
    perguntas = [p for c in contratos for p in c.perguntas]
    impossiveis = [p for p in perguntas if p.impossivel]
    spans = [s for p in perguntas for s in p.spans]
    categorias: dict[str, int] = {}
    for p in perguntas:
        categorias[p.categoria] = categorias.get(p.categoria, 0) + 1

    tamanhos = sorted(len(c.texto) for c in contratos)
    return {
        "n_contratos": len(contratos),
        "n_perguntas": len(perguntas),
        "n_impossiveis": len(impossiveis),
        "pct_impossiveis": round(len(impossiveis) / len(perguntas) * 100, 2) if perguntas else 0.0,
        "n_spans": len(spans),
        "n_categorias": len(categorias),
        "caracteres_contratos": sum(tamanhos),
        "contrato_chars": {
            "min": tamanhos[0] if tamanhos else 0,
            "mediana": tamanhos[len(tamanhos) // 2] if tamanhos else 0,
            "max": tamanhos[-1] if tamanhos else 0,
        },
        **validar_offsets(contratos),
        "categorias": dict(sorted(categorias.items(), key=lambda kv: -kv[1])),
    }


def gravar_jsonl(contratos: list[Contrato], saida: Path) -> tuple[Path, Path]:
    """Grava `contratos.jsonl` e `perguntas.jsonl` normalizados."""
    saida.mkdir(parents=True, exist_ok=True)
    p_contratos = saida / "contratos.jsonl"
    p_perguntas = saida / "perguntas.jsonl"

    with p_contratos.open("w", encoding="utf-8") as fh:
        for c in contratos:
            fh.write(
                json.dumps({"titulo": c.titulo, "texto": c.texto}, ensure_ascii=False) + "\n"
            )
    with p_perguntas.open("w", encoding="utf-8") as fh:
        for c in contratos:
            for p in c.perguntas:
                fh.write(
                    json.dumps(
                        {
                            "id": p.id,
                            "contrato": p.contrato,
                            "enunciado": p.enunciado,
                            "categoria": p.categoria,
                            "impossivel": p.impossivel,
                            "spans": [
                                {"texto": s.texto, "inicio": s.inicio, "fim": s.fim}
                                for s in p.spans
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return p_contratos, p_perguntas


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliza o CUAD e afere sua integridade.")
    parser.add_argument("--zip", type=Path, default=None, help="caminho do cuad.zip")
    parser.add_argument("--saida", type=Path, default=None, help="diretório dos JSONL")
    args = parser.parse_args()

    contratos = carregar(zip_path=args.zip)
    estat = estatisticas(contratos)
    saida = args.saida or (settings.data_processed / "cuad")
    p_contratos, p_perguntas = gravar_jsonl(contratos, saida)

    destino_report = settings.data_processed.parent.parent / "reports" / "fase6_cuad"
    destino_report.mkdir(parents=True, exist_ok=True)
    caminho = destino_report / "integridade.json"
    caminho.write_text(json.dumps(carimbar(estat), ensure_ascii=False, indent=2))

    print(f"contratos: {estat['n_contratos']:,}")
    print(f"perguntas: {estat['n_perguntas']:,}")
    print(f"  impossíveis: {estat['n_impossiveis']:,} ({estat['pct_impossiveis']}%)")
    print(f"  spans de resposta: {estat['n_spans']:,}")
    print(f"categorias de cláusula: {estat['n_categorias']}")
    print(
        f"offsets: {estat['spans_conferidos']:,} conferem, "
        f"{estat['spans_divergentes']:,} divergem"
    )
    print(f"jsonl: {p_contratos} | {p_perguntas}")
    print(f"report: {caminho}")


if __name__ == "__main__":
    main()
