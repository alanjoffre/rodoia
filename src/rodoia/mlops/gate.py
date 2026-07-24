"""Gate de avaliação — regressão de métrica FALHA o pipeline (Fase 5).

Lê os relatórios carimbados em `reports/` (produzidos pelas Fases 0–4) e compara cada
métrica-chave contra um **piso**. Se qualquer métrica cair abaixo do piso, o gate falha
(exit code 1) — é o portão de qualidade do CI. Não precisa de GPU/modelo: opera sobre os
JSONs já versionados, então roda no GitHub Actions em segundos.

**HONESTIDADE — o que este gate é e o que NÃO é.** Ele é um **guardrail de regressão de
ARTEFATO**: garante que um relatório commitado não seja substituído por outro pior sem alguém
notar. Ele **NÃO re-executa modelo nem regenera métrica** — confia no JSON versionado. A
*reprodução* de fato (regenerar a métrica a partir do modelo/dados e conferir contra o JSON) é
outra coisa, e roda no job `reproduzir` (ver `.github/workflows/reproduzir.yml`, runner
github-hosted em CPU, semanal), não neste gate barato de cada push.

Os pisos ficam um pouco ABAIXO dos valores atuais: toleram ruído de reexecução, mas pegam
qualquer regressão real. Atualizar um piso é uma decisão consciente (aparece no diff).

Uso:  python -m rodoia.mlops.gate         # imprime a tabela e sai 1 se algo regrediu
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rodoia.config import REPO_ROOT


@dataclass(frozen=True)
class Meta:
    nome: str            # rótulo legível
    relatorio: str       # caminho relativo ao repo
    caminho: str         # acesso pontilhado dentro do JSON (chaves e índices)
    op: str              # ">=", "<=" ou "=="
    piso: float | bool   # limiar (ou valor esperado, p/ "==")


# Pisos = valores atuais com folga p/ ruído; regressão real cai abaixo e falha o CI.
GATES: tuple[Meta, ...] = (
    Meta("F0 · MLP ROC-AUC", "reports/fase0_mlp/mlp.json", "roc_auc", ">=", 0.78),
    Meta("F1 · RAG hit@5 (híbrido)", "reports/fase1_retrieval/avaliacao_retrieval.json",
         "hibrido.hit_rate_at_k", ">=", 0.58),
    Meta("F1 · corpus (nº de normas)", "reports/fase1_rag/corpus.json", "n_normas", ">=", 100),
    Meta("F1 · κ humano (relevância)", "reports/fase1_rag/kappa_humano.json",
         "cohen_kappa", ">=", 0.6),
    Meta("F1 · κ humano (rótulo-gold fonte)", "reports/fase1_rag/kappa_gold_fonte.json",
         "cohen_kappa", ">=", 0.6),
    Meta("F1 · precisão de citação", "reports/fase1_geracao/avaliacao_geracao.json",
         "precisao_citacao_media", ">=", 0.85),
    # Segurança MEDIDA vira regressão-gated: a detecção da camada-1 do guardrail e o vazamento de
    # PII pós-masking (ver rag/redteam.py). Piso 0,95 (folga de 1 ataque sobre os 25); vazamento
    # com teto 0,0 — data-leakage não tem folga.
    Meta("F1 · red-team detecção (injeção)", "reports/fase1_seguranca/redteam.json",
         "guardrail.deteccao_guardrail.taxa", ">=", 0.95),
    Meta("F1 · red-team vazamento de PII", "reports/fase1_seguranca/redteam.json",
         "pii.vazamento_depois.taxa", "<=", 0.0),
    Meta("F2 · NER F1 (FT QLoRA)", "reports/fase2_ner/comparacao.json",
         "modelos.ft_qlora.f1_micro", ">=", 0.72),
    Meta("F2 · ganho FT vs base", "reports/fase2_ner/comparacao.json",
         "ganho_ft_vs_base", ">=", 0.55),
    # Piso com folga sob as 741.205 linhas atuais: a fonte da ANTT é revisada e pode encolher
    # de leve entre releases, mas uma queda abaixo de 700k denuncia ingestão truncada.
    Meta("F3 · linhas do fato (estrela)", "reports/fase3_dados/estrela.json",
         "n_linhas_fato", ">=", 700_000),
    Meta("F3 · Holt-Winters MAPE", "reports/fase3_dados/previsao.json",
         "modelos.holt_winters.mape_medio", "<=", 15.0),
    Meta("F3 · HW bate naïve (pareado)", "reports/fase3_dados/previsao.json",
         "comparacao_pareada.significativo", "==", True),
    Meta("F4 · roteamento (n=21, exato)", "reports/fase4_agente/roteamento.json",
         "resumo.acerto_roteamento", ">=", 0.85),
    Meta("F4 · juiz rota adequada", "reports/fase4_agente/avaliacao.json",
         "resumo.rota_ok_medio", ">=", 1.5),
    # Fase 6 — escala. O bulk da CFPB é republicado DIARIAMENTE e só cresce, então igualdade
    # contra literal quebraria amanhã: o piso fica com folga sob as 17.226.584 linhas do snapshot
    # de 2026-07-24 (sha256 carimbado no próprio report). Queda abaixo de 17M denuncia ingestão
    # truncada — o risco real ao ler um zip de 1,43 GB por streaming.
    Meta("F6 · linhas do CFPB (bulk)", "reports/fase6_escala/contagem_cfpb.json",
         "linhas_total", ">=", 17_000_000),
    # Narrativa é o insumo do corpus de texto (só 22,21% das linhas têm). Se a CFPB mudar a
    # política de publicação por consentimento, a queda aparece aqui ANTES de contaminar
    # qualquer métrica de recuperação. Piso sob as 3.825.572 medidas.
    Meta("F6 · narrativas do CFPB", "reports/fase6_escala/contagem_cfpb.json",
         "com_narrativa", ">=", 3_700_000),
    # CUAD — benchmark externo. Aqui igualdade É correta, ao contrário do CFPB acima: é um
    # dataset acadêmico CONGELADO, não uma série viva. Se 510 virar outro número, o espelho do
    # Kaggle mudou sob nossos pés e toda métrica de recuperação comparada a SOTA fica inválida.
    Meta("F6 · CUAD contratos", "reports/fase6_cuad/integridade.json",
         "n_contratos", "==", 510),
    # O portão mais importante da Fase 6: cada span de ouro precisa casar com o texto no offset
    # declarado. Gold desalinhado não quebra nada visivelmente — produz métrica plausível e
    # FALSA. Teto 0, sem folga, pelo mesmo motivo do vazamento de PII acima.
    Meta("F6 · CUAD offsets divergentes", "reports/fase6_cuad/integridade.json",
         "spans_divergentes", "<=", 0),
    # Baseline de recuperação (BM25, sem LLM) sobre o benchmark externo. Piso com folga sob os
    # 0,588 medidos: é o número que qualquer melhoria de arquitetura (denso, híbrido, rerank,
    # chunking por cláusula) precisa bater — e que uma regressão silenciosa derrubaria.
    Meta("F6 · CUAD recall@5 (BM25)", "reports/fase6_cuad/retrieval_bm25.json",
         "metricas.recall_at_5.media", ">=", 0.55),
    # Invariante do mapeamento span->chunk: toda pergunta respondível precisa ter ao menos um
    # chunk de gold. Se um refactor de chunking quebrar o alinhamento, o Recall cairia de forma
    # plausível em vez de falhar — este portão faz falhar. Teto 0, sem folga.
    Meta("F6 · CUAD perguntas sem gold", "reports/fase6_cuad/retrieval_bm25.json",
         "n_sem_gold", "<=", 0),
)


def _acessar(obj: Any, caminho: str) -> Any:
    """Navega um dict/list por caminho pontilhado ('a.b.0' → obj['a']['b'][0])."""
    atual = obj
    for parte in caminho.split("."):
        atual = atual[int(parte)] if parte.isdigit() else atual[parte]
    return atual


def _passou(valor: Any, op: str, piso: float | bool) -> bool:
    if op == ">=":
        return bool(valor >= piso)
    if op == "<=":
        return bool(valor <= piso)
    if op == "==":
        return bool(valor == piso)
    raise ValueError(f"operador inválido: {op!r}")


def avaliar(raiz: Path | None = None) -> tuple[bool, list[dict[str, Any]]]:
    """Retorna (tudo_ok, linhas). Uma métrica ausente/relatório faltando conta como FALHA."""
    raiz = raiz or REPO_ROOT
    linhas = []
    for g in GATES:
        p = raiz / g.relatorio
        try:
            valor = _acessar(json.loads(p.read_text(encoding="utf-8")), g.caminho)
            ok = _passou(valor, g.op, g.piso)
            erro = None
        except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
            valor, ok, erro = None, False, f"{type(e).__name__}: {e}"
        linhas.append({"nome": g.nome, "valor": valor, "op": g.op, "piso": g.piso,
                       "ok": ok, "erro": erro})
    return all(x["ok"] for x in linhas), linhas


def imprimir(linhas: list[dict[str, Any]]) -> None:
    for x in linhas:
        marca = "✓" if x["ok"] else "✗"
        alvo = "" if x["erro"] else f"{x['valor']} {x['op']} {x['piso']}"
        print(f"  [{marca}] {x['nome']:34} {alvo}{x['erro'] or ''}")


def main() -> int:
    tudo_ok, linhas = avaliar()
    print("Gate de avaliação (regressão de métrica falha o CI):")
    imprimir(linhas)
    n_ok = sum(x["ok"] for x in linhas)
    print(f"\n{n_ok}/{len(linhas)} portões OK — {'APROVADO' if tudo_ok else 'REPROVADO'}")
    return 0 if tudo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
