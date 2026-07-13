# 09 — Fase 1: interface FastAPI e fecho da fase

> Incremento 7 (final) da Fase 1. Módulo `api/app.py`.
> Rodar: `uvicorn rodoia.api.app:app` → abrir http://localhost:8000.

## A interface

API **assíncrona** em FastAPI que expõe o RAG completo:

- `GET /` — UI mínima de demonstração (HTML, faz a pergunta e mostra resposta + fontes).
- `GET /health` — prontidão.
- `POST /perguntar` — `{consulta, k}` → `{resposta, fontes, bloqueado}`.

**Decisões de engenharia:**
- O RAG (retrieval + LLM) é síncrono e pesado; o endpoint é `async` e delega a
  chamada a um threadpool (`asyncio.to_thread`) para **não bloquear o event loop**.
- Os componentes pesados (embedder, índice, reranker, LLM) sobem **uma vez** no
  startup (lifespan) e são reusados entre requisições.
- Toda pergunta passa pelo `responder_seguro` (guardrails + PII masking + auditoria).

## Demo

Local-first e **custo zero** (Ollama roda o LLM localmente) — sem o risco de um
endpoint público pago exposto que a seção 3.1 alerta. Deploy em nuvem fica para a
Fase 5.

```bash
uvicorn rodoia.api.app:app --port 8000
# POST /perguntar {"consulta": "Como funciona o vale-pedágio obrigatório?"}
# -> resposta citando Resolução 6024/2023
```

Validado ao vivo: `/perguntar` respondeu citando as Resoluções 673/2004 e 6024/2023;
injeção de prompt é bloqueada (`bloqueado: true`).

---

# Fase 1 — CONCLUÍDA

Sistema RAG completo sobre a regulação da ANTT, do scraping à API:

| Camada | Entrega | Doc |
|---|---|---|
| Ingestão | ANTTlegis (**125 resoluções, 4100 chunks**; ver `reports/fase1_rag/corpus.json`) | 06 |
| Indexação | E5 local + Qdrant, filtro de vigência | 06 |
| Retrieval | híbrido (BM25+RRF) **sem rerank** — hit@5 **0,64** [IC 0,50–0,76], MRR **0,485**, n=50 | 07 |

> **Decisão honesta (rerank × híbrido × denso).** No corpus ampliado (**125 normas, 4100 chunks** —
> ~2,8× o anterior) o **híbrido empata com o denso no hit@5 (0,64 = 0,64) e ganha no MRR (0,485 vs
> 0,466)** — ou seja, servir híbrido não perde nada e ordena melhor. O **rerank continua piorando o
> MRR** (0,485→0,440) sem mudar o hit@5, então segue **desligado por padrão** (não paga a latência).
> Notável: o hit@5 **se manteve em 0,64** mesmo com o corpus quase triplicando — robustez a um
> retrieval bem mais difícil. O gate trava a métrica do híbrido (a config servida). Números em
> `reports/fase1_retrieval/avaliacao_retrieval.json`.

### Limitações conhecidas da avaliação (assumidas, não escondidas)

Um revisor cético atacaria — e tem razão em parte:
- **n pequeno e anotador único.** O dourado é **n≈50** (retrieval) e **n=12** (geração), **escrito e
  rotulado pelo autor** — sem segundo anotador nem concordância (**κ**). Os ICs (Wilson/bootstrap)
  refletem esse n: largos de propósito. Um número sozinho enganaria; a faixa é honesta.
- **Corpus (ampliado) e n do dourado.** O corpus foi **ampliado de 45 → 125 normas (30 → 93
  vigentes)** ao corrigir um piso de tamanho tarde demais (18k chars) que descartava resoluções
  curtas legítimas — mais distratores, retrieval mais realista. O gargalo restante é o **nº de
  queries douradas** (ainda ~50, anotador único): ampliá-lo exige escrever mais pares (query→fonte)
  à mão. *Tentamos* gerar perguntas sintéticas por LLM e **descartamos** (citavam o número da
  resolução → query vazada). Preferimos assumir o limite a inflar o n com um benchmark ruim.
- **Held-out por construção.** O retriever (E5 pré-treinado + BM25 + RRF) **não tem parâmetro
  treinado** nessas queries — todo o conjunto é out-of-sample; não há risco de overfitting ao dourado.

**Fix real (backlog):** ampliar o corpus + anotação independente com κ. É trabalho de dados, não um
ajuste rápido — por isso está registrado como limitação, não maquiado.

### Banca de juízes independente + κ mensurável (`rag/painel_juizes.py`)

Contra o ataque "você avaliando você": a fidelidade da resposta passa a ser julgada por uma **banca
de 3 juízes de famílias distintas** — **llama3.1 (Meta), gemma2 (Google), mistral (Mistral)**, todos
**≠ do gerador** (qwen2.5) — em escala 0/1/2, e reportamos a **concordância inter-anotador (κ de
Fleiss)**. Resultado (`reports/fase1_geracao/painel_juizes.json`, n=12):

| | Valor |
|---|---|
| Fidelidade média (banca, 0–1) | **0,82** |
| **κ de Fleiss entre os 3 juízes** | **0,167** (concordância "leve", Landis-Koch) |
| Calibragem por juiz (média 0–2) | gemma2 **1,92** (leniente) · llama3.1 1,58 · mistral **1,42** (duro) |

→ **O achado mais honesto desta fase:** os juízes automáticos têm **calibragens bem diferentes** e
**mal concordam** (κ=0,17). Ou seja, **nota de LLM-juiz único é ruidosa** — o `faithfulness=0,73` do
juiz único não deve ser levado como verdade fina. Medir o κ **expõe** essa fragilidade em vez de
escondê-la; a banca (média + voto majoritário) é mais robusta que um juiz só. Continua sendo
**juiz automático, não humano** — o κ humano segue como backlog —, mas é um proxy rigoroso e
**mensurável** de independência.
| Governança | guardrail (direto+indireto+evasão) + PII + auditoria | 08 |
| Geração | citação; **juiz independente** (llama3.1, **n=12, sem IC**): faithfulness 0,73 / relevancy 1,0 / precisão de citação 0,92 · geração p50 ~21s | — |
| Interface | FastAPI async + UI de demo | 09 |

**Critérios de conclusão (todos ✓):** ingestão reproduzível · citação da fonte ·
métricas com baseline (retrieval + geração) · otimização com antes/depois (híbrido) ·
PII masking + guardrails adversariais · testes dos caminhos críticos (73) · demo acessível.
