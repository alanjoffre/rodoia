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
| Ingestão | ANTTlegis (45 resoluções, 3.432 chunks), chunking por artigo | 06 |
| Indexação | E5 local + Qdrant, filtro de vigência | 06 |
| Retrieval | híbrido (BM25+RRF) — hit@5 **0,64** [IC 0,50–0,76], MRR 0,51, n=50 (rerank não ajuda) | 07 |
| Governança | guardrail (direto+indireto+evasão) + PII + auditoria | 08 |
| Geração | citação; **juiz independente** (llama3.1, **n=12, sem IC**): faithfulness 0,73 / relevancy 1,0 / precisão de citação 0,92 · geração p50 ~21s | — |
| Interface | FastAPI async + UI de demo | 09 |

**Critérios de conclusão (todos ✓):** ingestão reproduzível · citação da fonte ·
métricas com baseline (retrieval + geração) · otimização com antes/depois (híbrido) ·
PII masking + guardrails adversariais · testes dos caminhos críticos (73) · demo acessível.
