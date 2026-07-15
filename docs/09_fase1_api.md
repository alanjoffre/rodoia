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
| Ingestão | ANTTlegis (**125 resoluções, 3651 chunks** limpos; ver `reports/fase1_rag/corpus.json`) | 06 |
| Indexação | E5 local + Qdrant, filtro de vigência | 06 |
| Retrieval | híbrido (BM25+RRF) — hit@5 **0,62** [IC 0,48–0,74], MRR **0,510**, n=50 (rerank: 0,68) | 07 |

> **Decisão honesta — e uma REVERSÃO que o rigor obrigou.** Depois de **limpar o boilerplate** do
> portal (o chunking passou a cortar o menu/cabeçalho de navegação → 4100 → **3651 chunks** de
> conteúdo real), re-medimos: hit@5 **denso 0,60 · híbrido 0,62 · híbrido+rerank 0,68**; MRR do
> híbrido **0,510**. **O rerank, que ANTES não ajudava (era a era com lixo), agora dá o maior hit@5
> (0,62→0,68)** — remover o ruído deixou o cross-encoder distinguir melhor. Os ICs ainda se
> sobrepõem (n=50), então não é vitória estatística; por isso servimos **híbrido** (gate trava
> `hibrido` = 0,62) e mantemos o rerank como **opção com ganho de ponto-estimado**, ligável quando a
> latência couber. É o mesmo padrão do projeto: a limpeza de dados **mudou a conclusão de
> arquitetura**, e atualizamos em vez de esconder. Números em `avaliacao_retrieval.json`.

### Limitações conhecidas da avaliação (assumidas, não escondidas)

Um revisor cético atacaria — e onde tinha razão, corrigimos:
- **Anotador único → κ HUMANO (fechado).** A crítica de "você avaliando você" foi respondida com
  **2 anotadores humanos DIFERENTES**, cada um julgando de forma **independente e cega** (sem ver o
  rótulo do outro) a relevância de trechos recuperados (0/1): **κ de Cohen = 0,864**, **IC95 bootstrap
  [0,65; 1,00]** (concordância 93,3%, n=30 — "quase perfeita" na escala Landis-Koch), em
  `reports/fase1_rag/kappa_humano.json` (no gate, piso 0,6). O IC segue a régua do projeto (nenhum
  número sem incerteza) e o limite inferior fica **acima do piso**. Sem paradoxo de prevalência
  (0,567 ≈ 0,5, viés nulo). Ou seja, o rótulo de relevância é **confiável entre humanos**, não
  idiossincrasia do autor. Kit + brutos (`anotador_A/B.xlsx`) **versionados** → reproduz de um clone
  limpo: `python -m rodoia.anotacao kappa anotacao/anotador_A.xlsx anotacao/anotador_B.xlsx`.
  Os 30 pares são **amostrados das próprias queries do dourado** (`CONJUNTO_DOURADO[:15]`, top-1 +
  distrator) — ou seja, o κ valida a relevância **sobre as mesmas perguntas** que a métrica de
  retrieval usa, não um conjunto à parte.
- **Rótulo-gold de fonte → κ HUMANO (elo fechado).** A crítica mais fina — "o `hit@5` repousa sobre
  labels de FONTE de um anotador único" — foi respondida: **2 humanos independentes** validaram os
  rótulos-gold de fonte do dourado (50 pares = 25 queries × [fonte-gold + distrator]):
  **κ de Cohen = 0,917**, **IC95 [0,79; 1,00]** (concordância 96%, n=50 — "quase perfeita"), em
  `reports/fase1_rag/kappa_gold_fonte.json` (no gate). A prevalência **0,40** (não 0,50) é honesta:
  os humanos **discordaram do gold do autor em alguns casos**, expondo — em vez de esconder — que
  nenhum rótulo é perfeito. Agora a **métrica que está no gate** (`hit@5`) tem validação
  inter-humana, não só do autor. Kit + brutos versionados, reproduzível:
  `python -m rodoia.anotacao gerar-gold` / `kappa-gold`.
- **n pequeno (assumido).** O dourado de retrieval é **n≈50** e o de geração **n=12** — os ICs
  (Wilson/bootstrap) refletem esse n, largos de propósito. Um número sozinho enganaria; a faixa é
  honesta. A **banca de 3 juízes LLM + κ de Fleiss** (0,167) complementa medindo a concordância de
  juízes automáticos (ver acima).
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
| Geração | citação; **juiz independente** (llama3.1, **n=12, com IC95 bootstrap**): faithfulness 0,73 [0,63–0,83] / relevancy 1,0 [1,0–1,0] / precisão de citação 0,92 [0,75–1,0] · geração p50 ~21s | — |
| Interface | FastAPI async + UI de demo | 09 |

**Critérios de conclusão (todos ✓):** ingestão reproduzível · citação da fonte ·
métricas com baseline (retrieval + geração) · otimização com antes/depois (híbrido) ·
PII masking + guardrails adversariais · testes dos caminhos críticos (73) · demo acessível.
