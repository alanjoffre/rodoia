<div align="center">

# 🛣️ RodoIA

**Engenharia de IA de ponta a ponta sobre a regulação e os dados abertos do transporte rodoviário brasileiro (ANTT)** — do *backpropagation* escrito à mão ao *serving* em produção com avaliação como portão de CI.

[![CI](https://github.com/alanjoffre/rodoia/actions/workflows/ci.yml/badge.svg)](https://github.com/alanjoffre/rodoia/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Tests](https://img.shields.io/badge/testes-155%20passando-brightgreen.svg)
![Gate](https://img.shields.io/badge/gate%20de%20avaliação-12%2F12-brightgreen.svg)
[![Demo](https://img.shields.io/badge/🔗_demo_ao_vivo-HF_Spaces-blue.svg)](https://huggingface.co/spaces/alanjoffre/rodoia-rag)

[**🔗 Demo ao vivo**](https://huggingface.co/spaces/alanjoffre/rodoia-rag) · [**📖 A história**](docs/HISTORIA.md) · [**🗺️ Arquitetura**](docs/ARQUITETURA.md) · [**🎓 Guia didático**](docs/GUIA_ENGENHARIA_IA.md) · [**📋 Plano mestre**](PROMPT_MESTRE.md)

</div>

---

> **🔗 Demo ao vivo (grátis):** busca semântica sobre a regulação da ANTT rodando **no seu navegador** (embeddings E5 via Transformers.js, sem servidor) — **[huggingface.co/spaces/alanjoffre/rodoia-rag](https://huggingface.co/spaces/alanjoffre/rodoia-rag)**.

Projeto de portfólio **público e open-source**. Objetivo: provar, **com código, métrica e deploy**, o perfil completo de um Engenheiro de IA — a trilha moderna (LLM · RAG · agentes · MLOps) **e** o núcleo clássico (ML/DL do zero · fine-tuning · serving de modelo próprio). A maioria dos portfólios para em *"chamei a API da OpenAI e funcionou"*. Este vai do **fundamento matemático** ao **ciclo de produção**.

## 📊 Resultados de relance

Cada fase é um marco publicável, testado e documentado antes da próxima começar.

| Fase | Entrega | Métrica-chave (com evidência versionada) |
|:---:|---|---|
| **0 · Fundamentos** | backprop + self-attention à mão (NumPy/PyTorch puro); MLP de severidade de acidentes | **ROC-AUC 0,81** |
| **1 · RAG** | busca híbrida (BM25+E5+RRF) sobre 125 normas / 3.651 chunks + guardrails | **hit@5 0,62** [0,48–0,74] · citação **0,92** · **κ humano 0,86 / 0,92** |
| **2 · Fine-tuning** | QLoRA (Qwen2.5-3B) p/ NER jurídico + serving vLLM fp8 | **F1 0,13 → 0,77** (SOTA 0,89) · **205 tok/s** |
| **3 · Dados** | esquema estrela DuckDB, 741k linhas, previsão de demanda | **Holt-Winters bate o naïve** Δ3,01pp (IC [1,76; 4,40]) |
| **4 · Agente** | grafo LangGraph com arestas condicionais reais (RAG+FT+dados) | **roteamento 0,95** (n=21, objetivo) |
| **5 · MLOps** | gate de avaliação no CI · MLflow · DVC · drift · custo | **gate 12/12** · **drift PSI 0,005** (estável) |

> **O diferencial não são os números altos — é o rigor ter corrigido os próprios números.** Uma auditoria κ inter-anotador **encontrou 16% dos rótulos-gold do hit@5 errados** e nós reportamos o impacto em vez de esconder. Ver [Decisões e trade-offs](#-decisões-e-trade-offs-o-arco-do-projeto).

## 🧭 Os cinco eixos

| Eixo | O que prova |
|---|---|
| **Fundamentos de ML/DL** | Modelo treinado do zero + atenção/backprop implementados à mão |
| **RAG avaliado** | Recuperação sobre a legislação da ANTT com medição de qualidade (LLM-juiz independente + IC + κ humano) |
| **Fine-tuning & serving** | LLM aberto adaptado (QLoRA) para NER jurídico — F1 0,13→0,77 vs. SOTA — quantizado (fp8) e servido em vLLM |
| **Agente orquestrado** | Raciocínio multi-etapa (LangGraph) combinando RAG + modelo próprio + dados estruturados |
| **MLOps & Cloud** | Versionamento, CI/CD com gate, observabilidade, custo, avaliação contínua e runbook de deploy |

## 🏗️ Arquitetura (visão de destino)

```
                         ┌──────────────────────────────┐
   Pergunta do usuário → │   Agente (LangGraph, Fase 4)  │
                         └──────┬───────────┬───────┬────┘
                                │           │       │
                 ┌──────────────▼──┐  ┌─────▼────┐  ▼ cálculo/agregação
                 │ RAG regulatório │  │  Modelo  │
                 │  (Fase 1)       │  │ fine-tun.│
                 │ embeddings +    │  │ (Fase 2, │
                 │ híbrido (RRF)   │  │  vLLM)   │
                 └────────┬────────┘  └──────────┘
                          │                 ▲
              ┌───────────▼──────┐   ┌───────┴────────┐
              │ Dados estrutur.  │   │ Fundamentos    │
              │ ANTT (Fase 3,    │   │ ML/DL (Fase 0) │
              │ DuckDB)          │   └────────────────┘
              └──────────────────┘
   MLOps transversal (Fase 5): DVC · MLflow · CI/CD · gate · observabilidade · drift
```

Mapa **módulo a módulo** de todo o código em **[docs/ARQUITETURA.md](docs/ARQUITETURA.md)**.

## 🗺️ Roadmap e status — todas as fases ✅

| Fase | Tema | Destaque |
|:---:|---|---|
| **0** | Fundamentos de ML/DL + higiene de repo | backprop/attention à mão com prova de equivalência ao autograd; MLP severidade **ROC-AUC 0,81** ([docs 00–05](docs/)) |
| **1** | RAG avaliado sobre a regulação da ANTT | híbrido BM25+RRF, 125 normas / 3.651 chunks; **hit@5 0,62** [0,48–0,74], citação **0,92**; κ humano **0,86/0,92**; guardrails ([docs 06–09](docs/09_fase1_api.md)) |
| **2** | Fine-tuning e serving de modelo próprio | QLoRA NER jurídico (LeNER-Br) **F1 0,13→0,77**, encostando no SOTA BERTimbau 0,89; vLLM fp8 205 tok/s. Precedido de um *estudo-baseline* honesto (FT **não** injeta conhecimento) → o arco negativo→pivô é a entrega ([docs/13](docs/13_fase2_ner.md)) |
| **3** | Dados estruturados abertos da ANTT | Volume de Pedágio (2010–2026, **741k linhas**), esquema estrela DuckDB + SQL analítico; **Holt-Winters bate o naïve** (pareado Δ3,01pp, IC [1,76; 4,40], vence em 73% das praças) ([docs/14](docs/14_fase3_dados_estruturados.md)) |
| **4** | Agente de orquestração (LangGraph) | grafo com **arestas condicionais reais** (guardrail + roteador) combinando RAG+FT+dados; **roteamento 0,95** (n=21); degradação graciosa testada ([docs/15](docs/15_fase4_agente.md)) |
| **5** | MLOps, Cloud e operação | **gate de avaliação** (regressão reprova o CI, 12/12) · GitHub Actions · MLflow + DVC · container · **drift PSI 0,005** · **custo R$/1k das 2 rotas** · **demo pública no ar** · deploy cloud = runbook ([docs/16](docs/16_fase5_mlops.md)) |

## ✅ Rastreabilidade requisito → fase

<details>
<summary><b>Cada requisito de uma vaga de Engenheiro de IA rastreado até a fase que o prova com código e evidência (clique para expandir)</b></summary>

<br>

| Requisito | Onde é provado | Evidência |
|---|---|---|
| Python sólido (async, tipagem, produção) | Todas | Código tipado, testado, `async` nos endpoints |
| Estruturas de dados, algoritmos, complexidade | Fase 0 + 1 | Análise de complexidade em decisões de retrieval |
| Matemática aplicada (álgebra, cálculo, prob./estat.) | Fase 0 | Derivações + gradiente/atenção manuais |
| SQL avançado e modelagem | Fase 3 | Esquema **estrela** (DuckDB, 741k linhas), window functions (LAG/RANK), camada de acesso testada + **previsão de demanda** (MAPE) |
| ML clássico (regressão, árvores, ensembles, clustering) | Fase 0 | Modelos treinados sobre dado tabular ANTT, com métricas |
| Redes neurais, backprop, arquitetura Transformer | Fase 0 | MLP em PyTorch puro + self-attention/backprop à mão |
| PyTorch | Fase 0 + 2 | Treino (F0) + fine-tuning QLoRA (F2) |
| Métricas e diagnóstico (overfitting, bias/variance) | Fase 0 | Curvas treino/validação documentadas |
| Engenharia de prompts avançada | Fase 1 + 4 | Prompts versionados, testados, com ablação |
| RAG, embeddings, banco vetorial (Qdrant) | Fase 1 | Híbrido (BM25+RRF) sobre **125 normas / 3.651 chunks** · avaliado com IC (hit@5 **0,62** [0,48–0,74], n=50; rerank **passou a ajudar** no corpus limpo, 0,68) · citação 0,92 · **κ humano** nos rótulos |
| Orquestração de agentes (LangGraph) | Fase 4 | Grafo (estado + arestas condicionais) combinando as 3 ferramentas · roteamento **0,95** (n=21) · juiz independente ([docs/15](docs/15_fase4_agente.md)) |
| Fine-tuning, LoRA/QLoRA, quantização | Fase 2 | QLoRA (Qwen2.5-3B) p/ **NER jurídico**: F1 **0,13→0,77** vs. SOTA 0,89 · **fp8** no vLLM (205 tok/s) · estudo-baseline em [docs/11](docs/11_fase2_resultados.md) |
| Avaliação de LLMs (LLM-as-judge, guardrails, hallucination) | Fase 1 + 2 + 4 | LLM-as-judge **independente** + faithfulness/relevancy (com IC) + precisão de citação · **κ humano inter-anotador 0,86** (relevância) + **0,92** (rótulo-gold do hit@5) — 2 anotadores independentes + **banca de 3 juízes LLM (κ de Fleiss)** · guardrails |
| Deploy/serving (FastAPI, vLLM, containers) | Fase 2 + 5 | vLLM + FastAPI async + container + demo client-side no ar |
| CI/CD para ML, versionamento (MLflow/DVC) | Fase 5 | GitHub Actions (lint+testes+**gate de regressão**) · MLflow (sqlite) · DVC ([docs/16](docs/16_fase5_mlops.md)) |
| Monitoramento, observabilidade, drift | Fase 1 + 5 | Latência/tokens medidos no RAG · **drift por PSI** (coorte, 0,005 estável) |
| Cloud (AWS/Azure/GCP) | Fase 5 | **Runbook de deploy** (Cloud Run justificado) + **modelo de custo R$/1k** das 2 rotas |
| Custo, latência, escalabilidade | Fase 5 | Custo R$/1k da vazão medida + trade-off scale-to-zero ([docs/16](docs/16_fase5_mlops.md) §6.1) |
| LGPD/GDPR, PII masking, auditoria | Fase 1 + 5 | Masking + trilha de auditoria |
| Segurança de IA (prompt injection, data leakage) | Fase 1 + 4 | Guardrails testados com casos adversariais |

</details>

## 🔬 Decisões e trade-offs (o arco do projeto)

O diferencial não são os números altos — é **o rigor ter corrigido os próprios números**. Cinco momentos em que a avaliação honesta mudou a conclusão:

- **Fase 1 — a auditoria que achou defeito.** Um κ humano inter-anotador sobre os rótulos-gold do `hit@5` **rejeitou 16% deles** (resoluções mal atribuídas por resíduo de numeração antiga). Em vez de esconder, **rerotulamos pela fonte correta** e reportamos a faixa real (**[0,70; 0,76]**) ao lado do número do gate (0,62, mantido conservador). Auditar a própria métrica é a disciplina.
- **Fase 2 — o pivô do fine-tuning.** Um *estudo-baseline* mostrou, com held-out, que o QLoRA **não injeta conhecimento factual** (in-sample melhorava, held-out piorava = memorização). Viramos para uma tarefa **objetiva** — NER jurídico — onde o FT vence com métrica dura (**F1 0,13→0,77**). O arco negativo→pivô é a entrega.
- **Fase 3 — a cereja e a inconsistência.** Um "MAPE 5,9%" **cereja** virou ~13% no backtest de 63 praças + IC. E um **erro metodológico meu** (naïve de 1-passo × Holt-Winters de 12-passos) foi corrigido para **multi-step justo** — aí o Holt-Winters **bate o naïve com significância** (Δ3,01pp, IC [1,76; 4,40]).
- **Fase 4 — o artefato do juiz.** O juiz penalizava "não rotear" nos casos fora-de-escopo/adversarial (onde declinar é o certo). Separar in-scope de declinados tirou o artefato: **roteamento 0,95** e juiz **rota 2,0/2**.
- **Fase 5 — o drift enganoso.** PSI sobre o volume **agregado** dava ~11 (a malha cresceu ~10×); trocar para a **coorte comum de praças** revelou o valor real — **0,005, estável**.

**Restrições assumidas conscientemente:** hardware de 6 GB (modelos 3B, time-slicing cérebro↔FT); alguns *n* pequenos (hit@5 n=50, juiz de geração n=12) — os ICs expõem isso; e **deploy em cloud não executado** (runbook pronto, [docs/16](docs/16_fase5_mlops.md) §7) por decisão de orçamento.

## 🚀 Como rodar

> Passo a passo completo (do clone à Fase 0 rodando): **[docs/RUNBOOK.md](docs/RUNBOOK.md)**.

```bash
git clone https://github.com/alanjoffre/rodoia && cd rodoia
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # núcleo + ferramentas de dev
pre-commit install               # ativa a barreira anti-segredo
cp .env.example .env             # preencha suas chaves (NUNCA comite o .env)
pytest                           # deve passar (teste-fumaça)
```

Dependências pesadas entram **por fase**, como extras: `".[fundamentos]"` (F0), `".[rag]"` (F1), `".[estruturados]"` (F3), `".[agente]"` (F4), `".[mlops]"` (F5) — quem só quer ler o RAG não precisa de PyTorch/vLLM.

## 🔒 Higiene de repositório público

Repo público desde o commit 1 (histórico Git imutável). Garantias em vigor:

- **Sem segredos** — `.gitignore` correto, `.env.example` versionado, `detect-secrets` no pre-commit bloqueando chaves antes do commit.
- **Sem dado bruto no Git** — reprodução canônica pelos **scripts de download** (fontes 100% públicas: `data/baixar_*`, `rag/baixar_normas`); apontadores DVC guardam o hash de conteúdo. Licença de cada fonte confirmada ([data/README.md](data/README.md)).
- **Fronteira de dados** — exclusivamente domínio público da ANTT. Zero dado/regra de qualquer empregador ou cliente ([`NOTICE`](NOTICE)).
- **Demo sem sangrar custo** — nada de API paga exposta; demo por retrieval client-side (R$0).

## 📚 Documentação

| Documento | Para quê |
|---|---|
| [**PROMPT_MESTRE.md**](PROMPT_MESTRE.md) | Plano completo, critérios de conclusão e regras de condução |
| [**docs/HISTORIA.md**](docs/HISTORIA.md) | Narrativa *problema → como resolvemos → resultado* de cada fase |
| [**docs/ARQUITETURA.md**](docs/ARQUITETURA.md) | Mapa módulo a módulo de `src/rodoia/**` |
| [**docs/GUIA_ENGENHARIA_IA.md**](docs/GUIA_ENGENHARIA_IA.md) | Não é da área? Cada termo e o fluxo mental de um Eng. de IA em linguagem acessível |
| [**docs/RUNBOOK.md**](docs/RUNBOOK.md) | Setup do zero, ambiente GPU (WSL2) |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/DATASET_CARD.md](docs/DATASET_CARD.md) | Governança do modelo FT e dos dados |

## 📄 Licença

[MIT](LICENSE). Dados e modelos de terceiros seguem suas próprias licenças (ver [`NOTICE`](NOTICE)).
