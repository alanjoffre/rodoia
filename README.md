# RodoIA

**Plataforma open-source de Engenharia de IA sobre a regulação e os dados abertos do transporte rodoviário brasileiro (ANTT) — dos fundamentos de ML/DL ao serving em produção.**

![CI](https://github.com/alanjoffre/rodoia/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> Projeto de portfólio público. Objetivo: provar, **com código, métrica e deploy**, o perfil completo de um Engenheiro de IA — cobrindo tanto a trilha moderna (LLM/RAG/agentes/MLOps) quanto o núcleo clássico de ML/DL, fine-tuning e serving de modelo próprio.

---

## Por que este projeto

A maioria dos portfólios de IA para em "chamei a API da OpenAI e funcionou". O RodoIA vai do **fundamento matemático** (um passo de backpropagation escrito à mão em NumPy, um bloco de atenção em PyTorch puro) até o **ciclo de produção** (fine-tuning próprio, serving em vLLM, CI/CD com avaliação como gate, deploy em cloud real e monitoramento de drift). O domínio — regulação e dados abertos da ANTT — é real, público e não-trivial.

## Os cinco eixos

| Eixo | O que prova |
|---|---|
| **Fundamentos de ML/DL** | Modelo treinado do zero + atenção/backprop implementados à mão |
| **RAG avaliado** | Recuperação sobre a legislação da ANTT com medição de qualidade (LLM-juiz independente + IC) |
| **Fine-tuning & serving** | LLM aberto adaptado (QLoRA) para NER jurídico — F1 0,13→0,77 vs. SOTA — quantizado (fp8) e servido em vLLM |
| **Agente orquestrado** | Raciocínio multi-etapa (LangGraph) combinando RAG + modelo próprio + dados estruturados |
| **MLOps & Cloud** | Versionamento, CI/CD, observabilidade, avaliação contínua e deploy em cloud |

## Arquitetura (visão de destino)

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
              │ DuckDB/Postgres) │   └────────────────┘
              └──────────────────┘
   MLOps transversal (Fase 5): DVC · MLflow · CI/CD · observabilidade · drift
```

## Rastreabilidade requisito → fase (o "100%")

Cada requisito de uma vaga de Engenheiro de IA é rastreado até a fase que o prova **com código e evidência**.

| Requisito | Onde é provado | Evidência |
|---|---|---|
| Python sólido (async, tipagem, produção) | Todas | Código tipado, testado, `async` nos endpoints |
| Estruturas de dados, algoritmos, complexidade | Fase 0 + 1 | Análise de complexidade em decisões de retrieval |
| Matemática aplicada (álgebra, cálculo, prob./estat.) | Fase 0 | Derivações + gradiente/atenção manuais |
| SQL avançado e modelagem | Fase 3 | Esquema **estrela** (DuckDB, 741k linhas), window functions (LAG/RANK), camada de acesso testada + **previsão de demanda** (RMSE/MAPE) |
| ML clássico (regressão, árvores, ensembles, clustering) | Fase 0 | Modelos treinados sobre dado tabular ANTT, com métricas |
| Redes neurais, backprop, arquitetura Transformer | Fase 0 | MLP em PyTorch puro + self-attention/backprop à mão (CNN/RNN fora de escopo) |
| PyTorch | Fase 0 + 2 | Treino (F0) + fine-tuning QLoRA (F2) |
| Métricas e diagnóstico (overfitting, bias/variance) | Fase 0 | Curvas treino/validação documentadas |
| Engenharia de prompts avançada | Fase 1 + 4 | Prompts versionados, testados, com ablação |
| RAG, embeddings, banco vetorial (pgvector/Qdrant) | Fase 1 | Hybrid search (BM25+RRF) · avaliado com IC (hit@5 0,64 [0,50–0,76], n=50; rerank não ajuda) · precisão de citação 0,92 |
| Orquestração de agentes (LangChain/LangGraph) | Fase 4 | Grafo LangGraph (estado + arestas condicionais) combinando as 3 ferramentas · roteamento **0,95** (n=21, objetivo) · juiz independente ([docs/15](docs/15_fase4_agente.md)) |
| Fine-tuning, LoRA/QLoRA, quantização | Fase 2 | QLoRA (Qwen2.5-3B) p/ **NER jurídico**: F1 **0,13→0,77** vs. SOTA BERTimbau 0,89 ([docs/13](docs/13_fase2_ner.md)) · **fp8** no vLLM (205 tok/s, NF4 ΔPPL +14%) · estudo-baseline (FT≠conhecimento) em [docs/11](docs/11_fase2_resultados.md) |
| Avaliação de LLMs (LLM-as-judge, guardrails, hallucination) | Fase 1 + 2 + 4 | LLM-as-judge **independente** + faithfulness/relevancy + precisão de citação (F1) · **banca de 3 juízes diversos com κ de Fleiss** (achado: juízes concordam pouco, κ=0,17 → juiz único é ruidoso) · juiz pareado c/ controle de viés (F2) · guardrails |
| Deploy/serving (FastAPI, vLLM, containers, k8s) | Fase 2 + 5 | vLLM + container + (opcional) k8s |
| CI/CD para ML, versionamento (MLflow/DVC/W&B) | Fase 5 | GitHub Actions (lint+testes+**gate de regressão**) · MLflow (sqlite, 5 runs) · DVC ([docs/16](docs/16_fase5_mlops.md)) |
| Monitoramento, observabilidade, drift | Fase 1 + 5 | Latência/tokens medidos no RAG (F1) · **drift por PSI** (coorte, 0,005 estável) na F5 |
| Cloud (AWS/Azure/GCP + serviços de ML) | Fase 5 | **Runbook de deploy** (Cloud Run, serviço justificado) — não executado por decisão de orçamento |
| Custo, latência, escalabilidade | Fase 5 | Métricas em runtime + trade-off documentado ([docs/16](docs/16_fase5_mlops.md) §7) |
| LGPD/GDPR, PII masking, auditoria | Fase 1 + 5 | Masking + trilha de auditoria |
| Segurança de IA (prompt injection, data leakage) | Fase 1 + 4 | Guardrails testados com casos adversariais |

## Roadmap e status

O projeto é faseado; **cada fase é um marco publicável por si só**. Nenhuma fase começa antes da anterior estar testada e documentada.

| Fase | Tema | Status |
|---|---|---|
| **0** | Fundamentos de ML/DL + higiene de repo público | ✅ concluída ([docs 00–05](docs/)) |
| **1** | RAG avaliado sobre a regulação da ANTT | ✅ concluída ([docs 06–09](docs/)) |
| **2** | Fine-tuning e serving de modelo próprio | ✅ concluída — **QLoRA vence com métrica dura**: NER jurídico (LeNER-Br), F1 **0,13 → 0,77** (base→FT), encostando no SOTA BERTimbau 0,89 ([resultados NER](docs/13_fase2_ner.md)). Serving fp8 no vLLM (205 tok/s). Antes, um *estudo-baseline* honesto ([docs/11](docs/11_fase2_resultados.md)) mostrou que FT **não** injeta conhecimento factual — o arco (negativo rigoroso → pivot p/ tarefa objetiva) é o diferencial. |
| **3** | Ingestão de dados estruturados abertos da ANTT | ✅ concluída ([docs/14](docs/14_fase3_dados_estruturados.md)) — Volume de Pedágio (2010–2026, 741k linhas) · **esquema estrela DuckDB** + SQL analítico (window) · camada de acesso testada (anti-injection) · **previsão de demanda** (backtest multi-step 12m em 63 praças + IC): **Holt-Winters bate o naïve com significância** (pareado Δ=3,01pp, IC95 [1,76; 4,40], vence em 73% das praças) |
| **4** | Agente de orquestração (LangGraph) | ✅ concluída ([docs/15](docs/15_fase4_agente.md)) — grafo com **arestas condicionais reais** (guardrail + roteador) que combina RAG (F1) + modelo FT NER (F2) + dados (F3) · **roteamento objetivo 0,95** (n=21: puros, combinado, ambíguo, fora-de-escopo, adversarial) · juiz independente (rota 2,0/2, n pequeno) · guardrails + degradação graciosa testados · demo `POST /agente` |
| **5** | MLOps, Cloud e operação | ✅ concluída ([docs/16](docs/16_fase5_mlops.md)) — **gate de avaliação** (regressão de métrica reprova o CI) · **GitHub Actions** (lint+testes+gate) · **MLflow** (sqlite) + **DVC** · **container** (Dockerfile+compose) · **drift** por PSI (0,005, estável) · **demo pública grátis** pronta ([HuggingFace Spaces](deploy/hf_space/README.md), R$0) · **deploy em cloud = runbook** (Cloud Run justificado, não executado por decisão de orçamento) |

O plano completo, os critérios de conclusão de cada fase e as regras de condução estão em **[PROMPT_MESTRE.md](PROMPT_MESTRE.md)**. Para um **mapa módulo a módulo** de todo o código (`src/rodoia/**` → o que cada arquivo faz), ver **[docs/ARQUITETURA.md](docs/ARQUITETURA.md)**. Não é da área? O **[guia didático](docs/GUIA_ENGENHARIA_IA.md)** explica cada termo e o fluxo mental de um Engenheiro de IA em linguagem acessível. Para a narrativa *problema → como resolvemos → resultado* de cada fase, ver **[docs/HISTORIA.md](docs/HISTORIA.md)**.

## Higiene de repositório público

Repo público desde o commit 1 (o histórico Git é imutável). Garantias em vigor:

- **Sem segredos** — `.gitignore` correto, `.env.example` versionado, `detect-secrets` no pre-commit bloqueando chaves/credenciais antes do commit.
- **Sem dado bruto no Git** — a reprodução canônica é rodar os **scripts de download** (fontes 100% públicas: `data/baixar_*`, `rag/baixar_normas`); os apontadores DVC guardam o **hash de conteúdo** para verificação. O remote DVC é **local por máquina** (`.dvc/config.local`, não publicado) — logo *não* prometemos `dvc pull` de terceiros; prometemos regeneração pelos scripts. Licença de cada fonte confirmada antes do uso (ver [data/README.md](data/README.md)).
- **Fronteira de dados** — exclusivamente domínio público da ANTT. Zero dado/regra de qualquer empregador ou cliente (ver `NOTICE`).
- **Demo sem sangrar custo** — nada de API paga exposta sem proteção; demo por modelo local/vLLM, rate-limit ou vídeo.

## Como rodar (setup inicial)

> Passo a passo completo (do clone à Fase 0 rodando): **[docs/RUNBOOK.md](docs/RUNBOOK.md)**.

```bash
git clone <url-do-repo> && cd rodoia
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # núcleo + ferramentas de dev
pre-commit install               # ativa a barreira anti-segredo
cp .env.example .env             # preencha suas chaves (NUNCA comite o .env)
pytest                           # deve passar (teste-fumaça)
```

Dependências pesadas entram por fase, como extras: `pip install -e ".[fundamentos]"` (Fase 0), `".[rag]"` (Fase 1), etc. — quem só quer ler o RAG não precisa instalar PyTorch/vLLM.

## Decisões e trade-offs (o arco do projeto)

O diferencial não são os números altos — é **o rigor ter corrigido os próprios números**. Quatro
momentos em que a avaliação honesta mudou a conclusão:

- **Fase 2 — o pivô do fine-tuning.** Um *estudo-baseline* mostrou, com held-out, que o QLoRA
  **não injeta conhecimento factual** (in-sample melhorava, held-out piorava = memorização). Em vez
  de esconder, viramos o problema para uma tarefa **objetiva** — NER jurídico — onde o FT vence com
  métrica dura (**F1 0,13→0,77**, encostando no SOTA 0,895). O arco negativo→pivô é a entrega.
- **Fase 3 — a cereja e a inconsistência.** O primeiro resultado de previsão (MAPE 5,9% numa praça)
  era **cereja**; o backtest em 63 praças + IC derrubou para ~13%. Depois, um **erro metodológico
  meu** (comparar naïve de 1-passo com Holt-Winters de 12-passos) foi corrigido para **multi-step
  justo** — aí o Holt-Winters **bate o naïve com significância** (pareado Δ=3,01pp, IC [1,76;4,40]).
- **Fase 4 — o artefato do juiz.** O juiz do agente penalizava "não rotear" nos casos fora-de-escopo
  e adversarial (onde declinar é o certo). Separar in-scope de declinados tirou o artefato: o
  roteamento é **1,0** e o juiz dá **rota 2,0/2**.
- **Fase 5 — o drift enganoso.** PSI sobre o volume **agregado** dava ~11 (a malha cresceu ~10×);
  trocar para a **coorte comum de praças** revelou o valor real — **0,005, estável**.

**Restrições assumidas conscientemente:** hardware de 6 GB (modelos 3B, time-slicing cérebro↔FT);
alguns *n* pequenos (hit@5 n=50, juiz in-scope n=4); e **sem deploy em cloud** — o runbook está
pronto ([docs/16](docs/16_fase5_mlops.md) §7), mas não foi executado por decisão de orçamento.

## Licença

[MIT](LICENSE). Dados e modelos de terceiros seguem suas próprias licenças (ver `NOTICE`).
