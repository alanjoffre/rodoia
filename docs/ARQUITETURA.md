# Arquitetura — mapa módulo a módulo

> Índice de referência: **cada arquivo `src/rodoia/**` → o que faz, funções-chave e a fase**.
> Para o *porquê* de cada decisão, ver os docs da fase (`docs/NN`). Para o *fluxo de requisito
> → prova*, ver a tabela de rastreabilidade no [README](../README.md).

## Visão de pacotes

```
src/rodoia/
├── fundamentos/   F0 · backprop e atenção à mão (PyTorch/NumPy puro)
├── ml/            F0 · ML clássico + MLP (severidade de acidente)
├── data/          F0/F3 · download + ingestão dos CSVs públicos da ANTT
├── rag/           F1 · corpus → chunking → índice → retrieval híbrido → geração → segurança → API
├── ft/            F2 · QLoRA, quantização, serving e avaliação (perplexidade, juízes, win-rate)
├── ner/           F2 · NER jurídico (LeNER-Br): generativo (FT) vs BERTimbau (SOTA)
├── dados/         F3 · esquema estrela DuckDB, SQL analítico, acesso, previsão
├── agente/        F4 · grafo LangGraph que orquestra RAG + modelo FT + dados
├── mlops/         F5 · gate de avaliação, rastreio MLflow, drift
├── api/           F1/F4 · FastAPI (RAG + agente)
└── (transversais) config.py · estat.py · proveniencia.py
```

## Transversais

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `config.py` | Config central (pydantic-settings); caminhos, modelos, `seed`, override por env | `Settings` |
| `estat.py` | Intervalos de confiança compartilhados (n pequeno) | `wilson`, `bootstrap_ic` |
| `proveniencia.py` | Carimbo de reprodutibilidade em todo report (seed/git_sha/versões/timestamp) | `carimbar`, `proveniencia`, `_git_sha`, `_versoes` |

## Fase 0 — Fundamentos, ML clássico e dados de acidentes

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `fundamentos/backprop_numpy.py` | Backpropagation à mão em NumPy (prova de fundamento + gradient checking) | `MLPNumpy`, `sigmoid`, `relu`, `bce_loss`, `gradient_check` |
| `fundamentos/attention.py` | Self-attention scaled-dot-product à mão em PyTorch puro (multi-cabeça) | `scaled_dot_product_attention`, `AutoAtencao`, `AtencaoMultiCabeca`, `mascara_causal` |
| `ml/classico.py` | Baseline de ML clássico (prever `houve_vitima`): pipelines sklearn + métricas | `avaliar`, `features`, `construir_modelos`, `construir_preprocessador` |
| `ml/mlp_torch.py` | MLP em PyTorch com laço de treino manual + limiar por F1 | `treinar`, `MLP`, `dispositivo`, `_limiar_max_f1` |
| `ml/diagnostico.py` | Diagnóstico (curvas de aprendizado/validação, calibração, clustering) | `diagnosticar`, `curva_aprendizado`, `diagrama_calibracao` |
| `data/baixar_acidentes.py` | Download reprodutível dos CSVs de Acidentes (CKAN) | `baixar_acidentes`, `listar_recursos_csv`, `baixar_recurso` |
| `data/esquema_acidentes.py` | Schema canônico + leitura robusta (latin-1, `;`, decimal `,`) | `validar_esquema`, `ler_csv_acidentes` |
| `data/ingestao_acidentes.py` | Consolida 39 concessionárias, deriva alvo e features | `consolidar`, `derivar_alvo`, `engenharia_features` |

## Fase 1 — RAG avaliado (`rag/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `rag/baixar_normas.py` | Baixa o corpus de resoluções (ANTTlegis) | `baixar_corpus` |
| `rag/fontes_antt.py` | Cliente ANTTlegis: lista atos, limpa HTML, checa vigência | `listar_atos`, `baixar_ato`, `limpar_html`, `esta_vigente` |
| `rag/chunking.py` | Chunking consciente da estrutura jurídica (por artigos + janela) | `chunk_norma`, `dividir_por_artigos`, `empacotar` |
| `rag/embeddings.py` | Embeddings locais (E5 multilingual, custo zero) | `E5Embedder`, `Embedder` (Protocol) |
| `rag/indice.py` | Índice vetorial Qdrant (modo local, sem servidor) | `criar_cliente`, `indexar`, `buscar` |
| `rag/construir_indice.py` | Constrói o índice a partir do corpus chunkado | `construir`, `carregar_chunks` |
| `rag/recuperador.py` | Recuperação híbrida densa+BM25 com fusão RRF + reranker | `RecuperadorHibrido`, `fundir_rrf`, `Reranker`, `tokenizar` |
| `rag/llm.py` | Interface `LLM` (Protocol) + backends Ollama e OpenAI-compat (vLLM) | `LLM`, `OllamaLLM`, `OpenAICompatLLM` |
| `rag/gerar.py` | Monta prompt ancorado, gera resposta com fontes, versão segura | `responder`, `responder_seguro`, `montar_prompt`, `montar_contexto` |
| `rag/seguranca.py` | Guardrails: anti-injection, PII masking, auditoria | `detectar_injection`, `mascarar_pii`, `registrar_auditoria` |
| `rag/avaliacao_retrieval.py` | Compara modos (denso/BM25/híbrido/rerank): hit@5 + MRR com IC | `comparar`, `avaliar_modo`, `carregar_recuperador` |
| `rag/avaliacao_geracao.py` | LLM-as-judge (estilo RAGAS): faithfulness/relevancy/citação | `avaliar_geracao`, `julgar`, `citacoes` |

## Fase 2 — Fine-tuning, serving e NER (`ft/`, `ner/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `ft/construir_dataset.py` | Gera o dataset de FT (Q&A) a partir das normas | `construir`, `gerar_pares_norma`, `para_chat` |
| `ft/split_dataset.py` | Split held-out por norma (avalia generalização) | `dividir` |
| `ft/treino_qlora.py` | Fine-tuning QLoRA (NF4 4-bit) do LLM aberto | `treinar` |
| `ft/merge_quantiza.py` | Merge do adaptador LoRA + quantização | `merge`, `quantizar_awq` |
| `ft/perplexidade.py` | Perplexidade de domínio (fit vs. generalização) | `perplexidade`, `_agregar_ppl` |
| `ft/quantizacao_qualidade.py` | Custo de qualidade da quantização (fp32 vs NF4) | `ppl` |
| `ft/gen_offline.py` | Gera respostas offline (vLLM) para o golden | `montar_conversas`, `montar_respostas` |
| `ft/aval_cite.py` | Métrica objetiva de citação, base vs. FT, com IC | `avaliar`, `cita_correta`, `cita_alguma` |
| `ft/juiz_factual.py` | Juiz factual com referência (correção base vs. FT) | `avaliar`, `_julgar_factual` |
| `ft/juiz_winrate.py` | LLM-as-judge win-rate com controle de comprimento | `comparar`, `_parse_veredito`, `_decidir` |
| `ft/benchmark_vllm.py` | Observabilidade de serving (tok/s, p50/p95, VRAM) | `benchmark`, `percentil`, `vram_usada_mb` |
| `ner/lener.py` | Carrega o LeNER-Br (CoNLL) | `carregar`, `baixar`, `_parse_conll` |
| `ner/generativo.py` | NER como extração generativa (JSON) para o LLM | `construir_dataset`, `entidades_bio`, `para_chat`, `como_conjunto` |
| `ner/avaliar_generativo.py` | F1 de entidade do NER generativo (vLLM), base vs. FT | `avaliar`, `metricas_ner`, `parse_entidades` |
| `ner/bertimbau.py` | NER com BERTimbau token-classification — a REFERÊNCIA SOTA | `treinar`, `_tokenizar_alinhar`, `_metricas` |

## Fase 3 — Dados estruturados (`dados/`, `data/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `data/baixar_volume.py` | Download do Volume de Pedágio (CKAN) | `baixar_volume`, `_recursos` |
| `data/ingestao_volume.py` | Consolida CSVs → parquet limpo (datas mistas, caixa, granularidade) | `consolidar`, `_para_data`, `_ler_csv` |
| `dados/estrela.py` | Constrói o esquema estrela (fato + dimensões) em DuckDB | `construir` |
| `dados/consultas.py` | SQL analítico (CTEs + window: LAG/RANK/sazonalidade/composição) | `rodar` |
| `dados/acesso.py` | Camada de acesso parametrizada (anti-injection) — ferramenta do agente | `ranking_pracas`, `volume_praca`, `serie_mensal`, `volume_por_ano` |
| `dados/previsao.py` | Previsão multi-step (12m) + backtest 63 praças + IC + teste pareado | `avaliar`, `_prever_praca`, `_gb_recursivo`, `_series_completas`, `_mape` |

## Fase 4 — Agente de orquestração (`agente/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `agente/estado.py` | Estado do grafo + contrato de ferramentas (injeção de dependência) | `EstadoAgente`, `DepsAgente` |
| `agente/roteador.py` | Decisão condicional: escolhe as ferramentas (LLM + fallback heurístico) | `rotear`, `_heuristica`, `_parse` |
| `agente/ferramentas.py` | Cascas sobre F1/F2/F3 + montagem das deps reais | `regulatorio_real`, `entidades_real`, `dados_real`, `deps_reais` |
| `agente/grafo.py` | Grafo LangGraph (nós + arestas condicionais) + `responder` | `construir_agente`, `responder`, `_no_guardrail`, `_no_roteador`, `_no_executar`, `_no_sintetizar` |
| `agente/casos.py` | Casos de domínio (puros, combinado, fora-de-escopo, adversarial) | `CASOS` |
| `agente/avaliar.py` | Avaliação de trajetória: roteamento objetivo + juiz independente | `avaliar`, `_julgar`, `_jaccard` |

## Fase 5 — MLOps (`mlops/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `mlops/gate.py` | Gate de avaliação: regressão de métrica falha o CI | `avaliar`, `GATES`, `_acessar`, `_passou` |
| `mlops/rastreio.py` | Consolida métricas das fases em runs MLflow (sqlite) | `coletar`, `registrar` |
| `mlops/drift.py` | Drift por PSI (coorte de praças, 12m vs 12m) | `psi`, `drift_volume`, `classificar` |

## API (`api/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `api/app.py` | FastAPI async: `/perguntar` (RAG), `/agente` (orquestrado), `/health`, UI | `perguntar`, `agente`, `_carregar`, `_carregar_agente` |

## Como as peças se conectam (fluxo do agente, Fase 4)

```
pergunta → api/app.py:/agente → agente/grafo.responder
   → guardrail (rag/seguranca.detectar_injection)
   → roteador (agente/roteador.rotear via rag/llm.OllamaLLM)
   → executar:
        regulatorio → rag/gerar.responder_seguro  (recuperador híbrido + LLM)
        entidades   → ner prompt + rag/llm.OpenAICompatLLM → vLLM (modelo FT)
        dados       → dados/acesso.* (DuckDB) + cálculo
   → sintetizar (LLM) → mascarar_pii → resposta + fontes + trajetória
```

## Onde ver as evidências

- **Métricas versionadas:** `reports/<fase>/*.json` (carimbadas por `proveniencia.carimbar`).
- **Gate de qualidade:** `src/rodoia/mlops/gate.py` (pisos por métrica) e o CI em `.github/workflows/ci.yml`.
- **Testes:** `tests/test_*.py` (um por área; fundamentos que exigem torch são pulados no CI via `tests/conftest.py`).
- **Narrativa por fase:** `docs/00`–`docs/16`; decisões/trade-offs no [README](../README.md#decisões-e-trade-offs-o-arco-do-projeto).
