<div align="center">

# 🗺️ Arquitetura — mapa módulo a módulo

**Cada arquivo `src/rodoia/**` → o que faz, funções-chave e a fase.** Índice de referência.

[← README](../README.md) · [📖 A história](HISTORIA.md) · [🎓 Guia didático](GUIA_ENGENHARIA_IA.md)

</div>

> Para o *porquê* de cada decisão, ver os docs da fase (`docs/NN`). Para o *fluxo de requisito → prova*, ver a tabela de rastreabilidade no [README](../README.md).

## 📦 Visão de pacotes

```
src/rodoia/
├── fundamentos/   F0 · backprop e atenção à mão (PyTorch/NumPy puro)
├── ml/            F0 · ML clássico + MLP (severidade de acidente)
├── ingestao/          F0/F3 · download + ingestão dos CSVs públicos da ANTT
├── rag/           F1 · corpus → chunking → índice → retrieval híbrido → geração → segurança → API
│                  F6 · benchmark externo CUAD (parser, avaliação BM25 vs denso)
├── ft/            F2 · QLoRA, quantização, serving e avaliação (perplexidade, juízes, win-rate)
├── ner/           F2 · NER jurídico (LeNER-Br): generativo (FT) vs BERTimbau (SOTA)
├── dominio/         F3 · esquema estrela DuckDB, SQL analítico, acesso, previsão
├── agente/        F4 · grafo LangGraph que orquestra RAG + modelo FT + dados
├── mlops/         F5 · gate de avaliação, rastreio MLflow, drift, custo, carga
├── api/           F1/F4 · FastAPI (RAG + agente)
└── (transversais) config.py · estat.py · proveniencia.py · observabilidade.py · anotacao.py
```

## 🔧 Transversais

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `config.py` | Config central (pydantic-settings); caminhos, modelos, `seed`, override por env | `Settings` |
| `estat.py` | ICs compartilhados (n pequeno) + concordância inter-anotador + **teste pareado** (χ² de Yates acima de 25 discordantes, **binomial exato** abaixo — o método usado sai no relatório) | `wilson`, `bootstrap_ic`, `cohen_kappa`, `cohen_kappa_ic95`, `fleiss_kappa`, `mcnemar` |
| `proveniencia.py` | Carimbo de reprodutibilidade em todo report (seed/git_sha/**git_dirty**/versões/timestamp) | `carimbar`, `proveniencia`, `_git_sha`, `_git_dirty`, `_versoes` |
| `observabilidade.py` | Cache LRU (corta p95) + métrica estruturada por requisição (serving) | `CacheLRU`, `registrar_metrica` |
| `anotacao.py` | Kit de anotação HUMANA (κ inter-anotador): relevância de trecho **e** rótulo-gold de fonte do hit@5 | `gerar_kit`, `computar_kappa`, `gerar_kit_gold`, `_ler` |

## 🧮 Fase 0 — Fundamentos, ML clássico e dados de acidentes

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `fundamentos/backprop_numpy.py` | Backpropagation à mão em NumPy (prova de fundamento + gradient checking) | `MLPNumpy`, `sigmoid`, `relu`, `bce_loss`, `gradient_check` |
| `fundamentos/attention.py` | Self-attention scaled-dot-product à mão em PyTorch puro (multi-cabeça) | `scaled_dot_product_attention`, `AutoAtencao`, `AtencaoMultiCabeca`, `mascara_causal` |
| `ml/classico.py` | Baseline de ML clássico (prever `houve_vitima`): pipelines sklearn + métricas | `avaliar`, `features`, `construir_modelos`, `construir_preprocessador` |
| `ml/mlp_torch.py` | MLP em PyTorch com laço de treino manual + limiar por F1 | `treinar`, `MLP`, `dispositivo`, `_limiar_max_f1` |
| `ml/diagnostico.py` | Diagnóstico (curvas de aprendizado/validação, calibração, clustering) | `diagnosticar`, `curva_aprendizado`, `diagrama_calibracao` |
| `ingestao/baixar_acidentes.py` | Download reprodutível dos CSVs de Acidentes (CKAN) | `baixar_acidentes`, `listar_recursos_csv`, `baixar_recurso` |
| `ingestao/esquema_acidentes.py` | Schema canônico + leitura robusta (latin-1, `;`, decimal `,`) | `validar_esquema`, `ler_csv_acidentes` |
| `ingestao/ingestao_acidentes.py` | Consolida 39 CSVs (37 concessionárias), deriva alvo e features | `consolidar`, `derivar_alvo`, `engenharia_features` |

## 🔎 Fase 1 — RAG avaliado (`rag/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `rag/baixar_normas.py` | Baixa o corpus de resoluções (ANTTlegis) | `baixar_corpus` |
| `rag/fontes_antt.py` | Cliente ANTTlegis: lista atos, limpa HTML, checa vigência | `listar_atos`, `baixar_ato`, `limpar_html`, `esta_vigente` |
| `rag/chunking.py` | Chunking consciente da estrutura jurídica (por artigos) + **de-boilerplate** | `chunk_norma`, `dividir_por_artigos`, `_cortar_ate_cabecalho`, `_e_boilerplate` |
| `rag/embeddings.py` | Embeddings locais, custo zero. **Classe por família** — o prefixo do E5 e do BGE difere e errar degrada em silêncio | `E5Embedder`, `BGEEmbedder`, `construir_embedder`, `Embedder` (Protocol) |
| `rag/indice.py` | Índice vetorial Qdrant (modo local, sem servidor) | `criar_cliente`, `indexar`, `buscar` |
| `rag/construir_indice.py` | Constrói o índice + emite a composição do corpus versionada | `construir`, `carregar_chunks`, `escrever_stats_corpus` |
| `rag/recuperador.py` | Recuperação híbrida densa+BM25 com fusão RRF + reranker | `RecuperadorHibrido`, `fundir_rrf`, `Reranker`, `tokenizar` |
| `rag/llm.py` | Interface `LLM` (Protocol) + `LLMStream` (streaming, opcional) + backends Ollama e OpenAI-compat (vLLM) | `LLM`, `LLMStream`, `OllamaLLM.gerar_stream`, `OpenAICompatLLM` |
| `rag/gerar.py` | Monta prompt ancorado, gera resposta com fontes, versão segura e **versão em fluxo (SSE) com o guardrail preservado** | `responder`, `responder_seguro`, `responder_seguro_stream`, `montar_prompt` |
| `rag/seguranca.py` | Guardrails: anti-injection, PII masking (inclusive **sobre fluxo**, com retaguarda para o padrão não escapar partido em tokens), auditoria | `detectar_injection`, `mascarar_pii`, `mascarar_pii_stream`, `registrar_auditoria` |
| `rag/avaliacao_retrieval.py` | Compara modos (denso/BM25/híbrido/rerank): hit@5 + MRR com IC · **hit@5 auditado pós-κ** | `comparar`, `avaliar_modo`, `avaliar_auditado`, `carregar_recuperador` |
| `rag/avaliacao_geracao.py` | LLM-as-judge (estilo RAGAS): faithfulness/relevancy/citação **com IC bootstrap** | `avaliar_geracao`, `ics_geracao`, `julgar`, `citacoes` |
| `rag/painel_juizes.py` | Banca de 3 juízes diversos (≠ gerador) + κ de Fleiss (independência mensurável) | `avaliar_painel`, `julgar` |

> **Auditoria de avaliação (Fase 1):** o κ humano dos rótulos é medido em [`anotacao.py`](../src/rodoia/anotacao.py) (transversal) e o efeito no `hit@5` em `avaliacao_retrieval.avaliar_auditado` → `reports/fase1_retrieval/hit5_auditado.json`.

## 🎯 Fase 2 — Fine-tuning, serving e NER (`ft/`, `ner/`)

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
| `ft/avaliar_ft.py` | Avaliação antes/depois do FT por juiz-com-referência | `comparar_modelos`, `julgar_correcao`, `carregar_referencias` |
| `ft/juiz_factual.py` | Juiz factual com referência (correção base vs. FT) | `avaliar`, `_julgar_factual` |
| `ft/juiz_winrate.py` | LLM-as-judge win-rate com controle de comprimento | `comparar`, `_parse_veredito`, `_decidir` |
| `ft/benchmark_vllm.py` | Observabilidade de serving (tok/s, p50/p95, VRAM) | `benchmark`, `percentil`, `vram_usada_mb` |
| `ner/lener.py` | Carrega o LeNER-Br (CoNLL) | `carregar`, `baixar`, `_parse_conll` |
| `ner/generativo.py` | NER como extração generativa (JSON) para o LLM | `construir_dataset`, `entidades_bio`, `para_chat`, `como_conjunto` |
| `ner/avaliar_generativo.py` | F1 de entidade do NER generativo (vLLM), base vs. FT | `avaliar`, `metricas_ner`, `parse_entidades` |
| `ner/bertimbau.py` | NER com BERTimbau token-classification — a REFERÊNCIA SOTA | `treinar`, `_tokenizar_alinhar`, `_metricas` |

## 📊 Fase 3 — Dados estruturados (`dominio/`, `ingestao/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `ingestao/baixar_volume.py` | Download do Volume de Pedágio (CKAN) | `baixar_volume`, `_recursos` |
| `ingestao/ingestao_volume.py` | Consolida CSVs → parquet limpo (datas mistas, caixa, granularidade) | `consolidar`, `_para_data`, `_ler_csv` |
| `dominio/estrela.py` | Constrói o esquema estrela (fato + dimensões) em DuckDB | `construir` |
| `dominio/consultas.py` | SQL analítico (CTEs + window: LAG/RANK/sazonalidade/composição) | `rodar` |
| `dominio/acesso.py` | Camada de acesso parametrizada (anti-injection) — ferramenta do agente | `ranking_pracas`, `volume_praca`, `serie_mensal`, `volume_por_ano` |
| `dominio/previsao.py` | Previsão multi-step (12m) + backtest 63 praças + IC + teste pareado | `avaliar`, `_prever_praca`, `_gb_recursivo`, `_series_completas`, `_mape` |

## 🤖 Fase 4 — Agente de orquestração (`agente/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `agente/estado.py` | Estado do grafo + contrato de ferramentas (injeção de dependência) | `EstadoAgente`, `DepsAgente` |
| `agente/roteador.py` | Decisão condicional: escolhe as ferramentas (LLM + fallback heurístico) | `rotear`, `_heuristica`, `_parse` |
| `agente/ferramentas.py` | Cascas sobre F1/F2/F3 + montagem das deps reais | `regulatorio_real`, `entidades_real`, `dados_real`, `deps_reais` |
| `agente/grafo.py` | Grafo LangGraph (nós + arestas condicionais) + `responder` | `construir_agente`, `responder`, `_no_guardrail`, `_no_roteador`, `_no_executar`, `_no_sintetizar` |
| `agente/casos.py` | Casos de domínio (puros, combinado, fora-de-escopo, adversarial) | `CASOS` |
| `agente/avaliar.py` | Avaliação de trajetória (juiz) + roteamento objetivo em n ampliado | `avaliar`, `avaliar_roteamento`, `_julgar`, `_jaccard` |

## ⚙️ Fase 5 — MLOps (`mlops/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `mlops/gate.py` | Gate de avaliação: regressão de métrica falha o CI (30 portões) | `avaliar`, `GATES`, `_acessar`, `_passou` |
| `mlops/rastreio.py` | Consolida métricas das fases em runs MLflow (sqlite) | `coletar`, `registrar` |
| `mlops/drift.py` | Drift por PSI (coorte de praças, 12m vs 12m) | `psi`, `drift_volume`, `classificar` |
| `mlops/reproduzir.py` | Reprodução real: re-executa o pipeline e confere contra o JSON commitado | `reproduzir_retrieval`, `reproduzir_previsao` |
| `mlops/carga.py` | Teste de carga do cache: mede p50/p95 sob concorrência (efeito medido, não afirmado) | `teste_carga`, `medir` |
| `mlops/custo.py` | Custo de serving **R$/1k req** da vazão medida (rota FT vazão · rota RAG latência) | `calcular`, `_linha`, `_linha_latencia` |
| `mlops/benchmark_motor.py` | F6 · benchmark DuckDB vs Spark sobre 17,2 M (mesmo SQL) + cross-check de correção | `comparar`, `rodar_duckdb`, `rodar_spark`, `_normalizar` |

## 📈 Fase 6 — Escala e benchmark externo (`ingestao/`, `rag/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `ingestao/baixar_cfpb.py` | Download do bulk CFPB (1,43 GB) com sha256 + manifesto; WAF invertido (UA `curl`) | `baixar_cfpb`, `consultar_metadados` |
| `ingestao/ingestao_cfpb.py` | Zip ZIP64 → Parquet particionado `ano=YYYY/` por streaming, sem materializar o CSV | `ingerir_cfpb`, `_FluxoZip`, `_data_iso`, `_EscritorParticionado` |
| `rag/baixar_cuad.py` | Download do CUAD (benchmark externo) via API pública do Kaggle, sem credencial | `baixar_cuad`, `consultar_metadados` |
| `rag/cuad.py` | Parser do `CUAD_v1.json` (SQuAD 2.0) → JSONL + aferição de integridade (offsets) | `carregar`, `validar_offsets`, `estatisticas` |
| `rag/avaliacao_cuad.py` | Recuperação BM25 dentro do contrato + métricas com IC; `consolidar` partilhado. Dois chunkers (janela / cláusula) e o **teto + recall normalizado** que torna a comparação entre eles legítima | `avaliar`, `consolidar`, `chunkar`, `chunkar_clausula`, `obter_chunker`, `_registrar`, `gold_da_pergunta`, `diagnostico_chunker` |
| `rag/avaliacao_cuad_denso.py` | Recuperação densa (e5) vs BM25 por categoria — forças complementares | `avaliar_denso`, `_ranquear_denso`, `_comparar_categorias` |
| `rag/avaliacao_cuad_hibrido.py` | Fusão RRF (reusa `recuperador.fundir_rrf`); tabela de 3 vias com IC | `avaliar_hibrido`, `fundir`, `_tabela_tres_vias` |
| `rag/avaliacao_cuad_rerank.py` | Pilha COMPLETA da Fase 1 (denso+BM25→RRF→cross-encoder) — único ganho com IC disjunto | `avaliar_rerank`, `rerankear`, `_delta_vs_hibrido` |
| `rag/avaliacao_cuad_geracao.py` | **Alucinação vs cobertura** (LLM local, API=0) · ablação de prompt · **comparação pareada** (McNemar) · rerank opcional no contexto · **auditoria do próprio detector de abstenção** (limite superior de falso positivo + backlog humano) | `avaliar_geracao`, `consolidar_geracao`, `comparar_pareado`, `auditar_deteccao`, `absteve`, `amostrar` |
| `rag/calibracao_abstencao.py` | Varre o limiar sobre o escore do cross-encoder e publica a **curva inteira**: AUC-ROC, melhor ponto (J de Youden) e a **economia de cascata** (chamadas de LLM evitadas × respondíveis perdidas) | `analisar`, `varrer_limiar`, `auc_roc`, `melhor_ponto`, `economia_cascata` |

## 🌐 API (`api/`)

| Arquivo | O que faz | Funções-chave |
|---|---|---|
| `api/app.py` | FastAPI async: `/perguntar` (RAG), **`/perguntar/stream` (SSE, guardrail preservado)**, `/agente` (orquestrado), `/health`, UI | `perguntar`, `perguntar_stream`, `agente`, `_carregar` |

## 🔗 Como as peças se conectam (fluxo do agente, Fase 4)

```
pergunta → api/app.py:/agente → agente/grafo.responder
   → guardrail (rag/seguranca.detectar_injection)
   → roteador (agente/roteador.rotear via rag/llm.OllamaLLM)
   → executar:
        regulatorio → rag/gerar.responder_seguro  (recuperador híbrido + LLM)
        entidades   → ner prompt + rag/llm.OpenAICompatLLM → vLLM (modelo FT)
        dados       → dominio/acesso.* (DuckDB) + cálculo
   → sintetizar (LLM) → mascarar_pii → resposta + fontes + trajetória
```

## 🔬 Onde ver as evidências

- **Métricas versionadas:** `reports/<fase>/*.json` (carimbadas por `proveniencia.carimbar`).
- **Gate de qualidade:** `src/rodoia/mlops/gate.py` (pisos por métrica, 30 portões) e o CI em `.github/workflows/ci.yml`.
- **Auditoria da avaliação:** κ humano em `anotacao.py` → `reports/fase1_rag/kappa_humano.json` e `kappa_gold_fonte.json`; efeito no hit@5 em `hit5_auditado.json`.
- **Testes:** `tests/test_*.py` (280 testes; 263 no CI — os 17 de fundamentos que exigem torch são pulados via `tests/conftest.py`).
- **Narrativa por fase:** `docs/00`–`docs/17`; decisões/trade-offs no [README](../README.md).
