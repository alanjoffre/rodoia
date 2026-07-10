# 12 — Backlog de rigor e plano das próximas fases

> Auditoria de nível especialista (2026-07-10) das Fases 0–2 entregues e plano de rigor
> para as Fases 3–5. Objetivo: manter todas as fases no padrão do `PROMPT_MESTRE`
> (avaliação com números, testes de caminhos críticos, observabilidade, reprodutibilidade).
> Itens marcados [ALTA]/[MÉDIA]/[BAIXA] por impacto no sinal de senioridade.

## Fase 0 — Fundamentos ML/DL (concluída; núcleo sólido)
**Forte:** backprop à mão provado por gradient-checking + equivalência autograd; SDPA de
1 cabeça provada vs. PyTorch; baseline com métricas certas p/ desbalanceado + CV + anti-leakage;
diagnóstico viés/variância + calibração + clustering. Já é nível especialista.

**Melhorias:**
- [ALTA] **Multi-head attention sem prova numérica** — só tem teste de forma; `docs/05` afirma equivalência não testada. → teste `AtencaoMultiCabeca(d,1)` vs. `AutoAtencao(d)` (mesmos pesos) + caso causal.
- [ALTA] **Caminho de treino não testado** — `MLP.treinar` não tem teste (o teste re-implementa o laço). → smoke com dados sintéticos: perda cai + artefatos escritos; idem `avaliar`/`diagnosticar`.
- [MÉDIA] **MLP avaliada só por ROC-AUC** — reusar `_metricas` do baseline e comparar MLP × HistGB na mesma tabela (PR-AUC/F1/calibração).
- [MÉDIA] **"Validação" da MLP = test set** — introduzir split treino/val/teste; teste intocado até o fim.
- [MÉDIA] **Reprodutibilidade** — sem lockfile; artefatos JSON sem proveniência (git SHA, seed, versões). → carimbar reports.
- [BAIXA] doc×código: MLP `epocas=15` default vs. 20 nos reports; limiar fixo 0.5 sem justificativa; caveat de determinismo GPU.

## Fase 1 — RAG (concluída; engenharia sênior)
**Forte:** reality-check de fontes ANTT (endpoints stateless, casca vs. texto, vigência), RRF por
posição, ablação denso→bm25→híbrido→rerank, chunking por artigo, segurança integrada, API async.

**Melhorias:**
- [ALTA] **Observabilidade ausente** — latência/tokens/custo não medidos (o `OllamaLLM` descarta `eval_count`/`total_duration`). Requisito explícito do PROMPT_MESTRE §3. → capturar e agregar (p50/p95, tokens/consulta) — barato e muito visível.
- [ALTA] **Juiz avalia o próprio gerador** (mesmo qwen2.5:7b gera e julga) → viés de auto-avaliação; usar juiz independente (como fizemos na Fase 2).
- [ALTA] **Golden set pequeno e circular** — n=10/n=8, perguntas paráfrases do título-alvo; "recall 0.80→0.90" = 1 pergunta. → ≥50 perguntas de intenção real + IC de Wilson/bootstrap.
- [ALTA] **Injeção indireta via contexto** não tratada (só a consulta é filtrada); teste com chunk envenenado.
- [MÉDIA] Guardrail 100% regex com testes circulares (as frases de ataque são as que o regex procura) → bateria de evasão (ofuscação, tradução). Métrica chamada `recall@k` é hit-rate@k. Ablação só de modo (sem varredura de k/chunk). Sem métrica de **correção de citação** (a joia de um RAG jurídico).
- [BAIXA] PII masking não pega CPF/CNPJ sem pontuação; `/perguntar` sem rate-limit (marcar dívida p/ Fase 5).

## Fase 2 — Fine-tuning/serving (concluída; avaliação rigorosa)
**Forte:** QLoRA em 6 GB, fp8 no vLLM, **3 medições** (PPL −18%, citação 0/0, win-rate com
controle de viés de comprimento), resultado negativo reportado com honestidade, 21 testes.

**Melhorias:**
- [MÉDIA] **Dataset de 84 exemplos** — a própria avaliação prova ser pequeno. → expandir p/ centenas.
- [MÉDIA] **Sem held-out real** — PPL é in-sample; retreinar 74/10 p/ PPL e win-rate de generalização.
- [BAIXA] Reativar o **juiz factual** quando `data/raw/normas.jsonl` (DVC) estiver disponível (Fase 5).

## Fase 3 — Ingestão de dados estruturados (a implementar, mesmo padrão)
- **Dados/licença:** Volume de Tráfego de Pedágio (carro-chefe), Praça+KMZ (geo), Receita; confirmar licença por dataset; pipeline `baixar_*` reproduzível (`sep=';'`, `latin-1`, `decimal=','`); brutos fora do Git (DVC).
- **Modelagem justificada:** esquema **estrela** em DuckDB (fato `volume_trafego` + dims praça/concessionária/tempo/receita); documentar grão e chaves; justificar cada trade-off.
- **SQL avançado:** queries com CTEs, window functions (LAG p/ MoM/YoY, RANK por praça), sazonalidade, JOIN geográfico; resultado esperado versionado.
- **Camada de acesso** tipada (Pydantic), parametrizada (anti-injection), pensada como **tool do agente (Fase 4)**; testes de schema/encoding + queries contra fixture.
- **Observabilidade/README:** métricas de ingestão (linhas, rejeições, cobertura temporal); README do modelo de dados (diagrama, dicionário, licença, reprodução); extra `estruturados` (duckdb) no pyproject.

## Fase 4 — Agente LangGraph (a implementar)
- [ ] Grafo com estado + **arestas condicionais reais** (roteamento, não linear).
- [ ] Integrar 3 ferramentas: RAG (F1) + FT no vLLM (F2) + acesso a dados (F3) + nó de cálculo.
- [ ] **FT + RAG combinados** — correção da citação 0/0 identificada na Fase 2.
- [ ] 2–3 casos ponta a ponta (texto regulatório + dado estruturado + cálculo).
- [ ] **Avaliação de trajetória** (LLM-judge no caminho, não só na resposta) — reusar o juiz pareado da F2.
- [ ] Tratamento de falha/fora-de-escopo/adversarial (reusar `rag/seguranca.py`).
- [ ] Diagrama do grafo no README + testes dos nós + demo de custo zero.

## Fase 5 — MLOps/Cloud (a implementar)
- [ ] Containerização (Docker/compose sobe a plataforma inteira; k8s opcional).
- [ ] **CI/CD com a suíte de avaliação como GATE** (regressão de métrica reprova) — ativar badges do README.
- [ ] MLflow + DVC (experimentos/prompts/configs versionados); migrar remote DVC → S3.
- [ ] Deploy em cloud com serviço gerenciado justificado; observabilidade (custo/latência/tokens/qualidade); drift.
- [ ] README final com rastreabilidade 100% preenchida; reativar juiz factual da Fase 2.

## Correções de documentação
**Aplicadas neste ciclo:** `PROMPT_MESTRE` "primeira tarefa" PENDENTE → concluída; `data/README`
"não há pipeline" → lista pipelines existentes (acidentes, normas); `pyproject` remove `autoawq`
(contradizia a decisão fp8 do doc 11) e eleva pisos p/ trl 1.x/transformers 5.x; `docs/10` ganha
aviso de "handoff superado por docs/11 (fp8, não AWQ)"; README rastreabilidade (fine-tuning →
fp8/vLLM/números; LLM-judge inclui Fase 2; remove TensorFlow) + status da Fase 2 enriquecido;
RUNBOOK Fase 2 "futura" → concluída.

**Pendentes (baixo impacto):** reconciliar "37 × 39 concessionárias" (RUNBOOK × doc 02);
`data/README` layout mostra `interim/` inexistente e a linha de Volume atribui "regressão" à F0
(foi classificação); carimbar proveniência (git SHA/seed/versões) nos JSON de `reports/`.
