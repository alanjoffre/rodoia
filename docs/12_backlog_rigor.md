# 12 — Backlog de rigor e plano das próximas fases

> Auditoria de nível especialista (2026-07-10) das Fases 0–2 entregues e plano de rigor
> para as Fases 3–5. Objetivo: manter todas as fases no padrão do `PROMPT_MESTRE`
> (avaliação com números, testes de caminhos críticos, observabilidade, reprodutibilidade).
> Itens marcados [ALTA]/[MÉDIA]/[BAIXA] por impacto no sinal de senioridade.

## Fase 0 — Fundamentos ML/DL (concluída; núcleo sólido)
**Forte:** backprop à mão provado por gradient-checking + equivalência autograd; SDPA de
1 cabeça provada vs. PyTorch; baseline com métricas certas p/ desbalanceado + CV + anti-leakage;
diagnóstico viés/variância + calibração + clustering. Já é nível especialista.

**✅ Resolvido neste ciclo:**
- [ALTA] **Multi-head attention** agora provada numericamente vs. `F.scaled_dot_product_attention` por cabeça + Wo, com caso causal (`tests/test_attention.py`).
- [ALTA] **Caminho de treino testado** — `tests/test_ml_treino.py` exercita `mlp_torch.treinar` (perda cai, métricas, artefatos) e `classico.avaliar` (com frame sintético).
- [MÉDIA] **MLP com o conjunto completo de métricas** (`_metricas` reusado): empata o HistGB em ROC-AUC/PR-AUC/F1/bal-acc, não só ROC-AUC.
- [MÉDIA] **Split três-vias** treino/val/teste — validação sintoniza o limiar e monitora a curva; teste medido 1× ao fim.
- [MÉDIA] **Proveniência** (`rodoia/proveniencia.py`) carimba seed/git_sha/versões/timestamp em `reports/fase0_*` + `versoes_nitro.txt` na F2.
- [BAIXA] épocas default 15→20; **limiar da MLP derivado na validação** (max-F1); caveat de determinismo GPU no `docs/04`.

**Pendente (baixo):** limiar 0.5 do **baseline clássico** (não da MLP) segue como ponto de operação documentado; lockfile único de todo o projeto (a F2 já tem `versoes_nitro.txt`).

## Fase 1 — RAG (concluída; engenharia sênior)
**Forte:** reality-check de fontes ANTT (endpoints stateless, casca vs. texto, vigência), RRF por
posição, ablação denso→bm25→híbrido→rerank, chunking por artigo, segurança integrada, API async.

**✅ Resolvido neste ciclo:**
- [ALTA] **Observabilidade** — `OllamaLLM`/`OpenAICompatLLM` capturam tokens (prompt/resposta) e latência (`ultima_metrica`); `responder` propaga; a avaliação agrega geração p50/p95 e tokens/resposta (10s p50, 137 tok).
- [ALTA] **Juiz independente** — gerador qwen2.5:7b × juiz **llama3.1:8b**; faithfulness caiu de 0.85 (auto-avaliação) para **0.73** — mais honesto.
- [ALTA] **Golden set 10→25** perguntas de intenção real (não paráfrase do título) + **IC de Wilson** (hit) e **bootstrap** (MRR): hit@5 caiu 0.90→**0.72** [0.52; 0.86]; ICs sobrepostos ⇒ sem vitória "clara"; **rerank não ajudou** (MRR 0.54 vs híbrido 0.62).
- [ALTA] **Injeção indireta** — contexto delimitado `<contexto>` + marcadores neutralizados + hierarquia de instrução no prompt; teste dedicado.
- [MÉDIA] `recall@k`→**`hit_rate@k`** (nome honesto); **precisão de citação 0.91** (citações ancoradas no contexto) + taxa de citar a fonte certa 0.75; guardrail com **normalização** + bateria de evasão (teto documentado).
- [BAIXA] PII: **CPF sem pontuação** mascarado (CNPJ bare já caía).

**Pendente (baixo):** golden ≥50 e por terceiro (estreitar ICs); ablação de hiperparâmetros (k_rrf/candidatos/chunk); `/perguntar` sem rate-limit (dívida p/ Fase 5).

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

**Também aplicadas:** "37 × 39 concessionárias" reconciliado (39 CSVs → 37 concessionárias) no
RUNBOOK; proveniência carimbada nos JSON de `reports/fase0_*`.
**Pendentes (baixo impacto):** `data/README` layout mostra `interim/` inexistente e a linha de
Volume atribui "regressão" à F0 (foi classificação de severidade).
