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

**✅ também:** golden **expandido para 50** (ICs mais estreitos): hit@5 híbrido **0,64 [0,50;0,76]**; confirmado com n=50 que o **rerank não ajuda** (MRR 0,47 < 0,51) → recomenda-se desligar (camada opcional).
**Pendente (baixo):** golden por terceiro (des-enviesar autoria); ablação de hiperparâmetros (k_rrf/candidatos/chunk); `/perguntar` sem rate-limit (dívida p/ Fase 5).

## Fase 2 — Fine-tuning/serving (concluída; NER é a manchete)
**Resultado principal:** o QLoRA vence com **métrica dura** no **NER jurídico (LeNER-Br)** —
F1 **0,13 → 0,77** (base→FT), encostando no SOTA BERTimbau 0,89 (ver `docs/13`). O estudo
generativo abaixo é o **baseline honesto** que motivou o pivot.
**Forte (baseline generativo):** QLoRA em 6 GB, fp8 no vLLM, avaliação multi-facetada com controle de viés.

**✅ Resolvido na 2ª rodada de rigor (`split_dataset.py`).** *Nota: os números abaixo são
da 2ª rodada (dataset 84→held-out −4%, n=25); foram **refinados** pela expansão a 158 ex.
(held-out +8%, n=50) mais adiante nesta seção — os valores finais valem.*
- [ALTA] **Held-out real** — 6 normas reservadas (nunca no treino); PPL **in-sample −16% × held-out −4%** ⇒ memorização ≫ generalização (o "−18%" antigo era só in-sample).
- [ALTA] **Proveniência** — `carimbar()` (com `_LIBS` estendido p/ transformers/peft/trl/vllm) em todos os reports de FT + `dataset_stats.json`.
- [MÉDIA] **`avaliar_ft.py` órfão** — grava em `avaliacao_ref_juiz.json` (não clobbera) e marcado INATIVO (precisa DVC).
- [MÉDIA] **Benchmark reprodutível** — `benchmark_vllm.py` versionado (percentis + `nvidia-smi`): **205 tok/s, p50 3.08s, VRAM 5168**.
- [MÉDIA] **`gen_offline` alinhado** ao `CONJUNTO_DOURADO` real (fonte única) + função pura testável; docs corrigidos.
- [MÉDIA] **IC** (Wilson) em citação (0/25 [0;0.13]) e win-rate (controlado **0.84 [0.65;0.94]**, agora significativo); helper `estat.py` compartilhado c/ a Fase 1.
- [MÉDIA] **Dataset determinístico** (temperatura 0) + `dataset_stats.json`.
- [BAIXA] **AWQ** deixou de ser default do merge (`--com-awq` opt-in); `_agregar_ppl`/`percentil`/`dividir` puras e testadas.

**✅ Ressalvas residuais também resolvidas:**
- **Custo de qualidade da quantização** medido (`quantizacao_qualidade.py`): PPL fp32 8.44 → NF4 9.64 (**ΔPPL +14%**); fp8 servido é mais preciso ⇒ custo ≤ 14% (cross-check fp8≈+4%).
- **Juiz factual com referência** reativado (`normas.jsonl` regenerado na F1): a métrica que a citação só aproximava.
- **Dataset expandido (84→158)** + **golden (25→50)** + re-treino + re-avaliação com n=50 — ICs mais estreitos.

**Achado refinado pela expansão (por que valeu a pena):** com 84 exemplos o FT era *muito* pior nos
fatos (factual 0.88→**0.52**, ICs disjuntos); com **158** recupera à **paridade** (0.85 vs 0.79,
ICs sobrepostos) — o dano factual era **artefato de small-data**. Mas o held-out PPL **piorou**
(−4% → **+8%**): mais fine-tuning = memoriza mais, **generaliza pior**. Win-rate estilo controlado
firme em **0.88 [0.76;0.94]**. Nenhuma ressalva de rigor pendente; expandir ainda mais é incremental.

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
**Pendentes:** nenhuma pendência de rigor nas Fases 0–2 (os itens de `data/README` — `interim/`
e a atribuição do Volume — já foram corrigidos).
