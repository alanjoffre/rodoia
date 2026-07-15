# A história do RodoIA — problema → como resolvemos → resultado

> Narrativa corrida do projeto, fase a fase. Para o mapa técnico dos arquivos ver
> [ARQUITETURA.md](ARQUITETURA.md); para os termos explicados, o [guia didático](GUIA_ENGENHARIA_IA.md);
> para as métricas cruas, `reports/<fase>/*.json`.

## O ponto de partida

A pergunta que move o projeto: **como provar, com código e número, o perfil completo de um
Engenheiro de IA** — do fundamento matemático ao sistema em produção — sobre dados **reais e
públicos**? Escolhemos o domínio da **ANTT** (regulação e dados abertos do transporte rodoviário
brasileiro): real, público e não-trivial. A regra de condução: **uma fase por vez, cada uma um
marco publicável, nenhuma começa antes da anterior estar testada.**

---

## Fase 0 — Fundamentos: "eu entendo o motor?"

**Problema.** Antes de usar IA, provar que entendo o que está por baixo. E, nos dados: o dataset
público de **Acidentes em rodovias concedidas** vinha em **39 CSVs** heterogêneos (latin-1, `;`,
decimal `,`), ~1,03 M linhas, sem um alvo de modelagem pronto.

**Como resolvemos.** Escrevi **backpropagation e self-attention à mão** (NumPy/PyTorch puro), com
prova de equivalência numérica contra o autograd. Consolidei os CSVs (37 concessionárias
reconciliadas), derivei o alvo `houve_vitima` e treinei ML clássico + uma MLP com laço de treino manual.

**Resultado.** Modelo de severidade com **ROC-AUC 0,81**, curvas de treino/validação documentadas.
Fundamento provado, não presumido.

---

## Fase 1 — RAG: "responder com base em documentos reais"

**Problema.** A legislação da ANTT **não tem API**: HTML em latin-1, normas antigas só em imagem
(exigem OCR), e é preciso saber se uma resolução está **vigente**. E responder sem **inventar**.

**Como resolvemos.** Pipeline de scraping + limpeza + checagem de vigência; chunking consciente da
estrutura jurídica (por artigos); **busca híbrida** (densa E5 + BM25 fundidos por RRF); geração
**ancorada** que cita a resolução; e guardrails (anti-injection, PII masking, auditoria). Medimos
tudo com **juiz LLM independente** e **intervalo de confiança**.

**Resultado.** **hit@5 0,62** [0,48–0,74], **precisão de citação 0,92**. E dois achados honestos: a
limpeza de boilerplate **reverteu** a conclusão do reranker (que passou a ajudar, 0,68); e uma
auditoria κ humana **achou 16% dos rótulos-gold errados** — rerotulados, com o impacto reportado.

---

## Fase 2 — Fine-tuning: "especializar um modelo e provar o ganho"

**Problema.** Como provar que **fine-tuning agrega**? A primeira tentativa — ensinar o modelo a
responder fatos da ANTT — **falhou**: com **held-out**, o ganho aparente virou **memorização** (ia
bem no que viu, mal no que não viu). Um resultado negativo, mas honesto.

**Como resolvemos.** Em vez de esconder, **pivotamos** para uma tarefa de **rótulo objetivo**: NER
jurídico sobre o **LeNER-Br** (dado público MIT). Fine-tuning com **QLoRA**, servido em **vLLM** com
**quantização fp8** (medindo o custo de qualidade da compressão), comparado contra o teto **BERTimbau
(SOTA)**.

**Resultado.** **F1 de entidade 0,13 → 0,77** (base → fine-tunado), encostando no SOTA 0,895 —
treinando em **1/5 dos dados** por via generativa. O arco *negativo rigoroso → pivô → vitória com
métrica dura* é o diferencial da fase.

---

## Fase 3 — Dados estruturados: "do CSV sujo à previsão que convence"

**Problema (o mais rico em dados).** O **Volume de Tráfego de Pedágio** (2010–2026) vinha **sujo**:
- **dois formatos de data** no mesmo dataset (`DD/MM/AAAA` nos anuais, `MM/AAAA` nos consolidados);
- **coluna divergente** entre anos (`categoria` vs `categoria_eixo`);
- **granularidade mista** (alguns anos vêm diários, outros mensais);
- **variantes de caixa** (`Passeio` vs `PASSEIO`) inflando as categorias.

**Como resolvemos.** Ingestão robusta que normaliza datas, reconcilia colunas, **trunca ao mês e
soma** (série mensal consistente) e padroniza a caixa. Modelagem em **esquema estrela** (DuckDB),
SQL analítico (window functions), camada de acesso parametrizada (anti-injection) e previsão de
demanda avaliada com **backtest multi-step em 63 praças + IC + teste pareado**.

**Resultado.** **741.205 linhas limpas** (197 meses, 50 concessionárias, 292 praças). E, na previsão,
o rigor **corrigiu o próprio número duas vezes**: um "MAPE 5,9%" que era **cereja** virou ~13% no
backtest; e uma comparação injusta (naïve de 1-passo × Holt-Winters de 12-passos) foi corrigida para
**multi-step justo** — aí o **Holt-Winters bate o naïve com significância** (pareado Δ=3,01 pp, IC
[1,76; 4,40], vence em 73% das praças).

---

## Fase 4 — Agente: "juntar tudo e decidir"

**Problema.** Ter três capacidades (RAG, modelo FT, dados) não basta — é preciso um sistema que
**decide** qual usar e **combina** as respostas, com segurança e sem cair quando algo falha.

**Como resolvemos.** Um grafo **LangGraph** com **arestas condicionais reais**: guardrail →
roteador (escolhe as ferramentas, podendo combinar) → execução (com degradação graciosa) → síntese
que cita fontes. Avaliação de **trajetória** com juiz independente.

**Resultado.** **Roteamento 0,95** em 21 casos (puros, combinado, ambíguo, fora-de-escopo, adversarial); juiz
**rota 2,0/2**. E caracterizamos o trade-off de hardware (7B na GPU vs 3B na CPU) com número — os
**três tools rodam simultaneamente** graças aos 32 GB de RAM.

---

## Fase 5 — MLOps: "não quebrar e não piorar"

**Problema.** IA regride **silenciosamente**. Como garantir que uma mudança não piora a qualidade?
E como levar isso a produção sem gastar?

**Como resolvemos.** Um **gate de avaliação** que lê os relatórios versionados e **reprova o CI** se
qualquer métrica-chave cair; **GitHub Actions** (lint + testes + gate); MLflow + DVC; **drift por
PSI** (corrigido de um sinal enganoso sobre agregado para a coorte estacionária de praças).

**Resultado.** **CI verde** com o gate barrando regressão (12/12 portões); drift **0,005 (estável)**.
O deploy em cloud fica como runbook (decisão de custo); a demo gratuita, pronta no HuggingFace Spaces.

---

## O fio condutor

O diferencial não é ter números altos — é o **rigor ter corrigido os próprios números** em toda
fase: o held-out derrubou a memorização (F2); o backtest+IC derrubaram a cereja (F3); o multi-step
justo revelou a vitória real (F3); a métrica pareada separou artefato de sinal (F4); a coorte
corrigiu o drift enganoso (F5). **Isso** é engenharia de IA a sério: deixar a evidência mandar,
mesmo quando ela contraria a narrativa que seria mais bonita.
