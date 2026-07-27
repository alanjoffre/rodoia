<div align="center">

# 📖 A história do RodoIA

**problema → como resolvi → resultado** — o projeto contado fase a fase.

[← README](../README.md) · [📅 Diário (passo a passo)](DIARIO.md) · [🗺️ Arquitetura](ARQUITETURA.md) · [🎓 Guia didático](GUIA_ENGENHARIA_IA.md) · [📊 métricas cruas](../reports/)

</div>

---

## O ponto de partida

A pergunta que move o projeto: **como provar, com código e número, o perfil completo de um Engenheiro de IA** — do fundamento matemático ao sistema em produção — sobre dados **reais e públicos**? Escolhi o domínio da **ANTT** (regulação e dados abertos do transporte rodoviário brasileiro): real, público e não-trivial.

> **A regra de condução:** uma fase por vez, cada uma um marco publicável, nenhuma começa antes da anterior estar testada.

### Linha do tempo

| Fase | Em uma linha | Resultado-chave |
|:---:|---|---|
| **0** · 🧮 | Entender o motor antes de usá-lo | ROC-AUC **0,81** |
| **1** · 🔎 | Responder a lei da ANTT sem inventar | hit@5 **0,62** · citação **0,92** · κ humano **0,86/0,92** |
| **2** · 🎯 | Especializar um modelo e provar o ganho | F1 **0,13 → 0,77** |
| **3** · 📊 | Do CSV sujo à previsão que convence | Holt-Winters bate o naïve (**Δ3,01pp**) |
| **4** · 🤖 | Juntar tudo e decidir | roteamento **0,95** (n=21) |
| **5** · ⚙️ | Não quebrar e não piorar | gate **27/27** · drift **0,005** · red-team **ASR 0** · **0 CVEs** |
| **6** · 📈 | E se o teste não for meu? | **17,2 M linhas** · CUAD (gold de terceiros) **0,588 → 0,652** (rerank, IC disjunto) · alucinação **1,3%** · DuckDB **6,7×** Spark |

---

## 🧮 Fase 0 — Fundamentos · *"eu entendo o motor?"*

**Problema.** Antes de usar IA, provar que entendo o que está por baixo. E, nos dados: o dataset público de **Acidentes em rodovias concedidas** vinha em **39 CSVs** heterogêneos (latin-1, `;`, decimal `,`), ~1,03 M linhas, sem um alvo de modelagem pronto.

**Como resolvi.** Escrevi **backpropagation e self-attention à mão** (NumPy/PyTorch puro), com prova de equivalência numérica contra o autograd. Consolidei os CSVs (37 concessionárias reconciliadas), derivei o alvo `houve_vitima` e treinei ML clássico + uma MLP com laço de treino manual.

**Resultado.** Modelo de severidade com **ROC-AUC 0,81**, curvas de treino/validação documentadas. Fundamento **provado, não presumido**.

---

## 🔎 Fase 1 — RAG · *"responder com base em documentos reais"*

**Problema.** A legislação da ANTT **não tem API**: HTML em latin-1, normas antigas só em imagem (exigem OCR), e é preciso saber se uma resolução está **vigente**. E responder sem **inventar**.

**Como resolvi.** Pipeline de scraping + limpeza + checagem de vigência; chunking consciente da estrutura jurídica (por artigos); **busca híbrida** (densa E5 + BM25 fundidos por RRF); geração **ancorada** que cita a resolução; e guardrails (anti-injection, PII masking, auditoria). Medi tudo com **juiz LLM independente** e **intervalo de confiança**.

**Resultado.** **hit@5 0,62** [0,48–0,74], **precisão de citação 0,92**.

> ⚖️ **O rigor corrigiu o número — duas vezes.** A limpeza de boilerplate **reverteu** a conclusão do reranker (que passou a *ajudar*, 0,68); e uma **auditoria κ humana** (2 anotadores independentes) **achou 16% dos rótulos-gold do hit@5 errados** — rerotulei pela fonte correta e reportei o impacto (hit@5 real [0,70; 0,76]) ao lado, sem maquiar o número do gate.

---

## 🎯 Fase 2 — Fine-tuning · *"especializar um modelo e provar o ganho"*

**Problema.** Como provar que **fine-tuning agrega**? A primeira tentativa — ensinar o modelo a responder fatos da ANTT — **falhou**: com **held-out**, o ganho aparente virou **memorização** (ia bem no que viu, mal no que não viu). Um resultado negativo, mas honesto.

**Como resolvi.** Em vez de esconder, **pivotei** para uma tarefa de **rótulo objetivo**: NER jurídico sobre o **LeNER-Br** (dado público MIT). Fine-tuning com **QLoRA**, servido em **vLLM** com **quantização fp8** (medindo o custo de qualidade da compressão), comparado contra o teto **BERTimbau (SOTA)**.

**Resultado.** **F1 de entidade 0,13 → 0,77** (base → fine-tunado), encostando no SOTA 0,895 — treinando em **1/5 dos dados** por via generativa.

> ⚖️ **O arco é a entrega:** *negativo rigoroso → pivô → vitória com métrica dura.* Um resultado negativo bem medido vale mais que um positivo inflado.

---

## 📊 Fase 3 — Dados estruturados · *"do CSV sujo à previsão que convence"*

**Problema (o mais rico em dados).** O **Volume de Tráfego de Pedágio** (2010–2026) vinha **sujo**:
- **dois formatos de data** no mesmo dataset (`DD/MM/AAAA` nos anuais, `MM/AAAA` nos consolidados);
- **coluna divergente** entre anos (`categoria` vs `categoria_eixo`);
- **granularidade mista** (alguns anos vêm diários, outros mensais);
- **variantes de caixa** (`Passeio` vs `PASSEIO`) inflando as categorias.

**Como resolvi.** Ingestão robusta que normaliza datas, reconcilia colunas, **trunca ao mês e soma** (série mensal consistente) e padroniza a caixa. Modelagem em **esquema estrela** (DuckDB), SQL analítico (window functions), camada de acesso parametrizada (anti-injection) e previsão de demanda avaliada com **backtest multi-step em 63 praças + IC + teste pareado**.

**Resultado.** **741.205 linhas limpas** (197 meses, 50 concessionárias, 292 praças).

> ⚖️ **O rigor corrigiu o próprio número duas vezes.** Um "MAPE 5,9%" que era **cereja** virou ~13% no backtest; e uma comparação injusta (naïve de 1-passo × Holt-Winters de 12-passos) foi corrigida para **multi-step justo** — aí o **Holt-Winters bate o naïve com significância** (pareado Δ=3,01 pp, IC [1,76; 4,40], vence em 73% das praças).

---

## 🤖 Fase 4 — Agente · *"juntar tudo e decidir"*

**Problema.** Ter três capacidades (RAG, modelo FT, dados) não basta — é preciso um sistema que **decide** qual usar e **combina** as respostas, com segurança e sem cair quando algo falha.

**Como resolvi.** Um grafo **LangGraph** com **arestas condicionais reais**: guardrail → roteador (escolhe as ferramentas, podendo combinar) → execução (com degradação graciosa) → síntese que cita fontes. Avaliação de **trajetória** com juiz independente.

**Resultado.** **Roteamento 0,95** em 21 casos (puros, combinado, ambíguo, fora-de-escopo, adversarial); juiz **rota 2,0/2**. Caracterizei o trade-off de hardware (7B na GPU vs 3B na CPU) com número — os **três tools rodam simultaneamente** graças aos 32 GB de RAM.

> ⚖️ **O rigor corrigiu o juiz.** Ele penalizava "não rotear" nos casos fora-de-escopo/adversarial (onde declinar é o certo). Separar in-scope de declinados **tirou o artefato** — só então o número ficou honesto.

---

## ⚙️ Fase 5 — MLOps · *"não quebrar e não piorar"*

**Problema.** IA regride **silenciosamente**. Como garantir que uma mudança não piora a qualidade? E como levar isso a produção sem gastar?

**Como resolvi.** Um **gate de avaliação** que lê os relatórios versionados e **reprova o CI** se qualquer métrica-chave cair; **GitHub Actions** (lint + testes + gate); MLflow + DVC; **drift por PSI**; e um **modelo de custo R$/1k** derivado da vazão medida.

**Resultado.** **CI verde** com o gate barrando regressão (**15 portões na época, 24 hoje**, 2 deles de segurança: detecção do red-team e vazamento de PII); drift **0,005 (estável)**. O deploy em cloud fica como runbook (decisão de custo); a demo gratuita, **no ar** no HuggingFace Spaces.

> ⚖️ **O rigor corrigiu o drift.** O PSI sobre o volume **agregado** dava ~11 (a malha cresceu ~10×); trocar para a **coorte comum de praças** revelou o valor real — **0,005, estável**.

---

## 📈 Fase 6 — Escala e benchmark externo · *"e se o teste não for meu?"*

**Problema.** Duas objeções que as fases anteriores não conseguiam responder. A primeira: o corpus da ANTT tem **3.647 chunks** — não exerce escala nenhuma. A segunda, mais incômoda: a avaliação toda roda contra **gold que eu mesmo rotulei**. A auditoria κ tratou isso por dentro (Fase 1), mas *"ele rotulou o próprio teste"* só morre com dado de terceiros.

**Como resolvi.** Dois eixos. **Escala:** o bulk da CFPB (EUA) — 1,43 GB de zip → Parquet particionado por streaming, sem materializar os 13,5 GB de CSV intermediários; e o motor de dados escolhido por **benchmark**, não por opinião. **Benchmark externo:** o CUAD — 510 contratos anotados por advogados, com **13.823 spans de gold cujo offset eu conferi um a um** antes de medir qualquer coisa.

**Resultado.** **17.226.584 linhas** ingeridas (13,5 GB → 1,1 GB, ~12:1). No CUAD, o arco completo com IC: BM25 **0,588** → denso **0,535** → híbrido **0,595**. E **DuckDB 6,7× mais rápido que Spark**, com **0 divergências** de resultado.

> ⚖️ **O rigor derrubou a premissa antes do código.** A ideia original era RAG sobre as manifestações de ouvidoria da ANTT. Antes de construir, fui **medir o dado**: o campo `mensagem` do SOU é um **ID numérico de 7 dígitos**, não o texto do cidadão — confirmado pelo dicionário oficial. **A premissa do projeto estava morta, e descobrir isso no dia 0 custou uma tarde em vez de dois meses.**

> ⚖️ **O rigor recusou a conclusão fácil, duas vezes.** (1) O denso **perde** do BM25 no agregado — mas vence exatamente nas categorias de **lacuna lexical** (metadados como *Document Name*), e perde onde o termo é raro e distintivo. Não é "denso é pior": são **forças complementares**, que é o argumento textual do híbrido. (2) O híbrido é o melhor em toda métrica — **mas o IC [0,584; 0,605] sobrepõe o do BM25 [0,577; 0,599]**. O ganho de +0,007 **não é estatisticamente distinguível**, e está reportado assim. Anunciar "o híbrido vence" seria a métrica maquiada que este projeto já reprovou uma vez.

> ⚖️ **O rigor derrubou uma previsão minha.** Eu havia escrito no doc que "um embedder inglês forte quase certamente levantaria o lado denso". Rodei o bge-large-en: **2,8× maior, 7,3× mais lento, e ICs sobrepostos — nenhuma diferença**. A previsão estava errada, e está registrada como errada. O que funcionou foi outra coisa: o **rerank cross-encoder**, único estágio cujo ganho tem **IC disjunto** (0,595 → 0,652).

> ⚖️ **O rigor pegou um viés que era meu.** Ao medir alucinação, o resultado foi **98,7% de não-alucinação** — que sozinho pareceria um triunfo. A segunda taxa desmentiu: **cobertura de 27%**, ou seja, o sistema recusava 3 de cada 4 perguntas que *tinham* resposta, ficando a 0,13 de um modelo que responde "não consta" a tudo. E parte da culpa era **do meu prompt**: suavizá-lo recuperou 10 pontos de cobertura **sem custo nenhum** de alucinação.

**O fecho do arco:** partindo do zero sobre um benchmark que não é meu, com IC em cada passo, a sequência BM25 → denso → híbrido → **rerank** **re-derivou a arquitetura da Fase 1** peça por peça — e mostrou por que o rerank é o estágio que faltava: RRF é **consenso, não seletor**, e compromete justamente onde os dois recuperadores discordam forte. Onde o RRF entregava *menos* que o BM25 sozinho (`Effective Date`, 0,689 → 0,567), o cross-encoder recupera e **supera** (0,776).

---

## 🧵 O fio condutor

O diferencial não é ter números altos — é o **rigor ter corrigido os próprios números** em toda fase (os callouts ⚖️ acima marcam cada correção). O apanhado completo — com o antes/depois de cada uma — vive numa fonte só: **[README § Decisões e trade-offs](../README.md#-decisões-e-trade-offs-o-arco-do-projeto)**.

**Isso** é engenharia de IA a sério: deixar a evidência mandar, **mesmo quando ela contraria a narrativa que seria mais bonita**.
