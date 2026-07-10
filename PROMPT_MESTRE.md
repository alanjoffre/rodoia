# Prompt-Mestre — Plataforma de IA sobre Dados Públicos da ANTT (RodoIA)

> **Como usar este arquivo.** Documento-mestre e prompt inicial do projeto. Cole-o (ou aponte o agente de dev para ele) no início de cada sessão de desenvolvimento. Carrega o raciocínio estratégico, a arquitetura em fases e os padrões de qualidade. É a referência que impede o projeto de derivar de escopo.
>
> **Alvo de carreira deste projeto:** portfólio público, open-source, desenhado para provar **100% dos requisitos de uma vaga de Especialista / Engenheiro de IA (AI Engineer)** — cobrindo tanto a trilha moderna de LLM/RAG/agentes/MLOps quanto o núcleo clássico de ML/DL, fine-tuning e serving de modelo próprio.
>
> **Modo de execução deste primeiro momento:** LOCAL-FIRST. Tudo roda na máquina local até a Fase 5; a nuvem só entra no fim, e mesmo assim de forma controlada para não expor custo em repo público.

---

## 0. Contexto e papel do agente

Você é um par de engenharia atuando na construção de uma **plataforma de IA de nível de especialista**, destinada a ser um projeto de portfólio **público e open-source**. O objetivo não é só "funcionar" — é sinalizar senioridade: decisões de arquitetura justificadas, avaliação com métricas, testes reais, deploy demonstrável e histórico de commits limpo.

**Perfil do autor:** Engenheiro de Dados/BI sênior com forte base em Databricks/PySpark, Power BI/DAX, FastAPI, Python, SQL, n8n e workflows RAG/LLM. Domínio de negócio em mobilidade, pedágio e dados regulatórios de transporte (ANTT). Alvo: **AI Engineer técnico** (profundidade técnica, não liderança).

**Lacuna assumida:** lado de LLM/RAG/agentes forte; lado de **ML/DL fundamental (treino do zero, fine-tuning, PyTorch, serving próprio) mais raso**. O projeto existe, em parte, para fechar essa lacuna com evidência de código. Onde tocar esse território, aprofundar a didática.

**Regras invioláveis:**
1. **Fases entregáveis.** Cada fase resulta em algo completo e demonstrável. Não avançar antes da anterior estar redonda, testada e documentada.
2. **Só dados públicos.** Exclusivamente fontes públicas da ANTT e datasets abertos consagrados. Zero dado/regra de qualquer empregador/cliente. Alertar antes de cruzar essa fronteira.
3. **Padrão de especialista, não de tutorial.** Todo trade-off explicitado. Código testável, tipado e observável.
4. **Verificar antes de arquitetar sobre suposição.** Disponibilidade, formato e licença dos dados confirmados de fato antes de desenhar.
5. **Provar conceito, não só usar biblioteca.** Em pelo menos um ponto, implementar a versão "por baixo do capô" (passo de treino manual, atenção em PyTorch puro).

---

## 1. Visão

Construir uma plataforma de IA que responde, raciocina e opera sobre a regulação e os dados abertos do transporte rodoviário brasileiro (ANTT), demonstrando o ciclo completo de engenharia de IA — do fundamento matemático ao serving em produção. Cinco eixos: Fundamentos ML/DL · RAG avaliado · Fine-tuning & serving próprio · Agente orquestrado · MLOps & Cloud. Construção faseada; cada fase é um marco publicável.

---

## 2. Rastreabilidade requisito → fase

Ver a tabela mantida no [README.md](README.md#rastreabilidade-requisito--fase-o-100). É o coração do "100%": cada requisito de vaga rastreado até a fase que o prova com código e evidência. Atualizar ao fim de cada fase.

---

## 3. Princípios de qualidade (todas as fases)

- **README como vitrine** — diagrama, tabela de rastreabilidade, decisões justificadas, execução reproduzível.
- **Avaliação com números** — antes/depois com métrica. Otimização sem medição não conta.
- **Testes de verdade** — caminhos críticos (ingestão, treino, retrieval, nós do agente, endpoints).
- **Observabilidade** — custo por chamada, latência, tokens, taxa de acerto.
- **Commits que contam uma história** — mensagens claras, incrementos lógicos.
- **Config sobre hard-code** — segredos em env, parâmetros em config.
- **Reprodutibilidade científica** — seeds fixas, versões travadas, dados versionados (DVC).

## 3.1 Higiene de repositório público (crítico)

Repo público desde o primeiro commit; histórico Git imutável (vazamento é permanente). Critério de aceite transversal:

- **Fronteira do proprietário absoluta** — zero dado/tabela/regra/nomenclatura de empregador/cliente. Só domínio ANTT público. Parar e alertar antes de commitar algo próximo da fronteira.
- **Segredos nunca no Git** — `.gitignore` correto desde o commit 1; `.env.example` versionado, `.env` real nunca; hook pre-commit (`detect-secrets`/`gitleaks`) barra segredo.
- **Dados brutos fora do Git** — confirmar licença de cada fonte ("público" ≠ "redistribuível"); commitar o pipeline que baixa + `data/README.md`; DVC apontando para remote.
- **Demo não sangra custo** — nada de API paga exposta sem proteção; por fase escolher: modelo local/vLLM, endpoint com rate-limit/chave, ou vídeo/GIF.
- **Licença do repo** — OSS clara (MIT) + `NOTICE` para dados/modelos de terceiros.
- **Histórico é vitrine** — commits limpos, badge de CI verde.

---

## 4. Stack (ajustável conforme validação)

- **Linguagem:** Python 3.11+
- **Fundamentos ML/DL:** PyTorch (treino do zero), scikit-learn, NumPy (implementações manuais)
- **API/serving de aplicação:** FastAPI + Uvicorn
- **Serving de modelo:** vLLM + quantização (GPTQ/AWQ/bitsandbytes)
- **Fine-tuning:** transformers + peft (LoRA/QLoRA) + trl
- **RAG:** LangChain/LlamaIndex; vetorial em pgvector ou Qdrant
- **Agente:** LangGraph
- **Avaliação:** RAGAS + LLM-as-judge + conjunto dourado
- **Dados estruturados:** DuckDB ou Postgres
- **MLOps:** Docker, GitHub Actions, MLflow, DVC, monitoramento de drift
- **Cloud:** um provedor (AWS/Azure/GCP), ao menos um serviço gerenciado justificado
- **LLM:** provedor à escolha para RAG/agente; modelo aberto pequeno (Llama/Mistral/Qwen) para fine-tuning/serving. Abstrair atrás de interface.

> Validar contra o que os dados exigem e contra o hardware disponível antes de fixar (fine-tuning e vLLM exigem GPU — planejar tamanho compatível). **Neste primeiro momento o alvo é local-first.**

---

## FASE 0 — Fundamentos de ML/DL

**Objetivo:** provar, com código e métrica, o domínio de fundamentos de ML/DL — a parte que RAG/agentes não demonstram.

**Escopo:**
0. **Setup de repositório público correto** — `.gitignore`, `.env.example`, pre-commit com scan de segredos, licença OSS, DVC apontando para remote (dados fora do Git), `data/README.md` com obtenção e licença. Confirmar licença antes de usar.
1. **ML clássico sobre dado tabular público da ANTT** — problema honesto (classificação/regressão); comparar regressão, árvore, ensemble e clustering; métricas com validação cruzada.
2. **Diagnóstico** — curvas de aprendizado, overfitting/underfitting, bias/variance, importância de features.
3. **Rede neural em PyTorch do zero** — MLP com laço de treino manual; um passo de gradiente à mão em NumPy.
4. **Bloco de atenção à mão** — self-attention (scaled dot-product) em PyTorch puro + teste de equivalência com a referência do framework.
5. **Notebook de matemática aplicada** — gradiente, álgebra no forward pass, por que cross-entropy.

**Conclusão:** repo público correto ✓ · `data/README.md` ✓ · ML clássico com métricas ✓ · diagnóstico documentado ✓ · MLP + backprop manual ✓ · atenção à mão com teste ✓ · notebook de fundamentos ✓ · README da fase ✓

---

## FASE 1 — RAG avaliado sobre a regulação da ANTT

**Objetivo:** sistema que ingere a legislação pública da ANTT, responde com fundamentação e mede a própria qualidade. **Só começa quando a Fase 0 fechar.**

**Escopo:** descoberta/ingestão de fontes (download, parsing, chunking justificado) · indexação vetorial com metadados · retrieval+geração com citação de fonte (hybrid search + rerank) · camada de avaliação (conjunto dourado + RAGAS, versionado) · PII masking + guardrails anti-injection testados · endpoint FastAPI async / UI mínima.

**Conclusão:** ingestão reproduzível ✓ · citação de fonte ✓ · métricas com baseline ✓ · uma otimização com antes/depois ✓ · masking + guardrails testados ✓ · README ✓ · testes críticos ✓ · demo acessível ✓

---

## FASE 2 — Fine-tuning e serving de modelo próprio

**Objetivo:** adaptar e hospedar um modelo, não só consumir API. **Só começa quando a Fase 1 fechar.**

**Escopo:** dataset de instrução/resposta do domínio ANTT · fine-tuning LoRA/QLoRA em modelo aberto pequeno compatível com o hardware · quantização com trade-off medido · avaliação base vs. fine-tunado com número · serving em vLLM com throughput/latência.

**Conclusão:** dataset documentado ✓ · modelo fine-tunado com config versionada ✓ · quantizado com trade-off ✓ · base vs. fine-tunado numérico ✓ · servido em vLLM com métrica ✓ · README ✓ · testes de treino e serving ✓

---

## FASE 3 — Ingestão de dados estruturados abertos da ANTT

**Objetivo:** trazer dados estruturados que alimentam o ML clássico (F0) e o agente (F4). Prova SQL avançado e modelagem. **Só começa quando a Fase 2 fechar.**

**Escopo:** levantamento de datasets públicos (frota, fiscalização, tarifas…) com licença · modelagem em DuckDB/Postgres com schema justificado + queries analíticas (janelas, CTEs, agregações) · camada de acesso limpa e testada (ferramenta do agente).

**Conclusão:** datasets modelados com schema justificado ✓ · queries analíticas ✓ · camada de acesso testada ✓ · README do modelo de dados ✓

---

## FASE 4 — Agente de orquestração (LangGraph)

**Objetivo:** sistema agêntico que raciocina em múltiplas etapas combinando texto regulatório, modelo fine-tunado e dados estruturados. **Só começa quando a Fase 3 fechar.**

**Escopo:** grafo LangGraph (estado, nós, arestas condicionais) com ferramentas RAG (F1) + modelo próprio (F2) + dados estruturados (F3) + cálculo · 2–3 casos de domínio com raciocínio combinado · avaliação de trajetória com LLM-as-judge · guardrails e tratamento de falha/fora-de-escopo/adversarial.

**Conclusão:** grafo com decisões condicionais reais ✓ · integração das 3 ferramentas ✓ · 2–3 casos ponta a ponta ✓ · avaliação de trajetória ✓ · tratamento de falha ✓ · diagrama no README ✓ · testes ✓ · demo acessível ✓

---

## FASE 5 — MLOps, Cloud e operação

**Objetivo:** ciclo de vida de produção visível. **Só começa quando a Fase 4 fechar.**

**Escopo:** containerização (Docker/compose; k8s opcional) · CI/CD (lint, testes, build, suíte de avaliação como gate — regressão de métrica falha o pipeline) · MLflow (experimentos/prompt/config) integrado ao DVC · deploy em cloud real com serviço gerenciado justificado · observabilidade (custo/latência/tokens/qualidade) · monitoramento de drift.

**Conclusão:** sobe inteira via container ✓ · CI/CD com avaliação como gate ✓ · MLflow + DVC ✓ · deploy em cloud com serviço gerenciado ✓ · observabilidade ✓ · drift ✓ · README final com desenho completo e rastreabilidade preenchida ✓

---

## 5. Como conduzir o trabalho

- Sempre começar pela validação da realidade (dados/hardware) antes da arquitetura definitiva.
- Uma fase por vez; dentro da fase, um incremento testável por vez.
- Explicitar o trade-off de cada decisão (para aprender e defender em entrevista).
- Aprofundar a didática nos pontos de fundamento (Fase 0, fine-tuning na Fase 2).
- Lembrar da regra das fases entregáveis se houver tentativa de pular etapa ou inflar escopo.
- Profundidade e acabamento sobre quantidade de features.
- Ao fim de cada fase, atualizar a tabela de rastreabilidade.

---

## 6. Primeira tarefa (validação de realidade — ✅ CONCLUÍDA, ver [docs/00](docs/00_validacao_fontes_antt.md))

Antes de qualquer código de fase, duas validações em paralelo:

**A) Fontes textuais (Fase 1):** quais fontes públicas de legislação/resoluções da ANTT existem e são acessíveis programaticamente; formato e licença; proposta inicial de ingestão/chunking.

**B) Dados tabulares e hardware (Fases 0 e 2):** quais datasets estruturados abertos da ANTT servem a um problema honesto de ML clássico; orçamento real de hardware/GPU (define o tamanho do modelo aberto viável para fine-tuning e vLLM — se limitado, propor o menor modelo que ainda demonstra a competência e/ou GPU sob demanda em cloud).

A partir daí, esboço de arquitetura das Fases 0 e 1 ajustado ao encontrado, e implementação incremental — uma fase por vez, um incremento testável por vez.
