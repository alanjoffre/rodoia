<div align="center">

# 🎓 Guia didático do RodoIA

**O que este projeto faz, por quê, e o trabalho de um Engenheiro de IA — explicado sem jargão.**

[← README](../README.md) · [📖 A história](HISTORIA.md) · [🗺️ Arquitetura](ARQUITETURA.md)

</div>

> **Para quem é:** quem quer entender **o que** este projeto faz e **por quê**, sem ser da área. Cada termo vem com uma **analogia simples** e o **porquê** de existir. Para o mapa técnico dos arquivos, ver [ARQUITETURA.md](ARQUITETURA.md).

---

## 1. O que é Engenharia de IA (em uma frase)

> Engenharia de IA é **pegar modelos de inteligência artificial e transformá-los num sistema que funciona de verdade** — com dados reais, medição de qualidade, custo controlado e funcionando em produção.

Não é "chamar a API da OpenAI"; é construir o encanamento inteiro, do fundamento matemático até o sistema no ar sendo monitorado. O RodoIA prova esse ciclo completo usando dados públicos da **ANTT** (a agência que regula o transporte rodoviário no Brasil).

---

## 2. O fluxo mental de um Engenheiro de IA

Pense num médico que primeiro entende o paciente, depois escolhe o exame, depois o tratamento, e por fim acompanha a recuperação. O Engenheiro de IA segue um fluxo parecido. Cada **fase** do RodoIA é uma etapa desse fluxo:

| Etapa | Pergunta que responde | No RodoIA |
|:---:|---|---|
| **0** · 🧮 Fundamentos | "Eu entendo o motor por dentro?" | Rede neural e atenção escritas à mão |
| **1** · 🔎 Dar fontes (RAG) | "Como respondo com base em documentos reais?" | Busca na legislação da ANTT |
| **2** · 🎯 Ensinar habilidade (fine-tuning) | "Como especializo o modelo numa tarefa?" | Extrair entidades de textos jurídicos |
| **3** · 📊 Lidar com números (dados) | "Como trabalho com tabelas, não só texto?" | Prever o volume de tráfego nas praças |
| **4** · 🤖 Juntar tudo (agente) | "Como faço o sistema decidir e combinar ferramentas?" | Um agente que escolhe qual ferramenta usar |
| **5** · ⚙️ Colocar em produção (MLOps) | "Como garanto que não quebra e não piora?" | Testes automáticos que barram regressão |

> **A ideia-chave:** cada etapa é um marco que vale por si, e nenhuma começa antes da anterior estar testada. É assim que um profissional constrói — **em camadas verificáveis**, não tudo de uma vez.

---

## 3. Glossário vivo — os termos, com analogia e o *porquê*

### 💬 Sobre modelos de linguagem

- **Modelo / LLM (Large Language Model)** — um programa treinado em bilhões de textos que prevê a próxima palavra. *Analogia:* um autocompletar gigante e muito bom. *Por quê:* é o "cérebro" que gera respostas.
- **Token** — pedaço de palavra que o modelo processa (ex.: "pedágio" pode virar "ped"+"ágio"). *Por quê:* velocidade e custo se medem em **tokens/segundo**.
- **Prompt** — o texto que você dá ao modelo como instrução. *Analogia:* o pedido que você faz ao garçom. Engenharia de prompt = fazer o pedido do jeito certo.

### 🔎 RAG — dar fontes ao modelo (Fase 1)

- **RAG (Retrieval-Augmented Generation)** — antes de responder, o sistema **busca trechos relevantes** num acervo e responde **com base neles**. *Analogia:* um advogado que consulta o código antes de opinar, em vez de responder de cabeça. *Por quê:* reduz invenção e permite citar a fonte.
- **Embedding / vetor** — transformar um texto numa lista de números que captura seu **significado**. Textos parecidos ficam com vetores próximos. *Analogia:* dar a cada frase uma "coordenada de GPS" do sentido. *Por quê:* permite buscar por significado, não só por palavra igual.
- **Chunk** — um pedaço do documento (ex.: um artigo da resolução). *Por quê:* o modelo lê melhor trechos curtos e focados do que documentos inteiros.
- **Busca densa vs. BM25** — densa = por significado (vetores); BM25 = por palavra-chave (clássica). *Por quê:* cada uma pega o que a outra perde.
- **Busca híbrida / RRF** — combinar as duas buscas numa lista só (RRF = a regra de fusão). *Analogia:* juntar a opinião de dois especialistas diferentes. *Por quê:* costuma achar mais.
- **Reranker** — um segundo modelo que reordena os resultados por relevância. *Por quê:* refina o topo da lista. *No RodoIA:* medimos que **antes não ajudava** (com o corpus sujo) e, depois de **limpar o boilerplate**, a conclusão **reverteu** — ele passou a ajudar (hit@5 0,62 → 0,68). Deixar a evidência mandar, mesmo quando muda a decisão.
- **hit@5** — "a resposta certa apareceu entre os 5 primeiros trechos?" (fração de acertos). **MRR** — quão no topo ela apareceu. *Por quê:* medem a qualidade da busca com número.

### 🛡️ Avaliação e segurança

- **LLM-as-judge** — usar **outro** modelo para dar nota às respostas. *Analogia:* um segundo professor corrigindo a prova. *Por quê:* avaliar em escala sem humano em tudo. Usamos um juiz **independente** (modelo diferente) para não "puxar sardinha".
- **κ inter-anotador (kappa)** — quando **dois humanos** rotulam a mesma coisa, o κ mede se eles concordam de verdade (descontando a sorte). *Analogia:* dois juízes de ginástica que dão a mesma nota. *Por quê:* prova que o rótulo é confiável, não a opinião de uma pessoa só. *No RodoIA:* essa auditoria κ **achou 16% dos rótulos-gold errados** — e nós corrigimos e reportamos.
- **Faithfulness / hallucination** — faithfulness = a resposta se apoia nas fontes; hallucination ("alucinação") = o modelo inventou. *Por quê:* inventar é o maior risco de um sistema de IA.
- **Guardrail** — trava de segurança. Ex.: bloquear **prompt injection** (quando o usuário tenta "ignore as instruções e faça outra coisa"). *Por quê:* sistema público precisa se defender.
- **PII** — dados pessoais (CPF, e-mail...). *Por quê:* mascaramos por LGPD.
- **Intervalo de confiança (IC)** — a faixa onde o valor real provavelmente está (ex.: hit@5 **0,62**, entre **0,48 e 0,74**). *Analogia:* a margem de erro de uma pesquisa eleitoral. *Por quê:* com poucos exemplos, um número sozinho engana; a faixa é honesta.

### 🎯 Fine-tuning — ensinar uma habilidade (Fase 2)

- **Fine-tuning** — pegar um modelo pronto e **treiná-lo mais um pouco** numa tarefa específica. *Analogia:* um clínico geral fazendo residência em cardiologia. *Por quê:* especializa sem treinar do zero (o que custaria milhões).
- **LoRA / QLoRA** — técnicas para fazer o fine-tuning **barato**, ajustando só uma fração do modelo. *Analogia:* trocar as marchas do carro sem refazer o motor. *Por quê:* cabe numa GPU pequena.
- **Quantização (fp8 / NF4)** — comprimir os números do modelo para ocupar menos memória. *Analogia:* salvar uma foto em qualidade menor pra caber no celular. *Por quê:* servir o modelo em hardware modesto — medimos o **custo de qualidade** dessa compressão.
- **Perplexidade** — o quanto o modelo fica "surpreso" com um texto (menor = mais familiar). *Por quê:* mede se o fine-tuning aproximou o modelo do jargão da ANTT.
- **Held-out / memorização** — separar exemplos que o modelo **nunca viu** para testar. Se ele vai bem no que viu mas mal no que não viu, **decorou** em vez de aprender. *Por quê:* é o teste que revela a verdade — e no RodoIA ele **derrubou** um resultado que parecia bom.
- **NER (Named Entity Recognition)** — achar entidades num texto (pessoas, leis, datas, órgãos). *Por quê:* foi a tarefa **objetiva** onde o fine-tuning venceu com número duro.
- **F1 / SOTA** — F1 = nota que equilibra acerto e cobertura; SOTA = "state of the art", o melhor resultado conhecido. *Por quê:* comparamos nosso modelo contra o SOTA para ser honesto sobre o nível.
- **vLLM** — servidor que roda o modelo de forma rápida e eficiente. *Por quê:* é o "motor de entrega" das respostas em produção.

### 📊 Dados estruturados (Fase 3)

- **Esquema estrela / DuckDB** — jeito de organizar tabelas (um "fato" central + "dimensões" ao redor) num banco rápido. *Analogia:* a planilha central de vendas ligada a tabelas de produto, loja e data. *Por quê:* deixa as perguntas analíticas (SQL) simples e baratas.
- **Window function (SQL)** — cálculo que olha "as linhas vizinhas" (ex.: crescimento vs. o mês anterior). *Por quê:* responde perguntas de tendência sem código extra.
- **Série temporal / previsão** — prever o futuro a partir do histórico (ex.: volume dos próximos 12 meses). **MAPE** = erro percentual médio da previsão (menor = melhor).
- **Baseline naïve** — a previsão "burra" de referência (ex.: "vai ser igual ao ano passado"). *Por quê:* se o modelo sofisticado não bate o naïve, ele não vale a pena. Um bom engenheiro **sempre** compara contra o naïve.
- **Backtest / teste pareado** — testar a previsão no passado, praça por praça, e comparar de forma justa. *Por quê:* evita "escolher a dedo" um caso bom (cereja) e mentir para si mesmo.
- **Holt-Winters** — método clássico de previsão que captura tendência e sazonalidade. *Por quê:* no RodoIA, ele **bateu o naïve com significância** — o resultado que convence.

### 🤖 Agente (Fase 4)

- **Agente** — um sistema de IA que **decide** quais passos e ferramentas usar para responder, em vez de seguir um roteiro fixo. *Analogia:* um assistente que decide se liga pro contador, pro advogado ou olha a planilha — conforme a pergunta.
- **LangGraph** — a "planta baixa" do agente: nós (etapas) e setas (decisões). *Por quê:* deixa o fluxo explícito, testável e com **decisões condicionais reais**.
- **Roteamento / trajetória** — roteamento = escolher a ferramenta certa; trajetória = o caminho que o agente percorreu. *Por quê:* avaliamos se ele roteia certo e se a resposta se apoia nas evidências.
- **Degradação graciosa** — se uma ferramenta falha, o agente **não cai**: registra e segue. *Por quê:* robustez é requisito de produção.

### ⚙️ MLOps — produção e operação (Fase 5)

- **MLOps** — as práticas para levar IA a produção e mantê-la saudável. *Analogia:* não basta construir o avião; precisa de manutenção, checklist e torre de controle.
- **CI/CD** — automação que, a cada mudança, roda **lint** (estilo), **testes** e um **gate**. *Por quê:* impede que código quebrado ou pior entre no projeto.
- **Gate de avaliação** — um teste que **reprova** a mudança se uma métrica-chave **piorar**. *Analogia:* o detector de metal do aeroporto para a qualidade do modelo. *Por quê:* IA pode regredir silenciosamente; o gate barra isso.
- **MLflow / DVC** — MLflow guarda o histórico de experimentos/métricas; DVC versiona dados e modelos (que são grandes demais pro Git). *Por quê:* reprodutibilidade e rastreio.
- **Drift / PSI** — drift = o mundo mudou e o modelo ficou desatualizado; PSI = a métrica que mede esse desvio. *Analogia:* a bússola que avisa quando o terreno mudou e é hora de recalibrar.
- **Custo por 1k requisições** — quanto custa servir mil respostas, calculado da **vazão medida** (não chutado). *Por quê:* um engenheiro sênior não diz "escala" sem a cifra.
- **Proveniência / seed** — proveniência = carimbar cada resultado com versão/data/código que o gerou; seed = fixar a aleatoriedade para o experimento ser **repetível**. *Por quê:* ciência de verdade é reproduzível.

---

## 4. Como esse fluxo aparece no RodoIA (resumo)

1. **Fundamentos** → provamos que entendemos o motor (rede neural à mão).
2. **RAG** → o sistema responde sobre a legislação **citando a fonte**, e medimos a qualidade.
3. **Fine-tuning** → especializamos um modelo para NER jurídico e provamos o ganho com F1.
4. **Dados** → modelamos as tabelas e prevemos a demanda, batendo o baseline de forma honesta.
5. **Agente** → um orquestrador decide entre as 3 ferramentas acima e combina as respostas.
6. **MLOps** → tudo isso roda com testes e um gate no CI que barra regressão.

> **O diferencial não são só os números altos — é o rigor ter corrigido os próprios números** ao longo do caminho (ver a seção "Decisões e trade-offs" no [README](../README.md), e os callouts ⚖️ na [história](HISTORIA.md)).

---

## 5. Por onde começar a ler

| Se você quer… | Vá para |
|---|---|
| A visão geral e os resultados | [README.md](../README.md) |
| A narrativa fase a fase | [HISTORIA.md](HISTORIA.md) |
| O mapa técnico dos arquivos | [ARQUITETURA.md](ARQUITETURA.md) |
| Cada fase em detalhe | `docs/00` a `docs/16` |
| A governança do modelo e dos dados | [MODEL_CARD.md](MODEL_CARD.md) · [DATASET_CARD.md](DATASET_CARD.md) |
