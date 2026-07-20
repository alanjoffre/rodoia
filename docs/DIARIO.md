<div align="center">

# 📅 O diário do RodoIA

**A construção contada linha a linha** — os 81 passos reais do projeto, na ordem em que aconteceram, cada um em linguagem clara.

[← README](../README.md) · [📖 A história (por fase)](HISTORIA.md) · [🗺️ Arquitetura](ARQUITETURA.md) · [🎓 Guia didático](GUIA_ENGENHARIA_IA.md)

</div>

---

## Como ler este diário

Se o [HISTORIA.md](HISTORIA.md) conta **o "porquê"** de cada fase, este diário conta **o "o quê", passo a passo**. Cada item abaixo é um passo real e datado da construção (um *commit* no histórico do código): o que fiz naquele momento e por que importava.

São **81 passos**, feitos entre **9 e 17 de julho de 2026**. Lidos em sequência, mostram como o projeto saiu do zero e chegou a um sistema completo — e, principalmente, como o **rigor** corrigiu os próprios números várias vezes pelo caminho.

**Legenda das fases:** 🧮 Fase 0 (fundamentos) · 🔎 Fase 1 (RAG) · 🎯 Fase 2 (fine-tuning) · 📊 Fase 3 (dados) · 🤖 Fase 4 (agente) · ⚙️ Fase 5 (MLOps) · 🔬 rigor/auditoria · 🚀 deploy · 🎨 apresentação

---

## 🗓️ 9 de julho — do zero à Fase 1 inteira

*O dia mais denso: montei a fundação, provei a matemática à mão e já entreguei o primeiro sistema de perguntas-e-respostas sobre a lei.*

1. **🧮 Montei o repositório público.** Estrutura de pastas, licença, ambiente Python — o esqueleto onde tudo seria construído, já pensado para ser aberto e reprodutível.
2. **🔎 Validei que as fontes da ANTT são reais.** Antes de escrever qualquer código de dados, conferi que os dados públicos existem, abrem e batem com a realidade — para não construir sobre uma fonte imaginária.
3. **🧮 Baixei os dados de acidentes.** Um programa que busca os 39 arquivos CSV de acidentes em rodovias concedidas, lê e valida cada um (encoding, separadores, colunas).
4. **🧮 Consolidei os dados e treinei o primeiro modelo.** Juntei os 39 arquivos numa base única (37 concessionárias reconciliadas) e treinei um classificador simples de "houve vítima?" — o ponto de partida (*baseline*).
5. **🧮 Diagnostiquei o modelo.** Medi se ele erra por falta de capacidade ou por excesso (viés/variância), se as probabilidades são confiáveis (calibração) e agrupei os dados (*clustering*) para entender padrões.
6. **📋 Escrevi o runbook.** Um guia passo a passo do "clonei o projeto" até "rodei a Fase 0", para qualquer pessoa reproduzir.
7. **🧮 Escrevi a rede neural à mão.** Implementei *backpropagation* do zero em NumPy (sem biblioteca fazer a conta por mim) e uma rede MLP em PyTorch — para provar que entendo o motor, não só usá-lo.
8. **🧮 Escrevi o *self-attention* à mão.** O mecanismo que está por trás dos modelos modernos de linguagem, implementado do zero e conferido contra a versão oficial (mesmo resultado numérico — prova de que está correto).
9. **🧮 Fechei a Fase 0.** Reuni a matemática aplicada num notebook e encerrei o bloco de fundamentos: **modelo de severidade com ROC-AUC 0,81**.
10. **🔎 Ingeri a legislação da ANTT.** Baixei e limpei as normas (a lei não tem API — é HTML antigo) e as quebrei em pedaços por artigo (*chunking*), respeitando a estrutura jurídica.
11. **🔎 Indexei e busquei com citação.** Transformei os trechos em vetores para busca semântica, com filtro de **vigência** (só normas em vigor) e citação da resolução-fonte.
12. **🔎 Busca híbrida.** Combinei busca por significado (densa) + por palavra exata (BM25), fundidas numa só ordenação (RRF), com um reordenador (*reranker*) e a primeira avaliação.
13. **🔎 Geração de resposta ancorada.** O modelo de linguagem (rodando localmente, via Ollama) passa a **responder citando a norma** — sem inventar.
14. **🔎 Guardrails de segurança.** Defesa contra injeção de comando, mascaramento de dados pessoais (PII) e trilha de auditoria.
15. **🔎 Avaliação por juiz-LLM.** Um segundo modelo avalia as respostas quanto a fidelidade à fonte e relevância.
16. **🔎 Fechei a Fase 1.** API web (FastAPI) + interface de demonstração no ar. O sistema de RAG está completo.
17. **🎯 Montei o dataset de fine-tuning.** 84 exemplos a partir de 29 normas — o material para especializar um modelo.
18. **🎯 Preparei os scripts para a GPU.** Roteiros de treino (QLoRA), quantização e *serving*, empacotados num "handoff" auto-contido para rodar na máquina com placa de vídeo (a "Nitro").

---

## 🗓️ 10 de julho — o fine-tuning na GPU e a primeira onda de rigor

*Rodei o treino de verdade na placa de vídeo e, logo em seguida, voltei em cada fase para endurecer a avaliação.*

19. **🎯 Rodei o treino na Nitro.** Fine-tuning com QLoRA, quantização fp8, serving com vLLM e avaliação — executados na máquina com GPU.
20. **🎯 Avaliei o modelo treinado com rigor.** Medi perplexidade no domínio e taxa de vitória (*win-rate*) julgada por um modelo independente.
21. **🔬 Controlei o viés do juiz.** Ajustei o win-rate para neutralizar preferências do avaliador e auditei os documentos, corrigindo inconsistências.
22. **🔬 Fechei lacunas de rigor na Fase 0.** Adicionei provas, testes do laço de treino, divisão de dados em 3 partes e rastreabilidade da origem dos dados.
23. **🔬 Fechei lacunas de rigor na Fase 1.** Observabilidade, juiz independente, conjunto-ouro (*golden set*) com intervalo de confiança e defesa contra injeção indireta.
24. **🔬 2ª rodada de rigor na Fase 2.** Teste com dados que o modelo nunca viu (*held-out* real), intervalo de confiança, benchmark reprodutível e proveniência.
25. **🔬 Resolvi ressalvas da Fase 2.** Juiz factual com referência e medição do custo de qualidade da compressão (quantização) do modelo.
26. **🔬 Onda 2 de rigor.** Expandi o golden set (para 50 casos) e o dataset de fine-tuning (para 158) e reavaliei com amostra maior.

---

## 🗓️ 11 de julho — o pivô de NER e as Fases 3, 4 e 5

*O dia em que um resultado negativo virou vitória, e em que as três últimas fases nasceram.*

27. **🎯 Estudo de NER no LeNER-Br.** Montei o reconhecimento de entidades jurídicas sobre um dataset público e medi o teto de qualidade: BERTimbau, F1 0,895.
28. **🔬 Corrigi a consistência (punch-list).** Acertei números que estavam desencontrados entre documentos (throughput, divisões de dados, licenças).
29. **🎯 O NER vira a manchete — o pivô.** A ideia original de fine-tuning falhava por memorização; pivotei para a tarefa objetiva de NER e **o modelo treinado venceu: F1 0,13 → 0,77**. Um negativo honesto virou entrega.
30. **🔬 Validação pós-pivô.** Corrigi 10 itens de revisão (navegação, contagens, afirmações exageradas) para o texto refletir o novo rumo.
31. **📊 Fase 3 — dados estruturados.** Modelei o volume de tráfego de pedágio num esquema estrela (DuckDB), com SQL analítico e a primeira previsão de demanda.
32. **📊 Previsão robusta.** Troquei a avaliação ingênua por *backtest* com intervalo de confiança e modelo Holt-Winters, e padronizei a caixa dos textos (Passeio vs PASSEIO).
33. **📊 Previsão multi-step justa.** Corrigi uma comparação injusta: passei a prever 12 meses à frente para todos os modelos e apliquei teste estatístico pareado.
34. **🤖 Fase 4 — o agente.** Um orquestrador (LangGraph) que decide qual capacidade usar — RAG, modelo treinado ou dados — e combina as respostas.
35. **⚙️ Fase 5 — MLOps.** Portão de avaliação, integração contínua (CI/CD), rastreamento de experimentos (MLflow), versionamento de dados (DVC), container e detecção de *drift*.
36. **🔬 Correções da revisão final.** Auditoria de consistência de ponta a ponta.
37. **🚀 Guia, arquitetura e demo grátis.** Escrevi o guia didático, o mapa de arquitetura, os *cards* de modelo/dataset e publiquei a demo gratuita no HuggingFace Spaces.

---

## 🗓️ 12 de julho — medindo o hardware e escrevendo a história

38. **🤖 Trade-off de hardware, medido.** Caracterizei com número a escolha entre cérebro de 7B na GPU e 3B na CPU — e mostrei que os três tools rodam ao mesmo tempo graças aos 32 GB de RAM.
39. **📖 Escrevi o HISTORIA.md.** A narrativa problema → solução → resultado, fase a fase.
40. **🔬 Correções da auditoria.** Acertos de consistência no README e nos documentos novos.

---

## 🗓️ 13 de julho — o especialista entra em cena (o dia da qualidade)

*Criei um revisor de padrão extremo e passei o dia inteiro respondendo aos furos que ele apontava — mais o deploy da demo.*

41. **🔬 Criei o subagente "Especialista de Engenharia de IA".** Um revisor automático calibrado no padrão mais alto, para caçar fraquezas no projeto.
42. **🔬 QA-1 + QA-2.** Apliquei as primeiras correções do especialista e reproduzi os resultados de verdade, do zero.
43. **🔬 QA-3 — honestidade sobre limites.** Em vez de inflar, **assumi** a limitação da avaliação onde ela existia.
44. **🤖 QA-4 — roteamento avaliado a sério.** Aumentei a avaliação do agente de 6 para **21 casos** (puros, combinados, ambíguos, fora de escopo, adversariais).
45. **🔬 Fechei residuais da reauditoria.** Drift, documentação, reprodução em máquina limpa (GitHub-hosted) e testes.
46. **🔬 Banca de 3 juízes.** Passei de um juiz para **três juízes independentes** com medida de concordância (κ de Fleiss) — rumo a um padrão sênior.
47. **🔎 Tripliquei o corpus.** Ampliei a base de normas (~3×), com 2ª âncora de reprodução e cache/observabilidade.
48. **🔬 Fechei o punch-list da 3ª reauditoria.** Evidência do corpus, 2ª âncora no CI e documentação de drift.
49. **⚙️ Teste de carga do cache.** Medi latência p50/p95 sob carga — reportando o número, sem afirmar o que não medi.
50. **🚀 Space turnkey.** Uma versão do deploy que baixa o corpus e constrói o índice sozinha, com passo a passo do HuggingFace.
51. **🚀 Alternativa em Docker.** Uma segunda via de deploy (SDK Docker, gratuita) na porta 7860.
52. **🎨 Limpeza de estilo.** Correção de formatação no app do Space.
53. **🚀 Demo estática client-side.** Como o HuggingFace passou a cobrar por Gradio/Docker, criei uma demo **que roda no navegador do visitante** (Transformers.js) — pública e de graça.
54. **🎨 Demo no ar.** Link e selo da demo ao vivo no README.
55. **🔎 De-boilerplate do corpus + reversão honesta.** O especialista achou lixo repetido (cabeçalhos/rodapés) nas normas; ao limpar, o reordenador **passou a ajudar** — e reportei essa reviravolta abertamente.
56. **🔬 Kit de anotação humana.** Ferramenta para dois anotadores humanos avaliarem em paralelo e medir a concordância real (κ de Cohen).
57. **🎨 Ajustes de estilo e versionamento do kit.** Correções de formatação e inclusão do arquivo de anotação no repositório.
58. **🔬 Kit em Excel.** O kit de anotação passou a aceitar planilha (.xlsx) além de CSV, para facilitar a vida dos anotadores.
59. **🔎 κ humano inter-anotador (0,86).** Registrei a concordância entre humanos como evidência versionada da qualidade da avaliação.
60. **🔬 Fechei 3 furos do κ.** Correções dos pontos que o especialista apontou na medida de concordância.
61. **⚙️ Custo de serving em R$/1k.** Modelo de custo derivado da vazão **medida** — sem gastar de fato.
62. **🔬 Intervalo de confiança no juiz + custo da rota RAG.** Refinos que elevaram o rigor da avaliação de geração e ligaram a concordância humana ao conjunto-ouro.
63. **🔬 Cobertura de testes e correção de docstring.** Detalhes finos apontados pelo especialista.
64. **🔎 Kit de validação dos rótulos-ouro do hit@5.** Ferramenta para conferir, na fonte, se os rótulos usados na métrica-chave da Fase 1 estavam certos.

---

## 🗓️ 15 de julho — a auditoria que corrigiu a métrica-chave + a repaginada visual

*Descobri que parte dos rótulos da métrica principal estava errada, corrigi pela fonte, e dei tratamento visual à documentação.*

65. **🔎 κ humano dos rótulos-ouro (0,92).** Dois anotadores conferiram os rótulos de fonte do hit@5 — o elo que faltava para confiar na métrica.
66. **🔬 Corrigi exagero e propaguei a auditoria.** Ajustei uma afirmação forte demais sobre a concordância e levei a auditoria até o número final do hit@5.
67. **🔎 hit@5 definitivo pós-auditoria.** A auditoria achou **16% dos rótulos errados**; **rerotulei pela fonte correta (não apaguei)** e reportei o impacto — o número honesto ao lado do número do portão.
68. **📖 Distinção lastro humano vs documental.** Deixei explícito no texto a diferença entre o que humanos refutaram e o que foi rerotulado por documento.
69. **🎨 README repaginado.** Reescrita à altura do projeto e sincronização de números que haviam ficado defasados.
70. **🎨 Tratamento visual do HISTORIA.md.** Cabeçalho, linha do tempo e destaques de rigor (os *callouts* ⚖️).
71. **🎨 Tratamento visual do guia didático.** Novos termos (κ, custo) e correções de números defasados.
72. **🎨 Tratamento visual da arquitetura.** Mapa atualizado com os módulos novos da sessão.
73. **🎨 Banner no topo do README.** A arte-cabeçalho (SVG) do RodoIA.
74. **🎨 Card para redes sociais (1200×1200).** Peça para post no LinkedIn.
75. **🎨 Card mostra o hit@5 auditado.** Troquei o número conservador pelo auditado (0,70–0,76) na peça social.
76. **🎨 Imagem de preview (1280×640).** A prévia que aparece ao compartilhar o link (GitHub/OpenGraph).

---

## 🗓️ 17 de julho — a auditoria de portfólio e a camada de segurança

*O fechamento: uma última auditoria, o endurecimento do CI e o red-team que achou um bug real.*

77. **🔬 Fechei os 5 achados da auditoria de portfólio.** A última passada de revisão de ponta a ponta.
78. **⚙️ Alinhei o Python em 3.12.** Ajuste que destravou o portão de checagem de tipos (mypy) no CI.
79. **🔎 2ª passada de de-boilerplate.** Removi um rodapé que ainda escapava da limpeza — sem alterar o hit@5.
80. **⚙️ Red-team adversarial.** Um ataque automatizado mediu a taxa de sucesso de invasão (ASR) — e **achou um bug real no guardrail**, que corrigi.
81. **⚙️ Segurança da cadeia de suprimentos.** Trava de dependências com hash, inventário de componentes (SBOM) e auditoria de vulnerabilidades (CVEs).

---

## 🧵 O que este diário mostra

Lido de ponta a ponta, o padrão fica claro: **construí rápido, mas voltei sem dó para endurecer cada número** — os passos de 🔬 rigor são metade do diário. O apanhado dessas correções (o antes/depois de cada uma) vive numa fonte só: **[README § Decisões e trade-offs](../README.md#-decisões-e-trade-offs-o-arco-do-projeto)**.

<div align="center">

[← README](../README.md) · [📖 A história (por fase)](HISTORIA.md) · [🗺️ Arquitetura](ARQUITETURA.md) · [🎓 Guia didático](GUIA_ENGENHARIA_IA.md)

</div>
