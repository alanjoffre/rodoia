# 17 — Fase 6: escala — ingestão de 17,2 M reclamações (CFPB)

> Prova **ingestão em escala com memória limitada**: 1,43 GB de zip → Parquet particionado,
> **sem materializar os 13,5 GB de CSV intermediários**. Entrega o corpus que as Fases 1–5 não
> conseguiam exercer (3.647 chunks contra **17.226.584 linhas**) e 2 portões novos no gate.
> Este doc cobre o **primeiro bloco** da Fase 6 (escala); o segundo — benchmark externo contra
> o CUAD — está em aberto (§8).

## 1. Por que um dataset americano num repositório de regulação brasileira

Enxerto precisa de justificativa, então aqui está a honesta. A metodologia de avaliação das
Fases 1 e 5 (hit@5 com IC de Wilson, κ de anotadores, gate de regressão) foi construída sobre
**gold rotulado pelo próprio autor** — a auditoria κ tratou isso por dentro (docs/16), mas a
objeção "ele rotulou o próprio teste" só morre com **benchmark de terceiros**. E o corpus da
ANTT, com 3.647 chunks, **não tem escala** para exercer particionamento, poda de partição ou
comparação de motor.

O Consumer Complaint Database da CFPB resolve as duas coisas: 17,2 M linhas e **narrativa livre
escrita pelo cidadão** — algo que a ANTT **não publica**. Isso foi verificado, não suposto: o
campo `mensagem` do Sistema de Ouvidoria (SOU) da ANTT é um **ID numérico de 7 dígitos**, não o
texto da manifestação, conforme o dicionário de dados oficial ("Campo numérico, de preenchimento
automático pelo sistema, com o código de identificação da mensagem"). O campo mais textual do SOU
tem 53 caracteres de média — é rótulo de taxonomia, não narrativa.

## 2. Dados — Consumer Complaint Database (CFPB, domínio público)

[CFPB](https://www.consumerfinance.gov/data-research/consumer-complaints/) — bulk único
`complaints.csv.zip`, **1,43 GB**, 16 colunas, domínio público (U.S. Government Works).
Pipeline reproduzível (`rodoia.ingestao.baixar_cfpb` → `rodoia.ingestao.ingestao_cfpb`);
dados brutos fora do Git.

**Quatro armadilhas reais, todas verificadas antes de escrever o módulo:**

- **WAF invertido.** O Akamai da CFPB responde **403 a User-Agent de navegador** (`Mozilla/...`)
  e **200 a `curl`/`Python-urllib`**. Tentar "parecer um navegador" quebra o download. O
  `_USER_AGENT` é deliberadamente `curl/8.4.0`, com comentário no código para ninguém "corrigir".
- **A API de busca é inutilizável.** `/consumer-complaints/search/api/v1/` devolve 403 ou o HTML
  da SPA; o endpoint Socrata retorna 404. Só o bulk é acessível programaticamente.
- **Três formatos de data na mesma coluna.** `YYYY-MM-DD`, **timestamp ISO completo**
  (`2026-07-24T09:08:01.000Z`) nas linhas recentes, e `MM/DD/YYYY` no snapshot de 2018 no Kaggle.
  Sem normalizar, o `min`/`max` compara representações por ordem lexicográfica e reporta um
  período que não existe — foi exatamente o defeito da 1ª execução (§5).
- **Taxonomia de `product` com variantes sobrepostas** ("Credit reporting", "Credit reporting,
  credit repair services…", "Credit reporting or other personal consumer reports") — legado de
  mudanças de formulário. A ingestão **não harmoniza**: grava o valor cru e deixa a harmonização
  explícita na camada de domínio. Esconder drift de rótulo dentro da ingestão seria maquiar a
  fase antes de ela começar.

**A narrativa existe em apenas 22,21% das linhas** (a CFPB publica só com consentimento do
consumidor), e a taxa **varia fortemente por ano**:

| ano | linhas | com narrativa |
|---|---:|---:|
| 2011–2014 | 336.030 | **0%** — o programa de publicação começou em 2015 |
| 2017 | 242.749 | **47,5%** (pico) |
| 2025 | 5.443.425 | 22,5% |
| 2026 | 4.543.639 | **2,1%** — a publicação depende da resposta da empresa |

> **Consequência de projeto:** o ano corrente **não serve para trabalho de texto** — não é queda
> real, é dado ainda não assentado. O corpus de recuperação corta em 2025.

Há também **explosão de volume**: 1,29 M (2023) → 2,73 M (2024) → 5,44 M (2025).

## 3. Ingestão em escala — `ingestao/ingestao_cfpb.py`

O CSV descomprimido não cabe confortavelmente no VHDX do WSL e não há motivo para gravá-lo. O zip
é lido, inflado e convertido em lotes, com memória limitada por `LINHAS_POR_LOTE = 50_000` e um
`ParquetWriter` aberto por ano.

Detalhe que obriga streaming manual: o bulk é **ZIP64**, então os campos de tamanho do cabeçalho
local vêm com o sentinela `0xFFFFFFFF` — não dá para confiar neles. O `_FluxoZip` infla com
`zlib.decompressobj(-15)` até o fim do fluxo em vez de contar bytes.

Layout **Hive** (`ano=YYYY/`) porque tanto DuckDB quanto Spark fazem *partition pruning* nele —
requisito do benchmark de motor (§8).

| | valor |
|---|---|
| Linhas | **17.226.584** |
| Com narrativa | **3.825.572** (22,21%) |
| Caracteres de narrativa | **3.906.519.554** |
| Período | 2011-12-01 .. 2026-07-24 |
| Partições | 16 (2011–2026) |
| Zip → CSV → Parquet | 1,43 GB → 13,5 GB → **1,1 GB** (zstd, ~12:1) |

> **Evidência versionada:** `reports/fase6_escala/contagem_cfpb.json`, carimbado com proveniência
> e com o **sha256 do snapshot** (`1025b803…`). Os três números acima reproduziram **idênticos em
> 3 execuções independentes** — um script de contagem descartável e duas execuções do módulo.

## 4. Portões — por que piso e não igualdade

O bulk é **republicado diariamente** e a contagem **só cresce**. Um portão `== 17.226.584`
quebraria no dia seguinte. O que é reproduzível não é o número sozinho: é o **par (sha256,
contagem)**, e é ele que o report carimba.

```
[✓] F6 · linhas do CFPB (bulk)     17226584 >= 17000000
[✓] F6 · narrativas do CFPB         3825572 >= 3700000
```

Pisos com folga sob o medido, no padrão do `F3 · linhas do fato`. Uma queda abaixo de 17 M
denuncia **ingestão truncada** — o risco real ao ler 1,43 GB por streaming. O segundo portão
vigia a política de publicação por consentimento: se a CFPB mudá-la, a queda aparece **antes** de
contaminar qualquer métrica de recuperação. Gate **15 → 17**.

## 5. Três defeitos que só a execução real pegou

Nenhum apareceu nos testes sintéticos — todos surgiram ao rodar os 17,2 M. Ficam registrados
porque o padrão se repete (docs/16 §2.1: *passar local não prova nada sobre o outro ambiente*).

1. **`periodo` misturava representações.** `max` saiu `2026-07-24T09:08:01.000Z` contra um `min`
   `2011-12-01`. O particionamento estava correto, mas o campo mentia. Corrigido com `_data_iso`,
   com teste para os três formatos. **O report foi regerado** — não se commita artefato com campo
   sabidamente errado.
2. **`pyarrow` não era carimbado** pelo `proveniencia.py`, e a versão **muda comportamento**:
   a 24 materializa a chave de partição Hive na leitura, a 25 não. Um teste que fixava a lista
   exata do schema passava no Windows (25) e falhava no WSL (24) — não era bug do código, o
   arquivo tem só as 16 colunas declaradas. Dividido em dois testes: um afirma o **subconjunto**
   (robusto a versão), outro usa `read_schema` para provar que `ano` vive no nome do diretório.
   `pyarrow` e `duckdb` entraram em `_LIBS`.
3. **`ruff format` local ia poluir o repositório.** O pre-commit fixa **ruff v0.8.4**; o ruff
   local era **0.15.22**, e o formatador novo explode o estilo compacto do `gate.py` — 151 linhas
   de churn escondendo 11 de mudança real. **A ferramenta local não é a autoridade, a pinada é.**
   O CI cobra `ruff check`, não `ruff format`.

E um defeito que não era do código: **o zip de 1,4 GB estava rastreável pelo Git**. Quem insere a
entrada em `data/raw/.gitignore` é o `dvc add`, e o DVC não estava instalado no venv do WSL —
então dado novo nasce rastreável. `/cfpb` adicionado seguindo o padrão de `/acidentes` e `/normas`.

## 6. Reproduzir

```bash
pip install -e ".[escala]"
python -m rodoia.ingestao.baixar_cfpb --verificar   # HEAD: confere se o WAF ainda libera
python -m rodoia.ingestao.baixar_cfpb               # ~1,43 GB, idempotente
python -m rodoia.ingestao.ingestao_cfpb             # -> reports/fase6_escala/contagem_cfpb.json
python -m rodoia.mlops.gate                         # 30/30
```

Custo medido na Nitro: ~2,5 min de download + ~20 min de parse.

## 7. Critérios de conclusão — bloco "escala"

- [x] Pipeline reproduzível com **proveniência do snapshot** (sha256 + last-modified + bytes)
- [x] Ingestão **com memória limitada**, sem materializar os 13,5 GB intermediários
- [x] Parquet **particionado** em layout consumível por DuckDB e Spark
- [x] Contagem **carimbada em artefato versionado** e defendida por 2 portões do gate
- [x] Armadilhas do dado **documentadas no código**, não só no doc (WAF, 3 formatos de data)
- [x] Testes dos caminhos críticos (9), **sem rede**, verdes em pyarrow 24 e 25
- [x] `dvc add` dos dados brutos e processados (ponteiros versionados, dado fora do Git)
- [x] **Benchmark externo ingerido e aferido** — CUAD com 13.823 spans de gold, **0 offsets
      divergentes** (§8), 2 portões adicionais
- [x] Avaliação de recuperação sobre o CUAD — BM25 (§9), denso (§10), híbrido (§11), tudo com IC
- [x] **Benchmark de motor (DuckDB vs Spark)** — DuckDB 6,7× mais rápido, **0 divergências** (§12)

## 8. Bloco "benchmark externo" — CUAD ingerido e aferido

[CUAD](https://www.atticusprojectai.org/cuad) (**CC BY 4.0**) — 510 contratos comerciais anotados
por advogados. `rag/baixar_cuad.py` → `rag/cuad.py` normaliza o `CUAD_v1.json` (SQuAD 2.0) para
`contratos.jsonl` + `perguntas.jsonl`, e afere a integridade:

> **Correção de licença.** Este documento afirmou *"Apache 2.0"* até 28/07/2026. O Atticus Project
> libera **todos os seus datasets sob CC BY 4.0** — o que **exige atribuição**, ao contrário do que
> a declaração errada implicava. A atribuição completa, com a citação do paper que os autores pedem,
> está no [`NOTICE`](../NOTICE); o cartão da fonte está em
> [`docs/DATASET_CARD.md`](DATASET_CARD.md). Uma licença afirmada de memória é o mesmo defeito que
> este documento persegue nas métricas — só que com consequência fora do repositório.
>
> Ressalva que o próprio Atticus repassa: a CC BY 4.0 cobre **a anotação**, não os contratos
> subjacentes, que são públicos e vêm do EDGAR (SEC/EUA); eles não declaram garantia sobre o status
> de licença deles.

| | medido |
|---|---:|
| Contratos | **510** |
| Perguntas | **20.910** |
| `is_impossible` | **14.208** (67,95%) |
| Spans de resposta | **13.823** |
| Categorias de cláusula | **41** |
| **Spans cujo offset confere com o texto** | **13.823 — 0 divergentes** |

> **A última linha é o portão que importa.** `answer_start` é offset de caractere no contrato
> inteiro; gold desalinhado **não quebra nada visivelmente** — produz métrica plausível e falsa.
> Conferir é barato e a ausência da conferência é cara, então `spans_divergentes` tem teto 0,
> sem folga, como o vazamento de PII da Fase 1.

Duas decisões de parsing registradas no código: **`is_impossible` não é descartado** (o reflexo
comum ao ver "impossible" jogaria fora 2/3 do benchmark e justamente a parte difícil — são elas
que medem **abstenção**), e **os offsets são preservados** em vez de só o texto do span, porque
o mapeamento span→chunk da avaliação depende deles e re-encontrar por busca de string é ambíguo
quando o mesmo trecho se repete.

**Contraste de portão, deliberado:** o CUAD usa **igualdade** (`n_contratos == 510`) enquanto o
CFPB usa **piso**. Não é inconsistência — o CUAD é dataset acadêmico congelado, e se 510 virar
outro número o espelho mudou sob nossos pés e toda comparação com SOTA fica inválida; o bulk da
CFPB é série viva que só cresce.

Também verificado: a **API pública do Kaggle dispensa credencial** (sem conta, sem `kaggle.json`,
sem o pacote `kaggle`), mas **responde 404 a HEAD** e 200 a GET — a consulta de metadados abre
com GET e fecha antes do corpo. E `User-Agent` com acento derruba a requisição com **400**:
cabeçalho HTTP não aceita não-ASCII.

## 9. Avaliação de recuperação — baseline BM25, zero LLM

`rag/avaliacao_cuad.py`: chunking com offsets → mapeamento span→chunk → BM25 → métricas com IC.
**510 contratos, 20.806 chunks, 6.702 perguntas respondíveis. 15,8 s de execução, custo de API
zero.**

**Recall@k aqui é Recall de verdade, não hit-rate.** A Fase 1 documenta honestamente que mede
*hit-rate* porque cada pergunta tem UMA fonte-gold. No CUAD o gold é exaustivo por pergunta, então
a fração de *todos* os relevantes recuperados é computável — e as duas métricas são reportadas
lado a lado, porque medem coisas diferentes.

| k | Recall@k (IC95 bootstrap) | Hit@k (IC95 Wilson) |
|---:|---|---|
| 1 | 0,275 [0,266; 0,284] | 0,431 [0,419; 0,443] |
| 3 | 0,491 [0,480; 0,502] | 0,625 [0,614; 0,637] |
| 5 | **0,588** [0,577; 0,599] | 0,705 [0,694; 0,716] |
| 10 | 0,724 [0,714; 0,733] | 0,812 [0,803; 0,821] |

**MRR 0,557** [0,547; 0,567]. E **`n_sem_gold = 0`**: toda pergunta respondível mapeou para pelo
menos um chunk — o alinhamento span→chunk fecha ponta a ponta, não por suposição.

### 9.1 O resultado negativo que importa — abstenção não funciona com BM25

O diagnóstico de abstenção compara o escore do top-1 nas duas populações:

| | p10 | mediana | p90 |
|---|---:|---:|---:|
| Respondível | 6,25 | **15,26** | 34,13 |
| `is_impossible` | 2,55 | **14,85** | 30,02 |

As medianas diferem em **+0,41 (2,7%)** e as distribuições se sobrepõem quase inteiramente.
**Nenhum limiar sobre o escore BM25 separa "o contrato tem essa cláusula" de "não tem."**

Isso não é falha do experimento — é o experimento funcionando. O diagnóstico foi escrito
justamente para detectar isso *antes* de construir uma política de abstenção sobre um sinal sem
informação. A razão é estrutural: BM25 devolve o chunk que melhor casa lexicalmente, e linguagem
contratual é homogênea — sempre há *algum* trecho que casa razoavelmente, exista a cláusula ou
não.

### 9.2 A média de 0,588 esconde um fator de 5,3×

Recall@5 por categoria de cláusula:

| melhores | | piores | |
|---|---:|---|---:|
| Renewal Term | 0,922 | Affiliate License-Licensor | 0,336 |
| Governing Law | 0,893 | Document Name | 0,322 |
| Termination For Convenience | 0,885 | Exclusivity | 0,277 |
| Notice Period To Terminate Renewal | 0,851 | Parties | 0,255 |
| Insurance | 0,827 | **Volume Restriction** | **0,173** |

O padrão nas piores é diagnóstico: `Document Name` e `Parties` são metadados — o nome do contrato
e quem o assina. O texto que os contém **não repete os termos da query**, então BM25 não tem por
onde casar. É o caso clássico de lacuna lexical, e é exatamente onde recuperação densa deve ganhar
— o que torna a comparação denso vs BM25 o próximo experimento com hipótese, não com esperança.

Reportar só a média de 0,588 esconderia essa estrutura inteira.

## 10. Denso vs BM25 — a média é o resumo errado

`rag/avaliacao_cuad_denso.py`: recuperação densa (multilingual-e5-small, 384 dim, na 4050) contra
o baseline BM25, **pela mesma máquina de métricas** (`consolidar` foi extraído para os dois
recuperadores compartilharem — o refator reproduz o 0,588 do BM25 bit a bit, senão a comparação
mediria o código). Sem Qdrant: recuperação dentro do contrato é produto escalar de vetores
normalizados. 2m48s de encode na GPU, custo de API zero.

**No agregado, o denso perde:**

| | BM25 | Denso | Δ |
|---|---:|---:|---:|
| recall@5 | 0,588 | 0,535 | **−0,053** |
| MRR | 0,557 | 0,496 | −0,061 |

Parar aqui seria a leitura preguiçosa. A média de −0,053 é a soma de **dois efeitos opostos** que
se cancelam — o denso vence em 12 das 41 categorias e o padrão de quem vence onde é o resultado:

| Denso VENCE (lacuna lexical) | BM25→Denso | | BM25 VENCE (termo raro/distintivo) | BM25→Denso |
|---|---:|---|---|---:|
| **Document Name** | 0,322 → 0,482 (**+0,161**) | | Unlimited/All-You-Can-Eat-License | 0,671 → 0,324 (−0,347) |
| No-Solicit Of Employees | 0,648 → 0,758 (+0,110) | | Non-Transferable License | 0,645 → 0,331 (−0,315) |
| **Volume Restriction** | 0,173 → 0,282 (**+0,108**) | | Irrevocable Or Perpetual License | 0,775 → 0,481 (−0,293) |
| Warranty Duration | 0,565 → 0,671 (+0,106) | | Effective Date | 0,689 → 0,411 (−0,278) |

**A hipótese da §9.2 se confirma com precisão.** As duas piores categorias do BM25 eram
`Volume Restriction` (0,173) e `Document Name` (0,322) — metadados cujo texto não repete os termos
da query. São exatamente os **dois maiores ganhos** do denso. Onde BM25 tem lacuna lexical, o
denso casa por significado.

**E as perdas são igualmente diagnósticas.** O denso despenca em `Unlimited/All-You-Can-Eat-License`,
`Non-Transferable License`, `Irrevocable Or Perpetual License` — categorias com **terminologia
jurídica rara e distintiva**, onde o casamento exato do BM25 é força e um modelo pequeno e
multilíngue **dilui** o token inglês raro. "Unlimited/All-You-Can-Eat-License" é uma expressão que
o BM25 acerta na mosca e o e5-small borra.

**A conclusão não é "denso perde" — é que os dois são complementares**, e é o argumento textual do
híbrido. A Fase 1 já escolheu BM25+E5+RRF (docs/07); este experimento **re-deriva o porquê do
híbrido sobre um benchmark de terceiros**, categoria a categoria, em vez de sobre o gold próprio.
A escolha arquitetural da Fase 1 fica validada externamente.

Dois caveats honestos, no relatório e aqui: (a) um embedder inglês forte (bge-large-en) quase
certamente levantaria o lado denso — mas o achado **estrutural** (forças complementares) independe
do modelo; (b) a abstenção continua impossível — as medianas do cosseno do top-1 são 0,836
(respondível) vs 0,827 (impossível), sobreposição ainda maior que a do BM25. Nenhum recuperador
isolado separa "tem cláusula" de "não tem".

## 11. Híbrido (RRF) — melhor em todo ponto, mas o ganho não é significativo

`rag/avaliacao_cuad_hibrido.py` funde BM25 + denso por **Reciprocal Rank Fusion**, reusando a
`fundir_rrf` da Fase 1 (`rag/recuperador.py`) — RRF funde POSIÇÃO no ranking, o que torna escore
BM25 (~15) e cosseno (~0,8) combináveis apesar de escalas incomparáveis. Mesma máquina de
métricas dos outros dois.

| | BM25 | Denso | **Híbrido** |
|---|---:|---:|---:|
| recall@5 | 0,588 [0,577; 0,599] | 0,535 | **0,595** [0,584; 0,605] |
| recall@10 | 0,724 [0,714; 0,733] | 0,699 | **0,740** [0,731; 0,750] |
| MRR | 0,557 [0,547; 0,567] | 0,496 | **0,562** [0,553; 0,572] |

**O híbrido é o melhor estimador de ponto em toda métrica** — a direção que a Fase 1 previu. Mas
a régua do projeto exige olhar o IC, não só o ponto: o IC do híbrido no recall@5 **[0,584; 0,605]
sobrepõe** o do BM25 **[0,577; 0,599]**. O ganho de +0,007 está **dentro da incerteza — não é
estatisticamente distinguível** do BM25 sozinho. O mesmo vale para MRR e recall@10 (ICs
sobrepostos). Reportar "híbrido vence" sem o IP seria a métrica maquiada que a auditoria da Fase 1
já reprovou uma vez.

**Por que o ganho é modesto — e é o achado sobre RRF.** O híbrido fica `>= melhor isolado` em
apenas **17 das 41 categorias**. O padrão de onde ganha e onde perde explica:

| Híbrido GANHA (rankers concordam) | bm25 / denso → hib | | Híbrido PERDE (rankers discordam forte) | bm25 / denso → hib |
|---|---|---|---|---|
| No-Solicit Of Customers | 0,475 / 0,467 → **0,552** | | Effective Date | 0,689 / 0,411 → 0,567 |
| Expiration Date | 0,753 / 0,758 → **0,804** | | Irrevocable/Perpetual License | 0,775 / 0,481 → 0,668 |
| Uncapped Liability | 0,806 / 0,708 → **0,836** | | Unlimited/All-License | 0,671 / 0,324 → 0,568 |

**RRF é um mecanismo de consenso, não um seletor do melhor.** Onde os dois rankers **concordam**
(ambos medíocres ou ambos bons), a fusão amplifica — os dois põem o gold perto do topo e o RRF o
concentra. Onde **discordam forte** — `Effective Date`, BM25 0,689 vs denso 0,411 — a fusão
**compromete**: arrasta o vencedor na direção do perdedor, porque RRF não tem como saber *qual*
ranker confiar naquela query. Ganha robustez, perde o pico.

**O arco fecha, e re-deriva a Fase 1 inteira.** Sobre um benchmark de terceiros, do zero:
BM25 → denso (complementar, §10) → híbrido RRF (melhor no agregado, mas o ganho é limitado pelo
lado denso fraco e pelo compromisso do RRF na discordância). O lever para tornar o ganho
significativo é exatamente o **próximo estágio que a Fase 1 já tem**: um **reranker cross-encoder**,
que lê query+trecho juntos e desempata *por query* — o que o RRF não consegue. A arquitetura de
recuperação da Fase 1 (denso + BM25 + RRF + rerank) não foi escolha de fé; este experimento a
justifica peça por peça, com IC, num dataset que não é o nosso.

## 12. Motor de dados — Spark medido e rejeitado com número, não com opinião

`mlops/benchmark_motor.py` roda **6 queries analíticas idênticas** (mesmo SQL) sobre os 17,2 M do
CFPB, nos dois motores, e compara tempo **e resultado**. A ingestão usa DuckDB (§3); este benchmark
justifica a escolha em vez de afirmá-la.

| query | DuckDB | Spark | speedup |
|---|---:|---:|---:|
| linhas por ano | 0,008s | 0,508s | **61,4×** |
| pruning (1 ano) | 0,003s | 0,089s | 27,2× |
| top-10 produtos | 0,029s | 0,515s | 18,0× |
| empresas distintas | 0,033s | 0,556s | 16,9× |
| top-10 empresas | 0,040s | 0,519s | 13,1× |
| taxa de narrativa/ano | 0,420s | 1,378s | **3,3×** |
| **total** | **0,533s** | **3,565s** | **6,7×** |

**DuckDB é 6,7× mais rápido no total** — a hipótese (num único nó com RAM suficiente, o overhead de
JVM/shuffle do Spark não se paga) confirmada com número. Mas a **variação do speedup é o achado**,
e é o que impede a leitura preguiçosa "Spark é ruim":

- **61,4× na contagem por ano** — query trivial, onde o custo é quase todo **overhead fixo por
  stage** do Spark (agendamento, serialização, JVM). DuckDB responde em 8 ms; o Spark paga 500 ms
  de infraestrutura para fazer 8 ms de trabalho.
- **3,3× na taxa de narrativa** — a query com **compute real** (CASE + avg sobre a coluna de
  narrativa inteira, 3,9 bi de caracteres). Aqui o paralelismo de 16 cores do Spark **faz trabalho
  de verdade** e a vantagem do DuckDB despenca de 61× para 3,3×.

O padrão mostra **exatamente onde o modelo do Spark começaria a pagar**: quanto mais compute por
query — e, sobretudo, quando o dado passar da RAM de um nó — o crossover se inverte. Abaixo disso,
que é o CFPB (17 M linhas, 1,1 GB de Parquet), Spark é overhead puro. A decisão de motor da
ingestão não foi enfeite de currículo evitado — foi **enfeite medido e rejeitado, com o número que
também mostra a condição em que ele venceria**.

**Correção acima do tempo: 0 divergências** nas 6 queries. Os dois motores dão a MESMA resposta
célula a célula (invariante gated) — então a escolha é sobre velocidade e operação, não correção.
Um motor rápido que errasse a conta seria descartado apesar do tempo.

*Reprodução:* pyspark exige JVM e briga com o stack CUDA da Fase 2, então roda num **venv isolado**
(`/tmp/spark_venv` + `openjdk-17-jre` + o extra `benchmark-motor`), nunca no ambiente de GPU. O
teste no CI cobre a comparação (pura) e o caminho DuckDB com um Parquet de fixture; `rodar_spark`
não entra no CI.

## 13. As extensões — uma previsão errada e uma confirmada

A §12 listou três extensões como "não mudam nenhuma conclusão de forma". Rodá-las mostrou que
**uma delas eu tinha previsto errado**, e que a outra não era extensão — era o estágio que faltava.

### 13.1 O embedder inglês forte NÃO ajudou — a previsão que falhou

A §10 registrou: *"um embedder inglês forte (bge-large-en) quase certamente levantaria o lado
denso"*. Medido:

| | BM25 | **e5-small** (118M, multilíngue) | **bge-large-en** (335M, inglês) |
|---|---:|---:|---:|
| recall@5 | 0,588 | **0,535** [0,524; 0,545] | **0,532** [0,522; 0,543] |
| MRR | 0,557 | 0,496 | 0,512 |
| tempo de encode | — | 2m48s | **20m32s** |

**Δ = −0,003, ICs sobrepostos.** Um modelo **2,8× maior**, específico para inglês, com o dobro de
dimensões e **7,3× mais lento**, não produziu diferença estatisticamente distinguível. A previsão
estava **errada na primeira metade e certa na segunda**: o modelo forte não levantou o denso, mas o
achado estrutural (complementaridade) sobreviveu.

O detalhe que fecha o argumento: nas categorias de **lacuna lexical** — onde o denso deveria
brilhar — o modelo maior é frequentemente **pior**.

| categoria | BM25 | e5-small | bge-large |
|---|---:|---:|---:|
| Document Name | 0,322 | **0,482** | 0,307 |
| Parties | 0,255 | **0,298** | 0,241 |
| Volume Restriction | 0,173 | **0,282** | 0,259 |

Três hipóteses, marcadas como hipóteses: (a) o gargalo é o **chunking** (janela cega), não o
embedder; (b) as queries são **rótulos de categoria**, não perguntas naturais, e bi-encoders são
treinados em perguntas; (c) linguagem contratual é homogênea e a similaridade satura. Testá-las é
trabalho futuro — o que está **medido** é que trocar de modelo não é o lever.

**Consequência prática:** o rerank abaixo roda com **e5-small**, escolhido pela medição e não pelo
tamanho. Um resultado que economiza 7× de compute é resultado.

### 13.2 O rerank cross-encoder — a hipótese confirmada, com IC disjunto

A §11 diagnosticou: RRF é **consenso, não seletor**, e o lever seria o cross-encoder, que desempata
**por query**. Rodando a pilha completa da Fase 1 (`denso + BM25 → RRF → rerank`):

| | BM25 | Híbrido | **+ Rerank** |
|---|---:|---:|---:|
| recall@1 | 0,275 | 0,275 | **0,313** |
| recall@5 | 0,588 [0,577; 0,599] | 0,595 [0,584; 0,605] | **0,652 [0,642; 0,662]** |
| recall@10 | 0,724 | 0,740 | **0,774** |
| MRR | 0,557 | 0,562 | **0,604** |

**Os ICs do rerank e do híbrido são DISJUNTOS** — o ganho de **+0,057** é, este sim,
estatisticamente significativo, ao contrário do +0,007 do híbrido sozinho. É a diferença entre
"melhor estimador de ponto" e "melhor, com evidência".

E o **mecanismo previsto se confirma na categoria certa**. As maiores vitórias do rerank sobre o
híbrido são exatamente onde o RRF comprometia por discordância dos rankers:

| categoria | BM25 | denso | híbrido (RRF) | **+rerank** |
|---|---:|---:|---:|---:|
| **Effective Date** | 0,689 | 0,411 | **0,567** ← RRF piorou | **0,776** |
| **Document Name** | 0,322 | 0,482 | **0,386** ← RRF piorou | **0,608** |
| Agreement Date | 0,430 | 0,482 | — | **+0,305** |

Em `Effective Date` o RRF entregou **menos que o BM25 sozinho** (0,567 vs 0,689) — o compromisso
custando caro. O cross-encoder não só recupera como **supera o melhor isolado** (0,776). Lendo
query e trecho juntos, ele decide *por query* qual sinal confiar; o RRF, por construção, não pode.

### 13.3 Abstenção — o escore finalmente separa (mas não o bastante)

O diagnóstico de abstenção rodou nos três recuperadores. A separação entre as medianas do escore
do top-1 (respondível − impossível):

| recuperador | resp. (mediana) | imposs. (mediana) | separação |
|---|---:|---:|---:|
| BM25 | 15,256 | 14,849 | **+0,407** (2,7%) |
| Híbrido (RRF) | 0,033 | 0,032 | **+0,001** ← escore RRF é comprimido, inútil aqui |
| **Rerank** | −3,184 | −6,942 | **+3,758** |

O cross-encoder é o primeiro sinal com separação **real** — ~9× a do BM25. **Mas as distribuições
ainda se sobrepõem:** o p10 dos respondíveis (−8,17) fica abaixo do p90 dos impossíveis (−2,31).
Um limiar erraria muito. Melhorou de "impossível" para "difícil", não para "resolvido" — e é assim
que está reportado.

### 13.4 Geração ancorada — a métrica que sozinha seria mentira

`rag/avaliacao_cuad_geracao.py` fecha o eixo: todas as métricas anteriores medem **recuperação**;
esta mede o que o gerador faz com o contexto. Roda no **LLM local** (Ollama, qwen2.5:7b) — **custo
de API zero**. Amostra estratificada por categoria, 150 por população, seed fixa.

O CUAD viabiliza a pergunta que quase nenhum portfólio responde: *quando a cláusula **não existe**,
o sistema se cala ou inventa?* Mas medir só isso seria maquiagem — um modelo que responde "não
consta" a **tudo** teria não-alucinação perfeita e seria inútil. Daí as **duas taxas**:

| | não-alucinação | cobertura | balanceada |
|---|---:|---:|---:|
| *Baseline trivial: abster de tudo* | *1,000* | *0,000* | *0,500* |
| Prompt **estrito** | 0,987 [0,953; 0,996] | 0,260 [0,196; 0,336] | 0,623 |
| Prompt **equilibrado** | 0,987 [0,953; 0,996] | **0,387** [0,312; 0,467] | **0,687** |

**O sistema está a +0,187 de um modelo que diz "não consta" a tudo.** Reportar apenas
"98,7% de não-alucinação" seria verdade e seria enganoso.

> **Sobre estes números terem mudado.** A primeira versão desta tabela trazia 0,273 e 0,373. O
> `OllamaLLM` roda a `temperatura=0.1` e **sem seed** — o que não é determinismo: duas execuções da
> mesma amostra dão respostas diferentes. Numa ablação isso é ruído contado como efeito. O backend
> passou a aceitar `seed` (default `None`, para não mexer nos outros chamadores) e a avaliação a
> fixá-la. Os números acima são os reprodutíveis; os anteriores eram um sorteio.

**E não é falha de recuperação.** O teto — fração de perguntas em que algum chunk de gold está no
top-5 (hit@5 do híbrido) — é **0,713**. A geração aproveita **54%** dele: em quase metade dos casos
o trecho certo **estava no contexto** e o modelo disse "não consta" mesmo assim.

**A ablação de prompt — um confundidor que eu mesmo introduzi.** O prompt inicial dizia *"Never
guess, never use outside knowledge"*. Sem testar, "o modelo abstém demais" seria indistinguível de
"o prompt do autor induziu abstenção". Suavizar o prompt recuperou **+0,100 de cobertura com
não-alucinação IDÊNTICA (0,987)** — ganho de graça, e prova de que **parte do problema era meu, não
do modelo**.

**O McNemar — a limitação que esta seção declarava em aberto, agora fechada.** A versão anterior
registrava: *"os ICs de cobertura se sobrepõem, então o ganho não é significativo; o desenho é
pareado, então um teste de McNemar seria bem mais potente — não foi computado"*. Foi.

Os ICs de cobertura de fato se sobrepõem ([0,196; 0,336] vs [0,312; 0,467]) — e **escondiam uma
diferença real**, porque comparar dois ICs ignora que as perguntas são as mesmas. Pareando por
pergunta:

| recorte | só o estrito acerta | só o equilibrado acerta | n discordantes | p |
|---|---:|---:|---:|---:|
| não-alucinação | 0 | 0 | 0 | 1,0 |
| **cobertura** | **2** | **21** | 23 | **6,6 × 10⁻⁵** |

**O prompt equilibrado é melhor, e agora com prova.** Ele não troca nada: a não-alucinação é
*literalmente idêntica* — zero perguntas em que as duas variantes discordam nessa população.

Duas notas sobre o teste em si. (a) Com **23 discordantes** a aproximação χ² **não vale** (a regra
prática pede ≥ 25) — `estat.mcnemar` detecta o regime e usa o **binomial exato**, que sob H₀ é
cara-ou-coroa em cada discordante e não aproxima nada. Qual dos dois foi usado sai no campo
`metodo`, porque um p-valor sem a proveniência do teste esconde a decisão mais importante dele.
(b) O McNemar só é legítimo aqui porque a **seed do LLM está fixa**; sem ela, respostas que mudam
entre rodadas entrariam contadas como discordância e inflariam a significância.

O que sobra depois de descontar o confundidor: mesmo no melhor prompt, **o gargalo é o gerador, não
a busca**. Levers plausíveis — modelo maior, few-shot, ou uma política de abstenção calibrada sobre
o escore do cross-encoder (§13.3, o primeiro sinal com separação real) — ficam medidos como
*próximos*, não como *feitos*.

### 13.5 Abstenção calibrada — o sinal com separação real é fraco demais para o trabalho

A §13.3 fechou com "melhorou de impossível para difícil". *Difícil quanto?* — a pergunta ficou em
aberto, e responder por adjetivo seria o inverso do que este documento faz. `rag/calibracao_abstencao.py`
varre o limiar sobre o escore do cross-encoder e reporta a **curva inteira**, não um ponto escolhido.

Sobre as **20.910 perguntas** (6.702 respondíveis + 14.208 impossíveis):

**AUC-ROC = 0,751.** É a probabilidade de uma pergunta respondível sorteada ter escore maior que
uma impossível sorteada — um resumo **independente de limiar**, que é o número que responde "esse
sinal presta?". 0,5 seria moeda. 0,751 é um sinal real, e insuficiente.

O melhor ponto por J de Youden (limiar −4,678): não-alucinação **0,729** [0,722; 0,736], cobertura
**0,634** [0,622; 0,645].

**A tradução para a decisão de engenharia.** A curva sozinha ainda não diz se vale a pena. O uso
real do limiar é uma **cascata**: abaixo dele, nem se chama o LLM. Então o que importa é quanto se
economiza e o que se perde — ponderado pelo prior do corpus (67,9% impossíveis):

| alvo de não-alucinação | limiar | chamadas de LLM evitadas | respondíveis perdidas |
|---:|---:|---:|---:|
| 0,50 | −6,726 | 42,0% | 19,4% |
| 0,70 | −4,970 | 58,8% | 34,4% |
| 0,80 | −3,800 | 68,8% | 44,3% |
| 0,90 | −2,045 | 81,9% | 62,0% |
| 0,95 | −0,874 | 88,6% | 74,4% |
| **0,99** | 1,467 | 97,2% | **92,9%** |

**A conclusão é negativa, e é a que importa.** O gerador da §13.4 opera a **0,987 de
não-alucinação com 0,387 de cobertura**. Para o limiar grátis chegar perto disso (0,99) ele barra
**92,9% das respondíveis** — cobertura ~0,071, cinco vezes pior que o gerador. **O limiar sobre o
escore de recuperação não substitui o gate do LLM.** O único sinal com separação real (§13.3) é
fraco demais para o trabalho, e agora isso está medido em vez de suposto.

O que ele *serve* é o outro extremo da cascata: a −6,726 evita **42% das chamadas** ao custo de
19,4% das respondíveis. Se a economia de inferência valer esse preço é decisão de produto, não de
métrica — mas o preço agora tem número.

### 13.6 Chunking por cláusula — o ganho é real e mora só no topo do ranking

A §13.1 levantou a hipótese (a): *o gargalo do denso é o chunking, não o embedder?* A janela de
1.500 caracteres corta no meio de cláusulas; `chunkar_clausula` divide na unidade semântica
natural — a mesma filosofia do `rag/chunking.py` da Fase 1, que divide por `Art. Nº`.

**A cobertura do detector foi medida antes de qualquer métrica** (`--diagnostico-chunker` →
`fronteiras_clausula.json`): **410 dos 510 contratos (80,4%)** têm 3+ fronteiras detectáveis,
mediana de 13. Os outros 100 caem no **fallback explícito para janela** — sem isso, "divide por
cláusula" seria uma afirmação sobre um corpus que não tem cláusulas detectáveis em 1/5 dos casos.
Chunks: 20.806 → 23.464.

O primeiro resultado, nos quatro recuperadores, parecia limpo:

| recuperador | recall@1 janela → cláusula | ICs |
|---|---|---|
| BM25 | 0,275 → 0,311 | [0,266; 0,284] vs [0,301; 0,321] — **disjuntos** |
| denso e5 | 0,225 → 0,262 | [0,216; 0,233] vs [0,253; 0,272] — **disjuntos** |
| híbrido RRF | 0,275 → 0,312 | [0,266; 0,284] vs [0,302; 0,322] — **disjuntos** |
| rerank | 0,313 → 0,355 | [0,303; 0,322] vs [0,344; 0,366] — **disjuntos** |

Quatro ICs disjuntos na mesma direção. Estava escrito como ganho — e estava errado.

#### O que desmontou: o MRR andou para o outro lado

Nos **mesmos quatro**, o MRR **caiu**: BM25 0,557 → 0,546, denso 0,496 → 0,491, híbrido 0,562 →
0,553, rerank 0,604 → 0,601. Recall@1 subindo com MRR descendo é incoerente para um mesmo
recuperador — a menos que o **denominador do recall** tenha mudado.

Mudou. `recall@k = |gold ∩ top-k| / |gold|`, e **|gold| depende do chunker**: a janela usa
`overlap=200`, então um span de fronteira intersecta **dois** chunks; o chunker por cláusula não
sobrepõe entre cláusulas. Medido sobre as 6.702 perguntas com gold:

| | |gold| médio | teto de recall@1 | teto de recall@5 |
|---|---:|---:|---:|
| janela | 1,908 | **0,717** | 0,989 |
| cláusula | 1,701 | **0,784** | 0,992 |

**O teto do recall@1 subiu 0,067 sozinho** — mais que qualquer ganho observado (+0,036 a +0,042).
Os quatro ICs disjuntos estavam medindo a régua, não o recuperador. Um span rotulado por advogado
não muda de tamanho porque eu troquei o fatiador; o número de chunks que ele toca, sim.

**Correção estrutural, não nota de rodapé.** `consolidar` agora reporta, em todo relatório:
`teto` por k, `gold_medio_por_pergunta`, e um **`recall_norm_at_k` = `|gold ∩ top-k| / min(k,|gold|)`**
— "do gold que **cabia** em k posições, quanto foi recuperado". Cada pergunta contribui numa escala
0–1 independente do seu |gold|, então a média **e o bootstrap** são comparáveis entre chunkings.
Normalizar só a média (razão de médias) daria o ponto sem o IC, que aqui é exatamente o que decide.
As **oito avaliações foram reexecutadas** com a métrica nova — nenhum número abaixo é derivado à
mão a partir dos relatórios antigos.

#### O resultado depois da correção

| recuperador | recall@1 **bruto** (janela → cláusula) | recall@1 **normalizado** |
|---|---|---|
| BM25 | 0,275 → 0,311 · **disjuntos** | 0,431 → 0,421 · sobrepõem |
| denso e5 | 0,225 → 0,262 · **disjuntos** | 0,352 → 0,356 · sobrepõem |
| híbrido RRF | 0,275 → 0,312 · **disjuntos** | 0,435 → 0,427 · sobrepõem |
| rerank | 0,313 → 0,355 · **disjuntos** | 0,473 → 0,472 · sobrepõem |

E as métricas que **nunca** dependeram do denominador concordam entre si, apontando de leve para o
outro lado — todas com ICs sobrepostos:

| | janela → cláusula |
|---|---|
| **hit@5** (algum gold no top-5) | BM25 0,705 → 0,694 · híbrido 0,713 → 0,700 · rerank 0,769 → 0,761 |
| **MRR** | BM25 0,557 → 0,546 · híbrido 0,562 → 0,553 · rerank 0,604 → 0,601 |

**A conclusão é efeito nenhum.** Não "ganho pequeno": em quatro recuperadores × quatro métricas
comparáveis, **nenhum IC se separa**, e em dois dos quatro o recall@1 normalizado até desce. Os
quatro ICs disjuntos da primeira tabela eram **integralmente** o denominador se mexendo. A
**hipótese (a) da §13.1 está refutada**: o chunking não era o gargalo do denso.

**O chunker fica no repositório** (`--chunker clausula`), como o `bge-large-en` da §13.1 — um
experimento negativo só é verificável se dá para rodá-lo de novo.

**Por que isto fica no documento em vez de sumir com o resultado.** O modo de falha aqui é o mesmo
que a §13.4 tem sobre o prompt e a §5 sobre o `ruff` local: um artefato do instrumento lido como
propriedade do objeto. Quatro ICs disjuntos e concordantes são exatamente o tipo de evidência que
se aceita sem conferir — e teriam sustentado uma alegação falsa no README. O que pegou não foi um
teste: foi uma métrica **secundária** (MRR) discordando da manchete. É o argumento para reportar
mais de uma métrica mesmo quando uma já conta a história que se queria contar.

> **Um segundo defeito, achado no meio da correção.** A primeira reexecução saiu com `teto: 0,0` e
> `recall_norm: 0,0` em tudo. Causa: os quatro módulos de avaliação reconstruíam o registro
> `Avaliada` campo a campo, e o `n_gold` recém-criado foi descartado pelos quatro em silêncio —
> passando por `ruff`, `mypy --strict` e 53 testes, porque o campo *existia*, só chegava zerado.
> As quatro reconstruções viraram um `_registrar` único com `dataclasses.replace`. O teste que
> guarda isso compara **todos** os campos via `fields()`, não o `n_gold`: travar o sintoma não
> impediria a próxima ocorrência.

### 13.7 O gerador maior — melhor na média, e **pior no que importa**

A §13.4 fechou apontando "modelo maior" como lever plausível. Rodado: **`gemma2:9b`** contra o
`qwen2.5:7b`, **mesmo prompt** (`equilibrado`), **mesma amostra**, **mesmas seeds** — de amostragem
e do LLM. 75 minutos a **6,3 tok/s** (5,4 GB de modelo em 6 GB de VRAM, com offload para CPU).

| | não-alucinação | cobertura | balanceada |
|---|---:|---:|---:|
| *Baseline trivial: abster de tudo* | *1,000* | *0,000* | *0,500* |
| `qwen2.5:7b` | **0,987** [0,953; 0,996] | 0,387 [0,312; 0,467] | 0,687 |
| `gemma2:9b` | 0,900 [0,842; 0,938] | **0,620** [0,540; 0,694] | **0,760** |

Pela média balanceada, o modelo maior vence com folga: **+0,073**. E a leitura de manchete pararia
aí. O McNemar pareado por recorte mostra o que a média esconde:

| recorte | só o qwen acerta | só o gemma acerta | n disc. | p | método |
|---|---:|---:|---:|---:|---|
| **cobertura** | 1 | **36** | 37 | **2,28 × 10⁻⁸** | χ² (Yates) |
| **não-alucinação** | **14** | 1 | 15 | **9,77 × 10⁻⁴** | binomial exato |
| geral | 15 | 37 | 52 | 3,59 × 10⁻³ | χ² (Yates) |

**As duas taxas se movem em direções opostas, e as duas com significância.** O `gemma2:9b`
responde muito mais (+23,3 pp de cobertura) **e inventa muito mais**: a alucinação sai de 1,3%
para **10,0%** — quase **8× mais**. Não é "melhor": é **outro ponto de operação**.

**Por que isso decide a escolha, e não a média balanceada.** A média trata os dois erros como
equivalentes, e neste domínio eles não são. Um contrato em que o sistema afirma existir uma
cláusula de rescisão que não existe é um erro de outra natureza que "não encontrei". A 10% de
invenção, uma em cada dez respostas afirmativas é ficção com aparência de citação — e o
`qwen2.5:7b`, com 1,3%, é o que se manda para produção jurídica. **O modelo maior perde por uma
razão que a métrica agregada premiava.**

**A justificativa metodológica desta seção inteira.** Foi para este caso que o McNemar foi feito
**por recorte** e não só no agregado: no "geral", os 36 acertos exclusivos de cobertura e os 14 de
não-alucinação **se cancelam parcialmente** (52 discordantes, p três ordens de grandeza mais fraco).
Um teste só do agregado teria reportado "o gemma é melhor, p = 0,004" — verdadeiro e enganoso.
Também é o caso que exigiu os **dois regimes** de teste: com 15 discordantes na não-alucinação a
aproximação χ² não vale, e o p vem do binomial exato — os dois números da tabela acima saem de
testes diferentes, e a coluna `método` existe para que isso não passe como detalhe.

> **Correção de reporte, achada ao escrever esta tabela.** O p da cobertura saía como **`0.0`** —
> `round(p, 6)` engolindo 2,28e-08. Um p-valor **nunca é zero**, e a perda de ordem de grandeza
> acontecia justamente onde a evidência é mais forte. Trocado por 3 algarismos significativos.

**O que o arco de geração fecha, então:** o prompt equilibrado é um ganho **de graça** (mais
cobertura, não-alucinação idêntica, p = 6,6 × 10⁻⁵) e entra; o gerador maior é um **trade-off**
medido, e **não entra** — o preço em invenção não vale a cobertura neste domínio. Os dois foram
listados como "próximos passos" na §13.4 com a mesma plausibilidade; medir separou um do outro.

## Fase 6 — encerrada

Os dois eixos fechados: **escala** (ingestão de 17,2 M + motor escolhido com benchmark) e
**benchmark externo** (CUAD, o arco completo BM25 → denso → híbrido → **rerank**, mais a geração
ancorada — tudo com IC, tudo sobre gold de terceiros, **custo de API zero**).

O arco de recuperação, medido de ponta a ponta:

| recuperador | recall@5 | ganho significativo? |
|---|---:|---|
| BM25 | 0,588 | — (baseline) |
| denso e5-small | 0,535 | perde no agregado, **complementar** por categoria |
| denso bge-large-en | 0,532 | **não** — 7,3× mais lento, ICs sobrepostos |
| híbrido RRF | 0,595 | **não** — IC sobrepõe o BM25 |
| **+ rerank cross-encoder** | **0,652** | **SIM — IC disjunto** |

Isso **re-deriva a arquitetura da Fase 1** (denso + BM25 + RRF + rerank) peça por peça, num dataset
que não é nosso — e mostra que o rerank não é enfeite: é o único estágio cujo ganho sobrevive ao IC.

**As três "extensões" da §12 renderam mais que o previsto:** uma **refutou minha própria previsão**
(o embedder forte não ajudou), uma **confirmou a hipótese com evidência** (rerank), e a terceira
expôs que **o gargalo migrou da busca para o gerador** — com uma ablação mostrando que parte do
problema era o prompt de quem media.

O que fica genuinamente aberto, e agora com número para priorizar: política de abstenção calibrada
sobre o escore do cross-encoder (§13.3), chunking consciente de cláusula (a hipótese (a) da §13.1),
e um gerador maior. Nenhum deles muda os achados **estruturais** já estabelecidos.
