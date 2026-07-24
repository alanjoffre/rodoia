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
python -m rodoia.mlops.gate                         # 17/17
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
- [ ] Avaliação de recuperação sobre o CUAD (Recall@k / MRR / abstenção)
- [ ] Benchmark de motor (DuckDB vs Spark) sobre o mesmo conjunto de queries

## 8. Bloco "benchmark externo" — CUAD ingerido e aferido

[CUAD](https://www.atticusprojectai.org/cuad) (Apache 2.0) — 510 contratos comerciais anotados
por advogados. `rag/baixar_cuad.py` → `rag/cuad.py` normaliza o `CUAD_v1.json` (SQuAD 2.0) para
`contratos.jsonl` + `perguntas.jsonl`, e afere a integridade:

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

### Próximo passo — avaliação, ainda sem LLM

Com o gold conferido, **Recall@k, MRR e nDCG saem sem uma única chamada de modelo**, reusando
`rag/avaliacao_retrieval.py` e o `estat.py` (Wilson/bootstrap). O que exige API é só a geração —
e só depois que a recuperação justificar o gasto.
