# Dados — obtenção, licença e versionamento

> **Regra de ouro deste diretório:** dado bruto **NÃO** entra no Git. O Git
> guarda (1) o pipeline que baixa os dados, (2) esta documentação e (3) os
> apontadores `.dvc`. Os dados em si vivem num **remote DVC**. Isso protege de
> problema de licença e mantém o repositório público leve.
>
> "Acesso público" **não** implica "direito de redistribuir". A licença de cada
> fonte é confirmada **antes** de qualquer uso — e registrada na tabela abaixo.

## Layout

```
data/
├── raw/         # baixado pelos pipelines; ignorado pelo Git, versionado por DVC
└── processed/   # pronto para treino/indexação; idem
```

## Como obter os dados

Pipelines de obtenção já publicados (reproduzíveis, dados brutos fora do Git via DVC):

```bash
# Fase 0 — acidentes rodoviários (ML clássico):
python -m rodoia.data.baixar_acidentes && python -m rodoia.data.ingestao_acidentes
# Fase 1 — resoluções da ANTT (RAG):
python -m rodoia.rag.baixar_normas
# Fase 3 — dados estruturados (volume de pedágio):
python -m rodoia.data.baixar_volume && python -m rodoia.data.ingestao_volume
```

### Datasets processados (`data/processed/*.jsonl`) — fora do Git, regeneráveis

Coerente com a regra de ouro, os datasets processados **não são versionados** (regeneram-se
pelo pipeline). Além disso, o **LeNER-Br** carrega **PII** (CPF/CNPJ/nomes) de decisões judiciais
públicas — legítimo sob a licença MIT, mas não redistribuímos num repo público. Regenerar:

```bash
python -m rodoia.ner.generativo          # -> ner_train.jsonl / ner_test.jsonl (baixa o LeNER-Br)
python -m rodoia.ft.construir_dataset    # -> ft_dataset*.jsonl (Q&A sintético das normas, via Ollama)
```

## Remote DVC (local-first)

Neste primeiro momento o remote DVC é **local** (um diretório fora do repo), e a
configuração fica em `.dvc/config.local` — **ignorada pelo Git**, para não vazar
caminho/usuário no repo público. Cada máquina configura o seu:

```bash
dvc remote add -d --local localstore "$HOME/dvc-remotes/rodoia"
```

Na Fase 5 (Cloud) este remote passa a apontar para um bucket (ex.: S3), sem
mudar nada no fluxo de trabalho.

## Fontes e licenças (VALIDADO em 09/07/2026)

Validação de realidade concluída — ver [docs/00_validacao_fontes_antt.md](../docs/00_validacao_fontes_antt.md)
para o levantamento completo, pegadinhas e itens ainda a confirmar por dataset.

**Veredito de licença:** repo público é seguro. Texto de norma = ato oficial fora
da proteção autoral (Lei 9.610/98, art. 8º, IV). Datasets = **CC-BY** (Decreto
8.777/2016) → exigem **atribuição à ANTT** (registrada no `NOTICE`).

| Fonte | Uso | Fase | Formato | Licença | Obs. |
|---|---|---|---|---|---|
| ANTTlegis — normas ANTT | RAG (texto) | 1 | HTML (latin-1), URL determinística, sem API | Ato oficial (domínio público, Lei 9.610/98) | scraping; OCR p/ normas antigas |
| LexML (dados abertos) | metadados/URN | 1 | JSON/dumps (SRU ao vivo tem anti-bot) | Licença aberta declarada | atribuir a base |
| Acidentes em rodovias concedidas | ML clássico (classificação) ⭐ | 0 | CSV (latin-1, `;`, decimal `,`) | CC-BY (confirmar por dataset) | 1 linha = 1 acidente; ~100k+/concessionária |
| Volume de Tráfego de Pedágio | SQL analítico + previsão de demanda (série temporal) | 3 | CSV (latin-1, `;`, decimal `,`) | CC-BY (confirmar por dataset) | ~142k linhas/ano; dados.antt.gov.br |
| Praça de Pedágio / Receita | dimensão + SQL | 3 | CSV + KMZ | CC-BY | JOINs geográficos/financeiros |
| LeNER-Br (NER jurídico PT-BR) | fine-tuning c/ rótulo objetivo | 2 | CoNLL | **MIT** (citar PROPOR 2018) | 7.827/1.176/1.389 sentenças; `rodoia.ner.lener` baixa |

> **Ler CSV da ANTT:** `pandas.read_csv(sep=';', encoding='latin-1', decimal=',')`.

## Fronteira de dados (inviolável)

Somente domínio público da ANTT e datasets públicos consagrados. **Zero** dado,
esquema, nomenclatura ou regra de negócio de qualquer empregador/cliente. Ver
`NOTICE` e a seção 3.1 do `PROMPT_MESTRE.md`.
