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
├── interim/     # intermediário; idem
└── processed/   # pronto para treino/indexação; idem
```

## Como obter os dados

Ainda **não** há pipeline publicado — este é o estado inicial do repositório
(Fase 0, item 0). A obtenção será, por fonte:

```bash
# (a preencher na Fase 1/Fase 3) — cada fonte terá um comando reproduzível, ex.:
# python -m rodoia.data.baixar_resolucoes_antt
# dvc add data/raw/resolucoes_antt && dvc push
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

## Fontes e licenças (VALIDAÇÃO DE REALIDADE PENDENTE)

Esta tabela é preenchida conforme a **Primeira Tarefa (Seção 6 do prompt-mestre)**
for executada. Nenhuma fonte abaixo foi ainda confirmada quanto a formato e
licença — por isso a coluna licença está como "a confirmar".

| Fonte | Uso | Fase | Formato | Licença | Status |
|---|---|---|---|---|---|
| Legislação/resoluções/portarias ANTT | RAG (texto regulatório) | 1 | a confirmar (PDF/HTML) | a confirmar | pendente |
| Dados abertos estruturados ANTT (frota, fiscalização, tarifas…) | ML clássico + agente | 0/3 | a confirmar (CSV/API) | a confirmar | pendente |

## Fronteira de dados (inviolável)

Somente domínio público da ANTT e datasets públicos consagrados. **Zero** dado,
esquema, nomenclatura ou regra de negócio de qualquer empregador/cliente. Ver
`NOTICE` e a seção 3.1 do `PROMPT_MESTRE.md`.
