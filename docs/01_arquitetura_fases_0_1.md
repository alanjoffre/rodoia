# 01 — Esboço de arquitetura das Fases 0 e 1

> Ajustado à [validação de fontes](00_validacao_fontes_antt.md). É um esboço de partida; cada incremento é validado e testado antes do próximo. Uma fase por vez, um incremento testável por vez.

---

## Fase 0 — Fundamentos de ML/DL (dataset: Acidentes em rodovias concedidas)

**Problema escolhido:** classificação binária de **severidade de acidente** (`com vítima` vs `sem vítima`, e/ou `houve morto`) — dado desbalanceado de propósito, para exercitar diagnóstico de verdade.

### Fluxo de dados

```
dados.antt.gov.br (CSV latin-1, ; , vírgula decimal)
   │  [pipeline de download versionado no Git]
   ▼
data/raw/acidentes/*.csv        ← DVC (fora do Git), 1 arquivo por concessionária
   │  [ingestão: normaliza encoding, unifica schema, valida colunas]
   ▼
data/processed/acidentes.parquet ← DVC
   │
   ├─► ML clássico (scikit-learn): LogReg · Árvore · RandomForest · GradientBoosting
   │      + clustering (KMeans) exploratório
   │      métricas: F1, ROC-AUC, matriz de confusão, validação cruzada, class_weight
   │
   ├─► Diagnóstico: curvas de aprendizado, bias/variance, importância de features
   │
   └─► MLP em PyTorch (MPS no Mac) — laço de treino manual
          + 1 passo de gradiente à mão em NumPy (prova de backprop)
```

### Componentes independentes de dado (provam fundamento, rodam standalone)

- **Bloco de self-attention** (scaled dot-product) em PyTorch puro + **teste de equivalência** com `torch.nn.functional.scaled_dot_product_attention`.
- **Notebook de matemática aplicada:** gradiente, álgebra no forward pass, por que cross-entropy é a loss natural.

### Estrutura de código (Fase 0)

```
src/rodoia/
├── data/
│   ├── baixar_acidentes.py      # download reproduzível (fonte pública) + dvc add
│   └── ingestao_acidentes.py    # latin-1→utf-8, unifica schema, parquet
├── ml/
│   ├── classico.py              # treino/compara modelos sklearn + métricas
│   ├── diagnostico.py           # curvas, bias/variance, feature importance
│   └── mlp_torch.py             # MLP + laço manual
├── fundamentos/
│   ├── backprop_numpy.py        # 1 passo de gradiente à mão
│   └── attention.py             # self-attention puro
notebooks/
├── 00_matematica_aplicada.ipynb
└── 01_acidentes_eda_diagnostico.ipynb
tests/
├── test_ingestao_acidentes.py   # schema, encoding, sanidade
├── test_attention.py            # equivalência com o framework
└── test_backprop_numpy.py       # gradiente manual ≈ autograd
```

**Rodagem:** tudo local no **Mac M3 Pro** (PyTorch MPS). Nenhuma GPU CUDA necessária nesta fase.

---

## Fase 1 — RAG sobre a regulação da ANTT

**Corpus inicial:** resoluções **vigentes** de transporte rodoviário (escopo enxuto, não o acervo inteiro).

### Pipeline

```
ANTTlegis (HTML latin-1, URL determinística abrirTextoAto)
   │  [scraping educado: varre tipo/número/ano; converte encoding]
   ▼
data/raw/normas/*.html + metadados (número, ano, tipo, situação, data DOU)
   │  [parsing → limpeza → chunking com estratégia justificada]
   ▼   (metadados enriquecidos com URN LexML via dumps de dados abertos)
data/processed/chunks/  ← DVC
   │  [embeddings]
   ▼
Banco vetorial (Qdrant local)  ── metadados p/ filtro (nº resolução, tema, vigência)
   │
   ▼
Retrieval (hybrid: denso + BM25) → rerank → montagem de contexto
   │
   ▼
Geração com CITAÇÃO da fonte (qual resolução/artigo embasou)
   │
   ├─► Avaliação: conjunto dourado + RAGAS (faithfulness, context precision, answer relevancy), versionado
   ├─► Governança: PII masking (ingestão+resposta) · guardrails anti-injection (testes adversariais) · trilha de auditoria
   └─► Interface: endpoint FastAPI async / UI mínima
```

### Decisões-chave (com trade-off a documentar na fase)

- **Só normas vigentes no corpus inicial** — evita o RAG citar norma revogada; o grafo de revogação completo fica para um incremento posterior.
- **Qdrant local** (não pgvector) para o local-first — sobe em container, sem servidor Postgres dedicado. Trade-off documentado no README da fase.
- **LLM/embeddings atrás de interface** — permite trocar provedor de API (Fase 1) pelo modelo próprio em vLLM (Fase 2) sem reescrever o RAG.
- **Demo sem custo aberto** (repo público): a demo ao vivo usará o modelo local da Fase 2, ou rate-limit, ou vídeo — decidido ao fechar a fase.

**Rodagem:** local no **Mac M3 Pro**. LLM/embeddings via API (com chave em `.env`) ou local; a substituição pelo modelo próprio vem na Fase 2.

---

## Sequência de execução imediata (Fase 0)

1. `baixar_acidentes.py` — baixar 1 concessionária, validar encoding/colunas, `dvc add`.
2. Confirmar **schema consistente** entre concessionárias (item aberto da validação) e unificar.
3. EDA + baseline de classificação (uma família de modelos, métricas honestas).
4. Diagnóstico (curvas, bias/variance, importância).
5. MLP PyTorch + backprop manual NumPy.
6. Self-attention à mão + teste de equivalência.
7. Notebook de matemática aplicada.
8. README da Fase 0 + atualizar a tabela de rastreabilidade.

Cada passo é um incremento testável, commitado com sua própria história.
