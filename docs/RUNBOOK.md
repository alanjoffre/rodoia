# RUNBOOK — como rodar o RodoIA do zero

> Passo a passo para colocar o projeto de pé numa máquina limpa e reproduzir toda
> a Fase 0 (dados → baseline → diagnóstico). Local-first: roda 100% na sua máquina.

## 0. Pré-requisitos

- **Python 3.12** (o piso declarado: é o que o CI, o container e a Nitro rodam — não afirmamos
  suporte a 3.11 porque não o exercitamos) · **git** · ~2 GB de disco livre.
- Acesso à internet na primeira execução (baixa os dados públicos da ANTT).
- Opcional (Fase 2, concluída — resultado principal: **NER, docs/13**; baseline em docs/10–11):
  máquina com GPU NVIDIA (Nitro/RTX 4050) em WSL2/CUDA para fine-tuning e vLLM. Fase 0 roda no Mac.

## 1. Clonar e preparar o ambiente

```bash
git clone <url-do-repo> rodoia && cd rodoia

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e ".[dev,fundamentos]"  # núcleo + ferramentas + libs de ML/DL da Fase 0
```

Os "extras" separam o peso por fase — quem só quer ler o RAG (Fase 1) não instala
PyTorch. Extras disponíveis: `fundamentos` (Fase 0), `rag` (Fase 1), `ft` (Fase 2),
`estruturados` (Fase 3), `agente` (Fase 4), `mlops` (transversal), `dev` (lint/teste).

## 2. Ativar a barreira anti-segredo (pre-commit)

```bash
pre-commit install     # passa a rodar detect-secrets + ruff a cada commit
```

## 3. Configurar segredos (opcional na Fase 0)

```bash
cp .env.example .env   # preencher só quando usar LLM (Fase 1+). NUNCA comitar o .env.
```

## 4. Obter os dados (duas opções)

**Opção A — reproduzir do zero (recomendada, não depende de remote):**
```bash
python -m rodoia.data.baixar_acidentes      # baixa 39 CSVs da ANTT -> data/raw/ (~126 MB)
dvc add data/raw/acidentes                   # (opcional) versiona no seu DVC
```

**Opção B — se você tem acesso ao remote DVC:**
```bash
dvc remote add -d --local localstore "$HOME/dvc-remotes/rodoia"   # ajuste o caminho
dvc pull                                                          # baixa dados + parquet
```

## 5. Rodar o pipeline da Fase 0

```bash
# 1) Consolida os 39 CSVs -> 37 concessionárias -> data/processed/acidentes.parquet (~1,03M acidentes)
python -m rodoia.data.ingestao_acidentes

# 2) Baseline: treina e compara 4 modelos -> reports/fase0_baseline/
python -m rodoia.ml.classico
#    (dev rápido: python -m rodoia.ml.classico --amostra 200000)

# 3) Diagnóstico: bias/variância, calibração, clustering -> reports/fase0_diagnostico/
python -m rodoia.ml.diagnostico
```

Saídas geradas: métricas em `reports/**/metrics.json` e `diagnostico.json`;
figuras `.png`; tabelas de comparação `.md`.

## 6. Rodar os testes

```bash
pytest                 # deve terminar tudo verde
```

## 7. (Futuro) Publicar no GitHub

O repositório ainda **não** tem remote. Para publicá-lo (público, como planejado):

```bash
gh repo create rodoia --public --source=. --remote=origin --push
# ou, manual:
# git remote add origin git@github.com:<user>/rodoia.git && git push -u origin main
```

Antes de publicar, conferir: nenhum `.env` rastreado (`git ls-files | grep .env`
deve mostrar só `.env.example`), CI verde e README revisado.

---

## Mapa de comandos por objetivo

| Quero… | Comando |
|---|---|
| Baixar os dados | `python -m rodoia.data.baixar_acidentes` |
| Gerar o parquet consolidado | `python -m rodoia.data.ingestao_acidentes` |
| Treinar/comparar modelos | `python -m rodoia.ml.classico` |
| Diagnosticar o modelo | `python -m rodoia.ml.diagnostico` |
| Rodar os testes | `pytest` |
| Ver o que cada fase prova | `README.md` (tabela de rastreabilidade) |
| Entender uma decisão | `docs/00`–`docs/03` |
