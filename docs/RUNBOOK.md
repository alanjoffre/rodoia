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
python -m rodoia.ingestao.baixar_acidentes      # baixa 39 CSVs da ANTT -> data/raw/ (~126 MB)
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
python -m rodoia.ingestao.ingestao_acidentes

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

## Ambientes efêmeros de verificação (receitas)

Três venvs **descartáveis** em `/tmp` sustentam verificações que **não podem** rodar no
ambiente principal — instalá-las no `.venv` da GPU quebraria o stack CUDA validado da Fase 2
(torch 2.11+cu130 / vLLM 0.24). Como vivem em `/tmp`, somem no reboot: as receitas ficam aqui
para que apagá-las não custe conhecimento.

**1. Reprodução do CI** — a rede que pegou a divergência `numpy 2.3.5 vs 2.5.1` e a ausência do
`pyarrow` no lock. **Rodar antes de todo push**: passar no `.venv` local não prova nada sobre o CI.

```bash
python3.12 -m venv /tmp/ci_repro
/tmp/ci_repro/bin/pip install --require-hashes -r requirements-ci.lock
/tmp/ci_repro/bin/pip install -e . --no-deps
/tmp/ci_repro/bin/ruff check . && /tmp/ci_repro/bin/mypy src \
  && /tmp/ci_repro/bin/pytest -q && /tmp/ci_repro/bin/python -m rodoia.mlops.gate
```

**2. DVC** — `dvc[s3]` arrasta muita dependência; isolado para não tocar no ambiente de GPU.

```bash
python3.12 -m venv /tmp/dvc_venv && /tmp/dvc_venv/bin/pip install "dvc>=3.51"
/tmp/dvc_venv/bin/dvc add data/raw/<novo> data/processed/<novo>
```

> ⚠️ **Quem insere a entrada no `data/*/.gitignore` é o `dvc add`.** Sem DVC instalado, dado novo
> nasce **rastreável** pelo Git — foi assim que o zip de 1,4 GB do CFPB chegou a um `git add -A` de
> entrar no repositório público.

**3. Benchmark de motor (Spark)** — exige JVM, que briga com o stack CUDA.

```bash
sudo apt-get update && sudo apt-get install -y openjdk-17-jre-headless   # o update é OBRIGATÓRIO
python3.12 -m venv /tmp/spark_venv
/tmp/spark_venv/bin/pip install "pyspark>=3.5" duckdb pyarrow pydantic-settings numpy certifi
/tmp/spark_venv/bin/pip install -e . --no-deps
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
/tmp/spark_venv/bin/python -m rodoia.mlops.benchmark_motor
```

> ⚠️ Duas armadilhas medidas: sem `apt-get update` o índice vem stale e o JRE dá **404**; e
> `pip install -e . --no-deps` **não** puxa as dependências transitivas do `rodoia`
> (`pydantic-settings`, `numpy`), então `config.py` estoura no import.

**O `openjdk-17-jre-headless` (184 MB) fica instalado no sistema de propósito:** sem ele,
`reports/fase6_escala/benchmark_motor.json` deixa de ser reproduzível — e um artefato versionado
que não se pode regenerar é exatamente o que este projeto não aceita.

---

## Mapa de comandos por objetivo

| Quero… | Comando |
|---|---|
| Baixar os dados | `python -m rodoia.ingestao.baixar_acidentes` |
| Gerar o parquet consolidado | `python -m rodoia.ingestao.ingestao_acidentes` |
| Treinar/comparar modelos | `python -m rodoia.ml.classico` |
| Diagnosticar o modelo | `python -m rodoia.ml.diagnostico` |
| Rodar os testes | `pytest` |
| **Reproduzir o CI antes do push** | ver *Ambientes efêmeros* §1 |
| Ingerir o CFPB (17,2 M linhas) | `python -m rodoia.ingestao.baixar_cfpb && python -m rodoia.ingestao.ingestao_cfpb` |
| Avaliar recuperação no CUAD | `python -m rodoia.rag.avaliacao_cuad` (BM25) · `..._denso` · `..._hibrido` · `..._rerank` |
| Medir alucinação (LLM local) | `python -m rodoia.rag.avaliacao_cuad_geracao --amostra 150` |
| Comparar motores de dados | ver *Ambientes efêmeros* §3 |
| Ver o que cada fase prova | `README.md` (tabela de rastreabilidade) |
| Entender uma decisão | `docs/00`–`docs/03` |
