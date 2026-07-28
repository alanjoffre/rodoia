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

## Lotes longos de avaliação (GPU)

As avaliações do CUAD levam de segundos (BM25) a ~30 min (rerank sobre os 510 contratos), e as de
geração dependem do **Ollama**, que é um serviço à parte. Duas regras aprendidas quebrando um lote:

**1. Recuperação antes de geração.** Só a geração depende de serviço externo. Um lote que roda
geração primeiro, com `set -e`, perde as re-execuções de recuperação quando o Ollama cai — que foi
exatamente o que aconteceu: `URLError: [Errno 110] Connection timed out` no meio da rodada do
`gemma2:9b` abortou oito re-execuções que não tinham relação nenhuma com o LLM.

**2. `set -e` é a armadilha, não o remédio.** Num lote de etapas independentes ele transforma uma
falha local em perda total. O padrão aqui é `set -uo pipefail` (sem `-e`), cada etapa numa função
que conta a falha e segue, e um `falhas=N` no fim.

O Ollama do WSL **não sobe sozinho** e não é serviço systemd. Antes de qualquer lote de geração:

```bash
pgrep -a ollama || (nohup ollama serve > /tmp/ollama.log 2>&1 &)
curl -s -m 10 http://127.0.0.1:11434/api/tags > /dev/null && echo OK
```

> ⚠️ `localhost` no WSL resolve para `::1` (só IPv6). Quando o servidor está fora do ar, o erro é
> **timeout** (Errno 110), não *connection refused* — o que faz parecer problema de rede em vez de
> serviço parado. Testar contra `127.0.0.1` desambigua.

**A seed do LLM não é opcional numa ablação.** `OllamaLLM` roda a `temperatura=0.1`, que **não é
determinismo**: duas rodadas das mesmas perguntas dão respostas diferentes. Num desenho pareado
(McNemar) esse ruído entra contado como discordância. `avaliacao_cuad_geracao` fixa `seed=--seed`;
qualquer comparação nova precisa fazer o mesmo.

---

## DVC com remote — passo a passo, **não executado**

**Estado real, para não haver ilusão:** o DVC rastreia os dados (`data/**/*.dvc` versionados,
conteúdo fora do Git), mas **não há remote configurado**. Na prática isso significa que os
`.dvc` guardam o *hash* do conteúdo — servem para **detectar** que um arquivo mudou — mas
**não há de onde baixá-lo**. `dvc pull` num clone novo falha. O dado vive numa única máquina.

Isso é backup ausente, não inconveniência: os artefatos regeneram pelos scripts de download
(fontes 100% públicas), então nada é irrecuperável — mas reconstruir o CFPB são **17,2 M linhas**
reingeridas do zero.

**Não executado por decisão de escopo** (o bucket custa e não é necessário para o portfólio
provar o que se propõe). O caminho, para quando for:

```bash
# 1) Bucket dedicado, versionamento ligado (protege contra push de dado corrompido)
aws s3 mb s3://rodoia-dvc --region sa-east-1
aws s3api put-bucket-versioning --bucket rodoia-dvc \
  --versioning-configuration Status=Enabled --region sa-east-1

# 2) Ciclo de vida: o DVC nunca apaga sozinho, e cache antigo acumula em silêncio
aws s3api put-bucket-lifecycle-configuration --bucket rodoia-dvc \
  --lifecycle-configuration file://lifecycle.json   # ex.: expirar versões antigas em 90 dias

# 3) Remote no repositório
pip install "dvc[s3]>=3.51"
dvc remote add -d origem s3://rodoia-dvc/cache
dvc remote modify origem region sa-east-1
git add .dvc/config && git commit -m "chore(dvc): remote S3"

# 4) Subir o que já está rastreado, e conferir num clone limpo
dvc push
dvc status -c        # deve dizer "Cache and remote are in sync"
```

> ⚠️ **Três armadilhas específicas deste repositório.**
> 1. **Credencial nunca no `.dvc/config`** — ele é versionado. Autenticação por perfil da AWS CLI
>    ou role; o `detect-secrets` do pre-commit barra chave, mas não confie nisso como controle.
> 2. **O CFPB tem 1,1 GB de Parquet.** O primeiro `dvc push` sobe tudo; em `sa-east-1` o
>    armazenamento é barato e a **transferência de saída** não. Vale medir antes de repetir.
> 3. **`dvc add` é quem escreve a entrada no `.gitignore`.** Sem o DVC instalado, dado novo nasce
>    **rastreável pelo Git** — foi assim que o zip de 1,43 GB do CFPB quase entrou no repositório
>    (docs/17 §5). Instalar o DVC não é opcional em quem for mexer nos dados.

---

## Regerar a imagem social (`assets/social-preview.png`)

A fonte é o **SVG** (`assets/social-preview.svg`); o PNG é derivado. Editar o PNG à mão faz os dois
divergirem, e é o PNG que o GitHub e o LinkedIn exibem.

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" `
  --headless=new --disable-gpu --hide-scrollbars `
  --screenshot="assets\social-preview.png" --window-size=1280,640 `
  "file:///D:/Dev/rodoia/assets/social-preview.svg"
```

1280×640 é o tamanho exato do *social preview* do GitHub. Qualquer navegador Chromium serve
(`chrome.exe` idem); não há dependência de build no projeto por causa disso.

> ⚠️ **Trocar o PNG no repositório NÃO troca o preview do GitHub.** Ele é carregado à parte, em
> *Settings → General → Social preview → Upload an image*. Os dois já divergiram: o arquivo dizia
> `gate 12/12` quando o gate tinha 30 portões, e a imagem publicada continuou a antiga.
>
> Este é o **único passo que nenhum teste cobre** — é ação na interface do GitHub, fora do
> repositório.
>
> ✅ **Os números do SVG SÃO verificados** por `tests/test_consistencia_docs.py`: o chip do gate
> contra o `gate.py`, a contagem de testes contra o badge do README, e `recall@5` e `F1` contra os
> relatórios que os produziram. Um teste também barra a volta da afirmação "serving **em
> produção**" — o deploy em nuvem não foi executado (§7.1 do docs/16). Se algum divergir, o CI
> reprova; resta regerar o PNG com o comando acima e reenviá-lo no GitHub.

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
| Comparar duas rodadas de geração (McNemar) | `..._geracao --comparar A.json B.json` (não chama o LLM) |
| Calibrar o limiar de abstenção | `..._rerank --dump-escores && python -m rodoia.rag.calibracao_abstencao` |
| Medir a cobertura do chunker por cláusula | `python -m rodoia.rag.avaliacao_cuad --diagnostico-chunker` |
| Comparar motores de dados | ver *Ambientes efêmeros* §3 |
| Ver o que cada fase prova | `README.md` (tabela de rastreabilidade) |
| Entender uma decisão | `docs/00`–`docs/03` |
