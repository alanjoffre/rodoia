# 16 — Fase 5: MLOps, operação e deploy

> Torna o **ciclo de vida de produção visível**: gate de avaliação, CI/CD, versionamento
> de experimentos e dados, containerização, observabilidade e drift — **executados de
> verdade, local, sem custo de nuvem**. O deploy em cloud fica como **runbook** pronto
> (não executado, por decisão de orçamento).

## 1. Gate de avaliação — o portão de qualidade (`mlops/gate.py`)

O coração da fase: um gate que lê os relatórios **já versionados** em `reports/` (produzidos
pelas Fases 0–4) e compara cada métrica-chave contra um **piso**. Se qualquer uma regredir, o
processo sai com código 1 e **reprova o CI**. Não precisa de GPU/modelo — opera sobre JSON, roda
em segundos no GitHub Actions.

```
$ python -m rodoia.mlops.gate
  [✓] F0 · MLP ROC-AUC                   0.813 >= 0.78
  [✓] F1 · RAG hit@5 (híbrido)           0.62 >= 0.58
  [✓] F1 · corpus (nº de normas)         125 >= 100
  [✓] F1 · κ humano (relevância)         0.8643 >= 0.6
  [✓] F1 · κ humano (rótulo-gold fonte)  0.9168 >= 0.6
  [✓] F1 · precisão de citação           0.917 >= 0.85
  [✓] F2 · NER F1 (FT QLoRA)             0.7735 >= 0.72
  [✓] F2 · ganho FT vs base              0.6429 >= 0.55
  [✓] F3 · Holt-Winters MAPE             13.25 <= 15.0
  [✓] F3 · HW bate naïve (pareado)       True == True
  [✓] F4 · roteamento (n=21, exato)      0.952 >= 0.85
  [✓] F4 · juiz rota adequada            2.0 >= 1.5
  12/12 portões OK — APROVADO
```

Os pisos ficam **abaixo** dos valores atuais (toleram ruído de reexecução, pegam regressão real).
Baixar um piso é uma decisão consciente — aparece no diff e no code review.

> **O que este gate é e o que NÃO é (honestidade).** É um **guardrail de regressão de artefato**:
> impede que um relatório commitado seja trocado por um pior sem revisão. Ele **não re-executa
> modelo** — confia no JSON versionado. A **reprodução real** (regenerar a métrica a partir do
> modelo/dados e conferir contra o commitado) é o job **`reproduzir`** (§2.1), que exige GPU e roda
> à parte. Separar os dois é proposital: o CI barato dá feedback em segundos; a reprodução cara roda
> sob demanda/agendada.

## 2. CI/CD — GitHub Actions (`.github/workflows/ci.yml`)

Em cada push/PR para `main`, três portões **bloqueantes**:

1. **Lint** — `ruff check .` (o repositório é ruff-clean; regras `E,F,I,UP,B,ASYNC`).
2. **Testes** — `pytest` (155 testes). Os testes de fundamentos que exigem PyTorch são
   **pulados na coleta** em CPU (`tests/conftest.py`) — validados localmente na Nitro; todo o
   resto (RAG, NER, dados, agente, gate, MLOps) roda no CI sem stack de GPU.
3. **Gate de avaliação** — `python -m rodoia.mlops.gate` (regressão de métrica falha o pipeline).

Instalação enxuta e CPU-only: `.[dev,agente,estruturados]` + `qdrant-client rank-bm25 fastapi
httpx uvicorn`. Nada de torch/vLLM/transformers (todos lazy ou "fakeados" nos testes).

### 2.1 Reprodução real — `.github/workflows/reproduzir.yml` (`mlops/reproduzir.py`)

O gate do §1 é barato e honesto sobre seu limite: **não regenera métrica**. A reprodução de fato
fica num job separado que **re-executa o pipeline** e falha se o resultado divergir do JSON
commitado — respondendo diretamente a "seu CI só lê números que você mesmo commitou".

- **Âncoras (2):** (1) `hit@5` do retrieval híbrido e (2) `MAPE` do Holt-Winters na previsão —
  ambas **determinísticas** e **CPU**, sem LLM; reproduzem **exatas** (Δ=0,0 contra os reports).
  O workflow **reconstrói do zero** tanto o índice (corpus público) quanto o DuckDB (volume público),
  então **as duas âncoras rodam no CI hosted**; a de previsão **pula graciosamente** (sem reprovar o
  job) só se o CKAN da ANTT estiver fora do ar. Evidência (com carimbo `git_sha`/`git_dirty`) em
  `reports/fase1_retrieval/reproducao.json`.
- **Onde roda:** runner **github-hosted** (`ubuntu-latest`), `workflow_dispatch` + agendado semanal.
  O job **baixa o corpus público** (`baixar_normas`), **reconstrói o índice** (`construir_indice`) e
  **re-executa** o retrieval — sem esconder atrás de GPU. Fica fora do `ci.yml` de cada push porque é
  lento e depende de rede externa (a fonte da ANTT).
- **Extensível:** o mesmo harness recebe âncoras mais pesadas (NER F1 via vLLM) num runner com GPU.

## 3. MLflow — rastreio de experimentos (`mlops/rastreio.py`)

Consolida as métricas das Fases 0–4 num store MLflow (backend **SQLite**, o file store foi
descontinuado), uma run por fase — versões, parâmetros e métricas navegáveis:

```bash
pip install -e ".[mlops]"
python -m rodoia.mlops.rastreio     # 5 runs -> mlruns/mlflow.db
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

`coletar()` (a extração das métricas) é pura e testada; `registrar()` faz o log. O `mlruns/`
fica fora do Git (é store local, regenerável).

## 4. DVC — dados e modelos versionados

`.dvc/` inicializado; dados brutos/processados e modelos (`models/`, `data/`) ficam **fora do
Git** com apontadores `.dvc` + `.gitignore` por diretório (ver [data/README.md](../data/README.md)).
O pipeline de cada fase reproduz os artefatos a partir do código + fontes públicas. Nenhum binário
pesado no histórico (o pre-commit `check-added-large-files` reforça, limite 1 MB).

## 5. Containerização (`Dockerfile`, `docker-compose.yml`)

Imagem da API (RAG + agente) em `python:3.12-slim`, torch **CPU-only** (a inferência pesada fica
nos serviços dedicados). O compose sobe a plataforma:

```bash
docker compose up --build                       # API em http://localhost:8080
docker compose exec ollama ollama pull qwen2.5:7b   # cérebro (1ª vez)
```

- **api** — FastAPI (`/perguntar` RAG, `/agente` orquestrado, `/health`), índice Qdrant embutido
  + DuckDB montados de `data/processed`.
- **ollama** — o cérebro (roteamento + síntese).
- **vllm** (comentado) — o modelo fine-tunado NER em GPU (rota de entidades); exige NVIDIA
  Container Toolkit.

> **Nota honesta:** o Docker Desktop não estava com integração WSL ativa neste ambiente, então o
> `docker build` **não foi executado aqui**; os arquivos foram escritos e revisados. Rodar o build
> é o passo do runbook (§7).

## 6. Observabilidade e drift (`mlops/drift.py`)

**Observabilidade** já vinha das fases: latência/tokens/qualidade medidos e salvos (RAG em
`avaliacao_geracao.json` → `geracao_p50_s`, `tokens_resposta_medio`; serving em
`benchmark_vllm.json` → tok/s; agente na trajetória). O serving agora emite também **uma métrica
estruturada por requisição** (`observabilidade.registrar_metrica`: latência, cache_hit, taxa de hit).
O gate transforma as métricas-chave em portão.

**Cache — efeito MEDIDO (não afirmado)** — `mlops/carga.py` faz um **teste de carga sob concorrência**
(backend simulado por `sleep`, latência divulgada — isola o efeito do cache, não do Ollama).
Resultado (`reports/fase5_mlops/carga.json`):

| Taxa de hit | Redução p50 | Redução p95 |
|---|---|---|
| 0,66 | **−100%** | −0% |
| 0,88 | **−100%** | −0% |
| 0,96 | **−100%** | **−100%** |

→ **Achado honesto:** o cache **colapsa a mediana** (consulta repetida = instantânea) em qualquer
taxa de hit; mas o **p95 só cai quando o hit passa de ~95%** — abaixo disso a cauda são os *misses*,
que pagam a geração inteira. Ou seja, o cache é um ganho de **mediana**, não de p95, a menos que o
tráfego seja muito repetitivo. Isso escala à latência real (p95 ≈ 30 s na F1): mesma relação, outra
magnitude. Reportar o limite do cache (não "resolve o p95") é a disciplina do projeto.

**Drift** — PSI (Population Stability Index) sobre o volume mensal **por praça**, últimos 12 meses
vs. os 12 anteriores, na **mesma coorte** de praças (isola drift de demanda do crescimento da
malha). Resultado atual:

```
PSI = 0.0051 (estável) -> monitorar
```

Faixas: < 0,1 estável · 0,1–0,25 moderado · > 0,25 relevante (dispararia re-treino da previsão).
A escolha da coorte comum foi deliberada: sobre o volume **agregado** o PSI dava ~11 (a rede
cresceu ~10× desde 2010) — um artefato de expansão, não drift de demanda. Corrigir o alvo para a
coorte estacionária é a mesma disciplina de rigor das fases anteriores.

### 6.1 Custo de serving — R$/1k req da vazão MEDIDA (`mlops/custo.py`)

Um staff não diz "escala" sem cifra. Derivamos o custo **da vazão real** do modelo FT em vLLM
(`benchmark_vllm.json`: **2,05 req/s**, 205 tok/s, 5,2 GB VRAM medidos) — aritmética, não cotação
ao vivo. A honestidade está em separar o **piso** (marginal, 100% de utilização) da **realidade**
(always-on a ~30%, pagando GPU ociosa). Premissas explícitas: câmbio R$5,40/US$, preços-hora de
GPU pequena ~2026 (`reports/fase5_mlops/custo.json`):

**Rota FT** (NER, `max_tokens=128`) — da **vazão concorrente medida** (2,05 req/s):

| GPU (premissa de preço) | R$/1k req (marginal) | R$/1k req (always-on 30%) | R$/mês (1 instância) |
|---|---|---|---|
| L4 24GB (spot) | **0,21** | 0,68 | 1.089 |
| RTX 4090 (comunidade) | 0,32 | 1,07 | 1.711 |
| L4 24GB (on-demand) | 0,51 | 1,71 | 2.722 |
| A10G 24GB (on-demand) | 0,73 | 2,44 | 3.888 |

**Rota RAG** (a de fato user-facing, 7B, geração longa) — da **latência p50 medida** (20,7 s),
single-stream (teto: sem batching medido; batching baixaria):

| GPU (premissa de preço) | R$/1k req (marginal) | R$/1k req (always-on 30%) |
|---|---|---|
| L4 24GB (spot) | **8,69** | 28,96 |
| RTX 4090 (comunidade) | 13,65 | 45,51 |
| L4 24GB (on-demand) | 21,72 | 72,40 |
| A10G 24GB (on-demand) | 31,03 | 103,43 |

→ **Achado:** servir o modelo FT é **barato** (centavos/1k) porque é pequeno (fp8, 5,2 GB); a rota
RAG é **~40× mais cara** — os 20,7 s de geração do 7B dominam. Em ambas, o custo real de um endpoint
de portfólio é **dominado pela GPU ociosa** (always-on ≈ 3,3× o marginal, = 1/utilização).
Conclusão de engenharia: para tráfego baixo, **escala-a-zero / sob demanda** vence always-on —
coerente com o runbook (§7, Cloud Run scale-to-zero). Ressalva honesta: scale-to-zero **adiciona
cold-start** (carregar o 7B na VRAM, dezenas de s na 1ª req após ociosidade) — o trade-off de escalar
a zero. Números em `reports/fase5_mlops/custo.json`.

## 7. Deploy — runbook

Tudo acima roda local. Há **dois caminhos**, e o gratuito está pronto:

**(0) Demo pública GRÁTIS — HuggingFace Spaces (recomendado p/ portfólio).** Hospeda o RAG da
Fase 1 no CPU free-tier (R$0, HTTPS + URL pública), com citação de fonte e guardrails; geração
opcional via Inference API. Arquivos e passo a passo em [`deploy/hf_space/`](../deploy/hf_space/README.md).
Resolve o "não tem demo viva" sem custo de nuvem.

**(1) Cloud gerenciada (produção/escala) — NÃO executado, por decisão de orçamento:**

1. **Build e push da imagem**
   ```bash
   docker build -t rodoia-api:1.0 .
   docker tag rodoia-api:1.0 <registry>/rodoia-api:1.0 && docker push <registry>/rodoia-api:1.0
   ```
2. **Serviço gerenciado justificado** — **Google Cloud Run** (ou AWS App Runner): container
   stateless, escala a zero (custo ~0 em repouso — ideal para portfólio), HTTPS gerenciado, deploy
   por imagem. A API é sem estado (índice/DuckDB montados; cérebro externo), então casa com
   serverless de contêiner. GPU (modelo FT) iria para um serviço à parte (Cloud Run GPU / EC2 g5)
   **sob demanda**, não 24/7.
3. **Cérebro em endpoint hosted** — em produção o roteamento/síntese vai para uma API hosted
   (a interface `OpenAICompatLLM` já suporta), liberando GPU e removendo o Ollama do caminho crítico.
4. **Segredos** — via secret manager do provedor (nunca no Git; `.env` só local).
5. **Custo/latência** — cifra derivada da vazão medida (§6.1): FT a **R$0,21–0,73/1k req**
   (marginal), dominado pela **GPU ociosa** no always-on → scale-to-zero é a escolha certa p/
   tráfego baixo. Cloud Run cobra por request/CPU-s; p95 de geração ~30 s (medido na F1) domina a
   latência → tuning de `max_tokens` e cache de respostas frequentes antes de escalar.
6. **Observabilidade em produção** — logs estruturados + as métricas do §6 exportadas; alerta se
   o gate (rodado periodicamente sobre uma amostra rotulada) cair, ou se o PSI de drift subir.

## 8. Critérios de conclusão

- [x] **Containerização** — Dockerfile + compose (API + Ollama + vLLM opcional). *Build não
      executado neste ambiente (Docker/WSL off) — documentado no runbook.*
- [x] **CI/CD com avaliação como gate** — GitHub Actions: lint + testes + **gate de regressão**.
- [x] **MLflow + DVC** — rastreio das métricas (sqlite, 5 runs) + dados/modelos via DVC.
- [~] **Deploy em cloud** — **runbook completo (§7), não executado** (decisão de orçamento).
- [x] **Observabilidade** — latência/tokens/qualidade medidos e versionados; gate sobre eles.
- [x] **Custo de serving** — R$/1k req derivado da vazão MEDIDA (§6.1, `custo.json`): marginal vs
      always-on, com premissas explícitas.
- [x] **Drift** — PSI por coorte (0,0051, estável) com faixa de ação.
- [x] **README final** com desenho completo e rastreabilidade preenchida.

### Encerramento
Com a Fase 5, o RodoIA cobre o ciclo completo — do backprop à mão ao gate de avaliação no CI. O
único item deliberadamente não executado é o deploy em nuvem (runbook pronto). Ver o apêndice de
**decisões e trade-offs** no [README](../README.md#decisões-e-trade-offs-o-arco-do-projeto).
