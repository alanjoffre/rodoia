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
  [✓] F1 · RAG hit@5 (híbrido)           0.64 >= 0.58
  [✓] F1 · precisão de citação           0.917 >= 0.85
  [✓] F2 · NER F1 (FT QLoRA)             0.7735 >= 0.72
  [✓] F2 · ganho FT vs base              0.6429 >= 0.55
  [✓] F3 · Holt-Winters MAPE             13.25 <= 15.0
  [✓] F3 · HW bate naïve (pareado)       True == True
  [✓] F4 · roteamento (acerto exato)     1.0 >= 0.9
  [✓] F4 · juiz rota adequada            2.0 >= 1.5
  9/9 portões OK — APROVADO
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
2. **Testes** — `pytest` (142 testes). Os testes de fundamentos que exigem PyTorch são
   **pulados na coleta** em CPU (`tests/conftest.py`) — validados localmente na Nitro; todo o
   resto (RAG, NER, dados, agente, gate, MLOps) roda no CI sem stack de GPU.
3. **Gate de avaliação** — `python -m rodoia.mlops.gate` (regressão de métrica falha o pipeline).

Instalação enxuta e CPU-only: `.[dev,agente,estruturados]` + `qdrant-client rank-bm25 fastapi
httpx uvicorn`. Nada de torch/vLLM/transformers (todos lazy ou "fakeados" nos testes).

### 2.1 Reprodução real — `.github/workflows/reproduzir.yml` (`mlops/reproduzir.py`)

O gate do §1 é barato e honesto sobre seu limite: **não regenera métrica**. A reprodução de fato
fica num job separado que **re-executa o pipeline** e falha se o resultado divergir do JSON
commitado — respondendo diretamente a "seu CI só lê números que você mesmo commitou".

- **Âncora atual:** `hit@5` do retrieval híbrido — **determinística**, roda em CPU, sem LLM; a
  reprodução bate exata (Δ=0,0 contra o `avaliacao_retrieval.json`). Verificado localmente.
- **Onde roda:** runner **self-hosted com GPU** (ex.: a Nitro), `workflow_dispatch` + agendado
  semanal — porque exige o corpus/índice (e, para âncoras futuras como o NER F1, a placa). O
  GitHub-hosted não tem dados nem GPU, então essa reprodução **não** cabe no CI barato.
- **Extensível:** o mesmo harness recebe âncoras de GPU (NER F1 via vLLM) quando o runner tem placa.

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
`benchmark_vllm.json` → tok/s; agente na trajetória). O gate transforma essas métricas em portão.

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
5. **Custo/latência** — Cloud Run: cobra por request/CPU-s; p95 de geração ~30 s (medido na F1)
   domina a latência → tuning de `max_tokens` e cache de respostas frequentes antes de escalar.
6. **Observabilidade em produção** — logs estruturados + as métricas do §6 exportadas; alerta se
   o gate (rodado periodicamente sobre uma amostra rotulada) cair, ou se o PSI de drift subir.

## 8. Critérios de conclusão

- [x] **Containerização** — Dockerfile + compose (API + Ollama + vLLM opcional). *Build não
      executado neste ambiente (Docker/WSL off) — documentado no runbook.*
- [x] **CI/CD com avaliação como gate** — GitHub Actions: lint + testes + **gate de regressão**.
- [x] **MLflow + DVC** — rastreio das métricas (sqlite, 5 runs) + dados/modelos via DVC.
- [~] **Deploy em cloud** — **runbook completo (§7), não executado** (decisão de orçamento).
- [x] **Observabilidade** — latência/tokens/qualidade medidos e versionados; gate sobre eles.
- [x] **Drift** — PSI por coorte (0,0051, estável) com faixa de ação.
- [x] **README final** com desenho completo e rastreabilidade preenchida.

### Encerramento
Com a Fase 5, o RodoIA cobre o ciclo completo — do backprop à mão ao gate de avaliação no CI. O
único item deliberadamente não executado é o deploy em nuvem (runbook pronto). Ver o apêndice de
**decisões e trade-offs** no [README](../README.md#decisões-e-trade-offs-o-arco-do-projeto).
