---
title: RodoIA — RAG ANTT
emoji: 🛣️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Demo pública gratuita — RAG sobre a regulação da ANTT

Hospeda a demo do RAG avaliado (Fase 1) no **HuggingFace Spaces (CPU free-tier, R$0)**: retrieval
híbrido (E5 + BM25 + RRF) com **citação de fonte**, guardrail anti-injection e PII masking. A
geração da resposta é **opcional** via HuggingFace Inference API (defina o secret `HF_TOKEN`); sem
token, a demo roda em modo *retrieval-only* (mostra os trechos citados) — leve e grátis.

> Este diretório **não** faz parte do pacote; são os arquivos que sobem para o Space.

## Passo a passo (uma vez, na sua conta HuggingFace)

**Pré-requisito:** uma conta grátis em huggingface.co e o `git` instalado. O `app.py` é **turnkey**:
na 1ª subida ele **baixa o corpus da ANTT e constrói o índice sozinho** (ver `_preparar_dados`). Você
só precisa subir os 3 arquivos.

1. **Crie o Space.** Em https://huggingface.co/new-space → dê um nome (ex.: `rodoia-rag`) → **SDK:
   Gradio** → **Hardware: CPU basic (free)** → Create.

2. **Clone o repositório do Space** (o HF cria um repo git para ele):
   ```bash
   git clone https://huggingface.co/spaces/<seu-usuario>/rodoia-rag
   cd rodoia-rag
   ```

3. **Copie os 3 arquivos deste diretório** para o repo do Space:
   ```bash
   cp /caminho/para/rodoia/deploy/hf_space/{app.py,requirements.txt,README.md} .
   ```
   > O `README.md` daqui já traz o cabeçalho YAML que o HF precisa (`sdk: gradio`, `app_file: app.py`).

4. **(Opcional, recomendado) Acelere o 1º boot** subindo o corpus junto (evita o scrape de ~10 min):
   ```bash
   # no seu clone do rodoia, gere o corpus e copie para o Space:
   python -m rodoia.rag.baixar_normas
   mkdir -p data/raw/normas && cp <rodoia>/data/raw/normas/normas.jsonl data/raw/normas/
   ```
   Sem este passo, o Space baixa o corpus sozinho na 1ª subida (só mais lento).

5. **(Opcional) Ligue a geração** de resposta: no Space → **Settings → Variables and secrets →
   New secret** → `HF_TOKEN` = um token de leitura (huggingface.co/settings/tokens). Sem token, a
   demo funciona em modo *retrieval-only* (mostra os trechos citados).

6. **Suba:**
   ```bash
   git add . && git commit -m "deploy: RodoIA RAG demo" && git push
   ```

7. **Acompanhe o build** na aba **Logs** do Space. Na 1ª vez ele instala as libs, (baixa o corpus) e
   constrói o índice — pode levar alguns minutos. Quando ficar **"Running"**, sua demo está no ar em
   `https://huggingface.co/spaces/<seu-usuario>/rodoia-rag`.

Pronto: a **demo viva** que faltava, **URL pública + HTTPS, sem custo de nuvem**. Cole o link no
README principal do projeto (badge/linha "Demo ao vivo").

## Por que Spaces (e não Cloud Run) para a demo
- **Grátis** (CPU free-tier) e **HTTPS + URL pública** prontos — ideal para portfólio.
- Sem cartão de crédito / sem risco de fatura.
- O deploy em cloud gerenciada (Cloud Run) continua como runbook em [docs/16 §7](../../docs/16_fase5_mlops.md)
  para o cenário de produção/escala.

## Limitações honestas
- CPU free-tier: latência de geração alta (por isso o modo retrieval-only é o padrão sem token).
- O corpus precisa ser provido no Space (passo 3) — o repo público não versiona dados brutos.
