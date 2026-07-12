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

## Como subir (uma vez, na conta HuggingFace do dono do projeto)

```bash
# 1) Crie um Space (SDK Gradio) em huggingface.co/new-space  ->  ex.: alanjoffre/rodoia-rag
# 2) Clone o repo do Space e copie estes arquivos:
git clone https://huggingface.co/spaces/<seu-usuario>/rodoia-rag && cd rodoia-rag
cp /caminho/rodoia/deploy/hf_space/{app.py,requirements.txt,README.md} .

# 3) O corpus de normas (data/raw/normas/normas.jsonl) NÃO está no Git público.
#    Opção A (recomendada): rode o pipeline e inclua o arquivo no Space:
#        python -m rodoia.rag.baixar_normas   # gera normas.jsonl
#        mkdir -p data/raw/normas && cp .../normas.jsonl data/raw/normas/
#    Opção B: adicione um subconjunto curado de resoluções para uma demo leve.

# 4) (Opcional) Ative a geração: no Space, Settings -> Secrets -> HF_TOKEN=<seu token de leitura>
# 5) Suba:
git add . && git commit -m "deploy: RodoIA RAG demo" && git push
```

O Space builda a imagem (instala `rodoia[rag]` do GitHub), sobe o Gradio e fica **público numa
URL** `https://huggingface.co/spaces/<seu-usuario>/rodoia-rag` — a demo viva que faltava, **sem
custo de nuvem**.

## Por que Spaces (e não Cloud Run) para a demo
- **Grátis** (CPU free-tier) e **HTTPS + URL pública** prontos — ideal para portfólio.
- Sem cartão de crédito / sem risco de fatura.
- O deploy em cloud gerenciada (Cloud Run) continua como runbook em [docs/16 §7](../../docs/16_fase5_mlops.md)
  para o cenário de produção/escala.

## Limitações honestas
- CPU free-tier: latência de geração alta (por isso o modo retrieval-only é o padrão sem token).
- O corpus precisa ser provido no Space (passo 3) — o repo público não versiona dados brutos.
