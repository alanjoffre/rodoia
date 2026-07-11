# 11 — Fase 2: estudo-baseline (fine-tuning generativo + quantização + serving na Nitro)

> 📌 **Enquadramento:** este é o **estudo-baseline** da Fase 2 — fine-tunar um LLM para
> *responder* sobre a ANTT (injetar conhecimento). Seu resultado é um **negativo honesto**
> (FT muda estilo, não injeta fato) que **motivou o pivot** para uma tarefa de rótulo
> objetivo onde o FT vence: **NER jurídico**, o resultado principal em [`docs/13`](13_fase2_ner.md).

> Execução real na **Acer Nitro V15 (RTX 4050 Laptop, 6 GB VRAM)** dentro do
> **WSL2/Ubuntu 24.04 + CUDA 13**. Este documento registra as decisões de treino e
> serving, os desvios em relação ao handoff (`docs/10`) forçados pelo ambiente, e os
> números medidos. Filosofia da fase: **provar o pipeline** (treinar → quantizar →
> servir → medir), reportando os resultados com honestidade — não "vencer" o base.

## 1. Ambiente efetivamente usado

| Componente | Versão / valor | Observação |
|---|---|---|
| GPU | RTX 4050 Laptop, 6141 MiB | ~4.95 GiB livres p/ o vLLM (resto: baseline Windows/WSL) |
| SO | Ubuntu 24.04 (WSL2), Python 3.12 | rede do WSL em **networking mirrored** (NAT estava quebrado) |
| torch | **2.11.0+cu130** | subiu de 2.6.0+cu124 ao instalar o vLLM |
| vLLM | **0.24.0** | |
| transformers / trl / peft | **5.13.0 / 1.8.0 / 0.19.1** | muito além dos mínimos do `pyproject` |
| bitsandbytes / accelerate / datasets | 0.49.2 / 1.14.0 / 5.0.0 | |
| CUDA toolkit | pip `nvidia/cu13` (nvcc 13.2) | `CUDA_HOME` apontado p/ ele (ver §5) |

### Adaptações de código necessárias (API nova)
- `treino_qlora.py` e `merge_quantiza.py`: `torch_dtype=` → **`dtype=`** (transformers 5.x).
- `treino_qlora.py`: `SFTConfig(max_seq_length=…)` → **`max_length=`** (trl 1.x).

Essas duas edições estão versionadas. O restante da API do TRL 1.x
(`SFTTrainer(processing_class=…, peft_config=…)`) permaneceu compatível.

## 2. Treino QLoRA — resultado

`python -m rodoia.ft.treino_qlora --modelo Qwen/Qwen2.5-3B-Instruct --epocas 3`

| Métrica | Valor |
|---|---|
| Método | QLoRA (NF4 4-bit + double-quant, cômputo bf16) + LoRA r=16, α=32 |
| Épocas / steps | 3 / 33 (batch 1, grad-accum 8) |
| Runtime | **~4 min 53 s** |
| VRAM no treino | **~5814 / 6141 MiB** (coube com folga mínima) |
| Loss | **~1.07 → ~0.61** (train_loss médio 0.98) |
| `mean_token_accuracy` | 0.766 → **0.842** |
| Artefato | adaptador LoRA (57 MB) em `models/qlora-antt/` |

O treino convergiu de forma saudável em 6 GB — o objetivo central (QLoRA rodando na
RTX 4050) foi atingido.

## 3. Merge + quantização — decisão e trade-off

- **Merge** (`merge_quantiza.py --pular-awq`) rodado **na CPU** (`CUDA_VISIBLE_DEVICES=""`)
  para não competir com os 6 GB: um 3B fp16 (~5.8 GB) não cabe na GPU junto do contexto
  CUDA. Gerou `models/antt-merged/` (**5.8 GB fp16**, `model.safetensors`).

- **AWQ → substituído por fp8.** O `autoawq` (previsto no handoff) está **descontinuado
  e fixa torch antigo**; instalá-lo rebaixaria o torch e **quebraria o vLLM 0.24/cu130**.
  O `llm-compressor` não estava disponível no índice. **Decisão de engenharia:** usar a
  **quantização fp8 in-flight do vLLM** (nativa na arquitetura Ada da RTX 4050), que
  entrega o mesmo objetivo — *modelo quantizado, servível em 6 GB, com trade-off medido*.
  Documentado como desvio consciente (a competência — quantizar e medir — é a mesma).

| | fp16 (merged) | **fp8 (servido)** |
|---|---|---|
| Peso em disco | 5.8 GB | — (quantizado no load) |
| VRAM servindo (peso+KV+overhead) | **não cabe** em 6 GB | **5168 / 6141 MiB** ✅ |

fp8 ~ metade da memória dos pesos → cabe o 3B **e** o KV cache do vLLM em 6 GB.

## 4. Serving no vLLM — throughput / latência

FT servido: `vllm serve models/antt-merged-ho --quantization fp8 --max-model-len 2048
--gpu-memory-utilization 0.80 --enforce-eager` (endpoint OpenAI-compat), medido pelo
harness reprodutível `benchmark_vllm.py` (ver §5.5):

| Métrica (24 req, concorrência 6, `max_tokens` 128) | Valor |
|---|---|
| Throughput | **205.5 tokens/s** |
| Requisições/s | 2.05 |
| Latência p50 / p95 | **3.08 s** / 3.59 s |
| VRAM em uso | 5168 / 6141 MiB |

## 5. Avaliação base vs. fine-tunado — rigorosa, com held-out e IC

### 5.0 Desenho: split held-out (`split_dataset.py`)
Para medir **generalização**, reservamos **6 normas inteiras** como *held-out* (split
determinístico, seed=42) e re-treinamos nas demais. Dataset **expandido para 158 exemplos**
(30 normas × ~5, `construir_dataset` com temperatura 0) → **124 treino / 34 held-out**. Split
versionado em `reports/fase2_ft/split_holdout.json`. O corpus `normas.jsonl` foi regenerado
(Fase 1), então a **correção factual com referência** voltou a ser medível (§5.4).

### 5.1 Perplexidade — *fit de domínio vs. generalização* (`perplexidade.py`, fp8)
PPL (↓ melhor) via `prompt_logprobs`, **in-sample** (treino) e **held-out** (normas NÃO vistas):

| PPL micro | Base | Fine-tunado | Δ |
|---|---|---|---|
| **in-sample** (124 textos) | 9.73 | **8.16** | **−16.2%** |
| **held-out** (34 textos) | 11.28 | 12.21 | **+8.3%** ⚠️ |

→ Resultado de manual: com **mais dados**, o FT baixa a PPL in-sample (**−16%**, memoriza o
registro) mas **PIORA no held-out (+8%)** — **generaliza pior que o base** em normas novas.
Overfitting ao estilo do domínio à custa da generalização. (Com 84 ex. o held-out era −4%;
com 158 virou +8% — mais fine-tuning, pior generalização.)

### 5.2 Acurácia de citação — *acerta a norma?* (`aval_cite.py`) — n=50, IC de Wilson
Sobre o `CONJUNTO_DOURADO` (**50** perguntas de intenção real, `temperature=0`):

| Métrica | Base | Fine-tunado |
|---|---|---|
| **Acurácia de citação** (norma **correta**) | 0/50 [0; 0.07] | **0/50 [0; 0.07]** |
| Taxa de citação (cita *alguma*) | 0.56 | 0.50 |

→ **Nenhum** acerta a norma (IC bem estreito com n=50) — o RAG (Fase 1), não o FT, é quem dá a
fonte correta.

### 5.3 Win-rate por juiz independente (`juiz_winrate.py`) — n=50, IC de Wilson
Juiz **`qwen2.5:7b`** (checkpoint distinto), pareado com **troca de posição**. Dois modos p/
isolar o viés de comprimento:

| Modo | Base | FT | Empates | FT win-rate (IC95) |
|---|---|---|---|---|
| **Bruto** | 39 | 3 | 8 | 0.06 [0.02; 0.16] |
| **Controlado** (mesmo tamanho) | 2 | **44** | 4 | **0.88 [0.76; 0.94]** |

→ O bruto era **quase todo viés de comprimento**. Controlando o tamanho, o FT **vence 44×2**
(IC **exclui 0.5**) — a igual conteúdo, a resposta concisa e assertiva do FT é
**significativamente** preferida em *estilo*.

### 5.4 Correção factual por juiz com referência (`juiz_factual.py`) — n=50, IC
Com o `normas.jsonl` regenerado, o juiz **independente** (llama3.1:8b) pontua 0–1 a **correção
factual** de cada resposta contra o TEXTO da norma-fonte (a métrica que a citação só aproximava):

| Correção factual (↑ melhor) | Base | Fine-tunado |
|---|---|---|
| média [IC95 bootstrap] | **0.85 [0.77; 0.93]** | **0.79 [0.68; 0.90]** |

→ Base e FT ficam **em paridade factual** (ganho −0.06, **ICs se sobrepõem**) — o FT **não é
factualmente pior**. Achado importante da expansão: com **84 exemplos** o FT chegou a ser
*muito* pior (0.88 → **0.52**, ICs disjuntos); com **158** a diferença sumiu. Ou seja, **a piora
factual era em boa parte artefato de dataset pequeno** — só medível porque expandimos e refizemos.

### 5.5 Serving fp8 + trade-off de quantização (`benchmark_vllm.py`, `quantizacao_qualidade.py`)
Serving: **205 tokens/s**, **p50 3.08s / p95 3.59s**, **VRAM 5168 / 6141 MiB**. Trade-off da
quantização medido nos **dois eixos**: *memória* (fp16 5.8 GB não cabe → fp8 cabe) e
*qualidade* — PPL held-out do merged em **fp32 8.44 → 4-bit NF4 9.64 (ΔPPL +14%)**; o fp8
servido é mais preciso que NF4, logo seu custo de qualidade é **≤ 14%** (cross-check: fp8 via
vLLM ≈ +4% vs. fp32).

### Interpretação (o resultado científico)
Retrato coerente, **não-óbvio** e — após a expansão do dataset — **mais nuançado**:
- **Estilo:** o FT deixa as respostas mais concisas/assertivas e, a igual comprimento, é
  **significativamente preferido** pelo juiz (win-rate 0.88 [0.76; 0.94]).
- **Fato:** com dataset pequeno o FT **degradava** os fatos (factual 0.88 → 0.52); com **158
  exemplos** recupera à **paridade** (0.85 vs 0.79). Ou seja, o dano factual era em boa parte
  **artefato de dados escassos**.
- **Conhecimento:** o FT **não acerta a citação** (0/50) e, pior, com mais dados **generaliza
  pior** a normas não vistas (PPL held-out **+8%**) — memoriza o registro em vez de aprender a
  norma. É a demonstração de **adaptação de estilo ≠ injeção de conhecimento**, e a justificativa
  quantitativa para **FT + RAG** (o RAG da Fase 1 dá a fonte factual).

Lições de método que só o rigor revelou: (1) sem **controlar o comprimento**, o win-rate diria o
oposto (0.06 → 0.88); (2) sem **held-out**, o "−18% de PPL" parecia generalização (era memorização
— o held-out ficou **+8%**); (3) sem **expandir o dataset**, a "piora factual" pareceria
definitiva (era small-data). Reports carimbados com proveniência em `reports/fase2_ft/`.

## 6. Como reproduzir (na Nitro)

```bash
# WSL2 Ubuntu, dentro do repo, com env preparado (venv + CUDA_HOME=nvidia/cu13 +
# VLLM_ATTENTION_BACKEND=TORCH_SDPA + VLLM_USE_FLASHINFER_SAMPLER=0 -> ver env.sh):
R=reports/fase2_ft
python -m rodoia.ft.split_dataset                                    # split held-out (124/34)
python -m rodoia.ft.treino_qlora --dataset data/processed/ft_dataset_treino.jsonl \
    --saida models/qlora-antt-ho --epocas 3
CUDA_VISIBLE_DEVICES="" python -m rodoia.ft.merge_quantiza \
    --adaptador models/qlora-antt-ho --merged models/antt-merged-ho   # merge-only é default
# perplexidade held-out E in-sample, base e FT:
python -m rodoia.ft.perplexidade Qwen/Qwen2.5-3B-Instruct $R/perplexidade_base_holdout.json \
    data/processed/ft_dataset_holdout.jsonl fp8
python -m rodoia.ft.perplexidade models/antt-merged-ho $R/perplexidade_ft_holdout.json \
    data/processed/ft_dataset_holdout.jsonl fp8
# citação + win-rate (juiz independente qwen2.5:7b via Ollama):
python -m rodoia.ft.gen_offline Qwen/Qwen2.5-3B-Instruct $R/respostas_base.json
python -m rodoia.ft.gen_offline models/antt-merged-ho     $R/respostas_ft.json
python -m rodoia.ft.aval_cite $R/respostas_base.json $R/respostas_ft.json $R/avaliacao_ft.json
python -m rodoia.ft.juiz_winrate $R/respostas_base.json $R/respostas_ft.json $R/winrate_ft.json
python -m rodoia.ft.juiz_winrate $R/respostas_base.json $R/respostas_ft.json \
    $R/winrate_ft_controlado.json --controlar-comprimento
# serving + benchmark reprodutível:
vllm serve models/antt-merged-ho --quantization fp8 --max-model-len 2048 \
    --gpu-memory-utilization 0.80 --enforce-eager --port 8001 --served-model-name antt-ft &
python -m rodoia.ft.benchmark_vllm antt-ft http://localhost:8001/v1 $R/benchmark_vllm.json
```

> **Notas do stack (fixar se reinstalar):** vLLM 0.24 puxa torch 2.11+cu130 e o
> **flashinfer faz JIT com nvcc** que falha por headers CUDA incompatíveis →
> desabilitar flashinfer (`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ATTENTION_BACKEND=TORCH_SDPA`)
> e apontar `CUDA_HOME`/`LD_LIBRARY_PATH` para o pacote pip `nvidia/cu13`. Em rede,
> o WSL exigiu **networking mirrored**. Modelos (`models/`) não vão para o Git.

## 7. Critérios de conclusão da Fase 2

- [x] Dataset de fine-tuning documentado (**158 exemplos**, `ft_dataset.jsonl` + `dataset_stats.json`)
- [x] Modelo fine-tunado com QLoRA, hiperparâmetros versionados
- [x] Modelo quantizado com trade-off medido nos **dois eixos**: memória (fp16 não cabe → fp8 5168 MiB) e qualidade (NF4 ΔPPL +14%)
- [x] Avaliação base vs. fine-tunado com **held-out + IC** (n=50): PPL in-sample −16% × held-out **+8%** (generaliza pior) · citação 0/50 · correção factual em paridade (0.85 vs 0.79) · win-rate estilo controlado **0.88 [0.76;0.94]**
- [x] Modelo servido via vLLM, throughput/latência por **harness reprodutível** (**205 tok/s; p50 3.08 s**)
- [x] `docs/11` com todas as decisões; reports carimbados com proveniência
- [x] Funções puras testadas (split/PPL/citação/win-rate/benchmark); execução real na Nitro

### Próximos passos sugeridos
1. **Combinar FT + RAG** (Fase 1) — o FT dá o estilo, o RAG dá a fonte factual.
2. **Reformular a tarefa de FT para rótulo objetivo** — este experimento (injetar
   conhecimento) prova o *pipeline* mas dá resultado fraco por design; a Fase 2 seguiu
   para um estudo de **NER no LeNER-Br** (métrica dura, FT vence) — ver `docs/13`.
