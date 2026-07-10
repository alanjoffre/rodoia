# 11 — Fase 2: resultados e fecho da fase (fine-tuning + quantização + serving na Nitro)

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

FT servido: `vllm serve models/antt-merged --quantization fp8 --max-model-len 2048
--gpu-memory-utilization 0.80 --enforce-eager` (endpoint OpenAI-compat).

| Métrica (24 req, concorrência 6, `max_tokens` 128) | Valor |
|---|---|
| Throughput | **101.1 tokens/s** |
| Requisições/s | 1.38 |
| Latência p50 | **3.35 s** |
| Latência p95 | 6.67 s |
| Tempo de carga do modelo | ~50 s |

## 5. Avaliação base vs. fine-tunado — rigorosa, com held-out e IC

### 5.0 Desenho: split held-out (`split_dataset.py`)
Para medir **generalização** (e não memorização), reservamos **6 das 29 normas inteiras**
como *held-out* (nenhum exemplo delas entra no treino; split determinístico, seed=42),
re-treinando o QLoRA nas 23 restantes (**66 exemplos**). O split é versionado em
`reports/fase2_ft/split_holdout.json`. Nota de dados: o corpus `normas.jsonl` (referência
factual do juiz) segue ausente (DVC sem remoto) — por isso a correção factual é medida por
proxy (citação da norma-fonte), não pelo `avaliar_ft.py` (inativo).

### 5.1 Perplexidade — *fit de domínio vs. generalização* (`perplexidade.py`, fp8)
PPL (↓ melhor) via `prompt_logprobs`, medida **in-sample** (respostas do treino) e
**held-out** (respostas de normas NÃO vistas):

| PPL micro | Base | Fine-tunado | Δ |
|---|---|---|---|
| **in-sample** (66 textos) | 9.79 | **8.24** | **−15.8%** |
| **held-out** (18 textos) | 9.12 | **8.78** | **−3.7%** |

→ O FT baixa a PPL **muito mais in-sample (−16%) que held-out (−4%)**: aprendeu o registro
dos exemplos vistos, mas **generaliza fracamente** a normas novas. O −18% relatado antes era
só in-sample e superestimava.

### 5.2 Acurácia de citação — *acerta a norma?* (`aval_cite.py`) — n=25, IC de Wilson
Sobre o `CONJUNTO_DOURADO` (25 perguntas de intenção real, `temperature=0`):

| Métrica | Base | Fine-tunado |
|---|---|---|
| **Acurácia de citação** (norma **correta**) | 0/25 [0; 0.13] | **0/25 [0; 0.13]** |
| Taxa de citação (cita *alguma*) | 0.56 | **0.84** |

→ **Nenhum** acerta a norma (ambos alucinam o número); o FT passou a **citar mais** (0.84 vs
0.56) — de novo estilo, não fato.

### 5.3 Win-rate por juiz independente (`juiz_winrate.py`) — n=25, IC de Wilson
Juiz **`qwen2.5:7b`** (checkpoint distinto), pareado com **troca de posição**. Dois modos p/
isolar o viés de comprimento (o base é ~4× mais longo):

| Modo | Base | FT | Empates | FT win-rate (IC95) |
|---|---|---|---|---|
| **Bruto** | 23 | 1 | 1 | 0.04 [0.01; 0.20] |
| **Controlado** (mesmo tamanho) | 1 | **21** | 3 | **0.84 [0.65; 0.94]** |

→ O bruto (base 23×1) era **quase todo viés de comprimento**. Controlando o tamanho, o FT
**vence 21×1** e o **IC exclui 0.5** — a igual conteúdo, a resposta concisa e assertiva do FT
é **significativamente** preferida.

### 5.4 Serving fp8 — benchmark reprodutível (`benchmark_vllm.py`)
**205 tokens/s**, latência **p50 3.08s / p95 3.59s**, **VRAM 5168 / 6141 MiB** (24 req,
concorrência 6). *Trade-off de quantização:* a memória está medida (fp16 5.8 GB não cabe →
fp8 cabe); o **custo de qualidade fp16×fp8 não foi medido** — o 3B em fp16 não carrega em
6 GB pelo mesmo caminho de serving (a própria restrição que forçou a quantização). Fica como
limite de hardware documentado.

### Interpretação (o resultado científico)
Coerente e **não-óbvia**: o fine-tuning **mudou a distribuição** (PPL in-sample −16%, respostas
−77% mais curtas, cita mais) e, **a igual comprimento, é significativamente preferido** pelo
juiz (0.84 [0.65; 0.94]) — mas **não injetou conhecimento factual** (citação 0/25) e
**generaliza fraco** a normas novas (PPL held-out só −4%). Com 66 exemplos e **sem RAG**, o
QLoRA ensinou o modelo a *soar* como especialista da ANTT sem *saber* a norma certa. É a
demonstração de manual de **adaptação de estilo ≠ injeção de conhecimento** — e a justificativa
quantitativa para **(a) FT + RAG** (a Fase 1 dá a fonte) e **(b) dataset muito maior**. Lição de
método: o win-rate (bruto 0.04 → controlado 0.84) prova que **medir sem controlar o confundidor
diz o oposto da verdade**. Todos os reports carimbados com proveniência em `reports/fase2_ft/`.

## 6. Como reproduzir (na Nitro)

```bash
# WSL2 Ubuntu, dentro do repo, com env preparado (venv + CUDA_HOME=nvidia/cu13 +
# VLLM_ATTENTION_BACKEND=TORCH_SDPA + VLLM_USE_FLASHINFER_SAMPLER=0 -> ver env.sh):
R=reports/fase2_ft
python -m rodoia.ft.split_dataset                                    # split held-out (66/18)
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

- [x] Dataset de fine-tuning documentado (84 exemplos, `ft_dataset.jsonl`)
- [x] Modelo fine-tunado com QLoRA, hiperparâmetros versionados
- [x] Modelo quantizado com trade-off de memória medido (**fp16 não-servível → fp8 5168 MiB**; custo de qualidade = limite de hardware documentado)
- [x] Avaliação base vs. fine-tunado com **held-out + IC**: PPL in-sample −16% × held-out −4% · citação 0/25 [0;0.13] · win-rate controlado **0.84 [0.65;0.94]**
- [x] Modelo servido via vLLM, throughput/latência por **harness reprodutível** (**205 tok/s; p50 3.08 s**)
- [x] `docs/11` com todas as decisões; reports carimbados com proveniência
- [x] Funções puras testadas (split/PPL/citação/win-rate/benchmark); execução real na Nitro

### Próximos passos sugeridos
1. **Combinar FT + RAG** (Fase 1) — o FT dá o estilo, o RAG dá a fonte correta.
2. **Expandir o dataset** (de 84 p/ centenas de exemplos) e reavaliar.
3. Disponibilizar `data/raw/normas.jsonl` (DVC) na máquina para reativar o juiz-LLM
   com referência do `avaliar_ft.py`.
