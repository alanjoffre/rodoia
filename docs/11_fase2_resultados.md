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

## 5. Avaliação base vs. fine-tunado — rigorosa e honesta

**Limitação de dados:** o corpus `data/raw/normas.jsonl` (referência factual) é gerido
por **DVC sem remoto configurado** nesta máquina; sem o texto das normas não há como o
juiz verificar *correção factual*. Compensamos com **três medições independentes**, cada
uma iluminando uma faceta diferente (dados brutos em `reports/fase2_ft/`).

### 5.1 Perplexidade de domínio — *o FT aprendeu o registro?* (`perplexidade.py`)
PPL do base vs. FT sobre as respostas de domínio (via `prompt_logprobs` do vLLM, fp8).

| PPL (↓ melhor) | Base | Fine-tunado | Δ |
|---|---|---|---|
| micro (por token) | 9.60 | **7.85** | **−18.3%** |
| macro (por texto) | 12.51 | **11.78** | −5.8% |

→ O QLoRA **aproximou mensuravelmente a distribuição do modelo ao registro jurídico da
ANTT**. (Nota: in-sample — mede *fit de domínio*, não generalização factual.)

### 5.2 Acurácia de citação — *acerta a norma certa?* (`aval_cite.py`)
Sobre o `CONJUNTO_DOURADO` (10 perguntas com a resolução-fonte conhecida), `temperature=0`.

| Métrica | Base | Fine-tunado |
|---|---|---|
| **Acurácia de citação** (resolução **correta**) | 0/10 | **0/10** |
| Taxa de citação (cita *alguma* resolução) | 50% | 50% |
| Comprimento médio | 946 chars | **219 chars** (−77%) |
| Respostas com hedge/ressalva | 2/10 | **0/10** |

→ **Nenhum** acerta a norma; ambos **alucinam** o número (ex.: vale-pedágio, esperado
6024/2023 → FT cita "6.088/2016"). O FT ficou muito mais **conciso e assertivo**.

### 5.3 Win-rate por juiz independente — *qual responde melhor?* (`juiz_winrate.py`)
Juiz **`qwen2.5:7b`** (Ollama, checkpoint distinto), pareado com **troca de posição**
(só conta vitória se consistente nas duas ordens). Critério: qualidade de resposta
regulatória (clareza/estrutura/formato), **não** correção factual. Rodamos **dois modos**
para isolar o viés de comprimento (o base é ~4× mais longo que o FT):

| Modo | Base vence | FT vence | Empates | FT win-rate |
|---|---|---|---|---|
| **Bruto** (respostas originais) | 9 | 0 | 1 | 0.00 |
| **Controlado** (truncado ao mesmo tamanho) | 1 | **4** | 5 | **0.40** |

→ **A vantagem 9×0 do base era quase toda viés de comprimento.** Com as respostas
truncadas ao mesmo tamanho, o juiz **inverte** e prefere o FT (4×1, 5 empates) — a
concisão assertiva do FT é levemente melhor *a igual quantidade de conteúdo*. Lição
metodológica: sem **controlar o confundidor (comprimento)**, o número diria o oposto
da verdade. (Ambos os relatórios: `winrate_ft.json`, `winrate_ft_controlado.json`.)

### Interpretação (o resultado científico)
As três medições contam uma história coerente e **não-óbvia**: o fine-tuning
**mudou a distribuição** do modelo (PPL de domínio −18%, respostas −77% mais curtas,
zero hedge) e, **a igual comprimento, é levemente preferido** pelo juiz (win-rate
controlado 0.40 × 0.10) — mas **não injetou conhecimento factual** (citação 0/0). Com 84
exemplos e **sem RAG**, o QLoRA ensinou o modelo a *soar* como um especialista da ANTT —
conciso e citando "Resolução nº X" — sem *saber* a norma certa, trocando as ressalvas do
base por afirmações assertivas (às vezes **alucinações confiantes**). É a demonstração de
manual de **adaptação de estilo ≠ injeção de conhecimento**, e a justificativa
quantitativa para **(a) combinar FT + RAG** (a Fase 1 fornece a fonte factual) e
**(b) expandir o dataset** muito além de 84 exemplos. O episódio do win-rate (bruto 0×9 →
controlado 4×1) também é uma lição de método: **medir sem controlar o confundidor engana**.

Relatórios completos: `reports/fase2_ft/` (`avaliacao_ft.json`, `perplexidade_*.json`,
`winrate_ft.json`, `respostas_*.json`, `comparacao.md`).

## 6. Como reproduzir (na Nitro)

```bash
# WSL2 Ubuntu, dentro do repo, com env preparado (venv + CUDA_HOME=nvidia/cu13 +
# VLLM_ATTENTION_BACKEND=TORCH_SDPA + VLLM_USE_FLASHINFER_SAMPLER=0 -> ver env.sh):
python -m rodoia.ft.treino_qlora --modelo Qwen/Qwen2.5-3B-Instruct --epocas 3
CUDA_VISIBLE_DEVICES="" python -m rodoia.ft.merge_quantiza --pular-awq \
    --adaptador models/qlora-antt --merged models/antt-merged
vllm serve models/antt-merged --quantization fp8 --max-model-len 2048 \
    --gpu-memory-utilization 0.80 --enforce-eager --port 8001 --served-model-name antt-ft
python -m rodoia.ft.gen_offline Qwen/Qwen2.5-3B-Instruct /tmp/ans_base.json   # respostas base
python -m rodoia.ft.aval_cite /tmp/ans_base.json /tmp/ans_ft.json reports/fase2_ft/avaliacao_ft.json
# avaliação rigorosa:
python -m rodoia.ft.perplexidade Qwen/Qwen2.5-3B-Instruct /tmp/ppl_base.json   # PPL base
python -m rodoia.ft.perplexidade models/antt-merged        /tmp/ppl_ft.json    # PPL FT
ollama serve & ollama pull qwen2.5:7b   # juiz independente
python -m rodoia.ft.juiz_winrate reports/fase2_ft/respostas_base.json \
    reports/fase2_ft/respostas_ft.json reports/fase2_ft/winrate_ft.json
```

> **Notas do stack (fixar se reinstalar):** vLLM 0.24 puxa torch 2.11+cu130 e o
> **flashinfer faz JIT com nvcc** que falha por headers CUDA incompatíveis →
> desabilitar flashinfer (`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ATTENTION_BACKEND=TORCH_SDPA`)
> e apontar `CUDA_HOME`/`LD_LIBRARY_PATH` para o pacote pip `nvidia/cu13`. Em rede,
> o WSL exigiu **networking mirrored**. Modelos (`models/`) não vão para o Git.

## 7. Critérios de conclusão da Fase 2

- [x] Dataset de fine-tuning documentado (84 exemplos, `ft_dataset.jsonl`)
- [x] Modelo fine-tunado com QLoRA, hiperparâmetros versionados
- [x] Modelo quantizado com trade-off medido (**fp16 não-servível → fp8 5168 MiB**)
- [x] Avaliação base vs. fine-tunado com números (**3 medições**: PPL −18% · citação 0/0 · win-rate bruto 0×9 → controlado FT 4×1)
- [x] Modelo servido via vLLM, com throughput/latência (**101 tok/s; p50 3.35 s**)
- [x] `docs/11` com todas as decisões de treino e serving
- [x] Partes testáveis do pipeline validadas; execução real confirmada na Nitro

### Próximos passos sugeridos
1. **Combinar FT + RAG** (Fase 1) — o FT dá o estilo, o RAG dá a fonte correta.
2. **Expandir o dataset** (de 84 p/ centenas de exemplos) e reavaliar.
3. Disponibilizar `data/raw/normas.jsonl` (DVC) na máquina para reativar o juiz-LLM
   com referência do `avaliar_ft.py`.
