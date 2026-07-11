# 10 — Fase 2: HANDOFF para execução na Nitro (fine-tuning + serving)

> ⚠️ **Handoff histórico — Fase 2 CONCLUÍDA.** Resultado **principal**: NER jurídico onde o
> fine-tuning vence (F1 0,13→0,77 vs. SOTA) — [`docs/13`](13_fase2_ner.md). Este handoff descreve
> o **estudo-baseline** (FT generativo), cujos resultados estão em [`docs/11`](11_fase2_resultados.md).
> A execução real substituiu AWQ por **fp8** e adaptou os scripts p/ trl 1.x / transformers 5.x —
> não siga o caminho AWQ literalmente.

> **Este documento é auto-contido.** Um agente de IA que nunca viu o resto da
> conversa deve conseguir, lendo isto + o código do repositório, executar a Fase 2
> de ponta a ponta na máquina com GPU NVIDIA (Acer Nitro V15, RTX 4050, 6 GB VRAM).
> Escrito porque os passos GPU **não rodam no Mac** (bitsandbytes e vLLM são CUDA-only).

---

## 0. Contexto do projeto (para quem chega agora)

**RodoIA** é uma plataforma de IA de portfólio, open-source, sobre a regulação e os
dados abertos do transporte rodoviário da ANTT. Repositório público:
https://github.com/alanjoffre/rodoia . Estado:

- **Fase 0 (concluída):** fundamentos de ML/DL — ML clássico, diagnóstico, backprop
  à mão (NumPy), MLP (PyTorch), self-attention à mão. Docs 00–05.
- **Fase 1 (concluída):** RAG sobre a regulação da ANTT — ingestão (ANTTlegis),
  Qdrant, retrieval híbrido, geração com citação, guardrails, FastAPI. Docs 06–09.
- **Fase 2 (ESTA, concluída):** fine-tuning QLoRA + quantização + serving vLLM
  de um modelo próprio, com avaliação antes/depois.

O código é tipado, testado (`pytest`) e passa por pre-commit (ruff + detect-secrets).
Regras invioláveis: só dados públicos da ANTT; nada de segredo no Git; commits com
mensagem clara em pt-br; cada incremento testável e documentado.

## 1. Objetivo da Fase 2 e o que provar

Adaptar um LLM aberto pequeno ao domínio ANTT e **hospedá-lo você mesmo** — provando
que sabe *operar* modelo, não só consumir API. Entregáveis:

1. Dataset de instrução do domínio ✅ (já pronto — ver §3).
2. Modelo fine-tunado com **QLoRA** (LoRA em 4-bit).
3. Modelo **quantizado** (AWQ) com trade-off medido.
4. **Avaliação base vs. fine-tunado** com número (o entregável científico).
5. Modelo **servido via vLLM**, com throughput/latência medidos.

## 2. O que JÁ ESTÁ PRONTO neste repositório (feito no Mac)

| Arquivo | O que faz | Onde roda |
|---|---|---|
| `data/processed/ft_dataset.jsonl` | 84 exemplos no plano original; **hoje 158** (ver docs/11) | pronto |
| `src/rodoia/ft/construir_dataset.py` | (re)gera/expande o dataset via Ollama | Mac ou Nitro |
| `src/rodoia/ft/treino_qlora.py` | fine-tuning QLoRA (NF4 + LoRA) | **Nitro** |
| `src/rodoia/ft/merge_quantiza.py` | funde LoRA + quantiza AWQ | **Nitro** |
| `src/rodoia/ft/avaliar_ft.py` | avaliação base vs. fine-tunado (LLM-as-judge) | **Nitro** |
| `src/rodoia/rag/llm.py::OpenAICompatLLM` | cliente p/ endpoint vLLM (OpenAI-compat) | qualquer |

Todos os scripts adiam os imports CUDA para dentro das funções, então **importam sem
erro** mesmo sem GPU (facilita testar a lógica). A execução real precisa da Nitro.

## 3. Hardware e escolhas (RTX 4050, 6 GB VRAM)

- **Modelo base: `Qwen/Qwen2.5-3B-Instruct`.** Em 4-bit cabe com folga em 6 GB;
  forte em português. Um 7B ficaria no limite — se der OOM, este é o primeiro
  parâmetro a reduzir (ou usar `Qwen/Qwen2.5-1.5B-Instruct`).
- **QLoRA:** treina só adaptadores LoRA sobre o modelo congelado em NF4 4-bit +
  double-quant + cômputo bf16 + `paged_adamw_8bit` + gradient checkpointing.
- **AWQ 4-bit para servir:** o merge gera fp16 (~6 GB, aperta); AWQ derruba para
  ~2-3 GB, sobrando VRAM para o KV cache do vLLM.

## 4. Setup do ambiente na Nitro

### 4a. Se a Nitro estiver com WINDOWS → use WSL2 (Ubuntu)

```powershell
# No PowerShell (admin):
wsl --install -d Ubuntu
# Reiniciar. O driver NVIDIA do Windows já expõe a GPU no WSL2 (não instalar driver DENTRO do WSL).
```
Dentro do Ubuntu (WSL2), instalar o **CUDA Toolkit** (sem driver) — seguir
https://developer.nvidia.com/cuda-downloads (distro: WSL-Ubuntu). Verificar `nvidia-smi`.

### 4b. Se a Nitro já estiver com LINUX

Instalar o driver NVIDIA + CUDA Toolkit da distro. Verificar `nvidia-smi`.

### 4c. Projeto + dependências (em ambos os casos, dentro do Linux/WSL2)

```bash
git clone https://github.com/alanjoffre/rodoia.git && cd rodoia
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ft,dev]"        # transformers, peft, trl, bitsandbytes, autoawq, vllm, accelerate, datasets
# Verificar a GPU:
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Versões:** o pyproject fixa mínimos (transformers>=4.44, peft>=0.12, trl>=0.11,
> bitsandbytes>=0.43, autoawq>=0.2, vllm>=0.6). As APIs do TRL/vLLM mudam com
> frequência — se algo quebrar, **fixar** a versão que funcionou e anotar aqui.

## 5. Passo a passo de execução (na Nitro)

```bash
# (opcional) expandir o dataset — precisa de Ollama rodando na Nitro:
#   ollama serve & ; ollama pull qwen2.5:7b
#   python -m rodoia.ft.construir_dataset --por-norma 4 --limite-normas 30

# 1) Fine-tuning QLoRA (~10-30 min em RTX 4050, 3B, 84 exemplos, 3 épocas)
python -m rodoia.ft.treino_qlora --modelo Qwen/Qwen2.5-3B-Instruct \
    --dataset data/processed/ft_dataset.jsonl --saida models/qlora-antt --epocas 3
# -> salva o adaptador LoRA em models/qlora-antt

# 2) Merge + quantização AWQ
python -m rodoia.ft.merge_quantiza --base Qwen/Qwen2.5-3B-Instruct \
    --adaptador models/qlora-antt --merged models/antt-merged --awq models/antt-awq

# 3) Servir base e fine-tunado no vLLM (dois terminais / duas portas)
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.5
vllm serve models/antt-awq --quantization awq --port 8001 --max-model-len 2048 --gpu-memory-utilization 0.5
# (na RTX 4050, servir os dois AO MESMO TEMPO pode não caber em 6 GB — se faltar
#  VRAM, avaliar um de cada vez: sobe o base, roda; sobe o ft, roda.)

# 4) Avaliação antes/depois (ajustar as portas em avaliar_ft.main se preciso)
python -m rodoia.ft.avaliar_ft
# -> reports/fase2_ft/avaliacao_ft.json com base_media, ft_media, ganho

# 5) Throughput/latência do vLLM: medir com o benchmark do vLLM ou um loop simples
#    de N requisições concorrentes, registrando tokens/s e latência p50/p95.
```

## 6. Troubleshooting (problemas prováveis)

- **OOM de VRAM no treino:** reduzir `--batch` (já é 1) → aumentar `--grad-accum`;
  reduzir `max_seq_length` (1024→512) no treino_qlora; ou trocar para 1.5B.
- **OOM ao servir dois modelos:** servir um de cada vez (§5 passo 3).
- **API do TRL mudou** (`SFTConfig`/`processing_class`/`max_seq_length`): checar a
  versão instalada do `trl` e ajustar; anotar a versão que funcionou.
- **AWQ falha em 3B:** alguns modelos precisam de `--trust-remote-code`; se AWQ não
  quantizar, servir o **merged fp16** com `--gpu-memory-utilization` mais alto e
  `--max-model-len` menor (o trade-off de quantização vira "não quantizado", documente).
- **bitsandbytes não acha CUDA no WSL2:** garantir `LD_LIBRARY_PATH` do CUDA e que
  `nvidia-smi` funciona dentro do WSL.

## 7. Ao terminar — registrar resultados (parte do portfólio)

1. Preencher `reports/fase2_ft/avaliacao_ft.json` (gerado) e **anotar os números**
   (base vs ft, ganho) + VRAM antes/depois da AWQ + throughput/latência do vLLM
   num novo `docs/11_fase2_resultados.md`.
2. Se o ganho do fine-tuning for pequeno/negativo, **reportar honestamente** — com
   84 exemplos é esperado; a competência provada é o *pipeline* (treinar, quantizar,
   servir, medir), não vencer o base. Sugerir expandir o dataset como próximo passo.
3. Atualizar o README (Fase 2 → concluída) e a tabela de rastreabilidade.
4. `git add`, commit em pt-br, e **push só com confirmação do Alan** (regra do repo).
5. Os modelos (`models/`) são grandes → **não** vão para o Git (já ignorados);
   documentar como reproduzir.

## 8. Critérios de conclusão da Fase 2

- [ ] Dataset de fine-tuning documentado ✅ (feito)
- [ ] Modelo fine-tunado com LoRA/QLoRA, config versionada (o script já versiona os hiperparâmetros)
- [ ] Modelo quantizado com trade-off medido (VRAM/latência/qualidade)
- [ ] Avaliação base vs. fine-tunado com números
- [ ] Modelo servido via vLLM, com throughput/latência
- [ ] `docs/11` com todas as decisões de treino e serving
- [ ] Testes do pipeline (as partes testáveis já têm; a execução é validada na Nitro)
