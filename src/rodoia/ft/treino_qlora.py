"""Fine-tuning QLoRA de um LLM aberto sobre o domínio ANTT (Fase 2).

>>> RODA NA NITRO (GPU NVIDIA / CUDA). NÃO roda no Mac (bitsandbytes é CUDA-only). <<<

Escolhas e trade-offs (defensáveis em entrevista):
- **Modelo base: Qwen2.5-3B-Instruct.** Na RTX 4050 (6 GB VRAM), um 3B em 4-bit
  cabe com folga (~2 GB de pesos + adaptadores + ativações); 7B ficaria no limite.
  Qwen2.5 é forte em português. Ver docs/10 para o teto de VRAM.
- **QLoRA (não full fine-tuning):** treina só adaptadores LoRA de baixo posto sobre
  o modelo congelado em 4-bit → cabe em 6 GB e é rápido. Full fine-tuning de 3B
  exigiria dezenas de GB. Trade-off: menos capacidade de mudança que full, mas
  suficiente para adaptar ESTILO/domínio, que é o objetivo.
- **NF4 + double quant + bf16 compute:** a receita padrão do paper QLoRA — 4-bit
  NormalFloat para os pesos, dupla quantização das constantes, cômputo em bf16.
- **LoRA r=16, alpha=32:** posto moderado; alpha/r=2 é um ponto de partida comum.

Uso (na Nitro, dentro do WSL2/Linux com CUDA):
    python -m rodoia.ft.treino_qlora \
        --modelo Qwen/Qwen2.5-3B-Instruct \
        --dataset data/processed/ft_dataset.jsonl \
        --saida models/qlora-antt --epocas 3

Dependências: transformers, peft, trl, bitsandbytes, accelerate, datasets
(pip install -e ".[ft]"). Versões testadas: ver docs/10.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def treinar(
    modelo: str = "Qwen/Qwen2.5-3B-Instruct",
    dataset: str = "data/processed/ft_dataset.jsonl",
    saida: str = "models/qlora-antt",
    epocas: int = 3,
    batch: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    r: int = 16,
    alpha: int = 32,
    seed: int = 42,
) -> None:
    # Imports pesados aqui dentro: só existem na Nitro (CUDA), não no ambiente do Mac.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA não disponível. Este script roda na Nitro (RTX 4050), não no Mac. "
            "Ver docs/10 (setup WSL2/Linux + CUDA)."
        )

    # 1) Quantização 4-bit (QLoRA): NF4 + double quant + cômputo bf16.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(modelo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        modelo, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
    )
    base = prepare_model_for_kbit_training(base)

    # 2) Adaptadores LoRA nas projeções de atenção e MLP.
    lora = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # 3) Dataset em formato de chat (messages) — o SFTTrainer aplica o chat template.
    ds = load_dataset("json", data_files=dataset, split="train")

    cfg = SFTConfig(
        output_dir=saida,
        num_train_epochs=epocas,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # otimizador paginado do QLoRA (economiza VRAM)
        seed=seed,
        report_to="none",
        max_seq_length=1024,
        packing=False,
    )

    trainer = SFTTrainer(
        model=base, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tokenizer
    )
    trainer.train()

    Path(saida).mkdir(parents=True, exist_ok=True)
    trainer.save_model(saida)  # salva o adaptador LoRA
    tokenizer.save_pretrained(saida)
    print(f"OK: adaptador LoRA salvo em {saida}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tuning QLoRA (Fase 2, roda na Nitro).")
    p.add_argument("--modelo", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset", default="data/processed/ft_dataset.jsonl")
    p.add_argument("--saida", default="models/qlora-antt")
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()
    treinar(
        modelo=args.modelo,
        dataset=args.dataset,
        saida=args.saida,
        epocas=args.epocas,
        batch=args.batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
