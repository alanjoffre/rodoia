"""NER com BERTimbau (token-classification) — a REFERÊNCIA SOTA da Fase 2.

Fine-tuning encoder (`neuralmind/bert-base-portuguese-cased`) para NER no LeNER-Br, com
métrica **F1 de entidade (seqeval)** por classe. É o padrão-ouro para NER e serve de teto
contra o qual o LLM generativo (QLoRA) é comparado.

Uso:  python -m rodoia.ner.bertimbau --epocas 3
"""
from __future__ import annotations

import argparse
import json

from rodoia.config import REPO_ROOT, settings
from rodoia.ner.lener import ID2LABEL, LABEL2ID, LABELS, carregar
from rodoia.proveniencia import carimbar

MODELO_BASE = "neuralmind/bert-base-portuguese-cased"
_REPORT = REPO_ROOT / "reports" / "fase2_ner"


def _tokenizar_alinhar(exemplos, tokenizer):
    """Tokeniza em subwords e alinha os rótulos: só o 1º subword de cada palavra recebe
    o rótulo; continuação e especiais recebem -100 (ignorados na perda)."""
    tok = tokenizer(exemplos["tokens"], truncation=True, max_length=256,
                    is_split_into_words=True)
    rotulos = []
    for i, tags in enumerate(exemplos["ner_tags"]):
        ids_palavra = tok.word_ids(batch_index=i)
        anterior, seq = None, []
        for wid in ids_palavra:
            if wid is None:
                seq.append(-100)
            elif wid != anterior:
                seq.append(tags[wid])
            else:
                seq.append(-100)
            anterior = wid
        rotulos.append(seq)
    tok["labels"] = rotulos
    return tok


def _metricas(pred):
    import numpy as np
    from seqeval.metrics import classification_report, f1_score

    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    y_pred, y_true = [], []
    for p_seq, l_seq in zip(preds, labels):
        yp, yt = [], []
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                yp.append(ID2LABEL[int(p)])
                yt.append(ID2LABEL[int(l)])
        y_pred.append(yp)
        y_true.append(yt)
    rel = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {"f1_micro": f1_score(y_true, y_pred), "relatorio": rel}


def treinar(epocas: int = 3, batch: int = 16, lr: float = 3e-5) -> dict:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    dsets = {}
    for split in ("train", "dev", "test"):
        ds = Dataset.from_list(carregar(split))
        dsets[split] = ds.map(lambda e: _tokenizar_alinhar(e, tokenizer), batched=True,
                              remove_columns=ds.column_names)

    modelo = AutoModelForTokenClassification.from_pretrained(
        MODELO_BASE, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    args = TrainingArguments(
        output_dir=str(settings.data_processed.parent.parent / "models" / "bertimbau-lener"),
        num_train_epochs=epocas, per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch, learning_rate=lr, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="no", logging_steps=50,
        bf16=torch.cuda.is_available(), report_to="none", seed=settings.seed,
    )
    trainer = Trainer(
        model=modelo, args=args, train_dataset=dsets["train"], eval_dataset=dsets["dev"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda p: {"f1_micro": _metricas((p.predictions, p.label_ids))["f1_micro"]},
    )
    trainer.train()

    pred = trainer.predict(dsets["test"])
    m = _metricas((pred.predictions, pred.label_ids))
    por_entidade = {e: round(m["relatorio"].get(e, {}).get("f1-score", 0.0), 4)
                    for e in ("ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL", "LEGISLACAO", "JURISPRUDENCIA")}
    res = carimbar({
        "modelo": MODELO_BASE, "abordagem": "encoder token-classification (SOTA)",
        "epocas": epocas, "n_teste": len(dsets["test"]),
        "f1_micro_teste": round(m["f1_micro"], 4),
        "f1_por_entidade": por_entidade,
    })
    _REPORT.mkdir(parents=True, exist_ok=True)
    (_REPORT / "bertimbau.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"BERTimbau F1-micro (teste) = {res['f1_micro_teste']} | por entidade: {por_entidade}")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="NER BERTimbau no LeNER-Br (Fase 2).")
    p.add_argument("--epocas", type=int, default=3)
    treinar(epocas=p.parse_args().epocas)


if __name__ == "__main__":
    main()
