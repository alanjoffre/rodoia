"""Split held-out do dataset de fine-tuning (Fase 2) — para avaliar GENERALIZAÇÃO.

Reserva N normas INTEIRAS como *held-out* (nenhum exemplo delas entra no treino), de
forma determinística (shuffle com seed). Assim as métricas de citação/PPL/win-rate
medidas sobre as perguntas dessas normas refletem generalização a normas NÃO vistas —
e não memorização in-sample. O split (quais normas) é versionado em
`reports/fase2_ft/split_holdout.json`, como o split 3-vias da Fase 0.

Uso:
    python -m rodoia.ft.split_dataset            # gera treino/holdout + registro
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from rodoia.config import REPO_ROOT, settings
from rodoia.proveniencia import carimbar

N_HOLDOUT = 6  # ~18 de 84 exemplos (~21%), 6 de 29 normas


def dividir(exemplos: list[dict], n_holdout: int = N_HOLDOUT, seed: int = 42) -> tuple[list, list, list]:
    """Separa exemplos por norma (`fonte`); reserva `n_holdout` normas p/ held-out.
    Determinístico dado o seed. Retorna (treino, holdout, normas_holdout)."""
    por_norma: dict[str, list] = defaultdict(list)
    for ex in exemplos:
        por_norma[ex["fonte"]].append(ex)
    normas = sorted(por_norma)  # ordem estável antes do shuffle
    random.Random(seed).shuffle(normas)
    holdout_normas = sorted(normas[:n_holdout])
    treino = [ex for n in normas[n_holdout:] for ex in por_norma[n]]
    holdout = [ex for n in holdout_normas for ex in por_norma[n]]
    return treino, holdout, holdout_normas


def main() -> None:
    origem = settings.data_processed / "ft_dataset.jsonl"
    exemplos = [json.loads(linha) for linha in origem.open(encoding="utf-8")]
    treino, holdout, holdout_normas = dividir(exemplos)

    def _escrever(caminho: Path, itens: list[dict]) -> None:
        caminho.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in itens) + "\n", encoding="utf-8"
        )

    _escrever(settings.data_processed / "ft_dataset_treino.jsonl", treino)
    _escrever(settings.data_processed / "ft_dataset_holdout.jsonl", holdout)

    registro = carimbar({
        "n_total": len(exemplos),
        "n_treino": len(treino),
        "n_holdout": len(holdout),
        "normas_holdout": holdout_normas,
    })
    saida = REPO_ROOT / "reports" / "fase2_ft" / "split_holdout.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")

    # Estatísticas de qualidade/diversidade do dataset (reprodutibilidade).
    por_norma: dict[str, int] = defaultdict(int)
    comprimentos = []
    for ex in exemplos:
        por_norma[ex["fonte"]] += 1
        for m in ex["messages"]:
            if m["role"] == "assistant":
                comprimentos.append(len(m["content"]))
    stats = carimbar({
        "n_exemplos": len(exemplos),
        "n_normas": len(por_norma),
        "exemplos_por_norma_min": min(por_norma.values()),
        "exemplos_por_norma_max": max(por_norma.values()),
        "resposta_len_chars_media": round(sum(comprimentos) / len(comprimentos)),
        "gerador": "qwen2.5:7b (Ollama, temperatura=0)",
    })
    (saida.parent / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"treino={len(treino)} holdout={len(holdout)} normas_holdout={holdout_normas}")
    print(f"registro: {saida} | stats: {saida.parent / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
