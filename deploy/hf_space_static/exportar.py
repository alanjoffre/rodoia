"""Exporta os embeddings do corpus (E5, prefixo 'passage') + metadados para a demo STATIC.

A demo do HF Static Space roda no NAVEGADOR (Transformers.js): ele embute a consulta com o mesmo
E5 (`Xenova/multilingual-e5-small`, ONNX) e compara contra os embeddings dos chunks pré-computados
aqui. Como os dois lados usam mean-pooling + normalização L2 e os prefixos query/passage, os vetores
ficam no mesmo espaço. Este script gera `dados.f32` (Float32 [N,384], little-endian) + `dados.json`
(metadados: número da resolução + trecho).

Uso:  python deploy/hf_space_static/exportar.py   # gera os assets no próprio diretório
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rodoia.rag.construir_indice import carregar_chunks
from rodoia.rag.embeddings import E5Embedder

AQUI = Path(__file__).resolve().parent


def _recortar(texto: str, max_texto: int) -> dict[str, object]:
    """Recorta o trecho para o preview e DIZ se cortou.

    Quem trunca é quem sabe que truncou: sem esta flag, a UI teria de adivinhar pelo tamanho
    (`len >= max_texto`) e carimbaria "…" em chunks que couberam inteiros — o mais curto do
    corpus tem 3 caracteres e aparecia como "do.…".
    """
    recorte = texto[:max_texto]
    return {"texto": recorte, "truncado": len(texto) > max_texto}


def exportar(saida_dir: Path = AQUI, apenas_vigentes: bool = True, max_texto: int = 320) -> dict:
    chunks = carregar_chunks()
    if apenas_vigentes:
        chunks = [c for c in chunks if c.get("vigente")]
    emb = E5Embedder().encode_passages([c["texto"] for c in chunks]).astype(np.float32)
    (saida_dir / "dados.f32").write_bytes(emb.tobytes())          # [N, dim] row-major float32
    meta = {"n": len(chunks), "dim": int(emb.shape[1]),
            "meta": [{"numero": c["numero"], **_recortar(c["texto"], max_texto)} for c in chunks]}
    (saida_dir / "dados.json").write_text(json.dumps(meta, ensure_ascii=False))
    n_trunc = sum(1 for m in meta["meta"] if m["truncado"])
    print(f"exportado: {len(chunks)} chunks (vigentes={apenas_vigentes}), dim {emb.shape[1]}, "
          f"{n_trunc} truncados -> {saida_dir}/dados.f32 + dados.json")
    return {"n": len(chunks), "dim": int(emb.shape[1]), "n_truncados": n_trunc}


if __name__ == "__main__":
    exportar()
