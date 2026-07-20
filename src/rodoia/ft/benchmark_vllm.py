"""Benchmark de serving do vLLM (Fase 2) — observabilidade REPRODUTÍVEL.

Mede throughput (tokens/s), latência (p50/p95) e VRAM contra um endpoint vLLM
OpenAI-compat já no ar. Antes os números existiam sem o harness que os gerou; aqui o
script fica versionado e o report é carimbado com proveniência.

Uso (com o vLLM servindo em :8001):
    python -m rodoia.ft.benchmark_vllm antt-ft http://localhost:8001/v1 \
        reports/fase2_ft/benchmark_vllm.json
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import subprocess
import sys
import time

import requests

from rodoia import estat
from rodoia.proveniencia import carimbar

PROMPT = ("Explique de forma detalhada as obrigacoes do transportador rodoviario "
          "de cargas segundo a ANTT.")


def percentil(valores: list[float], p: float) -> float:
    """Percentil nearest-rank (p em [0,1]), arredondado a 3 casas p/ os relatórios."""
    return round(estat.percentil(valores, p), 3)


def vram_usada_mb() -> int | None:
    """VRAM em uso (MiB) via nvidia-smi; None se indisponível."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _requisicao(base_url: str, modelo: str, max_tokens: int) -> tuple[float, int]:
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/chat/completions", json={
        "model": modelo, "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7, "max_tokens": max_tokens}, timeout=120)
    r.raise_for_status()
    return time.perf_counter() - t0, r.json().get("usage", {}).get("completion_tokens", 0)


def benchmark(modelo: str, base_url: str, n: int = 24, concorrencia: int = 6,
              max_tokens: int = 128) -> dict:
    vram = vram_usada_mb()
    lat, comp_tokens = [], 0
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concorrencia) as ex:
        for dt, ct in ex.map(lambda _: _requisicao(base_url, modelo, max_tokens), range(n)):
            lat.append(dt)
            comp_tokens += ct
    wall = time.perf_counter() - t0
    return carimbar({
        "modelo": modelo, "n_requests": n, "concorrencia": concorrencia, "max_tokens": max_tokens,
        "vram_usada_mb": vram,
        "wall_s": round(wall, 2),
        "throughput_tok_s": round(comp_tokens / wall, 1) if wall else None,
        "req_por_s": round(n / wall, 2) if wall else None,
        "lat_p50_s": percentil(lat, 0.50),
        "lat_p95_s": percentil(lat, 0.95),
        "total_completion_tokens": comp_tokens,
    })


def main() -> None:
    modelo, base_url, saida = sys.argv[1], sys.argv[2], sys.argv[3]
    res = benchmark(modelo, base_url)
    open(saida, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"throughput={res['throughput_tok_s']} tok/s  p50={res['lat_p50_s']}s  "
          f"p95={res['lat_p95_s']}s  VRAM={res['vram_usada_mb']}MiB  -> {saida}")


if __name__ == "__main__":
    main()
