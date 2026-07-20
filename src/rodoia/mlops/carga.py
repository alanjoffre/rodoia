"""Teste de carga do cache do serving (Fase 5 — latência MEDIDA, não afirmada).

O especialista apontou que o ganho do cache era *afirmado*, não medido. Aqui **medimos** o efeito
no p50/p95 **sob concorrência**, com um workload realista (algumas consultas "quentes" repetidas +
cauda), comparando **sem cache** vs **com cache**.

Honestidade sobre o método: o backend (a geração do LLM) é **simulado por um `sleep`** de latência
fixa e **divulgada** — assim o teste é rápido e reprodutível, e isola o efeito do CACHE (o que se
quer avaliar), não do Ollama. O efeito é **relativo e escala**: um hit passa de "latência do
backend" para ~microssegundos; na latência real medida na Fase 1 (p95 ≈ 30 s), a mesma taxa de hit
reduz a cauda proporcionalmente. Achado esperado e honesto: o cache **derruba a mediana** (consultas
repetidas ficam instantâneas); o **p95 só cai quando a taxa de hit é alta** (a cauda são os misses).

Uso:  python -m rodoia.mlops.carga
"""
from __future__ import annotations

import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from rodoia import estat
from rodoia.config import REPO_ROOT
from rodoia.observabilidade import CacheLRU
from rodoia.proveniencia import carimbar


def _workload(n_req: int, n_unicas: int, frac_quente: float, seed: int) -> list[str]:
    """Sequência de consultas com repetição: `frac_quente` das requisições caem em 3 consultas
    'quentes' (hot set); o resto na cauda. Modela tráfego real (poucas perguntas populares)."""
    rng = random.Random(seed)
    quentes = [f"q{i}" for i in range(3)]
    cauda = [f"q{i}" for i in range(3, max(4, n_unicas))]
    return [rng.choice(quentes) if rng.random() < frac_quente else rng.choice(cauda)
            for _ in range(n_req)]


def _servico(consulta: str, cache: CacheLRU | None, backend_s: float) -> str:
    """Espelha o /perguntar: consulta o cache; no miss, 'gera' (sleep) e cacheia."""
    if cache is not None and (v := cache.get(consulta)) is not None:
        return v
    time.sleep(backend_s)                      # simula a latência do backend (geração do LLM)
    r = f"resp:{consulta}"
    if cache is not None:
        cache.set(consulta, r)
    return r


def _percentil(vals: list[float], p: float) -> float:
    """Percentil nearest-rank; `p` em [0,100]. Delega ao contrato único em `estat`."""
    return estat.percentil(vals, p / 100)


def medir(reqs: list[str], backend_s: float, com_cache: bool, concorrencia: int = 10) -> dict:
    cache = CacheLRU(512) if com_cache else None

    def _um(c: str) -> float:
        t0 = time.perf_counter()
        _servico(c, cache, backend_s)
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=concorrencia) as ex:
        lat = list(ex.map(_um, reqs))
    return {"p50_s": round(_percentil(lat, 50), 3), "p95_s": round(_percentil(lat, 95), 3),
            "media_s": round(statistics.mean(lat), 3),
            "taxa_hit": round(cache.taxa_hit, 3) if cache else 0.0}


def teste_carga(backend_s: float = 0.3, n_req: int = 600, seed: int = 42) -> dict:
    cenarios = []
    for nome, n_unicas, frac in [("hit_baixo", 200, 0.3), ("hit_medio", 40, 0.6),
                                 ("hit_alto", 4, 0.5)]:
        reqs = _workload(n_req, n_unicas, frac, seed)
        sem = medir(reqs, backend_s, com_cache=False)
        com = medir(reqs, backend_s, com_cache=True)
        red_p50 = round(100 * (1 - com["p50_s"] / sem["p50_s"]), 1) if sem["p50_s"] else 0
        red_p95 = round(100 * (1 - com["p95_s"] / sem["p95_s"]), 1) if sem["p95_s"] else 0
        cenarios.append({
            "cenario": nome, "n_req": n_req, "taxa_hit": com["taxa_hit"],
            "p50_sem": sem["p50_s"], "p50_com": com["p50_s"],
            "p95_sem": sem["p95_s"], "p95_com": com["p95_s"],
            "reducao_p50_pct": red_p50, "reducao_p95_pct": red_p95,
        })
    return carimbar({
        "metodo": "backend simulado por sleep (latência divulgada); mede o efeito do CACHE sob "
                  "concorrência. Efeito escala à latência real (p95≈30s na Fase 1).",
        "backend_simulado_s": backend_s, "concorrencia": 10, "cenarios": cenarios,
    })


def main() -> None:
    res = teste_carga()
    saida = REPO_ROOT / "reports" / "fase5_mlops" / "carga.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"teste de carga (backend simulado={res['backend_simulado_s']}s, concorrência=10):")
    for c in res["cenarios"]:
        print(f"  {c['cenario']:10} hit={c['taxa_hit']:.2f} | "
              f"p50 {c['p50_sem']}→{c['p50_com']}s (-{c['reducao_p50_pct']}%) | "
              f"p95 {c['p95_sem']}→{c['p95_com']}s (-{c['reducao_p95_pct']}%)")
    print(f"relatório: {saida}")


if __name__ == "__main__":
    main()
