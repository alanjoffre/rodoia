"""Custo de serving — R$/1k requisições derivado do throughput vLLM MEDIDO (Fase 5).

Fecha o item que o especialista pediu: "derivar R$/1k req da vazão medida — é aritmética".
NÃO é cotação ao vivo. É um **modelo de custo** ancorado em (a) a vazão REAL do modelo
fine-tunado em vLLM (`reports/fase2_ft/benchmark_vllm.json`: `req_por_s`, medido na Nitro) e
(b) preços-hora de GPU de nuvem **assumidos** (premissas explícitas, ordem de grandeza ~2026 —
não uma cotação de fornecedor). A honestidade está em separar:

  • marginal (100% de utilização)  → PISO: só o tempo de GPU que a requisição consome;
  • always-on a baixa utilização    → REALIDADE de um endpoint de portfólio: paga-se a GPU
                                        ociosa, então o custo/req sobe por 1/utilização.

Escopo: vale para a rota do modelo FT (NER, `max_tokens=128`, geração curta). A rota RAG
completa tem p95≈30 s (geração longa) — custo/req proporcionalmente maior, mesma aritmética.

Uso:  python -m rodoia.mlops.custo      # imprime a tabela -> reports/fase5_mlops/custo.json
"""
from __future__ import annotations

import json

from rodoia.config import REPO_ROOT
from rodoia.proveniencia import carimbar

CAMBIO_BRL_POR_USD = 5.40          # PREMISSA (câmbio de referência, não cotação ao vivo)
UTILIZACAO_REALISTA = 0.30         # PREMISSA: endpoint de portfólio fica ~30% ocupado

# PREMISSAS de preço-hora on-demand (USD/h), tier de GPU pequena que comporta o modelo
# fp8 (~5,2 GB VRAM medidos). Ordem de grandeza ~2026 — não é cotação de fornecedor.
GPUS: tuple[tuple[str, float], ...] = (
    ("L4 24GB (spot)", 0.28),
    ("RTX 4090 (comunidade)", 0.44),
    ("L4 24GB (on-demand)", 0.70),
    ("A10G 24GB (on-demand)", 1.00),
)


def _linha(nome: str, usd_h: float, req_s: float, cambio: float, util: float) -> dict:
    seg_por_1k = 1000.0 / req_s
    marginal_brl = (seg_por_1k / 3600.0) * usd_h * cambio          # 100% de utilização (piso)
    mensal_brl = usd_h * 24 * 30 * cambio                          # 1 instância always-on/mês
    cap_req_mes = req_s * 3600 * 24 * 30                           # capacidade a 100%
    alwayson_1k_brl = mensal_brl / (cap_req_mes * util / 1000.0)   # custo real a `util`
    return {
        "gpu": nome,
        "usd_h": usd_h,
        "brl_por_1k_marginal": round(marginal_brl, 3),
        "brl_por_1k_alwayson": round(alwayson_1k_brl, 3),
        "brl_mensal_1_instancia": round(mensal_brl, 0),
    }


def _linha_latencia(nome: str, usd_h: float, lat_s: float, cambio: float, util: float) -> dict:
    """Custo por latência single-stream (rota RAG: só temos o p50 por requisição, não a vazão
    concorrente). Teto: sem batching medido, 1 requisição ocupa a GPU por ~lat_s. Batching
    baixaria o número."""
    marginal_brl = (lat_s * 1000 / 3600.0) * usd_h * cambio        # 1k req single-stream
    return {
        "gpu": nome,
        "usd_h": usd_h,
        "brl_por_1k_marginal": round(marginal_brl, 2),
        "brl_por_1k_alwayson": round(marginal_brl / util, 2),
    }


def calcular(raiz=None) -> dict:
    raiz = raiz or REPO_ROOT
    bench = json.loads((raiz / "reports/fase2_ft/benchmark_vllm.json").read_text(encoding="utf-8"))
    req_s = bench["req_por_s"]
    ger = json.loads((raiz / "reports/fase1_geracao/avaliacao_geracao.json").read_text(
        encoding="utf-8"))
    lat_rag = ger["observabilidade"]["geracao_p50_s"]
    ft = [_linha(n, p, req_s, CAMBIO_BRL_POR_USD, UTILIZACAO_REALISTA) for n, p in GPUS]
    rag = [_linha_latencia(n, p, lat_rag, CAMBIO_BRL_POR_USD, UTILIZACAO_REALISTA) for n, p in GPUS]
    return carimbar({
        "fontes": {
            "vazao_ft": "reports/fase2_ft/benchmark_vllm.json (medido, vLLM, modelo FT 3B)",
            "latencia_rag": "reports/fase1_geracao/avaliacao_geracao.json (p50 medido, geração 7B)",
        },
        "req_por_s_ft_medido": req_s,
        "latencia_p50_rag_medida_s": lat_rag,
        "premissas": {
            "cambio_brl_por_usd": CAMBIO_BRL_POR_USD,
            "utilizacao_realista": UTILIZACAO_REALISTA,
            "natureza": "modelo de custo — premissas explícitas, NÃO cotação ao vivo",
            "cold_start": ("scale-to-zero adiciona cold-start (carregar o modelo na VRAM, ~dezenas "
                           "de s p/ o 7B) na 1ª req após ociosidade — trade-off vs. always-on"),
        },
        "rota_ft": {"nota": "vazão concorrente medida (req/s)", "cenarios": ft},
        "rota_rag": {"nota": "latência single-stream p50 (teto; batching baixaria)",
                     "cenarios": rag},
    })


def main() -> int:
    res = calcular()
    saida = REPO_ROOT / "reports" / "fase5_mlops" / "custo.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"Custo de serving — rota FT (vazão medida {res['req_por_s_ft_medido']} req/s):")
    for c in res["rota_ft"]["cenarios"]:
        print(f"  {c['gpu']:24} R$ {c['brl_por_1k_marginal']:>6}/1k (marginal) · "
              f"R$ {c['brl_por_1k_alwayson']:>6}/1k (always-on 30%) · "
              f"R$ {c['brl_mensal_1_instancia']:>6.0f}/mês")
    print(f"Custo de serving — rota RAG (p50 medido {res['latencia_p50_rag_medida_s']}s, "
          "single-stream, teto):")
    for c in res["rota_rag"]["cenarios"]:
        print(f"  {c['gpu']:24} R$ {c['brl_por_1k_marginal']:>6}/1k (marginal) · "
              f"R$ {c['brl_por_1k_alwayson']:>6}/1k (always-on 30%)")
    print(f"-> {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
