"""Reprodução de métrica-âncora (Fase 5, P1) — o que o gate NÃO faz.

Enquanto `mlops/gate.py` só LÊ os JSONs commitados, aqui **re-executamos o pipeline** e
comparamos a métrica regenerada contra o valor versionado. Se divergir além da tolerância,
falha (exit 1). Isso fecha o ataque "seu CI não reproduz nada, só lê números que você commitou".

Âncora padrão: **hit@5 do retrieval híbrido** — determinística (mesmo índice + mesmas queries),
roda em CPU, sem LLM. Precisa do corpus/índice, regenerado no próprio job `reproduzir` por
`rag.baixar_normas` + `rag.construir_indice` — que roda em runner **github-hosted (CPU)**,
justamente para não esconder a reprodução atrás de hardware que ninguém pode auditar.
Âncoras que exigem GPU (ex.: NER F1 via vLLM) entram no mesmo harness quando há runner com placa.

Uso:  python -m rodoia.mlops.reproduzir            # reproduz a âncora e sai 1 se divergir
"""
from __future__ import annotations

import json
import sys

from rodoia.config import REPO_ROOT

# Tolerância pequena: o retrieval é determinístico, então a reprodução deve bater quase exato.
TOL_HIT = 0.02


def reproduzir_retrieval(tol: float = TOL_HIT) -> tuple[bool, dict]:
    """Re-roda o retrieval híbrido e compara hit@5 contra o report versionado."""
    from rodoia.rag.avaliacao_retrieval import avaliar_modo, carregar_recuperador

    rel = REPO_ROOT / "reports" / "fase1_retrieval" / "avaliacao_retrieval.json"
    esperado = json.loads(rel.read_text(encoding="utf-8"))["hibrido"]["hit_rate_at_k"]

    rec = carregar_recuperador(com_reranker=False)
    novo = avaliar_modo(rec, "hibrido", rerank=False)["hit_rate_at_k"]

    delta = abs(novo - esperado)
    ok = delta <= tol
    info = {"ancora": "retrieval_hit@5_hibrido", "commitado": esperado,
            "reproduzido": round(novo, 4), "delta": round(delta, 4), "tol": tol, "ok": ok}
    return ok, info


def reproduzir_previsao(tol: float = 0.1) -> tuple[bool, dict]:
    """2ª âncora: re-roda o backtest de previsão (determinístico, CPU) e compara o MAPE médio do
    Holt-Winters contra o report. Pula (sem falhar) se o DuckDB da Fase 3 não existir."""
    import numpy as np

    from rodoia.dados.estrela import DB

    rel = REPO_ROOT / "reports" / "fase3_dados" / "previsao.json"
    if not DB.exists():
        return True, {"ancora": "previsao_mape_holt_winters", "pulado": "DuckDB da Fase 3 ausente"}
    esperado = json.loads(rel.read_text(encoding="utf-8"))["modelos"]["holt_winters"]["mape_medio"]

    from rodoia.dados.previsao import _prever_praca, _series_completas
    vals = [v for _, s in _series_completas()
            if (v := _prever_praca(s).get("holt_winters")) is not None and np.isfinite(v)]
    novo = round(float(np.mean(vals)), 2)

    delta = abs(novo - esperado)
    ok = delta <= tol
    return ok, {"ancora": "previsao_mape_holt_winters", "commitado": esperado,
                "reproduzido": novo, "delta": round(delta, 4), "tol": tol, "ok": ok}


ANCORAS = (reproduzir_retrieval, reproduzir_previsao)


def main() -> int:
    from rodoia.proveniencia import carimbar

    print("Reprodução de métricas-âncora (re-executa o pipeline, não só lê o JSON):")
    infos, tudo_ok = [], True
    for fn in ANCORAS:
        ok, info = fn()
        infos.append(info)
        if "pulado" in info:
            print(f"  [–] {info['ancora']}: PULADO ({info['pulado']})")
            continue
        tudo_ok = tudo_ok and ok
        marca = "✓" if ok else "✗"
        print(f"  [{marca}] {info['ancora']}: commitado={info['commitado']} "
              f"reproduzido={info['reproduzido']} (Δ={info['delta']} ≤ {info['tol']})")
    # Evidência VERSIONADA de que a reprodução rodou (com carimbo git_sha/git_dirty).
    saida = REPO_ROOT / "reports" / "fase1_retrieval" / "reproducao.json"
    saida.write_text(json.dumps(carimbar({"ancoras": infos}), ensure_ascii=False, indent=2))
    print(f"evidência: {saida}")
    print("REPRODUZIDO" if tudo_ok else "DIVERGIU — alguma métrica regenerada não bate")
    return 0 if tudo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
