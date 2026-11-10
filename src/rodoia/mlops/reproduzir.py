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


def _mape_hw(series) -> float:
    import numpy as np

    from rodoia.dominio.previsao import _prever_praca
    vals = [v for _, s in series
            if (v := _prever_praca(s).get("holt_winters")) is not None and np.isfinite(v)]
    return round(float(np.mean(vals)), 2)


def reproduzir_previsao(tol: float = 0.1) -> tuple[bool, dict]:
    """2ª âncora: re-roda o backtest de previsão (determinístico, CPU) sobre o RECORTE
    VERSIONADO e compara o MAPE médio do Holt-Winters contra o report. O recorte tem de ser o
    mesmo do report (sha256), senão o número commitado não veio daquele dado.

    A tolerância cobre só variação de versão de biblioteca (o job instala sem lock). O dado ao
    vivo, se o DuckDB existir, entra como DERIVA informativa: a ANTT publica meses novos e
    revisa meses antigos, então ele muda sem que o código mude — isso não reprova a âncora."""
    from rodoia.dominio import previsao
    from rodoia.dominio.estrela import DB

    rel = json.loads((REPO_ROOT / "reports" / "fase3_dados" / "previsao.json")
                     .read_text(encoding="utf-8"))
    esperado = rel["modelos"]["holt_winters"]["mape_medio"]
    info: dict = {"ancora": "previsao_mape_holt_winters", "commitado": esperado, "tol": tol}
    if not previsao.RECORTE.exists():
        return False, {**info, "reproduzido": None, "delta": None, "ok": False,
                       "erro": "recorte versionado ausente"}
    sha = previsao.info_recorte()["sha256"]
    mesmo_recorte = sha == rel.get("recorte", {}).get("sha256")

    novo = _mape_hw(previsao.carregar_recorte())
    delta = abs(novo - esperado)
    ok = mesmo_recorte and delta <= tol
    info.update({"reproduzido": novo, "delta": round(delta, 4), "ok": ok,
                 "recorte_sha256": sha, "recorte_igual_ao_report": mesmo_recorte})
    if DB.exists():
        ao_vivo = _mape_hw(previsao._series_completas())
        info["deriva_ao_vivo"] = {"mape": ao_vivo, "delta": round(abs(ao_vivo - esperado), 4)}
    return ok, info


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
        if info.get("recorte_igual_ao_report") is False:
            print("      o recorte versionado não é o do report (sha256 diferente)")
        if "erro" in info:
            print(f"      {info['erro']}")
        if "deriva_ao_vivo" in info:
            d = info["deriva_ao_vivo"]
            print(f"      deriva do dado ao vivo (informativa, não reprova): MAPE={d['mape']} "
                  f"(Δ={d['delta']})")
    # Evidência VERSIONADA de que a reprodução rodou (com carimbo git_sha/git_dirty).
    saida = REPO_ROOT / "reports" / "fase1_retrieval" / "reproducao.json"
    saida.write_text(json.dumps(carimbar({"ancoras": infos}), ensure_ascii=False, indent=2))
    print(f"evidência: {saida}")
    print("REPRODUZIDO" if tudo_ok else "DIVERGIU — alguma métrica regenerada não bate")
    return 0 if tudo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
