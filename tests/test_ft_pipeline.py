"""Testes das funções puras do pipeline de Fase 2 (sem GPU, sem rede):
split held-out, agregação de perplexidade, percentil do benchmark, montagem de golden."""
import math

from rodoia.ft.benchmark_vllm import percentil
from rodoia.ft.gen_offline import montar_conversas, montar_respostas
from rodoia.ft.perplexidade import _agregar_ppl
from rodoia.ft.split_dataset import dividir


# ---- split held-out ----

def _exemplos():
    # 5 normas x 2 exemplos
    return [{"fonte": f"{n}/2020", "messages": [{"role": "assistant", "content": "x"}]}
            for n in range(5) for _ in range(2)]


def test_dividir_deterministico_e_separa_normas():
    tr1, ho1, normas1 = dividir(_exemplos(), n_holdout=2, seed=42)
    tr2, ho2, normas2 = dividir(_exemplos(), n_holdout=2, seed=42)
    assert normas1 == normas2  # determinístico
    assert len(normas1) == 2 and len(ho1) == 4 and len(tr1) == 6
    # nenhuma norma held-out aparece no treino (separação real)
    fontes_treino = {e["fonte"] for e in tr1}
    assert not (set(normas1) & fontes_treino)


# ---- agregação de perplexidade ----

class _LP:
    def __init__(self, lp):
        self.logprob = lp


def test_agregar_ppl_calcula_micro_macro():
    # 1 texto, 3 posições: None (1º token) + logprobs -1.0 e -2.0 → NLL=3, 2 tokens
    pls = [[None, {10: _LP(-1.0)}, {11: _LP(-2.0)}]]
    r = _agregar_ppl(pls)
    assert r["n_tokens"] == 2
    assert math.isclose(r["ppl_micro"], math.exp(1.5), rel_tol=1e-3)


def test_agregar_ppl_vazio():
    assert _agregar_ppl([[None]])["ppl_micro"] is None


# ---- benchmark: percentil ----

def test_percentil():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentil(vals, 0.5) == 3.0
    assert percentil(vals, 0.95) == 5.0
    assert percentil([], 0.5) == 0.0


# ---- gen_offline: montagem ----

def test_montar_conversas_e_respostas():
    golden = [{"consulta": "P1?", "fontes": ["6024/2023", "673/2004"]}]
    convs = montar_conversas(golden, sistema="SYS")
    assert convs[0][0] == {"role": "system", "content": "SYS"}
    assert convs[0][1]["content"] == "P1?"
    resp = montar_respostas(golden, ["  resposta  "])
    assert resp[0]["resposta"] == "resposta"  # strip
    assert resp[0]["fontes"] == ["6024/2023", "673/2004"]  # preserva multi-fonte
