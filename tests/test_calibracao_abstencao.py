"""Testes da calibração de abstenção e do McNemar (puros — sem GPU, sem dados)."""

from __future__ import annotations

from rodoia.estat import mcnemar
from rodoia.rag.calibracao_abstencao import (
    analisar,
    auc_roc,
    economia_cascata,
    melhor_ponto,
    varrer_limiar,
)


def test_auc_separacao_perfeita() -> None:
    """Respondíveis todos acima dos impossíveis -> AUC 1,0."""
    assert auc_roc([5.0, 6.0, 7.0], [1.0, 2.0, 3.0]) == 1.0


def test_auc_sinal_inutil() -> None:
    """Distribuições idênticas -> AUC 0,5 (moeda). É o caso que o relatório precisa
    denunciar: um escore que não separa nada."""
    assert auc_roc([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.5


def test_auc_separacao_invertida() -> None:
    assert auc_roc([1.0, 2.0], [5.0, 6.0]) == 0.0


def test_varredura_cobre_os_extremos_degenerados() -> None:
    """A curva tem que MOSTRAR os dois extremos: responder a tudo e abster de tudo.
    Escondê-los daria a impressão de que sempre existe um ponto bom."""
    curva = varrer_limiar([3.0, 4.0], [1.0, 2.0], n_pontos=20)
    assert curva[0].cobertura == 1.0        # limiar mínimo: responde a tudo
    assert curva[0].nao_alucinacao == 0.0
    assert curva[-1].nao_alucinacao == 1.0  # limiar máximo: abstém de quase tudo


def test_melhor_ponto_acha_a_separacao() -> None:
    curva = varrer_limiar([10.0, 11.0, 12.0], [1.0, 2.0, 3.0], n_pontos=40)
    m = melhor_ponto(curva)
    assert m is not None
    assert m.j_youden == 1.0  # separação perfeita
    assert 3.0 < m.limiar <= 10.0


def test_analisar_relatorio_completo() -> None:
    r = analisar({"respondivel": [5.0, 6.0], "impossivel": [1.0, 2.0]})
    assert r["auc_roc"] == 1.0
    assert r["n_respondiveis"] == 2
    assert r["melhor_ponto_youden"]["j_youden"] == 1.0
    assert len(r["curva"]) > 0


def test_analisar_lista_vazia_nao_quebra() -> None:
    r = analisar({"respondivel": [], "impossivel": []})
    assert r["auc_roc"] == 0.5
    assert r["curva"] == []


def test_cascata_pondera_pelo_prior_da_populacao() -> None:
    """A economia depende de QUANTAS perguntas são impossíveis. Com separação
    perfeita e 3 impossíveis para 1 respondível, um limiar que abstém de todas as
    impossíveis evita 75% das chamadas sem perder nenhuma respondível."""
    curva = varrer_limiar([10.0], [1.0, 2.0, 3.0], n_pontos=40)
    linhas = economia_cascata(curva, n_respondivel=1, n_impossivel=3, alvos=(0.99,))
    assert len(linhas) == 1
    assert linhas[0]["chamadas_llm_evitadas"] == 0.75
    assert linhas[0]["respondiveis_perdidas"] == 0.0


def test_cascata_contabiliza_a_perda_de_respondiveis() -> None:
    """Com distribuições que se sobrepõem, chegar a alta não-alucinação exige
    barrar respondíveis — e isso tem que aparecer como PERDA, não sumir dentro da
    economia. Aqui uma respondível (1,0) está na faixa das impossíveis."""
    curva = varrer_limiar([1.0, 5.0], [1.0, 2.0], n_pontos=40)
    linhas = economia_cascata(curva, n_respondivel=2, n_impossivel=2, alvos=(0.99,))
    assert linhas[0]["nao_alucinacao"] == 1.0
    assert linhas[0]["respondiveis_perdidas"] == 0.5


def test_cascata_pula_alvo_inatingivel() -> None:
    """Se nenhum limiar atinge o alvo, a linha SOME — em vez de reportar o ponto
    mais próximo como se tivesse atingido. Distribuições idênticas nunca chegam a
    99% de não-alucinação, e o relatório não pode fingir que chegam."""
    curva = varrer_limiar([1.0, 2.0], [1.0, 2.0], n_pontos=40)
    assert economia_cascata(curva, 2, 2, alvos=(0.99,)) == []
    assert economia_cascata([], 1, 1) == []


# --- McNemar ---------------------------------------------------------------


def test_mcnemar_conta_discordantes() -> None:
    a = [True, True, False, False]
    b = [True, False, True, False]
    r = mcnemar(a, b)
    assert r["b10"] == 1  # só a acerta
    assert r["b01"] == 1  # só b acerta
    assert r["n_discordantes"] == 2


def test_mcnemar_sem_discordancia() -> None:
    r = mcnemar([True, False], [True, False])
    assert r["n_discordantes"] == 0
    assert r["p_valor"] == 1.0


def test_mcnemar_usa_binomial_exato_em_amostra_pequena() -> None:
    """Com poucos discordantes a aproximação χ² não vale — mas o binomial exato vale,
    e é ele que responde. 5 discordantes todos na mesma direção: p = 2·(1/2)^5."""
    a = [True] * 10 + [False] * 5
    b = [True] * 10 + [True] * 5  # 5 discordantes, todos b01
    r = mcnemar(a, b)
    assert r["n_discordantes"] == 5
    assert r["metodo"] == "binomial_exato"
    assert r["p_valor"] == round(2 / 32, 6)


def test_mcnemar_caso_real_da_ablacao_de_prompt() -> None:
    """21 acertos exclusivos de um lado contra 2 do outro, 23 discordantes: fora do
    regime do χ² e dentro do binomial exato — que decide o que os ICs sobrepostos
    da §13.4 não decidiam."""
    a = [False] * 21 + [True] * 2 + [True] * 100
    b = [True] * 21 + [False] * 2 + [True] * 100
    r = mcnemar(a, b)
    assert r["n_discordantes"] == 23
    assert r["metodo"] == "binomial_exato"
    assert float(r["p_valor"]) < 0.001


def test_mcnemar_binomial_simetrico_nao_e_significativo() -> None:
    """Discordância equilibrada não vira significância só porque n é pequeno."""
    a = [False] * 5 + [True] * 5
    b = [True] * 5 + [False] * 5
    r = mcnemar(a, b)
    assert r["metodo"] == "binomial_exato"
    assert float(r["p_valor"]) == 1.0


def test_mcnemar_diferenca_grande_e_significativa() -> None:
    """40 itens onde só b acerta, 2 onde só a acerta -> p muito baixo."""
    a = [False] * 40 + [True] * 2 + [True] * 60
    b = [True] * 40 + [False] * 2 + [True] * 60
    r = mcnemar(a, b)
    assert r["n_discordantes"] == 42
    assert r["metodo"] == "qui2_yates"   # acima de 25, a aproximação vale
    assert float(r["p_valor"]) < 0.001


def test_mcnemar_vetores_de_tamanhos_diferentes() -> None:
    assert mcnemar([True], [True, False])["n_discordantes"] == 0


def test_p_valor_minusculo_nao_vira_zero() -> None:
    """`round(p, 6)` transformava 2,2e-08 em 0,0 — e p nunca é zero. O leitor perdia
    a ordem de grandeza exatamente onde a evidência é mais forte. Caso real: a
    comparação qwen2.5:7b × gemma2:9b na cobertura (36 × 1 discordantes)."""
    a = [False] * 36 + [True] * 1 + [True] * 200
    b = [True] * 36 + [False] * 1 + [True] * 200
    p = float(mcnemar(a, b)["p_valor"])
    assert p > 0.0
    assert p < 1e-6
