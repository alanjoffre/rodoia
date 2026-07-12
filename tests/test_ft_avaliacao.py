"""Testes das funções puras de avaliação da Fase 2 (sem GPU, sem rede)."""
from rodoia.ft.aval_cite import cita_alguma, cita_correta
from rodoia.ft.juiz_factual import _nota
from rodoia.ft.juiz_winrate import _decidir, _parse_veredito, _truncar_par

# ---- juiz_factual: parsing/clamp da nota ----

def test_nota_extrai_e_clampeia():
    assert _nota('{"nota": 0.9}') == 0.9
    assert _nota('lixo {"nota": 1.5} x') == 1.0   # clamp topo
    assert _nota('{"nota": -0.2}') == 0.0          # clamp piso
    assert _nota("sem json") == 0.0


# ---- aval_cite ----

def test_cita_correta_com_e_sem_ponto():
    assert cita_correta("conforme a Resolução nº 6024/2023", ["6024/2023"])
    assert cita_correta("nos termos da Resolução 6.024/2023", ["6024/2023"])  # ponto de milhar


def test_cita_correta_numero_errado():
    assert not cita_correta("a Resolução nº 6.088/2016 trata disso", ["6024/2023"])
    assert not cita_correta("Resolução 6024/2024", ["6024/2023"])  # ano diferente


def test_cita_correta_multi_fonte_casa_qualquer():
    # com múltiplas fontes esperadas, citar QUALQUER uma conta como correta
    assert cita_correta("ver a Resolução 673/2004 sobre vale-pedágio", ["6024/2023", "673/2004"])


def test_cita_correta_sem_citacao():
    assert not cita_correta("a ANTT regula o transporte de cargas", ["6024/2023"])


def test_cita_alguma():
    assert cita_alguma("calculado pela Resolução 5867/2020")
    assert not cita_alguma("a ANTT regulamenta os pisos mínimos de frete")


# ---- juiz_winrate: parsing do veredito ----

def test_parse_veredito_valido():
    assert _parse_veredito('{"melhor": "A"}') == "A"
    assert _parse_veredito('{"melhor": "B"}') == "B"
    assert _parse_veredito('texto {"melhor": "empate"} extra') == "empate"


def test_parse_veredito_invalido_vira_empate():
    assert _parse_veredito("sem json aqui") == "empate"
    assert _parse_veredito('{"melhor": "X"}') == "empate"
    assert _parse_veredito("") == "empate"


# ---- juiz_winrate: decisão pareada com troca de posição ----

def test_decidir_ft_vence_consistente():
    # ordem1 (A=base,B=ft) -> B ; ordem2 (A=ft,B=base) -> A  => FT ganhou nas duas
    assert _decidir("B", "A") == "ft"


def test_decidir_base_vence_consistente():
    assert _decidir("A", "B") == "base"


def test_decidir_inconsistente_ou_empate():
    assert _decidir("A", "A") == "empate"   # contradição -> empate
    assert _decidir("empate", "A") == "empate"
    assert _decidir("B", "B") == "empate"


def test_truncar_par_iguala_tamanho():
    a, b = _truncar_par("abcdefghij", "abc")
    assert a == "abc" and b == "abc"
    assert len(a) == len(b)
