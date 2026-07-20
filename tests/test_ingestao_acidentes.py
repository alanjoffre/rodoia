"""Testes da derivação de alvo e engenharia de features (sem rede, sem I/O)."""

from __future__ import annotations

import pandas as pd

from rodoia.ingestao.esquema_acidentes import COLUNAS_ESPERADAS
from rodoia.ingestao.ingestao_acidentes import derivar_alvo, engenharia_features


def _linha(**over) -> dict:
    """Uma linha com o schema canônico, zerada, com overrides."""
    base = dict.fromkeys(COLUNAS_ESPERADAS, 0)
    base.update(
        data="15/03/2021",
        horario="18:30:00",
        trecho="BR-116/SP",
        sentido="Norte",
        tipo_de_acidente="Colisão",
        tipo_de_ocorrencia="x",
        n_da_ocorrencia="1",
        km=100.0,
    )
    base.update(over)
    return base


def test_alvo_sem_vitima() -> None:
    df = derivar_alvo(pd.DataFrame([_linha()]))
    assert df.loc[0, "houve_vitima"] == 0
    assert df.loc[0, "houve_fatal"] == 0
    assert df.loc[0, "n_feridos"] == 0


def test_alvo_com_ferido_nao_fatal() -> None:
    df = derivar_alvo(pd.DataFrame([_linha(levemente_feridos=2, gravemente_feridos=1)]))
    assert df.loc[0, "houve_vitima"] == 1
    assert df.loc[0, "houve_fatal"] == 0
    assert df.loc[0, "n_feridos"] == 3


def test_alvo_fatal() -> None:
    df = derivar_alvo(pd.DataFrame([_linha(mortos=1)]))
    assert df.loc[0, "houve_vitima"] == 1
    assert df.loc[0, "houve_fatal"] == 1


def test_features_temporais_e_uf() -> None:
    df = engenharia_features(pd.DataFrame([_linha()]))
    assert df.loc[0, "ano"] == 2021
    assert df.loc[0, "mes"] == 3
    assert df.loc[0, "dia_semana"] == 0  # 15/03/2021 foi segunda-feira
    assert df.loc[0, "hora"] == 18
    assert df.loc[0, "uf"] == "SP"


def test_total_veiculos() -> None:
    df = engenharia_features(pd.DataFrame([_linha(automovel=2, caminhao=1, moto=1)]))
    assert df.loc[0, "total_veiculos"] == 4


def test_uf_ausente_vira_na() -> None:
    df = engenharia_features(pd.DataFrame([_linha(trecho="rodovia sem uf")]))
    assert df.loc[0, "uf"] == "NA"


def test_normaliza_caixa_das_categoricas() -> None:
    # 'Colisão Traseira' e 'colisão traseira' devem virar o mesmo valor.
    df = engenharia_features(
        pd.DataFrame([_linha(tipo_de_acidente="Colisão Traseira", sentido="Norte")])
    )
    assert df.loc[0, "tipo_de_acidente"] == "colisão traseira"
    assert df.loc[0, "sentido"] == "norte"
