"""Testes das funções puras do NER (Fase 2, LeNER-Br) — sem GPU/rede."""
from rodoia.ner.avaliar_generativo import metricas_ner, parse_entidades
from rodoia.ner.generativo import como_conjunto, entidades_bio
from rodoia.ner.lener import LABEL2ID


def test_entidades_bio_extrai_spans():
    tokens = ["MINISTÉRIO", "PÚBLICO", "propôs", "em", "2020"]
    tags = [LABEL2ID["B-ORGANIZACAO"], LABEL2ID["I-ORGANIZACAO"], LABEL2ID["O"],
            LABEL2ID["O"], LABEL2ID["B-TEMPO"]]
    ents = entidades_bio(tokens, tags)
    assert ("MINISTÉRIO PÚBLICO", "ORGANIZACAO") in ents
    assert ("2020", "TEMPO") in ents
    assert len(ents) == 2


def test_entidades_bio_ignora_I_orfao():
    # I- sem B- do mesmo tipo não inicia entidade
    tokens = ["x", "y"]
    tags = [LABEL2ID["O"], LABEL2ID["I-PESSOA"]]
    assert entidades_bio(tokens, tags) == []


def test_parse_entidades_json_valido_e_lixo():
    s = 'aqui: [{"texto": "João", "tipo": "PESSOA"}, {"texto": "STF", "tipo": "ORGANIZACAO"}] fim'
    assert parse_entidades(s) == {("joão", "PESSOA"), ("stf", "ORGANIZACAO")}
    assert parse_entidades("sem json") == set()
    # tipo inválido é descartado
    assert parse_entidades('[{"texto": "x", "tipo": "INVALIDO"}]') == set()


def test_metricas_ner_f1():
    preds = [como_conjunto([("João", "PESSOA"), ("Brasil", "LOCAL")])]
    golds = [como_conjunto([("João", "PESSOA")])]
    m = metricas_ner(preds, golds)
    # tp=1, fp=1, fn=0 -> precision 0.5, recall 1.0, F1 = 0.667
    assert abs(m["f1_micro"] - 0.6667) < 1e-3
    assert m["f1_por_entidade"]["PESSOA"] == 1.0
    assert m["f1_por_entidade"]["LOCAL"] == 0.0
