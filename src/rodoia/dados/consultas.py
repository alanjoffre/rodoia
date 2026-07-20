"""SQL analítico sobre o esquema estrela (Fase 3) — CTEs + window functions.

Consultas que exercitam SQL avançado (window: LAG, RANK, médias móveis) sobre o Volume
de Tráfego de Pedágio. Cada uma responde a uma pergunta de negócio e serve de base para a
camada de acesso (ferramenta do agente) e para a previsão. Resultados versionados em
`reports/fase3_dados/analitico.json`.

Uso:  python -m rodoia.dados.consultas
"""
from __future__ import annotations

import json

from rodoia.config import REPO_ROOT
from rodoia.dados.estrela import consultar_ro
from rodoia.proveniencia import carimbar

# Crescimento MoM/YoY do volume total da rede (LAG sobre a série mensal).
CRESCIMENTO = """
WITH serie AS (
    SELECT data, sum(volume_total) AS volume
    FROM fato_volume GROUP BY data
)
SELECT data, volume,
       round(100.0 * (volume - lag(volume, 1)  OVER (ORDER BY data))
                    / lag(volume, 1)  OVER (ORDER BY data), 2) AS mom_pct,
       round(100.0 * (volume - lag(volume, 12) OVER (ORDER BY data))
                    / lag(volume, 12) OVER (ORDER BY data), 2) AS yoy_pct
FROM serie ORDER BY data DESC LIMIT 6
"""

# Top praças por volume acumulado (RANK).
RANKING_PRACAS = """
WITH tot AS (
    SELECT p.praca, p.concessionaria, sum(f.volume_total) AS volume
    FROM fato_volume f JOIN dim_praca p USING (praca_id)
    GROUP BY p.praca, p.concessionaria
)
SELECT rank() OVER (ORDER BY volume DESC) AS posicao, praca, concessionaria, volume
FROM tot ORDER BY volume DESC LIMIT 10
"""

# Sazonalidade: volume médio por mês do ano (padrão sazonal).
SAZONALIDADE = """
SELECT t.mes, t.nome_mes, round(avg(m.volume)) AS volume_medio
FROM (SELECT data, sum(volume_total) AS volume FROM fato_volume GROUP BY data) m
JOIN dim_tempo t USING (data)
GROUP BY t.mes, t.nome_mes ORDER BY t.mes
"""

# Composição por tipo de veículo (Passeio × Comercial).
COMPOSICAO = """
SELECT tipo_de_veiculo,
       round(100.0 * sum(volume_total) / (SELECT sum(volume_total) FROM fato_volume), 1) AS pct
FROM fato_volume GROUP BY tipo_de_veiculo ORDER BY pct DESC
"""

CONSULTAS = {
    "crescimento_mom_yoy": CRESCIMENTO,
    "ranking_pracas": RANKING_PRACAS,
    "sazonalidade": SAZONALIDADE,
    "composicao_veiculo": COMPOSICAO,
}


def rodar(sql: str) -> list[dict]:
    return consultar_ro(sql)


def main() -> None:
    res = carimbar({nome: rodar(sql) for nome, sql in CONSULTAS.items()})
    saida = REPO_ROOT / "reports" / "fase3_dados"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "analitico.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str))
    print("=== sazonalidade (volume médio por mês) ===")
    for r in res["sazonalidade"]:
        print(f"  {r['nome_mes']:10} {r['volume_medio']:>14,.0f}")
    print("\n=== top 3 praças ===")
    for r in res["ranking_pracas"][:3]:
        print(f"  {r['posicao']}. {r['praca']} ({r['concessionaria']}) — {r['volume']:,.0f}")
    print(f"\nrelatório: {saida / 'analitico.json'}")


if __name__ == "__main__":
    main()
