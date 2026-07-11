# 14 — Fase 3: dados estruturados da ANTT — modelagem, SQL e previsão de demanda

> Prova **SQL avançado + modelagem dimensional + previsão com métrica dura** sobre um
> dataset público real e grande. Entrega o modelo de dados (DuckDB), consultas analíticas
> (window functions), uma **camada de acesso testada** (ferramenta do agente da Fase 4) e o
> resultado objetivo: **previsão de demanda de tráfego** (RMSE/MAPE, split temporal).

## 1. Dados — Volume de Tráfego nas Praças de Pedágio (ANTT)

[Portal de Dados Abertos ANTT](https://dados.antt.gov.br/dataset/volume-trafego-praca-pedagio)
— CSVs por ano, **2010–2026**. Pipeline reproduzível (`rodoia.data.baixar_volume` via API CKAN
→ `rodoia.data.ingestao_volume`); dados brutos fora do Git (DVC).

**Desafios reais tratados na ingestão** (observabilidade: 3 linhas rejeitadas de ~2 M):
- Encoding **ISO-8859-1**, separador `;`, decimal `,` (e milhar `.`).
- **Dois formatos de data**: anuais `DD/MM/AAAA`, consolidados `MM/AAAA` — normalizados.
- **Coluna divergente** (`categoria` vs `categoria_eixo`) — reconciliada.
- **Granularidade mista** (alguns anos vêm diários) → **normalizado ao mês** (trunc + soma),
  gerando série mensal consistente: **741.205 linhas, 197 meses, 50 concessionárias, 292 praças**.

## 2. Modelagem — esquema estrela (DuckDB)

Escolha: **estrela** (fato + dimensões) sobre a tabela achatada, porque as análises são
agregações por praça/tempo/categoria — a estrela dá JOINs baratos e SQL legível. DuckDB
(embutido, colunar) casa com o local-first.

- **`fato_volume`** (741.205) — grão: praça × mês × sentido × cobrança × categoria × tipo de
  veículo → `volume_total` (o grão mais fino da fonte; o agregado sai no SQL).
- **`dim_praca`** (383: praça → concessionária) · **`dim_tempo`** (197 meses → ano/mês/trimestre)
  · **`dim_categoria`** (categoria → tipo de veículo).

## 3. SQL analítico (CTEs + window functions) — `dados/consultas.py`

Resultados versionados em `reports/fase3_dados/analitico.json`:

- **Crescimento MoM/YoY** (LAG 1 e 12 sobre a série mensal): ex. mar/2026 **+10,1% MoM / +10,3% YoY**.
- **Ranking de praças** (RANK): 1º **P4 (Litoral Sul)** 281 M · 2º **Praça 01 BR-116/SP (NovaDutra)** 270 M.
- **Sazonalidade** (média por mês do ano): picos de **dezembro/janeiro** (~56 M, férias) e vale
  no meio do ano (~46 M) — padrão plausível de viagens.
- **Composição por veículo**: Passeio **63,5%** · Comercial **34,2%** · Moto **2,1%** (a fonte tem
  variantes de caixa residuais — observação de qualidade de dado).

## 4. Camada de acesso — `dados/acesso.py` (ferramenta do agente)

Funções tipadas e **parametrizadas** (placeholders `?`, nunca concatenação — anti-SQL-injection):
`ranking_pracas`, `volume_praca`, `serie_mensal`, `volume_por_ano`. Testadas com fixture DuckDB,
**incluindo um caso adversarial** (`"PA'; DROP TABLE …"` é tratado como valor literal, não executa).
Pronto para ser chamado como ferramenta pelo agente da Fase 4.

## 5. Previsão de demanda — o resultado objetivo — `dados/previsao.py`

Série mensal da praça de histórico mais longo (**P 04, 197 meses, 2010–2026**); **split temporal**
(últimos 12 meses = teste, sem vazamento — features usam só lags passados). Baselines vs. ML:

| Modelo | RMSE | MAPE |
|---|---|---|
| naïve (mês anterior) | 9.998 | 6,88% |
| sazonal-naïve (mesmo mês do ano anterior) | 46.785 | 27,99% |
| **Gradient Boosting** (lags 1/2/3/12 + médias móveis + mês + tendência) | **9.610** | **5,93%** |

→ **O ML vence** os dois baselines: MAPE **5,93%** vs. naïve 6,88% vs. sazonal 28%. Leitura honesta:
o volume mensal é **persistente** (o naïve é um baseline forte), então o ganho do ML sobre o naïve é
**modesto porém real** (RMSE −4%, MAPE −1 p.p.); o valor claro é **esmagar o sazonal-naïve** e provar
o pipeline de forecasting com métrica dura. Gráfico em `reports/fase3_dados/previsao.png`.

## 6. Reproduzir

```bash
pip install -e ".[estruturados]"
python -m rodoia.data.baixar_volume && python -m rodoia.data.ingestao_volume
python -m rodoia.dados.estrela        # esquema estrela -> data/processed/volume.duckdb
python -m rodoia.dados.consultas      # SQL analítico -> reports/fase3_dados/analitico.json
python -m rodoia.dados.previsao       # previsão -> reports/fase3_dados/previsao.json + .png
```

## 7. Critérios de conclusão (todos ✓)

- [x] Datasets modelados com **schema justificado** (estrela, grão documentado) + licença/pipeline
- [x] **Queries analíticas** com CTEs + window functions (LAG/RANK/médias) versionadas
- [x] **Camada de acesso** tipada, parametrizada (anti-injection) e testada — ferramenta do agente
- [x] **Resultado objetivo**: previsão de demanda com **RMSE/MAPE** e split temporal (ML bate baselines)
- [x] README do modelo de dados (este doc) + observabilidade de ingestão + reprodução
- [x] Testes dos caminhos críticos (acesso, ingestão, métricas)

### Próximo passo
Usar `dados/acesso.py` como **ferramenta do agente (Fase 4)** e cruzar com Praça/Receita (JOIN
geográfico/financeiro).
