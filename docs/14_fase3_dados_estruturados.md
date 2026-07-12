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
  gerando série mensal consistente: **741.205 linhas, 197 meses, 50 concessionárias, 292 praças**
  (nomes distintos; 383 pares praça×concessionária — ver `dim_praca` no §2).

## 2. Modelagem — esquema estrela (DuckDB)

Escolha: **estrela** (fato + dimensões) sobre a tabela achatada, porque as análises são
agregações por praça/tempo/categoria — a estrela dá JOINs baratos e SQL legível. DuckDB
(embutido, colunar) casa com o local-first.

- **`fato_volume`** (741.205) — grão: praça × mês × sentido × cobrança × categoria × tipo de
  veículo → `volume_total` (o grão mais fino da fonte; o agregado sai no SQL).
- **`dim_praca`** (383 linhas — uma por praça×concessionária; 292 nomes de praça distintos) ·
  **`dim_tempo`** (197 meses → ano/mês/trimestre)
  · **`dim_categoria`** (categoria → tipo de veículo).

## 3. SQL analítico (CTEs + window functions) — `dados/consultas.py`

Resultados versionados em `reports/fase3_dados/analitico.json`:

- **Crescimento MoM/YoY** (LAG 1 e 12 sobre a série mensal): ex. mar/2026 **+10,1% MoM / +10,3% YoY**.
- **Ranking de praças** (RANK): 1º **P4 (Litoral Sul)** 281 M · 2º **Praça 01 BR-116/SP (NovaDutra)** 270 M.
- **Sazonalidade** (média por mês do ano): picos de **dezembro/janeiro** (~56 M, férias) e vale
  no meio do ano (~46 M) — padrão plausível de viagens.
- **Composição por veículo**: Passeio **63,7%** · Comercial **34,3%** · Moto **2,1%** (a fonte tem
  variantes de caixa residuais — observação de qualidade de dado).

## 4. Camada de acesso — `dados/acesso.py` (ferramenta do agente)

Funções tipadas e **parametrizadas** (placeholders `?`, nunca concatenação — anti-SQL-injection):
`ranking_pracas`, `volume_praca`, `serie_mensal`, `volume_por_ano`. Testadas com fixture DuckDB,
**incluindo um caso adversarial** (`"PA'; DROP TABLE …"` é tratado como valor literal, não executa).
Pronto para ser chamado como ferramenta pelo agente da Fase 4.

## 5. Previsão de demanda — backtest multi-step + teste pareado — `dados/previsao.py`

Avaliação **robusta e justa**: **backtest em 63 praças** com histórico mensal **contíguo ≥ 100
meses**; para cada uma, **todos os modelos preveem 12 meses à frente a partir do fim do treino,
sem ver o teste** (multi-step honesto — a tarefa real de planejamento). O naïve vira *random walk*
(repete o último valor), o GB é **recursivo** (realimenta as próprias previsões) e o Holt-Winters
faz `forecast(12)`. MAPE por praça, **agregado com IC95 por bootstrap**; e a comparação decisiva —
o **teste pareado** melhor-modelo vs naïve (diferença por praça).

| Modelo | MAPE médio | IC95 (bootstrap, n=63) |
|---|---|---|
| **Holt-Winters** | **13,25%** | [8,48; 19,57] |
| Gradient Boosting (recursivo) | 15,43% | [11,06; 21,38] |
| naïve (random walk) | 16,26% | [11,31; 22,89] |
| sazonal-naïve | 17,87% | [12,96; 24,28] |

**Comparação pareada Holt-Winters vs naïve:** Δ = **3,01 pp** de MAPE, **IC95 [1,76; 4,40]** —
**não cruza 0 → o ganho é estatisticamente significativo**; o HW vence em **73% das praças**
(≈ **18% de redução relativa** de erro).

→ **Leitura honesta:** na tarefa realista (prever 12 meses à frente), o clássico **Holt-Winters
bate o baseline de forma significativa** (teste pareado, não só ICs marginais que se sobrepõem por
variância entre praças). Nota metodológica de rigor: uma versão anterior comparava o naïve/GB em
*1-passo-à-frente* (alimentados com o valor real recente) contra um HW *multi-step* — maçãs com
laranjas, que inflava o naïve para 13,7% e mascarava o ganho. Padronizar **todos em 12-passos** é o
justo **e** revelou o resultado que convence — mesma disciplina do held-out na Fase 2 (deixar o
rigor corrigir o próprio número). Gráfico da praça mais longa em `reports/fase3_dados/previsao.png`.

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
- [x] **Resultado objetivo**: previsão de demanda avaliada com **rigor** — backtest multi-step (12m à frente) em 63 praças, **MAPE + IC95**, 4 modelos; **Holt-Winters bate o naïve de forma significativa** (teste pareado Δ=3,01pp, IC95 [1,76; 4,40], vence em 73% das praças)
- [x] README do modelo de dados (este doc) + observabilidade de ingestão + reprodução
- [x] Testes dos caminhos críticos (acesso, ingestão, métricas)

### Próximo passo
Usar `dados/acesso.py` como **ferramenta do agente (Fase 4)** e cruzar com Praça/Receita (JOIN
geográfico/financeiro).
