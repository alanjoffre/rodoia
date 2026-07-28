# Dataset Card — fontes de dados do RodoIA

Cartões das fontes públicas usadas. Fronteira inviolável: **somente dado público** — domínio
público da ANTT nas Fases 0–4, mais datasets públicos consagrados de terceiros (LeNER-Br na
Fase 2; CFPB e CUAD na Fase 6), cada um com licença confirmada (ver `NOTICE` e
[data/README.md](../data/README.md)). Dados brutos/processados **não** vão para o Git
(regeneráveis pelo pipeline; DVC para o remoto).

## 1. Normas/resoluções da ANTT (ANTTlegis) — Fase 1 (RAG)
- **Uso:** corpus de recuperação (texto regulatório).
- **Licença:** ato oficial, **domínio público** (Lei 9.610/98, art. 8º, IV).
- **Obtenção:** `rodoia.rag.baixar_normas` (scraping determinístico; OCR p/ normas antigas).
- **Notas:** limpeza de HTML e checagem de vigência em `rag/fontes_antt.py`.

## 2. Volume de Tráfego nas Praças de Pedágio — Fase 3 (SQL + previsão)
- **Uso:** modelagem dimensional + previsão de demanda.
- **Licença:** **CC-BY** (Decreto 8.777/2016) → atribuição à ANTT (no `NOTICE`).
- **Escala:** 2010–2026; após ingestão → **741.205 linhas, 197 meses, 50 concessionárias, 292
  praças** (383 pares praça×concessionária).
- **Obtenção:** `rodoia.ingestao.baixar_volume` → `rodoia.ingestao.ingestao_volume`.
- **Qualidade tratada:** datas mistas (`DD/MM/AAAA` vs `MM/AAAA`), coluna `categoria`/`categoria_eixo`,
  granularidade diária→mensal, normalização de caixa.

## 3. Acidentes em rodovias concedidas — Fase 0 (ML clássico)
- **Uso:** classificação de severidade (`houve_vitima`).
- **Licença:** **CC-BY** (confirmar por dataset).
- **Escala:** ~1,03 M linhas; 39 CSVs → 37 concessionárias reconciliadas.
- **Obtenção:** `rodoia.ingestao.baixar_acidentes` → `rodoia.ingestao.ingestao_acidentes`.

## 4. LeNER-Br — Fase 2 (NER / fine-tuning)
- **Uso:** tarefa de rótulo objetivo (NER jurídico) para o fine-tuning e o baseline BERTimbau.
- **Licença:** **MIT** (citar PROPOR 2018).
- **Escala:** 7.827 / 1.176 / 1.389 sentenças (treino/val/teste); 6 tipos de entidade.
- **Obtenção:** `rodoia.ner.lener` baixa da fonte; datasets processados (`data/processed/*.jsonl`)
  são **regeneráveis** e não versionados.
- **⚠️ PII:** contém CPF/CNPJ/nomes de **registros judiciais públicos**. Legítimo sob a MIT, mas
  **não redistribuímos** no repositório público (gitignored) e **não deve ser usado para
  identificação/perfilamento**. Ver a nota de regeneração em [data/README.md](../data/README.md).

## 5. CFPB Consumer Complaint Database — Fase 6 (ingestão em escala)
- **Uso:** provar ingestão com memória limitada e escolher o motor de dados por benchmark. O
  corpus da ANTT tem 3.647 chunks e não exerce particionamento nem poda de partição.
- **Licença:** **domínio público** — *U.S. Government Works*, publicado por agência federal dos
  EUA. Sem exigência de atribuição.
- **Escala:** **17.226.584 linhas**, 16 colunas, 2011-12 a 2026-07; 1,43 GB de zip → 1,1 GB de
  Parquet particionado (Hive, `ano=YYYY`), sem materializar os 13,5 GB de CSV.
- **Obtenção:** `rodoia.ingestao.baixar_cfpb` → `rodoia.ingestao.ingestao_cfpb`.
- **Qualidade tratada:** três formatos de data na mesma coluna; taxonomia de `product` com
  variantes sobrepostas (gravada crua, harmonização explícita na camada de domínio); WAF que
  responde 403 a User-Agent de navegador.
- **⚠️ Viés de publicação:** só **22,21%** das linhas têm narrativa livre — a CFPB publica o
  texto apenas com consentimento do consumidor, e a taxa varia de **0% (2011–2014)** a 47,5%
  (2017) e 2,1% (2026, ainda não assentado). Qualquer trabalho de texto sobre esta base herda
  esse viés; ver `docs/17` §2.

## 6. CUAD v1 — Fase 6 (benchmark de recuperação de terceiros)
- **Uso:** medir recuperação contra **gold que não é do autor**. É o que responde à objeção
  *"ele rotulou o próprio teste"*, que a auditoria κ da Fase 1 só tratava por dentro.
- **Licença:** **CC BY 4.0** — **exige atribuição**. Obra: *Contract Understanding Atticus
  Dataset (CUAD) v1*, de **The Atticus Project**
  ([atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad)). Citação requerida:
  **Hendrycks, Burns, Chen & Ball, "CUAD: An Expert-Annotated NLP Dataset for Legal Contract
  Review", NeurIPS 2021.** Atribuição completa no [`NOTICE`](../NOTICE).
- **Escala:** 510 contratos comerciais · 20.910 perguntas · **13.823 spans** anotados por
  advogados · 41 categorias de cláusula · **14.208 (67,95%) `is_impossible`**.
- **Obtenção:** `rodoia.rag.baixar_cuad` → `rodoia.rag.cuad`.
- **Integridade aferida antes de qualquer métrica:** os **13.823 spans conferidos um a um** —
  **0 divergentes** entre o offset declarado e o texto. Gold desalinhado produz métrica
  plausível e falsa, que nenhum teste pegaria.
- **⚠️ Escopo da licença:** a CC BY 4.0 cobre **a anotação**. Os contratos subjacentes são
  públicos e vêm do **EDGAR (SEC/EUA)**; o Atticus Project não declara garantia sobre o status
  de licença deles, e essa ressalva é repassada aqui em vez de omitida.

## Ética e governança
- Atribuição registrada no `NOTICE`; licença confirmada **antes** do uso (validação em
  [docs/00](00_validacao_fontes_antt.md)).
- **Falha registrada (corrigida em 28/07/2026):** as duas fontes da Fase 6 rodaram sem cartão
  aqui, sem linha no `NOTICE` e com a licença do CUAD declarada **errada** no `docs/17`
  (*"Apache 2.0"* em vez de CC BY 4.0, que é a que exige atribuição). A regra "licença
  confirmada antes do uso" existia e não foi cumprida justamente na fase que trouxe dado de
  terceiros. Fica no cartão porque uma governança que só registra os acertos não é governança.
- PII mascarada nas respostas/logs da API (`rag/seguranca.mascarar_pii`).
- Zero dado/regra de empregador ou cliente (fronteira em `PROMPT_MESTRE.md` §3.1).
