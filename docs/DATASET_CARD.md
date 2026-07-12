# Dataset Card — fontes de dados do RodoIA

Cartões das fontes públicas usadas. Fronteira inviolável: **somente domínio público da ANTT e
datasets consagrados** (ver `NOTICE` e [data/README.md](../data/README.md)). Dados brutos/processados
**não** vão para o Git (regeneráveis pelo pipeline; DVC para o remoto).

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
- **Obtenção:** `rodoia.data.baixar_volume` → `rodoia.data.ingestao_volume`.
- **Qualidade tratada:** datas mistas (`DD/MM/AAAA` vs `MM/AAAA`), coluna `categoria`/`categoria_eixo`,
  granularidade diária→mensal, normalização de caixa.

## 3. Acidentes em rodovias concedidas — Fase 0 (ML clássico)
- **Uso:** classificação de severidade (`houve_vitima`).
- **Licença:** **CC-BY** (confirmar por dataset).
- **Escala:** ~1,03 M linhas; 39 CSVs → 37 concessionárias reconciliadas.
- **Obtenção:** `rodoia.data.baixar_acidentes` → `rodoia.data.ingestao_acidentes`.

## 4. LeNER-Br — Fase 2 (NER / fine-tuning)
- **Uso:** tarefa de rótulo objetivo (NER jurídico) para o fine-tuning e o baseline BERTimbau.
- **Licença:** **MIT** (citar PROPOR 2018).
- **Escala:** 7.827 / 1.176 / 1.389 sentenças (treino/val/teste); 6 tipos de entidade.
- **Obtenção:** `rodoia.ner.lener` baixa da fonte; datasets processados (`data/processed/*.jsonl`)
  são **regeneráveis** e não versionados.
- **⚠️ PII:** contém CPF/CNPJ/nomes de **registros judiciais públicos**. Legítimo sob a MIT, mas
  **não redistribuímos** no repositório público (gitignored) e **não deve ser usado para
  identificação/perfilamento**. Ver a nota de regeneração em [data/README.md](../data/README.md).

## Ética e governança
- Atribuição registrada no `NOTICE`; licença confirmada **antes** do uso (validação em
  [docs/00](00_validacao_fontes_antt.md)).
- PII mascarada nas respostas/logs da API (`rag/seguranca.mascarar_pii`).
- Zero dado/regra de empregador ou cliente (fronteira em `PROMPT_MESTRE.md` §3.1).
