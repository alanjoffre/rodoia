# 13 — Fase 2: fine-tuning que vence — NER jurídico (LeNER-Br) vs. SOTA

> O **resultado principal** da Fase 2. Prova que sei *quando* e *como* o fine-tuning
> agrega — com **métrica dura, benchmarkada** (F1 de entidade), não LLM-juiz sobre dado
> sintético. Executado na Nitro (RTX 4050) reusando a infra QLoRA + fp8 + vLLM da fase.

## 1. O arco científico (por que chegamos aqui)

A primeira tentativa da Fase 2 ([`docs/11`](11_fase2_resultados.md)) foi fine-tunar um LLM
para **responder** sobre a regulação da ANTT — injetar *conhecimento*. Avaliada com rigor
(held-out, IC, juiz factual com referência), deu um **negativo honesto**: o FT muda o
estilo mas **não injeta fato** (correção factual não melhora; generaliza pior a normas
novas). Diagnóstico: *fine-tuning adapta padrão/formato/estilo; conhecimento factual é
trabalho de RAG*.

A conclusão certa não é "fine-tuning é ruim" — é **escolher a tarefa onde ele comprovadamente
ganha**: uma tarefa de **rótulo objetivo**. Daí este estudo: **NER jurídico**, onde o modelo
aprende a *extrair entidades* — exatamente o tipo de padrão que o fine-tuning domina — e a
métrica é F1, comparável a um SOTA publicado.

## 2. Dados e método

- **Dataset: LeNER-Br** (Luz de Araujo et al., PROPOR 2018; **MIT**) — NER em textos jurídicos
  brasileiros: **7.827 / 1.176 / 1.389** sentenças (treino/dev/teste), 6 entidades
  (ORGANIZACAO, PESSOA, TEMPO, LOCAL, **LEGISLACAO**, **JURISPRUDENCIA**). Baixado por
  `rodoia.ner.lener` (CoNLL). Baseline publicado ~F1 92%.
- **Métrica:** **F1-micro de entidade** — uma entidade conta como acerto só se o par
  (texto, tipo) bate exatamente com o gold. Objetiva, sem juiz.
- **Três abordagens comparadas** (`reports/fase2_ner/comparacao.json`):
  1. **Base zero-shot** — Qwen2.5-3B-Instruct sem treino, instruído a extrair entidades em JSON.
  2. **FT QLoRA (generativo)** — o mesmo Qwen, fine-tunado (QLoRA, 1.500 sentenças) para
     produzir as entidades; servido em **fp8/vLLM** (reusa a infra da fase). `avaliar_generativo`
     parseia o JSON e calcula o F1.
  3. **BERTimbau (SOTA)** — `neuralmind/bert-base-portuguese-cased` com cabeça de
     token-classification (7.827 sentenças), o padrão-ouro para NER (seqeval).

## 3. Resultado — o fine-tuning vence, e com folga

| Modelo | Abordagem | Treino | **F1-micro (teste)** |
|---|---|---|---|
| Base zero-shot | LLM sem treino | 0 | **0,131** |
| **FT QLoRA** | LLM generativo | 1.500 | **0,774** |
| BERTimbau | encoder (SOTA) | 7.827 | **0,895** |

**Ganho do fine-tuning: F1 0,131 → 0,774 (+0,64, ~6×).** Com métrica objetiva, o FT tira o
modelo de "inútil" para "forte", **encostando no SOTA** apesar de treinar em **1/5 dos dados**
e por via **generativa** (mais difícil que token-classification).

F1 por entidade (FT QLoRA):

| Entidade | Base | **FT** | BERTimbau |
|---|---|---|---|
| PESSOA | 0,17 | **0,86** | 0,93 |
| LEGISLACAO | 0,01 | **0,82** | 0,95 |
| TEMPO | 0,09 | **0,82** | 0,94 |
| JURISPRUDENCIA | 0,02 | **0,74** | 0,85 |
| ORGANIZACAO | 0,22 | **0,71** | 0,86 |
| LOCAL | 0,10 | **0,60** | 0,68 |

## 4. Leitura honesta

- **É a vitória de fine-tuning que faltava** — objetiva, benchmarkada, com antes/depois claro.
- **Não bate o SOTA** (0,77 vs 0,89): esperado — o encoder é o padrão para NER e treinou em
  5× mais dados; o valor aqui é provar que **o mesmo LLM próprio, na mesma infra de serving**,
  aprende a tarefa. Treinar no dataset completo e mais épocas fecharia parte do gap
  (próximo passo barato).
- **LOCAL é o mais fraco** nas três abordagens (menos exemplos, 611 no treino) — coerente.
- **Complementaridade:** BERTimbau mostra domínio do caminho clássico; o LLM generativo mostra
  que a infra QLoRA/vLLM da fase serve para tarefas de extração — e ainda produz JSON pronto
  para o agente (Fase 4).

## 5. Reproduzir

```bash
python -m rodoia.ner.bertimbau --epocas 3                         # SOTA -> reports/fase2_ner/bertimbau.json
python -m rodoia.ner.generativo                                   # gera ner_train/test.jsonl
head -n 1500 data/processed/ner_train.jsonl > data/processed/ner_train_sub.jsonl
python -m rodoia.ft.treino_qlora --dataset data/processed/ner_train_sub.jsonl \
    --saida models/qlora-ner --epocas 2
CUDA_VISIBLE_DEVICES="" python -m rodoia.ft.merge_quantiza --adaptador models/qlora-ner \
    --merged models/antt-ner-merged
python -m rodoia.ner.avaliar_generativo Qwen/Qwen2.5-3B-Instruct reports/fase2_ner/generativo_base.json base
python -m rodoia.ner.avaliar_generativo models/antt-ner-merged   reports/fase2_ner/generativo_ft.json  ft
```

## 6. Critérios de conclusão (todos ✓)

- [x] Tarefa de **rótulo objetivo** com baseline público (LeNER-Br, licença MIT documentada)
- [x] **Fine-tuning com vitória mensurável**: F1 0,131 → **0,774** (+0,64) vs. base zero-shot
- [x] **Referência SOTA** (BERTimbau F1 0,895) contextualizando o resultado
- [x] Métrica objetiva por entidade (seqeval / match exato), sem LLM-juiz
- [x] Reusa a infra da fase (QLoRA + fp8 + vLLM); reports carimbados com proveniência
- [x] Funções puras testadas (`entidades_bio`, `metricas_ner`, `parse_entidades`)

### Próximo passo (opcional)
Treinar no LeNER-Br completo (7.827) por mais épocas para reduzir o gap ao SOTA; e usar o
extrator como **ferramenta do agente** (Fase 4).
