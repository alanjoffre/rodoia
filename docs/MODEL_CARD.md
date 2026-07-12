# Model Card — `rodoia-ner-ft` (NER jurídico PT-BR)

Cartão de modelo no padrão de governança (Mitchell et al., 2019), para o modelo fine-tunado
da Fase 2 do RodoIA.

## Detalhes do modelo
- **Base:** `Qwen/Qwen2.5-3B-Instruct` (Apache-2.0).
- **Adaptação:** **QLoRA** (NF4 4-bit) para **NER jurídico** como extração generativa (JSON).
- **Serving:** merge do adaptador + **quantização fp8**, servido em **vLLM** (OpenAI-compat).
- **Nome servido:** `rodoia-ner-ft` · **Fase:** 2 · **Docs:** [docs/13](13_fase2_ner.md).

## Uso pretendido
- **Para quê:** extrair entidades jurídicas (`ORGANIZACAO`, `PESSOA`, `TEMPO`, `LOCAL`,
  `LEGISLACAO`, `JURISPRUDENCIA`) de textos jurídicos/regulatórios em português — e servir como a
  **ferramenta de entidades do agente** (Fase 4).
- **Fora de escopo:** decisão jurídica automatizada, extração de PII para uso identificatório,
  domínios fora do jurídico-administrativo brasileiro.

## Dados de treino
- **LeNER-Br** (MIT; decisões judiciais públicas). Split held-out por documento.
- Treino do NER generativo com **~1.500 sentenças** (1/5 do conjunto do BERTimbau) — ver
  [DATASET_CARD](DATASET_CARD.md).

## Avaliação
| Modelo | F1-micro de entidade (LeNER-Br teste, 1.389 sentenças) |
|---|---|
| Base zero-shot (Qwen2.5-3B) | **0,131** |
| **FT (QLoRA, este modelo)** | **0,774** |
| BERTimbau (SOTA, referência) | 0,895 |

Métrica: match exato (texto+tipo). Ganho FT vs base: **+0,64**. Report carimbado em
`reports/fase2_ner/comparacao.json`. Serving: ~205 tok/s (fp8), ver `benchmark_vllm.json`.

## Limitações e riscos
- **Não bate o SOTA** (encoder dedicado) — a proposta é demonstrar fine-tuning com métrica dura
  treinando em 1/5 dos dados por via generativa, não superar o BERTimbau.
- **Alucinação de formato:** pode emitir JSON inválido em entradas atípicas (o parser é tolerante).
- **Viés do domínio:** treinado em decisões judiciais; performance cai fora desse registro.
- **PII:** entidades `PESSOA` podem conter nomes reais de registros públicos — não usar para
  identificação/perfilamento.

## Reprodutibilidade
`seed=42`; hiperparâmetros versionados em `src/rodoia/ft/treino_qlora.py`; proveniência
(git_sha/versões) carimbada nos reports. Pipeline: `ner.generativo` → `ft.treino_qlora` →
`ft.merge_quantiza` → `ner.avaliar_generativo`.

## Licença
Pesos derivados de Qwen2.5 (Apache-2.0) + LeNER-Br (MIT). Ver `NOTICE`.
