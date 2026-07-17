# 07 — Fase 1: retrieval híbrido, reranking e avaliação

> Incremento 3 da Fase 1. Módulos: `rag/recuperador.py`, `rag/avaliacao_retrieval.py`.
> Reproduzir: `python -m rodoia.rag.avaliacao_retrieval`.

## O problema com busca só semântica

Embeddings captam **significado** (bom para paráfrase), mas erram **termos exatos**
— número de resolução, siglas como RNTRC, jargão jurídico literal. BM25 (busca
léxica) é o oposto: acha a palavra exata, ignora sinônimos. Um RAG sério combina
os dois.

## A arquitetura

1. **Busca densa** — embeddings E5 no Qdrant (semântica).
2. **BM25** — índice léxico sobre os mesmos chunks.
3. **Reciprocal Rank Fusion (RRF)** — funde os dois rankings por
   `Σ 1/(k + posição)`. Usa a **posição**, não o score bruto (as escalas de cosseno
   e BM25 são incomparáveis) — por isso é robusto.
4. **Reranker cross-encoder** (`mmarco-mMiniLMv2`, multilíngue) — reordena os
   finalistas lendo *consulta + trecho juntos*. Mais preciso e mais caro, então só
   roda nos candidatos já filtrados.

## Avaliação (conjunto dourado)

**50 perguntas de intenção real** (como um usuário pergunta, não paráfrase do título —
para evitar circularidade), várias com múltiplas fontes válidas. Métricas com **IC 95%**:
**hit@5** (nome honesto — como o gold é único por pergunta, é *hit-rate*, não *recall*
verdadeiro) por IC de **Wilson**; **MRR** por **bootstrap**.

| Modo | hit@5 | IC95 (hit) | MRR | IC95 (MRR) |
|---|---|---|---|---|
| denso (só embeddings) | 0,60 | [0,46; 0,72] | 0,413 | [0,30; 0,53] |
| bm25 (só léxico) | 0,54 | [0,40; 0,67] | 0,399 | [0,28; 0,52] |
| **híbrido (RRF)** | **0,62** | [0,48; 0,74] | **0,510** | [0,39; 0,63] |
| híbrido + rerank | **0,68** | [0,54; 0,79] | 0,424 | [0,33; 0,52] |

> Números do corpus **limpo** (após o de-boilerplate que cortou menu/cabeçalho de navegação
> → 4.100 → **3.647 chunks** de conteúdo real). Um retrato anterior (corpus com boilerplate)
> dava hit@5 ~0,64 e o rerank *não* ajudava — a limpeza **reverteu** essa conclusão; ver abaixo.
>
> **2ª passada no de-boilerplate (2026-07-17).** Uma auditoria achou o que a 1ª deixou passar: o
> **rodapé** do portal (`Carregando... Voltar ao Topo`) escapava, porque `_e_boilerplate` exige 2+
> sinais de navegação e o rodapé tem só um. Resultado: **133 chunks (3,6%)** carregavam a
> navegação grudada no fim e **2 eram só isso** — recuperáveis numa busca. O corte agora é
> simétrico (cabeça e rodapé, no texto inteiro, antes de fatiar): **3.651 → 3.647 chunks**.
> **O hit@5 não mudou em nenhum modo** (híbrido 0,62; rerank 0,68) — o lixo era 0,08% dos
> caracteres. Só o MRR do denso oscilou 0,413→0,411, dentro do ruído. Limpeza sem custo de
> métrica: reportado assim porque medir e não mudar nada também é resultado.

## Leitura dos números (honesta)

- Com perguntas **realistas** e n=50, o hit@5 do híbrido fica em **0,62** [0,48; 0,74] — longe
  dos 0,90 do conjunto antigo (n=10, circular), que superestimava. Os ICs são largos de propósito
  (n=50): a faixa é honesta, um número solto enganaria.
- **O híbrido lidera o MRR** (0,510 vs. denso 0,413) e **bate o BM25 com folga** (0,399);
  fundir semântico + léxico acha a fonte mais cedo. Denso e híbrido têm ICs sobrepostos — o ganho
  do RRF sobre o denso é modesto, mas consistente no MRR.
- **REVERSÃO que o rigor obrigou — o rerank agora AJUDA.** No corpus com boilerplate o rerank não
  se justificava; removido o ruído, o cross-encoder passou a distinguir melhor e **dá o maior hit@5
  (0,62 → 0,68)**. Nuance honesta: ele **sobe o hit@5 mas baixa o MRR** (0,424 < 0,510) — puxa mais
  fontes *para dentro* do top-5, sem necessariamente levá-las ao top-1. Como os ICs ainda se
  sobrepõem (n=50), **servimos o híbrido** (o gate trava `hibrido` = 0,62) e mantemos o rerank como
  **camada opcional com ganho de ponto-estimado no hit@5** (`rerank=`), ligável quando a latência couber.

Lição de rigor: a **limpeza de dados mudou a conclusão de arquitetura** — o mesmo padrão do projeto
inteiro (medir → achar defeito → atualizar em vez de esconder). Detalhes e a decisão de serving em
[docs/09](09_fase1_api.md).

## Próximo incremento

Geração da resposta com **citação da fonte** a partir dos chunks recuperados —
precisa de um LLM (local-first: Ollama/llama.cpp). Depois: guardrails + PII masking
e a interface FastAPI.
