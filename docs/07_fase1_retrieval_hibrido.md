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
| denso (só embeddings) | 0,66 | [0,52; 0,78] | 0,499 | [0,38; 0,62] |
| bm25 (só léxico) | 0,54 | [0,40; 0,67] | 0,406 | [0,29; 0,53] |
| **híbrido (RRF)** | 0,64 | [0,50; 0,76] | **0,513** | [0,39; 0,63] |
| híbrido + rerank | 0,64 | [0,50; 0,76] | 0,473 | [0,35; 0,59] |

## Leitura dos números (honesta)

- Com perguntas **realistas** e n=50, o hit@5 fica em **0,64–0,66** — longe dos 0,90 do
  conjunto antigo (n=10, circular), que superestimava. Os **ICs estreitaram** (n=50 vs. 25).
- **O híbrido lidera o MRR** (0,513 vs. denso 0,499) e **bate o BM25 com folga** (0,406);
  fundir semântico + léxico ajuda a achar a fonte mais cedo. Denso e híbrido têm ICs
  sobrepostos — o ganho do RRF sobre o denso é modesto neste corpus.
- **O rerank NÃO ajuda** (MRR 0,473 < híbrido 0,513) — confirmado agora com n=50; a
  avaliação circular anterior escondia isso. Decisão honesta: a evidência **recomenda
  desligá-lo** neste corpus; fica como camada opcional (`rerank=`), a revalidar se o
  corpus crescer.

Lição de rigor: **ampliar (n=50) e des-enviesar o conjunto dourado + reportar IC** trocou
um "antes/depois" otimista por um retrato honesto — e mostrou que o **rerank não se
justifica** neste corpus e que o BM25 puro é o mais fraco.

## Próximo incremento

Geração da resposta com **citação da fonte** a partir dos chunks recuperados —
precisa de um LLM (local-first: Ollama/llama.cpp). Depois: guardrails + PII masking
e a interface FastAPI.
