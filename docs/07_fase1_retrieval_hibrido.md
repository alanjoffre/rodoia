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

10 perguntas naturais com a resolução-resposta conhecida (verificada nos títulos).
Métricas: **recall@5** (achou a fonte certa no top-5?) e **MRR** (achou cedo?).

| Modo | recall@5 | MRR |
|---|---|---|
| denso (só embeddings) | 0,80 | 0,625 |
| bm25 (só léxico) | 0,80 | 0,675 |
| **híbrido (RRF)** | **0,90** | 0,717 |
| **híbrido + rerank** | **0,90** | **0,725** |

## Leitura dos números (honesta)

- **O híbrido é a vitória clara:** recall **0,80 → 0,90** (+12,5%) e MRR **+15%**
  sobre o denso. Fundir semântico + léxico recupera casos que cada um sozinho perde.
- **O rerank dá ganho marginal** aqui (MRR 0,717 → 0,725). Esperado: com um conjunto
  dourado pequeno e candidatos já bons, sobra pouca margem. O reranker tende a
  brilhar em corpora maiores e consultas mais ambíguas — fica no pipeline como
  camada pronta, e a avaliação é o instrumento para revalidar quando o corpus crescer.

Este é o **antes/depois numérico** que o `PROMPT_MESTRE` exige: otimização (híbrido)
comprovada com métrica, não com "achismo".

## Próximo incremento

Geração da resposta com **citação da fonte** a partir dos chunks recuperados —
precisa de um LLM (local-first: Ollama/llama.cpp). Depois: guardrails + PII masking
e a interface FastAPI.
