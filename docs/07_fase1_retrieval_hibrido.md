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

**25 perguntas de intenção real** (como um usuário pergunta, não paráfrase do título —
para evitar circularidade), várias com múltiplas fontes válidas. Métricas com **IC 95%**
(dado o n ainda pequeno): **hit@5** (nome honesto — como o gold é único por pergunta, é
*hit-rate*, não *recall* verdadeiro) por IC de **Wilson**; **MRR** por **bootstrap**.

| Modo | hit@5 | IC95 (hit) | MRR | IC95 (MRR) |
|---|---|---|---|---|
| denso (só embeddings) | 0,72 | [0,52; 0,86] | 0,540 | [0,38; 0,70] |
| bm25 (só léxico) | 0,68 | [0,48; 0,83] | 0,535 | [0,36; 0,70] |
| **híbrido (RRF)** | **0,72** | [0,52; 0,86] | **0,620** | [0,46; 0,78] |
| híbrido + rerank | 0,72 | [0,52; 0,86] | 0,543 | [0,39; 0,70] |

## Leitura dos números (honesta)

- Com perguntas **realistas** (não paráfrases do título), o hit@5 fica em **0,72** — bem
  abaixo dos 0,90 do conjunto antigo (n=10, circular), que superestimava.
- **O híbrido lidera no MRR** (0,620 vs. 0,540 do denso) — fundir semântico + léxico ainda
  ajuda a achar a fonte *mais cedo*. Mas os **ICs se sobrepõem**: com n=25 não dá para
  cravar superioridade estatística — reportado como tendência, não vitória "clara".
- **O rerank NÃO ajudou aqui** (MRR 0,543 vs. híbrido 0,620) — chegou a piorar. A
  avaliação circular anterior escondia isso. Fica no pipeline como camada pronta, mas a
  decisão de mantê-lo ligado deve ser revalidada quando o corpus/golden crescer.

Lição de rigor: **ampliar e des-enviesar o conjunto dourado + reportar IC** transformou
um "antes/depois" otimista num retrato honesto (e mostrou que o rerank não se justifica
neste corpus). Próximo passo: golden ≥50 e por terceiro, para estreitar os ICs.

## Próximo incremento

Geração da resposta com **citação da fonte** a partir dos chunks recuperados —
precisa de um LLM (local-first: Ollama/llama.cpp). Depois: guardrails + PII masking
e a interface FastAPI.
