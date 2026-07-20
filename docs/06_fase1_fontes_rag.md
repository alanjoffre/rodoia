# 06 — Fase 1: fonte e ingestão do corpus regulatório (RAG)

> Incremento 1 da Fase 1. Reality-check da fonte textual + pipeline de ingestão e
> chunking das resoluções da ANTT. Módulos: `rag/fontes_antt.py`, `rag/baixar_normas.py`,
> `rag/chunking.py`.

## A decisão de fonte (reality-check)

O `PROMPT_MESTRE` manda a arquitetura se adaptar ao que existe. A validação em
campo do ANTTlegis (portal legado "datalegis") revelou:

- **A busca não é scriptável** por HTTP: `consultarAtos` dá **HTTP 500**;
  `abrirEmentarioANTT`/`consultarAtosInicial` redirecionam para a "casca" do portal.
- **Mas há dois endpoints públicos e STATELESS** (sem cookie/sessão) que resolvem:
  - **Texto** — `UrlPublicasAction.php?acao=abrirAtoPublico&sgl_tipo=RES&num_ato=...&seq_ato=000&vlr_ano=...&sgl_orgao=...` devolve o corpo integral da norma (HTML → texto digital, **sem OCR**).
  - **Enumeração** — páginas temáticas `TematicaAction.php?acao=abrirVinculos&cotematica=...` embutem `LinkTexto('RES','00000059','000','2002','DG/ANTT/MT')`, de onde extraímos a tupla exata de cada ato.

Fontes alternativas descartadas com evidência: **LexML** (bloqueado por proof-of-work),
**DOU/in.gov.br** (bom para publicações novas, fraco para acervo histórico),
**Querido Diário** (só diários municipais). Detalhe em [docs/00](00_validacao_fontes_antt.md).

### Armadilhas confirmadas (e como o código as trata)

- **`sgl_orgao` varia por ano** (ministério supervisor: MT/MTPA/MI). Solução: usar
  sempre o órgão que veio do `LinkTexto`, **nunca deduzir**.
- **Casca vs. texto real:** tupla errada devolve a casca do portal (~15 KB, sem
  cabeçalho). Detector: cabeçalho `RESOLUÇÃO Nº` **e** texto > 18 KB (`texto_valido`).
- **Vigência inline:** o cabeçalho marca "Revogada pela Resolução X". Distinguimos
  isso (ato morto) de "Revoga a Resolução X" (ato ativo) por regex — `esta_vigente`.
- **Servidor lento:** download educado, com delay entre requisições.

**Licença:** atos oficiais são domínio público (Lei 9.610/98, art. 8º IV) — livre
para repo público. Registrado no `manifesto.json` e no `NOTICE`.

## Pipeline de ingestão

```
páginas temáticas (abrirVinculos) ──regex LinkTexto──> tuplas (num, seq, ano, órgão)
        │  [dedupe por id, ordena por recência]
        ▼
abrirAtoPublico por ato ──limpar_html + detectar vigência──> data/raw/normas/normas.jsonl
        │  (1 linha/norma: número, ano, órgão, título, vigente, texto)   [DVC]
        ▼
chunking consciente da estrutura jurídica ──> chunks com metadados p/ citação
```

## Estratégia de chunking (justificada)

Chunk cego de N caracteres quebra regras no meio. Nossa estratégia (`rag/chunking.py`):

1. **Divide por artigo** (`Art. Nº`) — a unidade semântica da norma.
2. **Empacota** artigos consecutivos até um alvo de tamanho (`max_chars`, padrão 1500).
3. Artigo grande demais → **janela deslizante com sobreposição** (`overlap`, 200),
   para não perder contexto na fronteira.

Complexidade **O(n)** no tamanho do texto. Cada chunk carrega `numero/ano/órgão/
vigência` para a **citação da fonte** na resposta (requisito da Fase 1).

## Corpus inicial (baixado)

> ⏱️ **Registro do 1º incremento.** Os números abaixo são do corpus **inicial** (45 resoluções / 3.432 chunks). O corpus foi depois **expandido para 125 normas / 3.647 chunks** (valor atual, canônico em [`reports/fase1_rag/corpus.json`](../reports/fase1_rag/corpus.json) e no [DATASET_CARD](DATASET_CARD.md)). Mantido como histórico da decisão de partida.

- **Temas:** normas de cargas, produtos perigosos, carga lotação (transporte rodoviário).
- **130 atos únicos enumerados** → **45 resoluções** com texto integral (2004–2024),
  todas vigentes. **4 milhões de caracteres**; mediana 35 KB/norma; maior norma 1,7 MB
  (regulamento consolidado de produtos perigosos).
- **Chunking:** 3.432 chunks (média 1.283 chars) — a base de retrieval.

### Sobre a taxa de "casca" (85 de 130)

As falhas concentram-se em **atos antigos (2002–2004)** que não resolvem no
endpoint público `abrirAtoPublico` (a ANTT foi criada em 2001; atos muito antigos
usam códigos de órgão diferentes ou não estão no sistema de texto digital). Como o
alvo do RAG é a **regulação vigente** — que tende a ser recente —, o corpus foca no
conjunto resolvível e atual. Decisão honesta, documentada, não silenciosa: o
`manifesto.json` registra `n_casca_ou_erro`. O corpus pode crescer adicionando
mais temas (passageiros, infraestrutura) em incrementos futuros.

## Reproduzir

```bash
python -m rodoia.rag.baixar_normas --limite 250   # -> data/raw/normas/normas.jsonl
# (o chunking é aplicado na etapa de indexação — próximo incremento)
```
