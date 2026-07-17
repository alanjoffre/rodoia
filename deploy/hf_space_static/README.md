---
title: RodoIA — RAG ANTT
emoji: 🛣️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
---

# Demo pública GRÁTIS — RAG sobre a regulação da ANTT (roda no navegador)

O HuggingFace passou a cobrar por Spaces **Gradio/Docker** (compute); **Static** continua **grátis**.
Esta demo é **client-side**: a busca semântica roda **no navegador do visitante** (embeddings E5 via
`@xenova/transformers` em ONNX/WASM), comparando a pergunta contra os trechos das normas
pré-computados. **Retrieval-only** com citação da resolução-fonte — sem servidor, sem custo.

## Passo a passo (uma vez, na sua conta HuggingFace)

1. **Gere os assets** (embeddings do corpus) no seu clone do RodoIA:
   ```bash
   python deploy/hf_space_static/exportar.py      # -> dados.f32 + dados.json (neste diretório)
   ```
   (precisa do corpus: rode `python -m rodoia.rag.baixar_normas` antes, se ainda não tiver.)

2. **Crie o Space:** https://huggingface.co/new-space → nome `rodoia-rag` → **SDK: Static** → Create.

3. **Copie 4 arquivos** para o repo do Space:
   ```bash
   git clone https://huggingface.co/spaces/<seu-usuario>/rodoia-rag && cd rodoia-rag
   cp /d/Dev/rodoia/deploy/hf_space_static/{index.html,README.md,dados.f32,dados.json} .
   ```

4. **Suba:**
   ```bash
   git add . && git commit -m "deploy: RodoIA RAG demo (static, client-side)" && git push
   ```

5. Em segundos o Space fica **"Running"** em `https://huggingface.co/spaces/<seu-usuario>/rodoia-rag`.
   No 1º acesso o navegador baixa o modelo E5 (~30 MB, cacheado depois); as buscas seguintes são
   instantâneas. **URL pública + HTTPS, R$0.**

## Notas honestas
- **Client-side, retrieval-only:** a geração da resposta (com citação ancorada) roda no serving
  completo do projeto (FastAPI + Ollama/vLLM), não aqui. Esta demo prova a **recuperação** viva.
- **Só as normas VIGENTES** (93 das 125 do corpus → 2.994 dos 3.651 chunks): `exportar.py` roda com
  `apenas_vigentes=True`. Buscar norma revogada numa demo pública seria ruído, não recurso. O número
  exibido no topo da página é contado do próprio `dados.json`, nunca escrito à mão.
- Os embeddings dos chunks são gerados com o **mesmo E5** do backend (prefixo `passage`, mean-pool,
  normalização L2) e o navegador embute a pergunta com `query:` + o mesmo modelo — mesmo espaço
  vetorial. **Ressalva honesta:** o Transformers.js baixa o ONNX **quantizado (int8)** por padrão,
  enquanto as passagens foram embutidas em **fp32**. É uma assimetria real — desloca os scores na
  3ª casa e não muda o ranking na prática, mas não é "idêntico". Eliminá-la custaria 4× no download
  (`{ quantized: false }`) para um ganho que a demo não precisa.
- `dados.f32`/`dados.json` são **gerados** (não versionados no repo principal); só existem no Space.
  Os dois saem juntos do `exportar.py` e a página **recusa subir** se dessincronizarem
  (`byteLength !== n × dim × 4`) — sem essa checagem, o score viraria `NaN` silenciosamente.
