# Imagem da API do RodoIA (RAG + agente da Fase 4). O "cérebro" (Ollama) e o modelo
# fine-tunado (vLLM/GPU) rodam como serviços à parte — ver docker-compose.yml.
# torch é instalado na variante CPU-only: a API só faz embeddings/rerank (leves);
# a inferência pesada do LLM fica nos serviços dedicados.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Camada de dependências (cacheável): copia só o necessário p/ resolver os extras.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
       -e ".[rag,agente,estruturados]"

EXPOSE 8080

# Prontidão: o /health responde quando o recuperador está carregado.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "rodoia.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
