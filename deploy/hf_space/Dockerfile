# Alternativa ao SDK Gradio: Space com SDK **Docker** (também grátis no CPU basic).
# Use este arquivo se a criação do Space só te oferecer Docker/Static.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Instala o RodoIA (RAG) direto do repo público + Gradio.
RUN pip install --upgrade pip \
    && pip install "rodoia[rag] @ git+https://github.com/alanjoffre/rodoia.git" \
       gradio>=4.44 huggingface_hub>=0.25

COPY app.py .

# Se você subiu o corpus junto (data/raw/normas/normas.jsonl), ele é copiado; senão o app baixa
# sozinho no 1º boot.
COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
