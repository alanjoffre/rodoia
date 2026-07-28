"""API FastAPI (async) do RAG da ANTT — o endpoint que fecha a Fase 1.

Endpoints:
- GET  /            → UI mínima de demonstração (HTML).
- GET  /health      → prontidão.
- POST /perguntar   → {consulta, k} → resposta com fontes citadas.

O RAG (retrieval + LLM) é síncrono e pesado; para não travar o event loop, a
chamada roda num threadpool (`asyncio.to_thread`). Os componentes pesados
(embedder, índice, reranker) são carregados uma vez no startup (lifespan).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rodoia.config import REPO_ROOT
from rodoia.observabilidade import CacheLRU, registrar_metrica

_estado: dict[str, Any] = {}
_AUDITORIA = REPO_ROOT / "logs" / "auditoria.jsonl"
_METRICAS = REPO_ROOT / "logs" / "metricas.jsonl"      # observabilidade estruturada por requisição
# chave = (consulta normalizada, k); valor = o dict de `responder_seguro`
_CACHE: CacheLRU[tuple[str, int], dict[str, Any]] = CacheLRU(maxsize=256)   # corta o p95


def _carregar() -> None:
    """Carrega recuperador + LLM (sob demanda; reusado entre requisições)."""
    if "rec" not in _estado:
        from rodoia.rag.avaliacao_retrieval import carregar_recuperador
        from rodoia.rag.llm import OllamaLLM

        # rerank DESLIGADO por padrão: a avaliação (reports/fase1_retrieval) mostra que ele não
        # melhora hit@5 e piora o MRR — não justifica a latência. Ver docs/09.
        _estado["rec"] = carregar_recuperador(com_reranker=False)
        _estado["llm"] = OllamaLLM()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(_carregar)
    yield
    _estado.clear()


app = FastAPI(title="RodoIA — RAG sobre a regulação da ANTT", lifespan=lifespan)


class Pergunta(BaseModel):
    consulta: str
    k: int = 4


class Resposta(BaseModel):
    resposta: str
    fontes: list[str]
    bloqueado: bool


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "pronto": "rec" in _estado}


@app.post("/perguntar", response_model=Resposta)
async def perguntar(p: Pergunta) -> Resposta:
    from rodoia.rag.gerar import responder_seguro

    _carregar()
    chave = (p.consulta.strip().lower(), p.k)
    t0 = time.perf_counter()
    r = _CACHE.get(chave)
    cache_hit = r is not None
    if r is None:                    # miss: `r is None` (e não `not cache_hit`) p/ o mypy estreitar
        r = await asyncio.to_thread(
            responder_seguro, p.consulta, _estado["rec"], _estado["llm"], p.k, _AUDITORIA
        )
        _CACHE.set(chave, r)
    # observabilidade estruturada: latência, cache, resultado — uma linha JSON por requisição
    registrar_metrica({
        "endpoint": "perguntar", "latencia_s": round(time.perf_counter() - t0, 3),
        "cache_hit": cache_hit, "taxa_hit_cache": _CACHE.taxa_hit,
        "bloqueado": r["bloqueado"], "n_fontes": len(r["fontes"]),
    }, _METRICAS)
    return Resposta(resposta=r["resposta"], fontes=r["fontes"], bloqueado=r["bloqueado"])


@app.post("/perguntar/stream")
async def perguntar_stream(p: Pergunta) -> StreamingResponse:
    """Mesma resposta, em **Server-Sent Events** — um evento JSON por linha `data:`.

    **Por que existe.** O p95 da geração é ~30 s (docs/16 §6): sem streaming, isso é
    latência percebida inteira e a espera parece travamento. Com SSE o cliente recebe
    as **fontes antes do primeiro token** e o texto conforme sai — o
    time-to-first-token é o que o usuário sente. Também é o que remove o p95 da
    conta de timeout de qualquer plataforma (docs/16 §7.1).

    **Não passa pelo cache** de propósito: `CacheLRU` guarda a resposta pronta, e a
    razão de ser deste endpoint é justamente não esperar por ela. Quem quer a
    resposta cacheada usa `/perguntar`.
    """
    from rodoia.rag.gerar import responder_seguro_stream

    _carregar()
    t0 = time.perf_counter()

    def _fluxo() -> Iterator[str]:
        n_fontes, bloqueado = 0, False
        for evento in responder_seguro_stream(
            p.consulta, _estado["rec"], _estado["llm"], p.k, _AUDITORIA
        ):
            if evento["tipo"] == "fontes":
                n_fontes = len(evento["fontes"])
            elif evento["tipo"] == "bloqueio":
                bloqueado = True
            yield f"data: {json.dumps(evento, ensure_ascii=False)}\n\n"
        registrar_metrica({
            "endpoint": "perguntar_stream", "latencia_s": round(time.perf_counter() - t0, 3),
            "cache_hit": False, "taxa_hit_cache": _CACHE.taxa_hit,
            "bloqueado": bloqueado, "n_fontes": n_fontes,
        }, _METRICAS)

    async def _agregar() -> AsyncIterator[str]:
        # O RAG é síncrono e pesado; iterar direto travaria o event loop. Cada `next`
        # vai para o threadpool — o mesmo motivo pelo qual `/perguntar` usa `to_thread`.
        it = _fluxo()
        while True:
            pedaco = await asyncio.to_thread(next, it, None)
            if pedaco is None:
                return
            yield pedaco

    return StreamingResponse(
        _agregar(),
        media_type="text/event-stream",
        # `no-transform`/`X-Accel-Buffering`: proxies que bufferizam anulam o streaming
        # inteiro — a resposta chegaria de uma vez só, com a mesma latência de antes.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


class RespostaAgente(BaseModel):
    resposta: str
    fontes: list[str]
    rotas: list[str]
    trajetoria: list[dict[str, Any]]


def _carregar_agente() -> None:
    """Monta o agente (Fase 4) reusando o recuperador/LLM já carregados p/ o RAG."""
    if "agente_deps" not in _estado:
        _carregar()
        from rodoia.agente.estado import DepsAgente
        from rodoia.agente.ferramentas import dados_real, entidades_real, regulatorio_real
        from rodoia.rag.llm import OpenAICompatLLM

        cerebro = _estado["llm"]
        llm_ft = OpenAICompatLLM(modelo="rodoia-ner-ft", base_url="http://localhost:8001/v1")
        _estado["agente_deps"] = DepsAgente(
            llm_cerebro=cerebro,
            regulatorio=regulatorio_real(_estado["rec"], cerebro),
            entidades=entidades_real(llm_ft),
            dados=dados_real(),
        )


@app.post("/agente", response_model=RespostaAgente)
async def agente(p: Pergunta) -> RespostaAgente:
    """Agente orquestrado (Fase 4): roteia entre RAG + modelo FT + dados e sintetiza."""
    from rodoia.agente.grafo import responder as responder_agente

    _carregar_agente()
    r = await asyncio.to_thread(responder_agente, p.consulta, _estado["agente_deps"])
    return RespostaAgente(resposta=r["resposta"], fontes=r["fontes"],
                          rotas=r["rotas"], trajetoria=r["trajetoria"])


_HTML = """<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<title>RodoIA — RAG ANTT</title><style>
body{font-family:system-ui;max-width:760px;margin:40px auto;padding:0 16px;line-height:1.5}
textarea{width:100%;height:70px;padding:8px} button{padding:8px 16px;margin-top:8px}
#fontes{color:#666;font-size:.9em}
pre{white-space:pre-wrap;background:#f6f6f6;padding:12px;border-radius:6px}
</style></head><body>
<h2>RodoIA — regulação da ANTT</h2>
<p>Pergunte sobre resoluções de transporte rodoviário. As respostas citam a fonte.</p>
<textarea id="q" placeholder="Ex.: Como funciona o vale-pedágio obrigatório?"></textarea><br>
<button onclick="perguntar()">Perguntar</button>
<div id="out"></div>
<script>
async function perguntar(){
  const q=document.getElementById('q').value; const out=document.getElementById('out');
  out.innerHTML='Consultando...';
  const r=await fetch('/perguntar',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({consulta:q})});
  const d=await r.json();
  out.innerHTML='<pre>'+d.resposta+'</pre>'+(d.fontes.length?
    '<p id=fontes>Fontes: '+d.fontes.map(f=>'Resolução '+f).join(', ')+'</p>':'');
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return _HTML
