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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rodoia.config import REPO_ROOT

_estado: dict = {}
_AUDITORIA = REPO_ROOT / "logs" / "auditoria.jsonl"


def _carregar() -> None:
    """Carrega recuperador + LLM (sob demanda; reusado entre requisições)."""
    if "rec" not in _estado:
        from rodoia.rag.avaliacao_retrieval import carregar_recuperador
        from rodoia.rag.llm import OllamaLLM

        _estado["rec"] = carregar_recuperador(com_reranker=True)
        _estado["llm"] = OllamaLLM()


@asynccontextmanager
async def lifespan(app: FastAPI):
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
async def health() -> dict:
    return {"status": "ok", "pronto": "rec" in _estado}


@app.post("/perguntar", response_model=Resposta)
async def perguntar(p: Pergunta) -> Resposta:
    from rodoia.rag.gerar import responder_seguro

    _carregar()
    r = await asyncio.to_thread(
        responder_seguro, p.consulta, _estado["rec"], _estado["llm"], p.k, _AUDITORIA
    )
    return Resposta(resposta=r["resposta"], fontes=r["fontes"], bloqueado=r["bloqueado"])


class RespostaAgente(BaseModel):
    resposta: str
    fontes: list[str]
    rotas: list[str]
    trajetoria: list[dict]


def _carregar_agente() -> None:
    """Monta o agente (Fase 4) reusando o recuperador/LLM já carregados p/ o RAG."""
    if "agente_deps" not in _estado:
        _carregar()
        from rodoia.agente.ferramentas import (dados_real, entidades_real,
                                                regulatorio_real)
        from rodoia.agente.estado import DepsAgente
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
