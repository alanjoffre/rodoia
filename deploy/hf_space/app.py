"""Demo pública (HuggingFace Spaces, CPU free-tier) — RAG sobre a regulação da ANTT.

Reusa o pipeline do RodoIA (`rodoia.rag`): recuperação híbrida (denso E5 + BM25 + RRF)
sobre o corpus de normas e, se um `HF_TOKEN` estiver disponível, geração da resposta via
**HuggingFace Inference API** (sem precisar de GPU no Space). Sem token, a demo funciona em
modo **retrieval-only** (mostra os trechos citados) — leve e grátis.

Deploy: ver deploy/hf_space/README.md.
"""
from __future__ import annotations

import os

import gradio as gr

_estado: dict = {}


def _carregar():
    if "rec" not in _estado:
        from rodoia.rag.avaliacao_retrieval import carregar_recuperador
        # reranker desligado no free-tier (CPU) para latência; retrieval híbrido já é forte
        _estado["rec"] = carregar_recuperador(com_reranker=False)
    return _estado["rec"]


def _gerar_hosted(pergunta: str, contexto: str) -> str | None:
    """Geração opcional via Inference API (só se HF_TOKEN estiver setado no Space)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    from huggingface_hub import InferenceClient
    cli = InferenceClient(token=token)
    prompt = (f"Contexto (normas da ANTT):\n{contexto}\n\nPergunta: {pergunta}\n"
              "Responda em português, citando o número da resolução. Use só o contexto.")
    try:
        return cli.text_generation(prompt, model="Qwen/Qwen2.5-7B-Instruct",
                                   max_new_tokens=300, temperature=0.1)
    except Exception as e:
        return f"(geração indisponível: {type(e).__name__})"


def responder(pergunta: str) -> str:
    from rodoia.rag.seguranca import detectar_injection, mascarar_pii
    inj, _ = detectar_injection(pergunta or "")
    if inj:
        return "⚠️ Solicitação bloqueada (padrão suspeito de manipulação de instruções)."
    rec = _carregar()
    chunks = rec.buscar(pergunta, k=4, modo="hibrido")
    if not chunks:
        return "Não encontrei base nas normas disponíveis."
    fontes = list(dict.fromkeys(c["numero"] for c in chunks))
    contexto = "\n\n".join(f"[Resolução {c['numero']}] {c['texto'][:600]}" for c in chunks)
    resposta = _gerar_hosted(pergunta, contexto)
    cab = f"**Fontes:** {', '.join('Resolução ' + f for f in fontes)}\n\n"
    if resposta:
        return cab + mascarar_pii(resposta)
    return cab + "**Trechos recuperados:**\n\n" + mascarar_pii(contexto)


demo = gr.Interface(
    fn=responder,
    inputs=gr.Textbox(label="Pergunte sobre a regulação da ANTT",
                      placeholder="Ex.: Como funciona o vale-pedágio obrigatório?"),
    outputs=gr.Markdown(label="Resposta"),
    title="RodoIA — RAG sobre a regulação da ANTT",
    description="Demo do RAG avaliado (Fase 1). Retrieval híbrido (E5+BM25+RRF) com citação de "
                "fonte. Geração via HuggingFace Inference API quando HF_TOKEN está configurado.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
