"""Geração da resposta do RAG: recupera trechos, monta um prompt ancorado e pede
ao LLM uma resposta COM CITAÇÃO da fonte.

Anti-alucinação (grounding): o prompt instrui o modelo a responder SÓ com base nos
trechos fornecidos e a dizer que não encontrou quando a resposta não estiver ali.
Cada trecho é numerado com a resolução de origem, para o modelo citar.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rodoia.rag.recuperador import RecuperadorHibrido
from rodoia.rag.seguranca import detectar_injection, mascarar_pii, registrar_auditoria

PROMPT_SISTEMA = (
    "Você é um assistente jurídico especializado na regulação da ANTT (transporte "
    "rodoviário). Responda em português, de forma objetiva, USANDO EXCLUSIVAMENTE os "
    "trechos de resoluções fornecidos no contexto. Cite a resolução (número/ano) que "
    "embasa cada afirmação, no formato (Resolução X/AAAA). Se a resposta não estiver "
    "nos trechos, diga claramente que não encontrou base nas normas fornecidas. "
    "NUNCA invente números de resolução, artigos ou regras."
)


def montar_contexto(chunks: list[dict]) -> str:
    """Formata os trechos recuperados como fontes numeradas para o prompt."""
    partes = []
    for i, c in enumerate(chunks, 1):
        vig = "vigente" if c.get("vigente") else "revogada"
        partes.append(f"[Fonte {i} — Resolução {c['numero']} ({vig})]\n{c['texto']}")
    return "\n\n".join(partes)


def montar_prompt(consulta: str, chunks: list[dict]) -> str:
    return (
        f"Contexto (trechos de resoluções da ANTT):\n\n{montar_contexto(chunks)}\n\n"
        f"Pergunta: {consulta}\n\n"
        "Responda com base apenas no contexto acima, citando as resoluções."
    )


def responder(
    consulta: str,
    recuperador: RecuperadorHibrido,
    llm,
    k: int = 5,
    apenas_vigentes: bool = True,
    rerank: bool = True,
) -> dict:
    """Executa o RAG ponta a ponta e devolve resposta + fontes citadas + trechos."""
    chunks = recuperador.buscar(
        consulta, k=k, modo="hibrido", rerank=rerank and recuperador.reranker is not None
    )
    if apenas_vigentes:
        chunks = [c for c in chunks if c.get("vigente")] or chunks
    if not chunks:
        return {
            "resposta": "Não encontrei base nas normas disponíveis.",
            "fontes": [],
            "chunks": [],
        }

    resposta = llm.gerar(montar_prompt(consulta, chunks), sistema=PROMPT_SISTEMA)
    fontes = list(dict.fromkeys(c["numero"] for c in chunks))  # únicas, na ordem
    return {"resposta": resposta, "fontes": fontes, "chunks": chunks}


def responder_seguro(
    consulta: str,
    recuperador: RecuperadorHibrido,
    llm,
    k: int = 5,
    auditoria: Path | None = None,
) -> dict:
    """RAG com a camada de segurança: bloqueia prompt injection, mascara PII na
    resposta e registra a consulta na trilha de auditoria."""
    injecao, motivo = detectar_injection(consulta)
    if injecao:
        resultado = {
            "resposta": "Sua solicitação foi bloqueada por conter um padrão suspeito de "
            "manipulação de instruções. Reformule a pergunta sobre a regulação da ANTT.",
            "fontes": [],
            "chunks": [],
            "bloqueado": True,
            "motivo": motivo,
        }
    else:
        r = responder(consulta, recuperador, llm, k=k)
        resultado = {
            "resposta": mascarar_pii(r["resposta"]),
            "fontes": r["fontes"],
            "chunks": r["chunks"],
            "bloqueado": False,
            "motivo": None,
        }

    if auditoria is not None:
        registrar_auditoria(
            {
                "ts": datetime.now(UTC).isoformat(),
                "consulta": mascarar_pii(consulta),
                "bloqueado": resultado["bloqueado"],
                "motivo": resultado["motivo"],
                "fontes": resultado["fontes"],
            },
            auditoria,
        )
    return resultado
