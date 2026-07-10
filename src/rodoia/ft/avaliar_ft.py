"""Avaliação antes/depois do fine-tuning por juiz-com-referência (Fase 2).

>>> INATIVO nesta máquina: exige `data/raw/normas.jsonl` (DVC, sem remoto) como
referência factual + dois endpoints vLLM + Ollama. A avaliação ATIVA da fase é a
combinação `perplexidade.py` + `aval_cite.py` + `juiz_winrate.py` (ver docs/11).
Quando o corpus DVC estiver disponível, este script complementa com a nota factual;
grava em `avaliacao_ref_juiz.json` (NÃO no `avaliacao_ft.json` consolidado).

Compara o modelo BASE vs. o FINE-TUNADO nas mesmas perguntas de domínio, SEM RAG
(mede o que o modelo "sabe" após a adaptação). Um LLM juiz, que recebe o texto da
resolução de referência, pontua a correção de cada resposta (0-1). O ganho (ou a
ausência dele, reportada honestamente) é o resultado da fase.

A lógica é testável no Mac; a execução real usa dois endpoints vLLM na Nitro
(base e fine-tunado), via `OpenAICompatLLM`.
"""

from __future__ import annotations

import json
import re

from rodoia.config import REPO_ROOT, settings

_PROMPT_JUIZ = (
    "Avalie a CORREÇÃO da resposta em relação à pergunta, usando o TEXTO DE "
    "REFERÊNCIA (trecho da resolução correta) como verdade. Dê uma nota de 0.0 a "
    "1.0 (1.0 = correta e fundamentada; 0.0 = errada ou inventada). Responda APENAS "
    'com JSON: {{"nota": <0-1>}}\n\n'
    "PERGUNTA: {pergunta}\n\nREFERÊNCIA:\n{referencia}\n\nRESPOSTA:\n{resposta}"
)
_RE_JSON = re.compile(r"\{[^{}]*\}", re.S)


def carregar_referencias(max_chars: int = 4000) -> dict[str, str]:
    """Mapa fonte -> trecho do texto da norma, para o juiz usar como verdade."""
    refs = {}
    for linha in settings.normas_jsonl.open(encoding="utf-8"):
        r = json.loads(linha)
        refs[r["numero"]] = r["texto"][:max_chars]
    return refs


def _nota(saida: str) -> float:
    m = _RE_JSON.search(saida)
    if not m:
        return 0.0
    try:
        return float(json.loads(m.group(0)).get("nota", 0.0))
    except (ValueError, TypeError):
        return 0.0


def julgar_correcao(pergunta: str, resposta: str, referencia: str, juiz) -> float:
    prompt = _PROMPT_JUIZ.format(pergunta=pergunta, referencia=referencia, resposta=resposta)
    return _nota(juiz.gerar(prompt))


def comparar_modelos(
    base_llm, ft_llm, juiz, golden: list[dict], referencias: dict[str, str]
) -> dict:
    """Para cada caso: gera resposta do base e do ft, julga ambas contra a
    referência da fonte esperada. Retorna médias + casos."""
    casos = []
    for caso in golden:
        pergunta, fonte = caso["consulta"], caso["fontes"][0]
        ref = referencias.get(fonte, "")
        r_base = base_llm.gerar(
            pergunta, sistema="Responda sobre a regulação da ANTT, citando a resolução."
        )
        r_ft = ft_llm.gerar(
            pergunta, sistema="Responda sobre a regulação da ANTT, citando a resolução."
        )
        casos.append(
            {
                "pergunta": pergunta,
                "fonte": fonte,
                "nota_base": julgar_correcao(pergunta, r_base, ref, juiz),
                "nota_ft": julgar_correcao(pergunta, r_ft, ref, juiz),
            }
        )
    n = len(casos) or 1
    base_media = sum(c["nota_base"] for c in casos) / n
    ft_media = sum(c["nota_ft"] for c in casos) / n
    return {
        "base_media": base_media,
        "ft_media": ft_media,
        "ganho": ft_media - base_media,
        "n": len(casos),
        "casos": casos,
    }


def main() -> None:
    from rodoia.rag.avaliacao_retrieval import CONJUNTO_DOURADO
    from rodoia.rag.llm import OllamaLLM, OpenAICompatLLM

    # Base e fine-tunado servidos por vLLM (portas distintas) na Nitro; juiz local.
    base = OpenAICompatLLM("Qwen/Qwen2.5-3B-Instruct", base_url="http://localhost:8000/v1")
    ft = OpenAICompatLLM("antt-awq", base_url="http://localhost:8001/v1")
    juiz = OllamaLLM()  # ou outro OpenAICompatLLM

    res = comparar_modelos(base, ft, juiz, CONJUNTO_DOURADO, carregar_referencias())
    b, f, g = res["base_media"], res["ft_media"], res["ganho"]
    print(f"base: {b:.2f} | fine-tunado: {f:.2f} | ganho: {g:+.2f}")
    saida = REPO_ROOT / "reports" / "fase2_ft"
    saida.mkdir(parents=True, exist_ok=True)
    # NÃO grava em avaliacao_ft.json (esse é o report consolidado da avaliação ativa) —
    # este é o juiz-com-referência, INATIVO até `data/raw/normas.jsonl` (DVC) existir.
    (saida / "avaliacao_ref_juiz.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"relatório: {saida / 'avaliacao_ref_juiz.json'}")


if __name__ == "__main__":
    main()
