"""Self-attention (scaled dot-product) implementada À MÃO em PyTorch puro —
o mecanismo central do Transformer, base de todos os LLMs das Fases 1–4.

Ideia: cada posição da sequência gera três vetores — Query (o que procuro),
Key (o que ofereço) e Value (o que entrego). A atenção compara cada Query com
todas as Keys, transforma as similaridades em pesos (softmax) e retorna a soma
ponderada dos Values. Fórmula:

    Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

O `√d_k` estabiliza os gradientes: sem ele, os produtos internos crescem com a
dimensão e jogam o softmax para regiões saturadas (gradiente ~0).

Prova de fundamento: um teste confirma que esta implementação bate, numericamente,
com `torch.nn.functional.scaled_dot_product_attention` (a referência do framework).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def mascara_causal(n: int, device=None) -> torch.Tensor:
    """Máscara triangular inferior (n, n): a posição i só vê i e o que veio antes.
    É o que impede um LLM de 'espiar o futuro' ao prever o próximo token."""
    return torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mascara: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Atenção scaled dot-product. Formas: (..., T, d_k) para q/k e (..., T, d_v) v.
    Retorna (saída, pesos_de_atenção). `mascara` True = permitido, False = bloqueado.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # similaridade Q·Kᵀ
    if mascara is not None:
        scores = scores.masked_fill(~mascara, float("-inf"))  # bloqueia posições
    pesos = torch.softmax(scores, dim=-1)  # vira distribuição (soma 1)
    return pesos @ v, pesos  # soma ponderada dos Values


class AutoAtencao(nn.Module):
    """Self-attention de uma cabeça: projeta a entrada em Q, K, V e aplica a atenção."""

    def __init__(self, d_modelo: int) -> None:
        super().__init__()
        self.Wq = nn.Linear(d_modelo, d_modelo, bias=False)
        self.Wk = nn.Linear(d_modelo, d_modelo, bias=False)
        self.Wv = nn.Linear(d_modelo, d_modelo, bias=False)

    def forward(self, x: torch.Tensor, mascara: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        saida, _ = scaled_dot_product_attention(q, k, v, mascara)
        return saida


class AtencaoMultiCabeca(nn.Module):
    """Multi-head attention à mão: várias atenções em paralelo, cada uma num
    subespaço, concatenadas e projetadas. Deixa o modelo atender a vários tipos
    de relação ao mesmo tempo."""

    def __init__(self, d_modelo: int, n_cabecas: int) -> None:
        super().__init__()
        if d_modelo % n_cabecas != 0:
            raise ValueError("d_modelo deve ser divisível por n_cabecas")
        self.n_cabecas = n_cabecas
        self.d_cabeca = d_modelo // n_cabecas
        self.Wq = nn.Linear(d_modelo, d_modelo, bias=False)
        self.Wk = nn.Linear(d_modelo, d_modelo, bias=False)
        self.Wv = nn.Linear(d_modelo, d_modelo, bias=False)
        self.Wo = nn.Linear(d_modelo, d_modelo, bias=False)

    def _dividir_cabecas(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_cabecas, self.d_cabeca).transpose(1, 2)  # (B,H,T,dh)

    def forward(self, x: torch.Tensor, mascara: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self._dividir_cabecas(self.Wq(x))
        k = self._dividir_cabecas(self.Wk(x))
        v = self._dividir_cabecas(self.Wv(x))
        saida, _ = scaled_dot_product_attention(q, k, v, mascara)  # (B,H,T,dh)
        saida = saida.transpose(1, 2).reshape(B, T, -1)  # reconcatena as cabeças
        return self.Wo(saida)
