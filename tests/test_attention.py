"""Testes da self-attention à mão. O central: equivalência numérica com a
implementação de referência do PyTorch (`F.scaled_dot_product_attention`)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rodoia.fundamentos.attention import (
    AtencaoMultiCabeca,
    AutoAtencao,
    mascara_causal,
    scaled_dot_product_attention,
)


def _qkv(b=2, t=5, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    forma = (b, t, d)
    return (torch.randn(forma, generator=g, dtype=torch.float64) for _ in range(3))


def test_equivalencia_com_referencia_pytorch() -> None:
    """Nossa atenção == F.scaled_dot_product_attention (sem máscara)."""
    q, k, v = _qkv()
    nossa, _ = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(nossa, ref, atol=1e-9)


def test_equivalencia_causal() -> None:
    """Com máscara causal, também bate com a referência (is_causal=True)."""
    q, k, v = _qkv()
    t = q.size(1)
    nossa, _ = scaled_dot_product_attention(q, k, v, mascara=mascara_causal(t))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(nossa, ref, atol=1e-9)


def test_pesos_somam_um() -> None:
    """Cada linha da matriz de atenção é uma distribuição (soma 1)."""
    q, k, v = _qkv()
    _, pesos = scaled_dot_product_attention(q, k, v)
    somas = pesos.sum(dim=-1)
    assert torch.allclose(somas, torch.ones_like(somas), atol=1e-9)


def test_mascara_causal_bloqueia_futuro() -> None:
    """Posição i não pode atender a j > i (peso zero acima da diagonal)."""
    q, k, v = _qkv()
    t = q.size(1)
    _, pesos = scaled_dot_product_attention(q, k, v, mascara=mascara_causal(t))
    triangulo_superior = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    assert (pesos[..., triangulo_superior] == 0).all()


def test_auto_atencao_preserva_forma() -> None:
    x = torch.randn(2, 5, 16)
    saida = AutoAtencao(d_modelo=16)(x)
    assert saida.shape == x.shape


def test_multicabeca_preserva_forma_e_roda() -> None:
    x = torch.randn(2, 7, 32)
    saida = AtencaoMultiCabeca(d_modelo=32, n_cabecas=4)(x)
    assert saida.shape == x.shape
