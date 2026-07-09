"""Testes da MLP em PyTorch (leves; o treino real é validado por execução)."""

from __future__ import annotations

import torch

from rodoia.ml.mlp_torch import MLP, dispositivo


def test_forward_tem_forma_certa() -> None:
    modelo = MLP(d_entrada=10)
    x = torch.randn(32, 10)
    saida = modelo(x)
    assert saida.shape == (32,)  # 1 logit por exemplo


def test_dispositivo_valido() -> None:
    dev = dispositivo()
    assert dev.type in {"mps", "cuda", "cpu"}


def test_um_passo_reduz_perda() -> None:
    """Sanidade do ciclo de otimização num problema sintético separável."""
    torch.manual_seed(0)
    X = torch.randn(256, 6)
    y = (X[:, 0] + X[:, 1] > 0).float()
    modelo = MLP(6, ocultas=(16,), p_dropout=0.0)
    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-2)

    perda0 = crit(modelo(X), y).item()
    for _ in range(100):
        opt.zero_grad()
        loss = crit(modelo(X), y)
        loss.backward()
        opt.step()
    assert loss.item() < perda0  # aprendeu
