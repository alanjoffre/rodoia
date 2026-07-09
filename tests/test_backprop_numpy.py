"""Testes do backprop manual: gradient checking, aprendizado e equivalência com
o autograd do PyTorch. Se a derivação (regra da cadeia) estivesse errada, estes
testes falhariam."""

from __future__ import annotations

import numpy as np
import torch

from rodoia.fundamentos.backprop_numpy import MLPNumpy, bce_loss, sigmoid


def _dados(n: int = 64, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


def test_gradient_check_bate_com_numerico() -> None:
    """Gradiente analítico ≈ gradiente numérico (erro relativo ~1e-7)."""
    from rodoia.fundamentos.backprop_numpy import gradient_check

    X, y = _dados()
    modelo = MLPNumpy(d_entrada=X.shape[1], d_oculta=8)
    erros = gradient_check(modelo, X, y)
    for nome, err in erros.items():
        assert err < 1e-6, f"{nome}: erro relativo {err:.2e} alto demais"


def test_treino_reduz_perda() -> None:
    X, y = _dados()
    modelo = MLPNumpy(d_entrada=X.shape[1], d_oculta=8)
    historico = modelo.treinar(X, y, epocas=300, lr=0.5)
    assert historico[-1] < historico[0]  # aprendeu algo
    assert historico[-1] < 0.4  # convergiu para perda baixa


def test_equivalencia_com_autograd_pytorch() -> None:
    """Nossos gradientes manuais == os do PyTorch, para os MESMOS pesos."""
    X, y = _dados()
    modelo = MLPNumpy(d_entrada=X.shape[1], d_oculta=8)

    # Gradientes analíticos (nossos).
    grads_np = modelo.backward(modelo.forward(X), y)

    # Réplica em PyTorch com os mesmos parâmetros e mesmo forward.
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)
    W1 = torch.tensor(modelo.W1, requires_grad=True)
    b1 = torch.tensor(modelo.b1, requires_grad=True)
    W2 = torch.tensor(modelo.W2, requires_grad=True)
    b2 = torch.tensor(modelo.b2, requires_grad=True)

    a1 = torch.relu(Xt @ W1 + b1)
    p = torch.sigmoid(a1 @ W2 + b2)
    perda = torch.nn.functional.binary_cross_entropy(p, yt)  # média, como o nosso
    perda.backward()

    for nome, tensor in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
        assert np.allclose(
            grads_np[nome].reshape(tensor.shape), tensor.grad.numpy(), atol=1e-8
        ), nome


def test_sigmoid_estavel_em_valores_extremos() -> None:
    z = np.array([-1000.0, 0.0, 1000.0])
    p = sigmoid(z)
    assert np.isfinite(p).all()
    assert p[0] == 0.0 and p[2] == 1.0

    # sanidade da perda
    assert bce_loss(np.array([0.9]), np.array([1.0])) < bce_loss(np.array([0.1]), np.array([1.0]))
