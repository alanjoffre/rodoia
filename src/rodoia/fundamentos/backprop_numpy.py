"""Backpropagation implementada À MÃO, em NumPy puro — prova de fundamento.

Uma rede de 2 camadas para classificação binária:

    X ──(W1,b1)──> z1 ──ReLU──> a1 ──(W2,b2)──> z2 ──sigmoid──> p ≈ P(y=1)

O objetivo NÃO é performance — é demonstrar que entendo o que `loss.backward()`
faz por dentro. Todos os gradientes são derivados na mão pela **regra da cadeia**
e depois validados por **gradient checking** (comparação com o gradiente numérico
por diferenças finitas). Se a derivação estivesse errada, o teste falharia.

Convenções: X tem forma (N, d) — N exemplos, d features. Perda = binary
cross-entropy média.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid numericamente estável: aplica cada fórmula só na metade adequada,
    evitando `exp()` de número grande positivo (que estouraria)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))  # z≥0: exp(-z) ≤ 1
    ez = np.exp(z[~pos])  # z<0: exp(z) < 1
    out[~pos] = ez / (1.0 + ez)
    return out


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def bce_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy média. É a perda 'natural' da classificação binária
    porque é a log-verossimilhança negativa de uma Bernoulli (ver docs/04)."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class MLPNumpy:
    """MLP de 1 camada oculta, com forward, backward manual e passo de treino."""

    def __init__(self, d_entrada: int, d_oculta: int = 16, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Inicialização de He (adequada a ReLU): escala 1/sqrt(fan_in) evita que os
        # sinais exploda/desapareçam ao atravessar a camada.
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_entrada), (d_entrada, d_oculta))
        self.b1 = np.zeros(d_oculta)
        self.W2 = rng.normal(0, np.sqrt(2.0 / d_oculta), (d_oculta, 1))
        self.b2 = np.zeros(1)

    # ---- forward -----------------------------------------------------------
    def forward(self, X: np.ndarray) -> dict:
        z1 = X @ self.W1 + self.b1  # (N, h)  combinação linear 1
        a1 = relu(z1)  # (N, h)  não-linearidade
        z2 = a1 @ self.W2 + self.b2  # (N, 1)  combinação linear 2
        p = sigmoid(z2)  # (N, 1)  probabilidade
        return {"X": X, "z1": z1, "a1": a1, "z2": z2, "p": p}

    # ---- backward (a parte que prova o entendimento) -----------------------
    def backward(self, cache: dict, y: np.ndarray) -> dict:
        """Gradientes da perda em relação a cada parâmetro, pela regra da cadeia.

        Fato-chave (derivado no notebook docs/04): para sigmoid + BCE, a derivada
        da perda em relação ao logit z2 colapsa para (p - y)/N. É por isso que
        essa dupla é a escolha 'natural' — o gradiente fica limpo.
        """
        X, a1, z1, p = cache["X"], cache["a1"], cache["z1"], cache["p"]
        N = X.shape[0]
        y = y.reshape(-1, 1)

        dz2 = (p - y) / N  # (N,1)  ∂L/∂z2
        dW2 = a1.T @ dz2  # (h,1)  ∂L/∂W2 = a1ᵀ · dz2
        db2 = dz2.sum(axis=0)  # (1,)   ∂L/∂b2

        da1 = dz2 @ self.W2.T  # (N,h)  propaga o erro para a camada oculta
        dz1 = da1 * (z1 > 0)  # (N,h)  ∂ReLU/∂z1 = 1 se z1>0, senão 0
        dW1 = X.T @ dz1  # (d,h)  ∂L/∂W1
        db1 = dz1.sum(axis=0)  # (h,)   ∂L/∂b1

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ---- treino ------------------------------------------------------------
    def passo_treino(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1) -> float:
        """Um passo de gradiente descendente: forward → perda → backward → update."""
        cache = self.forward(X)
        perda = bce_loss(cache["p"], y.reshape(-1, 1))
        grads = self.backward(cache, y)
        for nome, g in grads.items():
            getattr(self, nome)[...] -= lr * g  # θ ← θ − lr · ∂L/∂θ
        return perda

    def treinar(self, X, y, epocas: int = 200, lr: float = 0.1) -> list[float]:
        return [self.passo_treino(X, y, lr) for _ in range(epocas)]

    def prever_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["p"].ravel()


def gradient_check(
    modelo: MLPNumpy, X: np.ndarray, y: np.ndarray, eps: float = 1e-6
) -> dict[str, float]:
    """Valida o backward comparando o gradiente ANALÍTICO (regra da cadeia) com o
    gradiente NUMÉRICO por diferenças finitas centrais:

        ∂L/∂θ ≈ (L(θ+eps) − L(θ−eps)) / (2·eps)

    Retorna o erro relativo por parâmetro — deve ser ~1e-7 (praticamente zero).
    Este é o teste que um framework NÃO faz por você: prova que a matemática está
    certa, não só que o código roda.
    """
    grads = modelo.backward(modelo.forward(X), y)
    erros = {}
    for nome in ("W1", "b1", "W2", "b2"):
        theta = getattr(modelo, nome)
        g_analitico = grads[nome]
        g_numerico = np.zeros_like(theta)
        it = np.nditer(theta, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            orig = theta[i]
            theta[i] = orig + eps
            l_mais = bce_loss(modelo.forward(X)["p"], y.reshape(-1, 1))
            theta[i] = orig - eps
            l_menos = bce_loss(modelo.forward(X)["p"], y.reshape(-1, 1))
            theta[i] = orig
            g_numerico[i] = (l_mais - l_menos) / (2 * eps)
            it.iternext()
        # Erro relativo normalizado (evita divisão por zero quando ambos ~0).
        num = np.linalg.norm(g_analitico - g_numerico)
        den = np.linalg.norm(g_analitico) + np.linalg.norm(g_numerico) + 1e-12
        erros[nome] = float(num / den)
    return erros
