"""MLP em PyTorch para a classificação de severidade, com LAÇO DE TREINO MANUAL
(forward → loss → backward → step) rodando na GPU do Mac (MPS).

Contraponto ao `backprop_numpy.py`: lá provei o gradiente à mão; aqui uso o
autograd do framework, mas escrevendo o loop de treino explicitamente (nada de
`.fit()` mágico) — para deixar claro o ciclo de otimização. Compara-se ao baseline
sklearn (HistGB ROC-AUC 0,813) no mesmo split.

Desbalanceamento tratado via `pos_weight` no BCEWithLogitsLoss (equivalente ao
`class_weight='balanced'` do sklearn).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from rodoia.config import settings
from rodoia.ml.classico import carregar_dados, construir_preprocessador

_REPORT_DIR = settings.data_processed.parent.parent / "reports" / "fase0_mlp"


def dispositivo() -> torch.device:
    """MPS (GPU Metal) no Mac; senão CPU. CUDA seria usado na Nitro (Fase 2)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLP(nn.Module):
    """Perceptron multicamada: 2 camadas ocultas com ReLU e dropout."""

    def __init__(self, d_entrada: int, ocultas=(64, 16), p_dropout: float = 0.2) -> None:
        super().__init__()
        camadas: list[nn.Module] = []
        d = d_entrada
        for h in ocultas:
            camadas += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_dropout)]
            d = h
        camadas.append(nn.Linear(d, 1))  # logit (sem sigmoid: a loss cuida disso)
        self.rede = nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x).squeeze(1)


def _preparar_tensores(amostra: int | None):
    """Carrega, faz split, ajusta o pré-processador NO TREINO (sem vazar o teste)
    e devolve tensores float32."""
    X, y = carregar_dados(amostra=amostra)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=settings.seed
    )
    prep = construir_preprocessador().fit(X_tr)  # fit só no treino!

    def to(A) -> torch.Tensor:
        return torch.tensor(np.asarray(A, dtype=np.float32))

    return (
        to(prep.transform(X_tr)),
        to(y_tr.to_numpy()),
        to(prep.transform(X_te)),
        to(y_te.to_numpy()),
    )


def treinar(
    amostra: int | None = None, epocas: int = 15, batch: int = 4096, lr: float = 1e-3
) -> dict:
    torch.manual_seed(settings.seed)
    dev = dispositivo()
    Xtr, ytr, Xte, yte = _preparar_tensores(amostra)
    Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
    print(f"dispositivo: {dev} | treino: {len(Xtr):,} | features: {Xtr.shape[1]}")

    modelo = MLP(Xtr.shape[1]).to(dev)
    # pos_weight = nº negativos / nº positivos → penaliza mais o erro na classe rara.
    pos_weight = ((ytr == 0).sum() / (ytr == 1).sum()).to(dev)
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    n = len(Xtr)
    hist_treino, hist_val = [], []
    for ep in range(epocas):
        modelo.train()
        perm = torch.randperm(n, device=dev)
        perda_acum = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb, yb = Xtr[idx], ytr[idx]

            # ---- o ciclo de otimização, explícito ----
            otimizador.zero_grad()  # zera gradientes da iteração anterior
            logits = modelo(xb)  # forward
            perda = criterio(logits, yb)  # loss
            perda.backward()  # backward (autograd)
            otimizador.step()  # atualiza os pesos
            perda_acum += perda.item() * len(idx)

        modelo.eval()
        with torch.no_grad():
            val_loss = criterio(modelo(Xte), yte).item()
        hist_treino.append(perda_acum / n)
        hist_val.append(val_loss)
        print(f"  época {ep + 1:2}/{epocas}  treino={hist_treino[-1]:.4f}  val={val_loss:.4f}")

    modelo.eval()
    with torch.no_grad():
        prob = torch.sigmoid(modelo(Xte)).cpu().numpy()
    roc = float(roc_auc_score(yte.cpu().numpy(), prob))

    _salvar(hist_treino, hist_val, roc, str(dev))
    print(f"ROC-AUC (teste): {roc:.3f}  |  baseline HistGB: 0.813")
    return {"roc_auc": roc, "dispositivo": str(dev), "epocas": epocas}


def _salvar(hist_treino, hist_val, roc, dev) -> None:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(hist_treino) + 1), hist_treino, "o-", label="treino")
    ax.plot(range(1, len(hist_val) + 1), hist_val, "o-", label="validação")
    ax.set(
        xlabel="época",
        ylabel="BCE loss",
        title=f"Curva de treino da MLP ({dev}) — ROC-AUC teste {roc:.3f}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(_REPORT_DIR / "curva_treino_mlp.png", dpi=110)
    plt.close(fig)
    (_REPORT_DIR / "mlp.json").write_text(
        json.dumps(
            {"roc_auc": roc, "dispositivo": dev, "loss_treino": hist_treino, "loss_val": hist_val},
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP em PyTorch (Fase 0).")
    parser.add_argument("--amostra", type=int, default=None)
    parser.add_argument("--epocas", type=int, default=15)
    args = parser.parse_args()
    treinar(amostra=args.amostra, epocas=args.epocas)


if __name__ == "__main__":
    main()
