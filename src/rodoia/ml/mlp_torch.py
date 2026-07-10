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
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from torch import nn

from rodoia.config import settings
from rodoia.ml.classico import _metricas, carregar_dados, construir_preprocessador
from rodoia.proveniencia import carimbar

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
    """Split TRÊS-VIAS treino/validação/teste (64/16/20), pré-processador ajustado
    SÓ no treino (sem vazar val/teste). A validação sintoniza limiar e monitora a
    curva; o TESTE fica intocado e só é medido uma vez, no fim. Devolve 6 tensores."""
    X, y = carregar_dados(amostra=amostra)
    # 1) separa o teste (20%), intocado
    X_resto, X_te, y_resto, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=settings.seed
    )
    # 2) do restante, separa validação (20% do restante = 16% do total)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_resto, y_resto, test_size=0.2, stratify=y_resto, random_state=settings.seed
    )
    prep = construir_preprocessador().fit(X_tr)  # fit só no treino!

    def to(A) -> torch.Tensor:
        return torch.tensor(np.asarray(A, dtype=np.float32))

    return (
        to(prep.transform(X_tr)), to(y_tr.to_numpy()),
        to(prep.transform(X_val)), to(y_val.to_numpy()),
        to(prep.transform(X_te)), to(y_te.to_numpy()),
    )


def _limiar_max_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Limiar de decisão que maximiza o F1 NA VALIDAÇÃO (em vez do 0.5 arbitrário,
    ainda mais enganoso com pos_weight que distorce a probabilidade)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    # precision_recall_curve devolve len(thr) = len(prec)-1
    return float(thr[max(range(len(thr)), key=lambda i: f1[i])])


def treinar(
    amostra: int | None = None, epocas: int = 20, batch: int = 4096, lr: float = 1e-3
) -> dict:
    torch.manual_seed(settings.seed)
    dev = dispositivo()
    Xtr, ytr, Xval, yval, Xte, yte = _preparar_tensores(amostra)
    Xtr, ytr = Xtr.to(dev), ytr.to(dev)
    Xval, yval = Xval.to(dev), yval.to(dev)
    Xte, yte = Xte.to(dev), yte.to(dev)
    print(f"dispositivo: {dev} | treino: {len(Xtr):,} | val: {len(Xval):,} | "
          f"teste: {len(Xte):,} | features: {Xtr.shape[1]}")
    if dev.type != "cpu":
        print("  (caveat: treino em GPU pode ter pequena não-reprodutibilidade numérica)")

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
            val_loss = criterio(modelo(Xval), yval).item()  # VALIDAÇÃO, não o teste
        hist_treino.append(perda_acum / n)
        hist_val.append(val_loss)
        print(f"  época {ep + 1:2}/{epocas}  treino={hist_treino[-1]:.4f}  val={val_loss:.4f}")

    # limiar sintonizado na VALIDAÇÃO; métricas medidas no TESTE, uma única vez.
    modelo.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(modelo(Xval)).cpu().numpy()
        prob_te = torch.sigmoid(modelo(Xte)).cpu().numpy()
    limiar = _limiar_max_f1(yval.cpu().numpy(), prob_val)
    metricas = _metricas(yte.cpu().numpy(), (prob_te >= limiar).astype(int), prob_te)
    metricas["limiar"] = round(limiar, 4)

    _salvar(hist_treino, hist_val, metricas, str(dev))
    print(f"MLP teste — ROC-AUC {metricas['roc_auc']:.3f} | PR-AUC {metricas['pr_auc']:.3f} | "
          f"F1 {metricas['f1']:.3f} | bal_acc {metricas['balanced_accuracy']:.3f} "
          f"(limiar {limiar:.2f})  vs. baseline HistGB ROC-AUC 0.813")
    return {"dispositivo": str(dev), "epocas": epocas, **metricas}


def _salvar(hist_treino, hist_val, metricas: dict, dev: str) -> None:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(hist_treino) + 1), hist_treino, "o-", label="treino")
    ax.plot(range(1, len(hist_val) + 1), hist_val, "o-", label="validação")
    ax.set(
        xlabel="época",
        ylabel="BCE loss",
        title=f"Curva de treino da MLP ({dev}) — ROC-AUC teste {metricas['roc_auc']:.3f}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(_REPORT_DIR / "curva_treino_mlp.png", dpi=110)
    plt.close(fig)
    report = carimbar(
        {"dispositivo": dev, **metricas, "loss_treino": hist_treino, "loss_val": hist_val}
    )
    (_REPORT_DIR / "mlp.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP em PyTorch (Fase 0).")
    parser.add_argument("--amostra", type=int, default=None)
    parser.add_argument("--epocas", type=int, default=20)
    args = parser.parse_args()
    treinar(amostra=args.amostra, epocas=args.epocas)


if __name__ == "__main__":
    main()
