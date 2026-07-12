"""Configuração de coleta dos testes.

No CI (CPU, sem GPU) o PyTorch não é instalado — os testes de fundamentos da Fase 0
(atenção/backprop/MLP, que exigem torch) são IGNORADOS na coleta em vez de quebrarem.
Localmente, com torch presente, todos rodam normalmente. Assim o CI valida todo o resto
(RAG, NER, dados, agente, gate) sem precisar do stack pesado de GPU.
"""
from __future__ import annotations

import importlib.util

# Testes que dependem de torch (validados localmente na Nitro; pulados no CI sem torch).
_DEPENDEM_DE_TORCH = [
    "test_attention.py",
    "test_backprop_numpy.py",
    "test_mlp_torch.py",
    "test_ml_treino.py",
]

collect_ignore = [] if importlib.util.find_spec("torch") else list(_DEPENDEM_DE_TORCH)
