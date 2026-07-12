"""Fase 5 — MLOps: gate de avaliação, rastreio (MLflow), drift.

Tudo roda local e sem GPU: o gate lê os relatórios já versionados em `reports/` e
falha se uma métrica-chave regredir — é o que torna a "avaliação como gate" do CI real.
"""
