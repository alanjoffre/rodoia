# Validação dos rótulos-gold de FONTE do hit@5 — instruções (κ inter-anotador)

Objetivo: fechar o **último elo** de rigor da Fase 1. O `hit@5` do retrieval é medido contra um
**conjunto dourado** de perguntas cuja "fonte correta" (a resolução da ANTT que responde) foi, até
aqui, definida por **um único anotador** (o autor). Aqui **dois avaliadores humanos** julgam, de
forma independente, se cada resolução é **de fato** uma fonte correta para a pergunta — e medimos o
**κ de Cohen** dessa concordância. Isso valida, entre humanos, os rótulos que a métrica do gate usa.

## O que fazer (cada avaliador, SEM ver o do outro)

1. Abra o **seu** arquivo no Excel/LibreOffice/Google Sheets:
   - Avaliador 1 → **`gold_fonte_A.xlsx`**
   - Avaliador 2 → **`gold_fonte_B.xlsx`**
2. Para cada linha, leia a **`consulta`** (pergunta do usuário), a **`resolucao`** (número), o
   **`titulo`** e o **`trecho`** (início da resolução da ANTT). Preencha a coluna **`relevante`**:
   - **`1`** = esta resolução **é uma fonte correta** para responder a pergunta (trata do mesmo
     assunto/regra que a pergunta pede).
   - **`0`** = **não é** a fonte certa (é de outro tema, mesmo que também seja da ANTT).
   - Julgue pelo conteúdo, não pelo número. Na dúvida, decida pela resolução que **você citaria**
     ao responder essa pergunta.
3. Preencha **todas as 50 linhas**. Não combine respostas com o outro avaliador — a independência é
   o que dá valor ao κ.
4. **Salve** (Ctrl+S) mantendo o formato **.xlsx**. Não mude o cabeçalho nem a coluna `relevante`.

## Depois (quem coordena)

Com os dois arquivos preenchidos:

```bash
python -m rodoia.anotacao kappa-gold anotacao/gold_fonte_A.xlsx anotacao/gold_fonte_B.xlsx
```

Gera `reports/fase1_rag/kappa_gold_fonte.json` com `cohen_kappa`, `cohen_kappa_ic95`,
`concordancia_percentual` e `n_pares` (interpretação Landis-Koch: 0,41–0,60 moderada · 0,61–0,80
substancial · >0,80 quase perfeita).

## Notas

- São **50 pares** = 25 perguntas × (a fonte-**gold** do autor + 1 **distrator**: a fonte-gold de
  outra pergunta). A mistura é de propósito — sem casos "não relevante", o κ seria degenerado. Um κ
  alto significa que os avaliadores **concordam com os rótulos-gold** (marcam a gold como relevante e
  o distrator como não), validando entre humanos os labels que o `hit@5` usa.
- Textos são **domínio público** (atos oficiais da ANTT). Sem PII.
- Discordância em alguns casos é esperada e honesta — é o que o κ mede. Reproduz de um clone limpo:
  os pares são regeneráveis com `python -m rodoia.anotacao gerar-gold` (seed fixa).
