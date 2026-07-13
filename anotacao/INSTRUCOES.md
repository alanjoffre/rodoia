# Anotação humana de relevância — instruções (κ inter-anotador)

Objetivo: obter um **κ de Cohen humano** — a concordância entre **dois avaliadores** julgando, de
forma independente, se um trecho recuperado é **relevante** para a pergunta. Isso substitui o
"anotador único" por evidência de concordância humana.

## O que fazer (cada avaliador, SEM ver o do outro)

1. Abra o **seu** arquivo no Excel/LibreOffice/Google Sheets:
   - Avaliador 1 → **`anotador_A.xlsx`**
   - Avaliador 2 → **`anotador_B.xlsx`**
2. Para cada linha, leia a **`consulta`** (pergunta do usuário) e o **`trecho`** (pedaço de uma
   resolução da ANTT). Preencha a coluna **`relevante`**:
   - **`1`** = o trecho **ajuda a responder** a pergunta (é sobre o mesmo assunto/regra).
   - **`0`** = o trecho **não ajuda** (é de outro tema, ou genérico demais).
   - Na dúvida entre "ajuda um pouco" e "não ajuda", decida pelo que **você usaria** para responder.
3. Preencha **todas as 30 linhas**. Não combine respostas com o outro avaliador (a graça é a
   independência).
4. **Salve** (Ctrl+S) mantendo o formato **.xlsx**. Não mude o cabeçalho nem a coluna `relevante`.
   (O cálculo do κ também aceita `.csv`, se preferir exportar.)

## Depois (quem coordena)

Com os dois arquivos preenchidos, rode:

```bash
python -m rodoia.anotacao kappa anotacao/anotador_A.xlsx anotacao/anotador_B.xlsx
```

Isso gera `reports/fase1_rag/kappa_humano.json` com:
- **`cohen_kappa`** — a concordância corrigida pelo acaso (interpretação Landis-Koch: 0,21–0,40
  razoável · 0,41–0,60 moderada · 0,61–0,80 substancial · >0,80 quase perfeita);
- **`concordancia_percentual`** e **`n_pares`**.

## Notas
- São **30 pares** = ~15 perguntas × (1 trecho provavelmente relevante + 1 distrator), embaralhados.
  A mistura é de propósito: sem variedade, o κ não seria informativo.
- Textos são **domínio público** (atos oficiais da ANTT). Sem PII.
- É esperado que anotadores humanos **discordem em alguns casos** — é justamente isso que o κ mede,
  honestamente. Um κ "moderado/substancial" já é um sinal forte de que o rótulo é confiável.
