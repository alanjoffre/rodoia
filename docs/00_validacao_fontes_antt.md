# 00 — Validação de realidade das fontes da ANTT

> **Status:** concluída em 09/07/2026 (pesquisa web com URLs verificadas ao vivo).
> **Por que este documento existe:** o `PROMPT_MESTRE.md` (Seção 6, "Primeira tarefa") exige confirmar disponibilidade, formato e licença dos dados **antes** de desenhar arquitetura. A arquitetura se adapta ao que existe, não ao que imaginamos. Este é o registro dessa validação — a base factual das Fases 0, 1 e 3.
>
> **Como ler:** cada fonte traz URL verificada, formato, acesso programático, licença e as pegadinhas reais encontradas. Ao fim, as **decisões** tomadas e os **itens a confirmar** (honestos — o que a pesquisa não conseguiu fechar).

---

## Parte A — Fontes textuais (para o RAG da Fase 1)

Objetivo: legislação/normas da ANTT (resoluções, portarias, deliberações) como corpus do RAG.

### A.1 ANTTlegis — sistema oficial de normas *(fonte primária de texto)*

- **URL:** https://anttlegis.antt.gov.br/ (no ar, HTTP 200). Plataforma "datalegis".
- **Conteúdo:** acervo canônico e completo da ANTT — Resoluções, Portarias, Deliberações, Instruções, Súmulas, Decisões, Editais. Indexado por tipo, número, ano, órgão, tema e **situação (vigente/revogado/anulado)**.
- **Formato:** HTML por ato, servido por **URL determinística** (`ActionDatalegis.php?acao=abrirTextoAto&tipo=RES&numeroAto=NNNNNNNN&valorAno=AAAA&...`), o que permite varredura previsível por tipo/número/ano.
- **Acesso programático:** **sem API REST**. Existe só a "Ferramenta Push" (assinatura por e-mail). A ingestão viável é **scraping educado** do padrão `abrirTextoAto`.
- **Licença:** sem termo explícito na página; o conteúdo é ato oficial → domínio público por lei (ver A.4).
- **Pegadinhas confirmadas:**
  - Encoding **ISO-8859-1 (latin-1)**, não UTF-8 → converter na ingestão.
  - Resoluções **antigas** (início dos anos 2000) tendem a ser **PDF-imagem** → exigem **OCR**. Atos recentes vêm em HTML/texto.
  - Precisa modelar o **grafo de revogação** (altera/revoga/é revogada por) para não responder com base em norma revogada.

### A.2 LexML Brasil *(camada de metadados / identidade estável)*

- **URL:** https://www.lexml.gov.br/ · docs https://projeto.lexml.gov.br/
- **Valor:** identificador canônico **URN LexML** por norma + metadados (ementa, assunto, autoridade, data), ótimo como **ID único e deduplicador** entre fontes. Declara **licença aberta** explícita (a mais favorável das fontes textuais).
- **Acesso programático:** API **SRU/CQL → XML**; APIs de **dados abertos JSON** (`/apidata/...`); dumps de metadados (inclusive no Kaggle e no Dados Abertos do Senado).
- **Pegadinha crítica confirmada:** o `www.lexml.gov.br` está hoje atrás de um **desafio anti-bot proof-of-work** (haphash/SHA-256 em JS). curl e fetch simples recebem a **página de desafio, não o XML**. → Preferir os **dumps/JSON de dados abertos** (que não passam pelo muro) a depender do SRU ao vivo.

### A.3 DOU / Imprensa Nacional (in.gov.br) *(proveniência e recência)*

- **URL:** https://www.in.gov.br/ · busca https://www.in.gov.br/consulta/
- **Valor:** fonte primária de publicação (toda norma da ANTT sai no DOU) → boa para **data de publicação/vigência** e captar atos recentes.
- **Acesso programático:** a API oficial (WS-INCom) é **só para publicar**, restrita a órgãos — inútil para leitura. A consulta pública (`in.gov.br/consulta/`, base do projeto **Ro-DOU**) retorna JSON, mas **deu timeout/bloqueio (HTTP 000)** no teste — provável rate-limit. Alternativa conveniente: espelho **Base dos Dados** (BigQuery, SQL/Python).
- **Ruído:** o texto no DOU vem no contexto do jornal (outras matérias) → não usar como texto principal.

### A.4 Base legal do domínio público *(o que torna seguro redistribuir)*

- **Lei 9.610/1998, Art. 8º, IV:** não são objeto de proteção autoral "os textos de tratados ou convenções, **leis, decretos, regulamentos, decisões judiciais e demais atos oficiais**". Resoluções/portarias da ANTT são atos normativos oficiais → **texto livre para copiar e redistribuir**.
- **Ressalva:** isso cobre o *texto oficial*. Uma *base de terceiros* que compila esses textos pode ter proteção sobre a **compilação/estrutura** (não sobre o texto). → Se usar dumps do LexML/Base dos Dados, **atribuir** essas bases.

### Volume estimado (Parte A)

~**6.000 resoluções** (numeração saltou para 6.000 em 12/2022; já em 6.079 em 03/2026), muitas revogadas, + milhares de portarias/deliberações. O subconjunto **vigente e relevante para transporte rodoviário** é bem menor (centenas a poucos milhares) — bom, cabe num RAG focado.

---

## Parte B — Datasets tabulares (para o ML clássico da Fase 0 e o SQL da Fase 3)

### Portal primário

- **Dados Abertos da ANTT** — https://dados.antt.gov.br/ — instância **CKAN** ativa, temas: Rodovias (~45), Passageiros (~20), Cargas (~10), Fiscalização (~10), Ferrovias (~6).
- **dados.gov.br** espelha/aponta para o portal da ANTT (o próprio é mais completo).
- **Kaggle/HuggingFace:** sem dataset oficial da ANTT re-hospedado → puxar direto do portal.
- **Licença geral:** padrão **CC-BY** (Decreto 8.777/2016). Seguro para repo público **com atribuição** — mas confirmar o campo licença de cada dataset no uso.
- **Pegadinha técnica transversal (confirmada baixando arquivos):** CSVs em **latin-1**, separador **`;`**, decimal com **vírgula**. Ler com `pandas.read_csv(sep=';', encoding='latin-1', decimal=',')`.

### Datasets relevantes (verificados)

| # | Dataset | URL | Granularidade | Apto p/ ML? |
|---|---|---|---|---|
| 1 | **Volume de Tráfego nas Praças de Pedágio** ⭐ | `/dataset/volume-trafego-praca-pedagio` | por praça/eixo/veículo/mês (2010–2026); ~142k linhas/ano | **Regressão** (alvo `volume_total`) + série temporal |
| 5 | **Acidentes em rodovias concedidas** ⭐⭐ | `/dataset/acidentes-rodovias` | **1 linha = 1 acidente**; centenas de milhares somando concessionárias | **Classificação** de severidade — melhor candidato |
| 3 | Praça de Pedágio (cadastro + KMZ) | `/dataset/praca-de-pedagio` | dimensão geográfica | Dimensão p/ JOIN (SQL) |
| 4 | Receita de Pedágio | `/dataset/receita-pedagio` | anual/concessionária | SQL analítico (agregado demais p/ ML) |
| 7 | SIFAMA — Autos de Infração (Cargas) | `/dataset/sifama-autos-de-infracao-cargas` | **agregado** (`quantidade_autos` por UF/mês/código) | Regressão de contagem / SQL |
| 9 | RNTRC / RNTRC-Veículos | `/dataset/rntrc-veiculos` | frota nacional (ativos) | Potencial (headers a confirmar) |
| 10 | Transporte Rod. de Passageiros (MONITRIIP) | `/dataset/transporte-rodoviario-de-passageiros` | viagens/bilhetes/linhas | Potencial (granularidade a confirmar) |

### Detalhe dos dois carro-chefe

**Acidentes (`acidentes-rodovias`)** — colunas reais (header verificado):
`data;horario;n_da_ocorrencia;tipo_de_ocorrencia;km;trecho;sentido;tipo_de_acidente;automovel;bicicleta;caminhao;moto;onibus;outros;tracao_animal;transporte_de_cargas_especiais;trator_maquinas;utilitarios;ilesos;levemente_feridos;moderadamente_feridos;gravemente_feridos;mortos`
- **Correção importante:** a página sugere "demonstrativo/agregado", mas o **CSV é por acidente individual**.
- Só a Autopista Fernão Dias tem ~117.900 linhas (2010→).
- **Distinção PRF × ANTT:** estes acidentes são reportados pelas **concessionárias**, só das **rodovias federais concedidas** (escopo ANTT). A base nacional de todas as BRs é da **PRF** (outra organização/licença) — fora do escopo "ANTT puro"; fica como complemento opcional.

**Volume de Tráfego (`volume-trafego-praca-pedagio`)** — header verificado:
`concessionaria;mes_ano;sentido;praca;tipo_cobranca;categoria_eixo;tipo_de_veiculo;volume_total`

---

## Decisões desta validação

1. **RAG (Fase 1) — fontes em camadas:**
   - **Texto primário:** ANTTlegis (scraping do `abrirTextoAto`, latin-1→UTF-8, capturando metadados e situação vigente/revogado).
   - **Identidade/metadados:** LexML via **dumps/JSON de dados abertos** (não o SRU ao vivo, por causa do proof-of-work) — para URN canônica e deduplicação.
   - **Recência/proveniência:** DOU via Ro-DOU ou espelho Base dos Dados, só para data — não como texto.
   - **Escopo inicial enxuto:** começar pelas **resoluções vigentes de transporte rodoviário** (não o acervo inteiro), para um RAG focado e defensável.

2. **ML clássico (Fase 0) — problema carro-chefe:**
   - **Dataset:** Acidentes em rodovias concedidas (`acidentes-rodovias`).
   - **Problema honesto:** **classificação binária de severidade** (`houve vítima fatal` / ou `com vítima` vs `sem vítima`) a partir de horário, km, trecho, sentido, `tipo_de_acidente` e composição de veículos.
   - **Por que é honesto e forte:** por instância, centenas de milhares de linhas, features reais, e **desbalanceamento natural** (fatais são minoria) — pretexto perfeito para demonstrar F1/ROC-AUC/`class_weight` em vez de acurácia ingênua (casa com o critério "diagnóstico de modelo" da Fase 0).

3. **SQL analítico (Fase 3) — dataset carro-chefe:**
   - Volume de Tráfego de Pedágio + JOINs com Praça de Pedágio (geografia) e Receita (financeiro) — exercita janelas, CTEs e agregações sobre dado real.

4. **Licença — veredito:** repo público é **seguro**. Texto de norma = ato oficial fora da proteção autoral (Lei 9.610/98, art. 8º, IV). Datasets = CC-BY (Decreto 8.777/2016), exigem **atribuição à ANTT**. Registrar proveniência (URL + data de coleta) de tudo em `data/README.md` e `NOTICE`.

---

## Itens a confirmar (honesto — antes de fechar cada fase)

- [ ] **Schema consistente entre CSVs de Acidentes** (cada concessionária é um arquivo; validar se todas têm as mesmas colunas antes de concatenar).
- [ ] **Licença individual** de cada dataset usado (confirmar o campo na página, não assumir CC-BY).
- [ ] **LexML:** endpoint de dados abertos que de fato entrega normas da ANTT, e cobertura real de resoluções de agência (pode privilegiar leis/decretos federais).
- [ ] **in.gov.br/consulta:** formato do JSON e limites de rate — deu timeout no teste.
- [ ] **Headers a inspecionar** antes de prometer alvo: SIFAMA Passageiros/Trânsito, RNTRC-Veículos, MONITRIIP.
- [ ] **Fatia de PDFs que exigem OCR** no ANTTlegis (resoluções antigas) — dimensionar.
- [ ] **Redação literal** do Art. 8º, IV colada do Planalto na doc final (a citação aqui está correta em substância).

---

## Fontes verificadas

Parte A: [anttlegis.antt.gov.br](https://anttlegis.antt.gov.br/) · [lexml.gov.br](https://www.lexml.gov.br/) · [projeto.lexml.gov.br](https://projeto.lexml.gov.br/) · [in.gov.br](https://www.in.gov.br/) · [Lei 9.610/98](https://www.planalto.gov.br/ccivil_03/leis/l9610.htm)

Parte B: [dados.antt.gov.br/dataset](https://dados.antt.gov.br/dataset) · [Volume de Tráfego](https://dados.antt.gov.br/dataset/volume-trafego-praca-pedagio) · [Acidentes](https://dados.antt.gov.br/dataset/acidentes-rodovias) · [Praça de Pedágio](https://dados.antt.gov.br/dataset/praca-de-pedagio) · [Receita](https://dados.antt.gov.br/dataset/receita-pedagio) · [SIFAMA Cargas](https://dados.antt.gov.br/dataset/sifama-autos-de-infracao-cargas) · [RNTRC-Veículos](https://dados.antt.gov.br/dataset/rntrc-veiculos)
