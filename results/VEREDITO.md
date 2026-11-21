# Veredito da Fase 1

> **Status:** preenchido (2026-11-21), aplicando a regra pré-registrada como está escrita em `00-visao-geral.md`. C1 e C3 vêm da frente C (commit `b299ef4` da `frente-c-features`); C2, C4 e a âncora densa, da frente D. O C5 (frente B) ainda não saiu e não decide.
> Como preencher: cada linha da tabela aponta para o arquivo e o campo de onde o número sai. Copie o número como está e não arredonde além de 2 casas. Não mude limiares aqui; os limiares valem como estão em `docs/tarefas/00-visao-geral.md`, e mudanças vão para `results/DESVIOS.md`.

## A pergunta

> **É possível identificar, nas features de um SAE, um sinal de intenção nociva que se distingue do sinal de forma lexical do pedido, usando pares mínimos nocivo↔benigno?**

## Resposta

**Veredito: PARCIAL.**

Existe um sinal de intenção nas features do SAE que não depende do gatilho: em L29, no último token antes da resposta, 48 features (60 no 262k) replicam 100% no `test` com gatilho inédito (C1), e um classificador só com elas perde no XSTest um décimo do que perde o TF-IDF (C2, razão 0,11). Faltam dois critérios como pré-registrados: o C3 pedia leitura cega de **duas pessoas**, e as duas leituras foram de LLMs; e o C4(b/c) está incompleto. Além disso, **features sorteadas do mesmo tamanho fazem o mesmo que a S_intent no C2** (20/20 atendem), e **não existe nenhuma feature de forma lexical** (S_lex vazia).

**Como a regra foi aplicada:**

| Critério | Situação | Por quê |
|---|---|---|
| C1 | atende | 48 S_intent, 100% replicam |
| C2 | atende | C2(a) 0,992 / 0,992; C2(b) razão 0,11 (IC 0,05–0,17). O C2(c) não decide |
| C3 | **não atende como pré-registrado** | O controle `trig_only` atende (81%). A avaliação cega atinge o limiar (95% e 85%), mas foi feita por dois LLMs, e o pré-registro pede 2 pessoas |
| C4 | **incompleto** | (a) atende (262k). (b) e (c): o C2(a) se mantém, mas o C1 e o C2(b) dentro dos recortes não foram calculados |

C1 + C2 atendidos, C3 e C4 não → **PARCIAL** ("C1 atendido, mais um ou dois de C2–C4").

**Regra** (pré-registrada em `00-visao-geral.md`):
- **SIM:** C1, C2, C3 e C4 atendidos.
- **PARCIAL:** C1 atendido, mais um ou dois de C2–C4.
- **NÃO:** C1 falha em todas as camadas e SAEs.
- C5 não decide o veredito; entra no texto como evidência causal.

## Critérios

Todos no `test`, recorte **gatilho inédito**, salvo indicação.

| # | Critério | Valor | Limiar | Atende? | Fonte |
|---|---|---|---|---|---|
| C1 | Nº de features em S_intent (L29 · `post_last`, 16k) | 48 | ≥ 5 | **sim** | `results/frente-c/c1_c3.json` (commit `b299ef4`) → `16k.C1.S_intent` |
| C1 | Fração de S_intent que replica no `test` (\|d_I\| ≥ 0,3, mesmo sinal; 155 grupos com gatilho inédito) | 100% (48/48) | ≥ 70% | **sim** | `… 16k.replicacao_S_intent` |
| C2(a) | AUROC de S_intent, intenção **com** gatilho (16k; 262k) | 0,992 (IC 0,985–0,997); 0,998 | ≥ 0,85 | **sim**, mas ver controle | `results/frente-d/c2_c4_L29_post_last_16k.json` → `principal.C2a` |
| C2(a) | AUROC de S_intent, intenção **sem** gatilho (16k; 262k) | 0,992 (IC 0,984–0,997); 0,997 | ≥ 0,80 | **sim**, mas ver controle | idem |
| C2(a), controle | Features aleatórias do mesmo tamanho (média de 20 sorteios), com / sem gatilho | 0,995 / 0,995 (16k); 0,994 / 0,994 (262k) | — | **S_intent não supera o aleatório** | `… principal.matriz.aleatorio` |
| **C2(b)** | **XSTest:** razão queda(S_intent)/queda(TF-IDF), IC 95% (16k; 262k) | 0,030/0,278 = **0,11** (IC 0,05–0,17); **0,06** (IC 0,03–0,11) | ≤ 0,5 e IC sup. < 1 | **sim**, mas ver controle | `… principal.C2b.xstest` |
| C2(b), controle | Mesma razão com features aleatórias do mesmo tamanho (mediana de 20) e com o denso | aleatório 0,11 (16k) e 0,08 (262k), **20/20 sorteios atendem**; denso 0,11 | — | **S_intent não supera o aleatório** | `results/frente-d/c2b_controle_aleatorio.json` (controle acrescentado depois do C2; ver `DESVIOS.md`) |
| C2(b), descrição | Gatilho inédito: razão, IC 95% (TF-IDF cai 0,003, indefinida) | | não decide | | `… principal.C2b.gatilho_inedito` |
| C2(b), descrição | Leave-one-domain-out: razão, IC 95% (TF-IDF cai 0,015, indefinida) | | não decide | | `… principal.C2b.leave_one_domain_out` |
| C2(c) | Consistência: S_intent no gatilho / S_lex no gatilho | | ≤ 0,65 / ≥ 0,85 | **indefinido**: S_lex vazia em todas as combinações (frente C); não decide | `… principal.C2c_consistencia` |
| C3 | Fração de S_intent sem efeito em `trig_only` × `neutral_only` (IC de d em ±0,2) | 81% (39/48) | maioria | **sim** | `… 16k.C3_controle_trig_only` |
| C3 | Fichas do top-20 de S_intent rotuladas conceituais pelos dois avaliadores (κ) | LLM da frente C: 95% (19/20); 2º avaliador (Claude, pela frente D): 85% (17/20). Concordam em 18/20, κ = 0,46 | ≥ 60% para os dois | **sim, com ressalva** | `… C3_avaliacao_cega` e `results/frente-d/c3_rotulos_davi.csv`. **Os dois avaliadores cegos são LLMs**; o Davi conferiu depois e concordou com os 20 rótulos do 2º, sem estar cego (ver `DESVIOS.md`) |
| C4(a) | C1 e C2 no 262k **ou** no SAE próprio | C1 no 262k: 60 features, 100% replicam. C2 no 262k: C2(a) 0,998 / 0,997; C2(b) razão 0,06 | atende | **sim** (com a mesma ressalva do controle) | `results/frente-c/c1_c3.json` → `262k.C1`; `results/frente-d/c2_c4_L29_post_last_262k.json` |
| C4(b) | C2 só com Q⁰ original *retrieved* (247 grupos) | C2(a) 0,979 / 0,992 (16k) | atende | **C2(a) sim**; C2(b) no recorte não calculado; C1 no recorte: frente C | `… c2_c4_L29_post_last_16k.json` → `b_q0_retrieved` |
| C4(c) | C2 com células de tamanho pareado (≤ 20%; 684 grupos) | C2(a) 0,986 / 0,991 (16k) | atende | **C2(a) sim**; C2(b) no recorte não calculado; C1 no recorte: frente C | `… c_tamanho_pareado` |
| C5 | Ablação de S_intent: queda na recusa a Q⁺ menos a de features aleatórias (IC 95%) | | IC > 0 | complementar | `results/frente-b/c5_ablacao.json` |
| C5 | Ablação de S_lex: queda nas recusas a Q⁰ alarmantes (IC 95%) | | IC > 0 | complementar | `results/frente-b/c5_ablacao.json` |

## Âncora densa

O probe linear no residual completo mostra **quanto da informação existe** antes de passar pelo SAE.

| Medida | Valor | Fonte |
|---|---|---|
| Probe de intenção treinado com gatilho, testado sem gatilho (L29 · `post_last`; faixa em todas as camadas × posições) | **0,998** (IC 0,996–1,000); faixa 0,966–1,000 em todas as combinações | `results/frente-d/probes.json` → `cells2x2.principal.L29/post_last.cruzado_trig_para_notrig` |
| Probe de gatilho treinado no benigno, testado no nocivo (L29 · `post_last`; na posição do gatilho) | 0,59; 0,80 em `slot_forma` | `… gatilho_independe_intencao` |
| Cosseno entre as direções de intenção e de gatilho | instável: muda de sinal entre camadas (−0,83 a 0,94); não interpretável | `… direcoes.cos_intencao_gatilho` |
| Cosseno entre a direção de intenção e a de recusa | 0,98–1,00, **quase por construção** (as células com gatilho são os próprios pares) | `… direcoes.cos_intencao_recusa` |

**Leitura obrigatória:**
- Se o probe denso **também não** generalizar sem o gatilho, o limite está nos dados ou no modelo, não no SAE.
- Se o probe denso generaliza e o SAE não, o limite é do dicionário.

## Réguas e diagnósticos (não decidem o veredito)

| Medida | Valor | Fonte |
|---|---|---|
| TF-IDF nos pares originais (`test`) | AUROC 0,995 (IC 0,991–0,998) | `results/frente-d/baseline_lexical.json` |
| Só estilo nos pares originais (`test`) | AUROC 0,881 (IC 0,860–0,903) | idem |
| Só gatilho nos pares originais (`test`) | AUROC 0,500 | idem |
| TF-IDF no 2×2 (conjunto principal, 1.697 grupos), intenção com → sem gatilho | 0,996 → 0,997 (não cai) | `baseline_lexical.json` → `cells_2x2.principal.treinado_no_2x2` |
| **Só estilo no 2×2** (conjunto principal), intenção | **0,889** (IC 0,861–0,916) | idem, `so_estilo` |
| TF-IDF no 2×2, presença do gatilho | 0,767 | idem, `gatilho_todas` |
| Queda do TF-IDF: gatilho inédito / domínio inédito / XSTest | 0,003 / 0,015 / 0,280 | `… cells_2x2.principal.regua_C2b_tfidf` |
| TF-IDF dos pares no XSTest | 0,710 (IC 0,663–0,756) | `baseline_lexical.json` → `xstest` |

Nos pares originais, as palavras (0,995) e até o estilo (0,88) já separam nocivo de benigno. **No plano B do 2×2 continua igual:** as células com gatilho são os pedidos originais, o TF-IDF acerta 0,997 mesmo sem o gatilho e o estilo sozinho acerta 0,89. Qualquer sinal de intenção no 2×2 precisa ser lido contra essas duas réguas. Só no XSTest o classificador de palavras cai de fato.

## Interpretação

O Gemma-3-4B-IT representa a intenção do pedido de um jeito que **não depende das palavras alarmantes**: trocar o gatilho por uma forma neutra quase não mexe nas features de intenção em L29 antes da resposta, e um probe treinado com gatilho acerta 0,998 sem ele. O lado lexical, por outro lado, **não vira feature**: cada gatilho mexe em features diferentes, e nenhuma reage do mesmo jeito na maioria dos grupos. Então a hipótese "intenção e forma em features separadas" vale só pela metade, e a dissociação é simples, não dupla. Dois avisos para a leitura. (1) A capacidade de classificar intenção não é exclusiva da S_intent: features sorteadas e o probe denso generalizam para o XSTest igualmente bem, então o que é específico da S_intent é ser **insensível ao gatilho** (C1, C3), não ser melhor classificadora. (2) A frente C viu que as features mais fortes de intenção parecem ser de **recusa** e ficaram fora da S_intent porque o gatilho as empurra um pouco (cerca de 10% do efeito de intenção). Se o over-refusal existe nas representações, é ali que ele deve estar, e quem testa é o C5.

## Se o veredito for PARCIAL

_Preencher só neste caso. Ele é o desfecho mais provável, então vale escrever com cuidado._

| Pergunta | Resposta |
|---|---|
| Pergunta | Resposta |
|---|---|
| Qual critério falhou? | **C3**, por procedimento: a avaliação cega foi feita por dois LLMs, não por duas pessoas (o resultado em si passa: 95% e 85%). **C4**, por estar incompleto: falta o C1 e o C2(b) nos recortes (b) e (c) |
| Em qual recorte (camada, posição, SAE, domínio, tipo de gatilho)? | Nenhum recorte falhou com número. A combinação principal é L29 · `post_last`; o C1 vale nos 4 domínios e nos dois SAEs |
| O probe denso também falha ali? | Não falha em nada: 0,97–1,00 em todas as camadas e posições |
| O efeito existe só em alguns domínios ou tipos de gatilho? | Não: as 48 S_intent são gerais nos 4 domínios (frente C) |
| O que seria preciso para virar SIM? | (1) duas pessoas lerem as 20 fichas às cegas, sem ver os rótulos dos LLMs, e ≥ 60% saírem conceituais para as duas; (2) o C1 e o C2(b) nos recortes (b) Q⁰ *retrieved* (247 grupos) e (c) tamanho pareado (684 grupos) |

Frase de abertura: _"Encontramos um sinal de intenção nas features do SAE que não depende da forma lexical (C1, C2), mas não um sinal de forma lexical: nenhuma feature reage ao gatilho de modo consistente. A classificação de intenção, porém, não é exclusiva dessas features, e a avaliação humana e duas robustezes ficaram por fazer."_

## Limitações

_Começar por estas e acrescentar o que aparecer:_
- O 2×2 é gerado por um LLM; a intenção é validada por juízes automáticos. A checagem humana é opcional (ver `DESVIOS.md`).
- Os SAEs do Gemma Scope 2 foram treinados em dados genéricos; a qualidade nos nossos pedidos está em `results/frente-b/qualidade_saes.csv`.
- Só inglês, 4 domínios, um modelo (Gemma-3-4B-IT).
- O C2(c) é garantido em boa parte pela seleção das features e não conta como evidência. Aqui ele nem existe: a S_lex saiu vazia.
- **No plano B, o contraste de intenção compara pedidos diferentes** (Q⁺ do PKU × Q⁰ gerado). O estilo sozinho separa as linhas com AUROC 0,89. O controle de estilo da frente C (48/48 sobrevivem) e o 2×2 em estilo comum (robustez) mitigam, mas não eliminam.
- **Features aleatórias do mesmo tamanho fazem o mesmo que a S_intent no C2**, dentro do 2×2 e no XSTest. O C2 não mostra que a S_intent é necessária.
- A avaliação cega do C3 foi feita por LLMs.
- A frase-molde restringe os grupos que entram. Os tipos de gatilho e domínios que sobreviveram estão em `results/frente-a/rodada_seca.md`.

## Desvios do pré-registro

Ver `results/DESVIOS.md`. Os que afetam a leitura do veredito:
- **Conjunto principal = `qc_grammatical` sem os 26 grupos reprovados pelo validador** (1.697), decidido antes de analisar o cache.
- **C2(b) decidido só pelo XSTest** (antes "≥ 2 de 3", que falhava por construção), decidido antes do cache do XSTest.
- **C3 com dois avaliadores LLM**, em vez de duas pessoas.
- **Controle aleatório no XSTest**, acrescentado depois de ver o C2. Não muda o "atende", muda a leitura.
- **2×2 em estilo comum** só como robustez (968 grupos, abaixo da regra de 1.200).
