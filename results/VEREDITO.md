# Veredito da Fase 1

> **Status:** em preenchimento (2026-11-21). C1 e C3 vêm da frente C (commit `b299ef4` da `frente-c-features`); C2 e a âncora densa, da frente D. Os números entram quando cada frente publicar seus arquivos. **Os quatro revisam antes do merge final.**
> Como preencher: cada linha da tabela aponta para o arquivo e o campo de onde o número sai. Copie o número como está e não arredonde além de 2 casas. Não mude limiares aqui; os limiares valem como estão em `docs/tarefas/00-visao-geral.md`, e mudanças vão para `results/DESVIOS.md`.

## A pergunta

> **É possível identificar, nas features de um SAE, um sinal de intenção nociva que se distingue do sinal de forma lexical do pedido, usando pares mínimos nocivo↔benigno?**

## Resposta

**Veredito:** _SIM / PARCIAL / NÃO (a preencher)_

_Uma frase com a resposta e o número que mais pesa. Exemplo de forma: "PARCIAL: existe um sinal de intenção em L22 (12 features, 83% replicam no test), mas ele cai quase tanto quanto o TF-IDF quando o gatilho sai (razão 0,9)."_

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

_Um parágrafo. O que a resposta significa para a hipótese de over-refusal: o modelo representa intenção separada da forma? Em que camada e posição? A recusa acompanha qual dos dois sinais (C5)?_

## Se o veredito for PARCIAL

_Preencher só neste caso. Ele é o desfecho mais provável, então vale escrever com cuidado._

| Pergunta | Resposta |
|---|---|
| Qual critério falhou? | |
| Em qual recorte (camada, posição, SAE, domínio, tipo de gatilho)? | |
| O probe denso também falha ali? | sim → limite do modelo ou dos dados · não → limite do SAE |
| O efeito existe só em alguns domínios ou tipos de gatilho? | |
| O que seria preciso para virar SIM? | |

Frase de abertura sugerida: _"Encontramos um sinal de intenção nas features do SAE (C1), mas ele [não se separa do lexical / não sobrevive a outro dicionário / depende do estilo das fontes]: [critério] falhou em [recorte] ([número])."_

## Limitações

_Começar por estas e acrescentar o que aparecer:_
- O 2×2 é gerado por um LLM; a intenção é validada por juízes automáticos. A checagem humana é opcional (ver `DESVIOS.md`).
- Os SAEs do Gemma Scope 2 foram treinados em dados genéricos; a qualidade nos nossos pedidos está em `results/frente-b/qualidade_saes.csv`.
- Só inglês, 4 domínios, um modelo (Gemma-3-4B-IT).
- O C2(c) é garantido em boa parte pela seleção das features e não conta como evidência.
- A frase-molde restringe os grupos que entram. Os tipos de gatilho e domínios que sobreviveram estão em `results/frente-a/rodada_seca.md`.

## Desvios do pré-registro

Ver `results/DESVIOS.md`. Liste aqui só os que afetam a leitura do veredito, com uma linha cada.
