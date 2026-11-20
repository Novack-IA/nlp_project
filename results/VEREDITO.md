# Veredito da Fase 1

> **Status:** esqueleto (2026-11-20). Os números entram quando cada frente publicar seus arquivos. **Os quatro revisam antes do merge final.**
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
| C1 | Nº de features em S_intent (melhor camada) | | ≥ 5 | | `results/frente-c/c1_c3.json` |
| C1 | Fração de S_intent que replica no `test` (\|d_I\| ≥ 0,3, mesmo sinal) | | ≥ 70% | | `results/frente-c/c1_c3.json` |
| C2(a) | AUROC de S_intent, intenção **com** gatilho | | ≥ 0,85 | | `results/frente-d/c2_c4.json` → `principal.C2a.auroc_com_gatilho` |
| C2(a) | AUROC de S_intent, intenção **sem** gatilho | | ≥ 0,80 | | `… principal.C2a.auroc_sem_gatilho` |
| **C2(b)** | Razão queda(S_intent)/queda(TF-IDF), com IC 95% | | ≤ 0,5 e IC sup. < 1 | | `… principal.C2b.razao`, `ic95` |
| C2(c) | Consistência: S_intent no gatilho / S_lex no gatilho | | ≤ 0,65 / ≥ 0,85 | não decide | `… principal.C2c_consistencia` |
| C3 | Fração de S_intent sem efeito em `trig_only` × `neutral_only` (IC de d em ±0,2) | | maioria | | `results/frente-c/c1_c3.json` |
| C3 | Fichas do top-20 de S_intent rotuladas conceituais pelos dois avaliadores (κ) | | ≥ 60% | | `results/frente-c/c1_c3.json` |
| C4(a) | C1 e C2 no 262k **ou** no SAE próprio | | atende | | `results/frente-c/c1_c3.json` (robustez) |
| C4(b) | C1 e C2 só com Q⁰ original *retrieved* | | atende | | `… c2_c4.json` → `b_q0_retrieved` |
| C4(c) | C1 e C2 com células de tamanho pareado (≤ 20%) | | atende | | `… c2_c4.json` → `c_tamanho_pareado` |
| C5 | Ablação de S_intent: queda na recusa a Q⁺ menos a de features aleatórias (IC 95%) | | IC > 0 | complementar | `results/frente-b/c5_ablacao.json` |
| C5 | Ablação de S_lex: queda nas recusas a Q⁰ alarmantes (IC 95%) | | IC > 0 | complementar | `results/frente-b/c5_ablacao.json` |

## Âncora densa

O probe linear no residual completo mostra **quanto da informação existe** antes de passar pelo SAE.

| Medida | Valor | Fonte |
|---|---|---|
| Probe de intenção treinado com gatilho, testado sem gatilho (melhor camada × posição) | | `results/frente-d/probes.json` → `cells2x2.L*/*.cruzado_trig_para_notrig` |
| Probe de gatilho treinado no benigno, testado no nocivo | | `… gatilho_independe_intencao` |
| Cosseno entre as direções de intenção e de gatilho | | `… direcoes.cos_intencao_gatilho` |
| Cosseno entre a direção de intenção e a de recusa | | `… direcoes.cos_intencao_recusa` |

**Leitura obrigatória:**
- Se o probe denso **também não** generalizar sem o gatilho, o limite está nos dados ou no modelo, não no SAE.
- Se o probe denso generaliza e o SAE não, o limite é do dicionário.

## Réguas e diagnósticos (não decidem o veredito)

| Medida | Valor | Fonte |
|---|---|---|
| TF-IDF nos pares originais (`test`) | AUROC 0,995 (IC 0,991–0,998) | `results/frente-d/baseline_lexical.json` |
| Só estilo nos pares originais (`test`) | AUROC 0,881 (IC 0,860–0,903) | idem |
| Só gatilho nos pares originais (`test`) | AUROC 0,500 | idem |
| TF-IDF no 2×2: com gatilho → sem gatilho (queda) | | `baseline_lexical.json` → `cells_2x2.queda_auroc_tfidf` |
| TF-IDF no XSTest | | `baseline_lexical.json` → `xstest` |

Nos pares originais, as palavras (0,995) e até o estilo (0,88) já separam nocivo de benigno. Por isso esses números são diagnóstico e o veredito usa o 2×2.

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
