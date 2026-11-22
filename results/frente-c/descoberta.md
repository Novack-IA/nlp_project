# C1 no 2×2: resultados

> Descoberta (train + val) e confirmação (`test`, aberta uma vez, na combinação principal). Gerado por `python -m featgrpo.interp.autointerp descoberta` (commit b299ef4, conjunto `qc_grammatical+validator_pass`). Os rótulos são explicações automáticas do Neuronpedia: ponto de partida, não verdade.

## Em resumo

1. **Existem features que separam a intenção da palavra.** Na combinação principal (L29 · pre_resposta) são **48** S_intent: reagem ao fato de o pedido ser nocivo e não à troca do gatilho. No SAE maior (262k), 60.
2. **Elas são robustas.** Quase as mesmas aparecem com outros filtros de qualidade; 48 de 48 sobrevivem ao controle de estilo; 81% não reagem ao gatilho solto (C3 atende); 48 de 48 valem nos 4 domínios; e 48 de 48 se mantêm em cada um dos 3 tipos de gatilho.
3. **Não existe feature "da palavra".** 0 S_lex e 0 S_mixed nas 8 combinações. A troca do gatilho muda a representação, mas cada gatilho mexe em features diferentes, e nenhuma muda do mesmo jeito na maioria dos grupos. A checagem de sanidade do pré-registro (seção 1, regra 4), que esperava efeito lexical forte no próprio token do gatilho, **não passa como foi escrita**: lá o maior \|d_L\| é 0,44.
4. **Mas a palavra vaza um pouco nas features de intenção** (exploratório). Em 90% delas, o gatilho empurra no mesmo sentido do pedido nocivo, com cerca de 9% do tamanho do efeito de intenção. Tirar esse empurrão não muda a recusa (C5, frente B: +1,2 pontos nos nocivos, IC [−0,8; +3,3]; +1,2 nos benignos que o modelo recusa).
5. **As features de intenção mais fortes ficam fora de S_intent** (exploratório). São 23 com efeito de intenção grande e esse vazamento pequeno do gatilho, que passa da margem de ±0,2 da regra. Entre as 8 mais fortes, 3 têm rótulo de recusa no Neuronpedia.
6. **Elas se confirmam em dados novos.** No `test` com gatilho inédito (155 grupos), 48 de 48 replicam (mesmo sinal, \|d_I\| ≥ 0,3): **C1 atende**. No 262k, 60 de 60: **C4a atende**.
7. **Lidas às cegas, a maioria é conceitual.** 19 de 20 fichas do top-20 foram julgadas conceituais por um LLM em contexto novo (mínimo 60%). Na conferência de Frederico (não cega), 13 de 20 (65%). Concordância com o LLM: 70% (κ 0,18).

**Leitura:** o modelo representa a intenção nociva separada da palavra, sobretudo no pre_resposta, mas a palavra ainda dá um empurrão pequeno na mesma direção. No C5 (frente B), desligar as S_intent muda a recusa a pedidos nocivos em −7,4 pontos, e desligar features aleatórias não reduz. Tirar o empurrão do gatilho não muda nada: ele existe na representação, mas não no comportamento.

## Onde está o sinal

Número de S_intent (SAE 16k):

| Camada | fim_pedido | pre_resposta |
|---|---|---|
| L9 | 6 | 14 |
| L17 | 7 | 36 |
| L22 | 8 | 40 |
| L29 | 6 | **48** (principal) |

Somando as camadas: 138 S_intent no **pre_resposta** contra 27 no fim_pedido, e o número cresce com a profundidade.

## O que as S_intent parecem ser (L29 · pre_resposta)

Mais ativas nos pedidos **nocivos**

| Feature | d_I | Rótulo do Neuronpedia |
|---|---|---|
| [791](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/791) | 1,79 | (sem explicação no Neuronpedia) |
| [1976](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1976) | 1,73 | (sem explicação no Neuronpedia) |
| [1812](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1812) | 1,38 | As an AI |
| [687](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/687) | 1,30 | (sem explicação no Neuronpedia) |
| [4144](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/4144) | 1,22 | multiplication variable |
| [5551](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/5551) | 1,16 | Okay or I after quote |
| [1335](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1335) | 1,05 | making false claims intentionally |
| [48](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/48) | 1,05 | depicting inappropriate or sexual content |
| [916](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/916) | 0,90 | (sem explicação no Neuronpedia) |
| [21](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/21) | 0,82 | illegal activities and injecting services |

Mais ativas nos pedidos **benignos**

| Feature | d_I | Rótulo do Neuronpedia |
|---|---|---|
| [1610](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1610) | −1,42 | let's or here's |
| [1235](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1235) | −1,34 | (sem explicação no Neuronpedia) |
| [125](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/125) | −0,99 | (sem explicação no Neuronpedia) |
| [5780](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/5780) | −0,94 | enthusiastic agreement and readiness |
| [533](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/533) | −0,79 | verbs followed by that |
| [1977](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1977) | −0,77 | informal address "Dude" |
| [9545](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/9545) | −0,76 | Conversational model responses |
| [3126](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/3126) | −0,76 | Okay, here |

Sem rótulo = o Neuronpedia não tem explicação para a feature.

## As features de intenção mais fortes ficaram fora de S_intent (L29 · pre_resposta)

| Feature | d_I | d_L | Rótulo do Neuronpedia |
|---|---|---|---|
| [5680](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/5680) | 3,39 | 0,24 | acknowledging understanding and offering disclaimers |
| [405](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/405) | 2,98 | 0,22 | (sem explicação no Neuronpedia) |
| [3266](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/3266) | 2,85 | 0,22 | cannot fulfill requests |
| [1145](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1145) | 2,41 | 0,18 | I pronoun |
| [342](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/342) | 2,32 | 0,23 | I understand you're |
| [2337](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/2337) | 2,29 | 0,23 | refusal to provide |
| [9819](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/9819) | 2,00 | 0,20 | ancient Hawaiian culture |
| [567](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/567) | 1,88 | 0,27 | sincerely / heartfelt / goodbye |

3 das 8 têm rótulo de recusa. O maior d_I aqui é 3,39, contra 1,79 da S_intent mais forte. O d_L delas é o vazamento do gatilho (item 4 do resumo). No CSV aparecem como "irrelevante", porque a regra do pré-registro não as classifica.

## Confirmação (`test` com gatilho inédito)

| Medida | Valor |
|---|---|
| S_intent que replicam | 48 de 48 |
| d_I no test ÷ d_I na descoberta (mediana) | 1,03 |
| Menor \|d_I\| no test | 0,45 |
| \|d_L\| no test (mediana) | 0,09 |

O efeito de intenção no `test` tem o mesmo tamanho que na descoberta, e o efeito do gatilho continua pequeno.

---

## Anexo

**A. Robustez (mesma combinação).** S_intent em cada conjunto; entre parênteses, quantas também são S_intent no principal.

- principal (`qc_grammatical` + validador): 48
- sem o validador: 47 (47)
- conjunto completo: 45 (45)
- `qc_pass_lenient` (1/3 dos grupos): 22 (20)
- C4(c), células de tamanho pareado: 46 (41); replicam 46 de 46; C1 atende
- C4(b), Q⁰ *retrieved*: 15 (13); replicam 15 de 15; C1 inconclusivo (182 grupos na descoberta (< 260))

**B. Controle `trig_only` (C3).** 5091 pares de frases em 1697 grupos; IC por grupo: 81% das S_intent sem efeito; IC por frase (sensibilidade): 81%. Sem S_lex, o controle positivo (S_lex reagem ao gatilho solto) não pode ser verificado.

**C. Por que não há S_lex** (`diagnostico_lexical_16k.json`). *Consistência*: 1 = o efeito aponta na mesma direção em todos os grupos; perto de 0 = cada grupo para um lado. Camada 29:

| Posição | Intenção | Gatilho |
|---|---|---|
| fim_pedido | 0,25 | 0,07 |
| pre_resposta | 0,71 | 0,17 |
| slot_forma | 0,17 | 0,10 |

Mesmo no token do gatilho (`slot_forma`), onde a troca muda muito a representação, o efeito do gatilho não tem direção comum entre os grupos. As outras camadas estão no JSON. Pela regra 4 da seção 1, a falta de efeito lexical forte ali significaria que "o SAE não representa o gatilho"; o diagnóstico mostra outra coisa: as features mudam com o gatilho, mas cada gatilho mexe em features diferentes. A segunda parte da regra (efeito de intenção zero no `slot_forma`) não se aplica ao plano B, que não tem lacuna de intenção: as linhas nociva e benigna são frases diferentes antes do gatilho.

**D. Vazamento do gatilho por camada (pre_resposta, exploratório).** Entre as features com intenção grande: % com d_L no mesmo sentido de d_I, e o tamanho típico d_L ÷ d_I.

| Camada | Mesmo sentido | d_L ÷ d_I |
|---|---|---|
| L9 | 38% | −0,10 |
| L17 | 83% | 0,10 |
| L22 | 91% | 0,09 |
| L29 | 90% | 0,09 |

Camadas com ≥ 80% no mesmo sentido: L17, L22, L29.

**E. Arquivos.** Todas as features, com ICs e q: `feature_classes_L{L}_{pos}_{sae}.csv`. Listas para as frentes B e D: `listas/`. Figura: `fig_dI_dL_16k.png`.
