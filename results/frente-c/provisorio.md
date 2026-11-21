# Modo provisório nos pares originais (C1 provisório)

> **Diagnóstico, fora do veredito** (pré-registro, regra 4 da seção 5). Nos pares originais, só o estilo do texto já separa nocivo de benigno com AUROC 0,88 (`results/frente-d/baseline_lexical.json`), então features "de intenção" aqui podem ser features de estilo ou de assunto. Serve para depurar o código e ter uma primeira ideia. Só a descoberta (train + val).
> Gerado por `python -m featgrpo.interp.autointerp provisorio` a partir de `provisorio_16k.json` (commit 8b1439f). Os rótulos são as explicações automáticas do Neuronpedia (`gemma-3-4b-it/{L}-gemmascope-2-res-16k`), escritas por um LLM a partir de texto genérico: são ponto de partida, não verdade.

Intenção provisória = ativação em Q⁺ − ativação em Q⁰, pareada por grupo. "Efeito grande" = \|d\| ≥ 0,5 e q < 0,01 (BH na família das 8 combinações). "Sobrevivem ao estilo" = controle de estilo (`DESVIOS.md`, 2026-11-21).

## Resumo

| Combinação | Pares na descoberta | Efeito grande | Mais ativas em Q⁺ | Mais ativas em Q⁰ | Sobrevivem ao estilo | R² do estilo (mediana) |
|---|---|---|---|---|---|---|
| L9 · fim_pedido | 1784 | 9 | 8 | 1 | 9 de 9 | 0,05 |
| L9 · pre_resposta | 2081 | 26 | 19 | 7 | 26 de 26 | 0,16 |
| L17 · fim_pedido | 1784 | 16 | 12 | 4 | 16 de 16 | 0,04 |
| L17 · pre_resposta | 2081 | 56 | 20+ | 20+ | 56 de 56 | 0,08 |
| L22 · fim_pedido | 1784 | 12 | 10 | 2 | 12 de 12 | 0,04 |
| L22 · pre_resposta | 2081 | 64 | 20+ | 20+ | 64 de 64 | 0,09 |
| L29 · fim_pedido | 1784 | 7 | 6 | 1 | 7 de 7 | 0,04 |
| L29 · pre_resposta | 2081 | 63 | 20+ | 18 | 63 de 63 | 0,07 |

"20+" = há mais de 20 features desse lado; abaixo aparecem só as 20 de maior \|d\|.

## L9 · fim_pedido

**Mais ativas em Q⁺ (nocivo)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [3812](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/3812) | 0,93 | 72% | deceptive manipulation |
| 2 | [4449](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/4449) | 0,60 | 34% | making generalizations |
| 3 | [4013](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/4013) | 0,59 | 33% | VPN provider |
| 4 | [10809](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/10809) | 0,57 | 45% | can be done easily |
| 5 | [279](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/279) | 0,57 | 100% | (sem explicação no Neuronpedia) |
| 6 | [14842](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/14842) | 0,55 | 66% | how to perform an action |
| 7 | [87](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/87) | 0,53 | 95% | (sem explicação no Neuronpedia) |
| 8 | [4963](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/4963) | 0,52 | 32% | sexually explicit |

**Mais ativas em Q⁰ (benigno)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [270](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/270) | −1,01 | 100% | (sem explicação no Neuronpedia) |

## L9 · pre_resposta

**Mais ativas em Q⁺ (nocivo)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [5624](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/5624) | 1,46 | 95% | YAML configuration |
| 2 | [174](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/174) | 1,03 | 100% | (sem explicação no Neuronpedia) |
| 3 | [11712](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/11712) | 0,97 | 89% | preventive precautions |
| 4 | [759](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/759) | 0,97 | 100% | (sem explicação no Neuronpedia) |
| 5 | [2737](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/2737) | 0,95 | 82% | helpline and support numbers |
| 6 | [13](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/13) | 0,80 | 100% | sentence endings |
| 7 | [436](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/436) | 0,78 | 100% | (sem explicação no Neuronpedia) |
| 8 | [1610](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/1610) | 0,78 | 100% | Okay |
| 9 | [4647](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/4647) | 0,72 | 86% | It, You, Here, Here, Cru, You, Kam, Here |
| 10 | [5524](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/5524) | 0,70 | 39% | hate speech and discrimination |
| 11 | [312](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/312) | 0,65 | 100% | (sem explicação no Neuronpedia) |
| 12 | [1499](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/1499) | 0,65 | 84% | (sem explicação no Neuronpedia) |
| 13 | [3](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/3) | 0,62 | 99% | blog post Review |
| 14 | [42](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/42) | 0,62 | 100% | AI model response |
| 15 | [75](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/75) | 0,61 | 35% | items for purchase |
| 16 | [216](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/216) | 0,61 | 84% | (sem explicação no Neuronpedia) |
| 17 | [294](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/294) | 0,57 | 100% | (sem explicação no Neuronpedia) |
| 18 | [344](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/344) | 0,56 | 28% | (sem explicação no Neuronpedia) |
| 19 | [669](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/669) | 0,54 | 100% | (sem explicação no Neuronpedia) |

**Mais ativas em Q⁰ (benigno)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [314](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/314) | −1,56 | 100% | (sem explicação no Neuronpedia) |
| 2 | [578](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/578) | −1,02 | 100% | quantifiers and articles |
| 3 | [449](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/449) | −1,01 | 95% | (sem explicação no Neuronpedia) |
| 4 | [383](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/383) | −0,96 | 100% | hey! conversational opener |
| 5 | [2816](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/2816) | −0,92 | 59% | international context or locations |
| 6 | [14557](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/14557) | −0,72 | 100% | mathematical explanation starters |
| 7 | [323](https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/323) | −0,52 | 100% | (sem explicação no Neuronpedia) |

## L17 · fim_pedido

**Mais ativas em Q⁺ (nocivo)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [384](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/384) | 1,07 | 75% | (sem explicação no Neuronpedia) |
| 2 | [21](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/21) | 0,96 | 77% | absence and negation |
| 3 | [875](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/875) | 0,94 | 64% | (sem explicação no Neuronpedia) |
| 4 | [2516](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2516) | 0,91 | 57% | (sem explicação no Neuronpedia) |
| 5 | [4103](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4103) | 0,90 | 57% | model refusing harmful content |
| 6 | [4886](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4886) | 0,90 | 61% | clandestine illegal activities |
| 7 | [117](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/117) | 0,76 | 84% | (sem explicação no Neuronpedia) |
| 8 | [2537](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2537) | 0,70 | 65% | (sem explicação no Neuronpedia) |
| 9 | [1249](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1249) | 0,65 | 58% | terror and violent actions |
| 10 | [891](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/891) | 0,64 | 57% | (sem explicação no Neuronpedia) |
| 11 | [252](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/252) | 0,63 | 65% | (sem explicação no Neuronpedia) |
| 12 | [7518](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/7518) | 0,50 | 28% | resemble scratching! |

**Mais ativas em Q⁰ (benigno)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [48](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/48) | −0,72 | 100% | (sem explicação no Neuronpedia) |
| 2 | [234](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/234) | −0,55 | 90% | (sem explicação no Neuronpedia) |
| 3 | [3](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3) | −0,54 | 37% | abstract contexts and consequences |
| 4 | [2120](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2120) | −0,52 | 34% | presenting options |

## L17 · pre_resposta

**Mais ativas em Q⁺ (nocivo)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [9690](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/9690) | 3,06 | 99% | medical and chemical terms |
| 2 | [4103](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4103) | 3,04 | 100% | model refusing harmful content |
| 3 | [891](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/891) | 2,59 | 99% | (sem explicação no Neuronpedia) |
| 4 | [1395](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1395) | 2,00 | 88% | (sem explicação no Neuronpedia) |
| 5 | [5516](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/5516) | 1,56 | 73% | detect and infer |
| 6 | [21](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/21) | 1,42 | 77% | absence and negation |
| 7 | [4886](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4886) | 1,41 | 75% | clandestine illegal activities |
| 8 | [101](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/101) | 1,41 | 78% | (sem explicação no Neuronpedia) |
| 9 | [10916](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/10916) | 1,36 | 74% | cannot fulfill |
| 10 | [153](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/153) | 1,35 | 93% | (sem explicação no Neuronpedia) |
| 11 | [2301](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2301) | 1,13 | 59% | disclaimers and important notices |
| 12 | [1086](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1086) | 1,10 | 70% | our / our / our |
| 13 | [5784](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/5784) | 1,09 | 67% | disclaimers in multiple languages |
| 14 | [2546](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2546) | 0,97 | 65% | (sem explicação no Neuronpedia) |
| 15 | [1249](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1249) | 0,91 | 65% | terror and violent actions |
| 16 | [6515](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/6515) | 0,90 | 54% | avoiding negative outcomes |
| 17 | [107](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/107) | 0,90 | 100% | greetings and polite phrases |
| 18 | [3115](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3115) | 0,77 | 42% | model's responses |
| 19 | [7371](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/7371) | 0,71 | 39% | assessing qualities or inappropriateness |
| 20 | [50](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/50) | 0,69 | 98% | only or exceptions |

**Mais ativas em Q⁰ (benigno)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [502](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/502) | −1,84 | 98% | (sem explicação no Neuronpedia) |
| 2 | [3419](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3419) | −1,81 | 90% | Okay, here's an article |
| 3 | [3329](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3329) | −1,36 | 82% | Okay, let's |
| 4 | [1935](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1935) | −1,30 | 69% | risk-adjusted return |
| 5 | [3386](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3386) | −1,12 | 76% | (sem explicação no Neuronpedia) |
| 6 | [3094](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3094) | −1,07 | 98% | Okay, let's |
| 7 | [279](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/279) | −1,03 | 100% | (sem explicação no Neuronpedia) |
| 8 | [1003](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1003) | −0,88 | 53% | (sem explicação no Neuronpedia) |
| 9 | [471](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/471) | −0,88 | 64% | (sem explicação no Neuronpedia) |
| 10 | [14967](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/14967) | −0,77 | 38% | frustration and questions |
| 11 | [641](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/641) | −0,77 | 98% | VBA code explanations |
| 12 | [2535](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2535) | −0,76 | 52% | initiating structured business responses |
| 13 | [215](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/215) | −0,70 | 82% | (sem explicação no Neuronpedia) |
| 14 | [589](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/589) | −0,68 | 100% | introduces facts or historical context |
| 15 | [10860](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/10860) | −0,64 | 42% | poignant emotions and vulnerability |
| 16 | [883](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/883) | −0,63 | 100% | (sem explicação no Neuronpedia) |
| 17 | [4149](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4149) | −0,62 | 28% | intensifiers for scope or scale |
| 18 | [3801](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3801) | −0,61 | 31% | after but here |
| 19 | [8560](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/8560) | −0,58 | 27% | explaining or breaking down subjects |
| 20 | [3769](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3769) | −0,53 | 26% | explicitly describing characters |

## L22 · fim_pedido

**Mais ativas em Q⁺ (nocivo)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [604](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/604) | 1,10 | 85% | harmful unethical behavior |
| 2 | [441](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/441) | 0,70 | 51% | (sem explicação no Neuronpedia) |
| 3 | [8861](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/8861) | 0,60 | 46% | illegal activities and pornography |
| 4 | [2309](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/2309) | 0,58 | 41% | (sem explicação no Neuronpedia) |
| 5 | [9542](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/9542) | 0,56 | 44% | negative actions or outcomes |
| 6 | [561](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/561) | 0,56 | 31% | being or becoming |
| 7 | [6768](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/6768) | 0,54 | 34% | illegal acts and crimes |
| 8 | [1138](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1138) | 0,53 | 48% | (sem explicação no Neuronpedia) |
| 9 | [2987](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/2987) | 0,53 | 40% | (sem explicação no Neuronpedia) |
| 10 | [800](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/800) | 0,51 | 41% | (sem explicação no Neuronpedia) |

**Mais ativas em Q⁰ (benigno)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [250](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/250) | −0,71 | 95% | (sem explicação no Neuronpedia) |
| 2 | [883](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/883) | −0,64 | 53% | explaining complex topics |

## L22 · pre_resposta

**Mais ativas em Q⁺ (nocivo)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [6369](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/6369) | 2,55 | 97% | refusing explicit content requests |
| 2 | [807](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/807) | 2,51 | 96% | (sem explicação no Neuronpedia) |
| 3 | [2309](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/2309) | 2,46 | 99% | (sem explicação no Neuronpedia) |
| 4 | [5344](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/5344) | 2,44 | 93% | response to harmful prompts |
| 5 | [1579](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1579) | 2,13 | 90% | text HOME to |
| 6 | [619](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/619) | 2,03 | 93% | legal disclaimer advice |
| 7 | [1028](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1028) | 2,01 | 93% | declining harmful requests |
| 8 | [1083](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1083) | 2,01 | 99% | Okay, let's explain honestly |
| 9 | [1096](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1096) | 1,98 | 87% | first-person pronouns |
| 10 | [604](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/604) | 1,82 | 88% | harmful unethical behavior |
| 11 | [6446](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/6446) | 1,64 | 87% | expressing regret or unfortunate news |
| 12 | [392](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/392) | 1,51 | 93% | (sem explicação no Neuronpedia) |
| 13 | [3114](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/3114) | 1,49 | 78% | offering help or resources |
| 14 | [7876](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/7876) | 1,23 | 62% | sequences followed by you or we |
| 15 | [48](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/48) | 1,18 | 100% | electric vehicles and futures |
| 16 | [5639](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/5639) | 1,17 | 65% | I cannot and will not |
| 17 | [1929](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1929) | 1,15 | 60% | opening quote or enclosure |
| 18 | [15198](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/15198) | 1,14 | 64% | friend and buddy greetings |
| 19 | [6217](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/6217) | 1,11 | 76% | cannot fulfill request |
| 20 | [16166](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/16166) | 1,11 | 65% | (sem explicação no Neuronpedia) |

**Mais ativas em Q⁰ (benigno)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [6199](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/6199) | −1,88 | 92% | Okay, let's break down |
| 2 | [491](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/491) | −1,71 | 89% | (sem explicação no Neuronpedia) |
| 3 | [49](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/49) | −1,25 | 73% | positive evaluations |
| 4 | [1658](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1658) | −1,24 | 82% | "let" |
| 5 | [41](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/41) | −1,16 | 100% | acknowledges questions and starts answering |
| 6 | [13813](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/13813) | −1,02 | 59% | okay, alright, absolutely |
| 7 | [12400](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/12400) | −0,88 | 51% | congratulations |
| 8 | [3419](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/3419) | −0,82 | 61% | Okay, here |
| 9 | [1488](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1488) | −0,80 | 53% | Okay |
| 10 | [5568](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/5568) | −0,73 | 38% | dude, guy, crap |
| 11 | [534](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/534) | −0,72 | 66% | numbered lists |
| 12 | [1120](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1120) | −0,68 | 47% | title or heading markers |
| 13 | [1415](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1415) | −0,67 | 82% | (sem explicação no Neuronpedia) |
| 14 | [1171](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1171) | −0,60 | 61% | (sem explicação no Neuronpedia) |
| 15 | [1655](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1655) | −0,59 | 29% | novels and fiction |
| 16 | [1177](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1177) | −0,58 | 28% | characters and protagonists |
| 17 | [4019](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/4019) | −0,56 | 90% | Okay, acknowledging question |
| 18 | [1830](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/1830) | −0,55 | 29% | introduces lists |
| 19 | [281](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/281) | −0,54 | 100% | (sem explicação no Neuronpedia) |
| 20 | [602](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/602) | −0,53 | 41% | programming languages |

## L29 · fim_pedido

**Mais ativas em Q⁺ (nocivo)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [21](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/21) | 0,85 | 83% | illegal activities and injecting services |
| 2 | [441](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/441) | 0,81 | 99% | (sem explicação no Neuronpedia) |
| 3 | [1415](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1415) | 0,73 | 56% | (sem explicação no Neuronpedia) |
| 4 | [1335](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1335) | 0,66 | 38% | making false claims intentionally |
| 5 | [1813](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1813) | 0,58 | 35% | for malicious |
| 6 | [735](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/735) | 0,51 | 84% | (sem explicação no Neuronpedia) |

**Mais ativas em Q⁰ (benigno)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [1082](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1082) | −0,51 | 31% | Okay, let's tackle |

## L29 · pre_resposta

**Mais ativas em Q⁺ (nocivo)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [5680](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/5680) | 3,01 | 99% | acknowledging understanding and offering disclaimers |
| 2 | [405](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/405) | 2,71 | 98% | (sem explicação no Neuronpedia) |
| 3 | [3266](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/3266) | 2,65 | 99% | cannot fulfill requests |
| 4 | [1145](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1145) | 2,25 | 94% | I pronoun |
| 5 | [342](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/342) | 2,20 | 93% | I understand you're |
| 6 | [2337](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/2337) | 2,16 | 94% | refusal to provide |
| 7 | [9819](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/9819) | 1,90 | 82% | ancient Hawaiian culture |
| 8 | [567](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/567) | 1,80 | 89% | sincerely / heartfelt / goodbye |
| 9 | [3571](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/3571) | 1,65 | 79% | code installation commands |
| 10 | [955](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/955) | 1,64 | 80% | offers to help |
| 11 | [1976](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1976) | 1,62 | 84% | (sem explicação no Neuronpedia) |
| 12 | [791](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/791) | 1,62 | 88% | (sem explicação no Neuronpedia) |
| 13 | [745](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/745) | 1,61 | 81% | (sem explicação no Neuronpedia) |
| 14 | [660](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/660) | 1,61 | 96% | , let |
| 15 | [6086](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/6086) | 1,56 | 82% | (sem explicação no Neuronpedia) |
| 16 | [9394](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/9394) | 1,51 | 77% | structured lists starting with "I" or "1" |
| 17 | [975](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/975) | 1,39 | 83% | greetings |
| 18 | [4009](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/4009) | 1,38 | 73% | players like Best |
| 19 | [1812](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1812) | 1,27 | 73% | As an AI |
| 20 | [714](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/714) | 1,18 | 75% | (sem explicação no Neuronpedia) |

**Mais ativas em Q⁰ (benigno)**

| # | Feature | d | Ativa em | Rótulo do Neuronpedia |
|---|---|---|---|---|
| 1 | [1610](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1610) | −1,27 | 66% | let's or here's |
| 2 | [1235](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1235) | −1,23 | 71% | (sem explicação no Neuronpedia) |
| 3 | [13924](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/13924) | −1,02 | 82% | Okay, this is |
| 4 | [5780](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/5780) | −0,88 | 56% | enthusiastic agreement and readiness |
| 5 | [125](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/125) | −0,86 | 92% | (sem explicação no Neuronpedia) |
| 6 | [2869](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/2869) | −0,83 | 74% | giving you |
| 7 | [533](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/533) | −0,74 | 50% | verbs followed by that |
| 8 | [1977](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1977) | −0,71 | 41% | informal address "Dude" |
| 9 | [3126](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/3126) | −0,71 | 44% | Okay, here |
| 10 | [9545](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/9545) | −0,69 | 58% | Conversational model responses |
| 11 | [694](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/694) | −0,66 | 38% | (sem explicação no Neuronpedia) |
| 12 | [796](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/796) | −0,63 | 32% | (sem explicação no Neuronpedia) |
| 13 | [5194](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/5194) | −0,62 | 32% | affirmative certainty |
| 14 | [436](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/436) | −0,61 | 31% | (sem explicação no Neuronpedia) |
| 15 | [348](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/348) | −0,60 | 100% | (sem explicação no Neuronpedia) |
| 16 | [6436](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/6436) | −0,57 | 38% | Okay, here's |
| 17 | [1293](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/1293) | −0,56 | 57% | (sem explicação no Neuronpedia) |
| 18 | [6797](https://www.neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-16k/6797) | −0,52 | 27% | greetings like dear and hi |
