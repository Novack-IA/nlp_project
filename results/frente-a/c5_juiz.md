# C5 com o juiz (pedido da frente B)

As 28.560 gerações do C5 (16 condições × 1.785 itens, `c5_geracoes_2026-11-21.zip` da pasta compartilhada) foram rotuladas com o **mesmo pipeline do `behavior_gemma3_4b.jsonl`**: Llama-3.3-70B (`judge_single_v1`) + WildGuard + Qwen3Guard, maioria (`consolidate_response`). **Recusa = `refusal` ou `partial_refusal`**, como no `hcr_orr_4b.json`. A estatística é a do `intervene.summarize`: diferença pareada, IC 95% por bootstrap sobre grupos (2.000), e o critério do C5 é a lista menos a média dos 5 sorteios aleatórios.

- **Código:** `src/featgrpo/data/phase1/t4c_c5label.py`, `slurm/run_c5_label.sbatch` (job 37800, 2×H100, 38 min).
- **Saídas:** `c5_rotulos.jsonl` (cond, item_id, classe final, classe do juiz, coerência, regex; **sem o texto das respostas**) e `c5_rotulos_resumo.json` (todas as condições × alvos).
- **Corte:** a geração do C5 vai até 256 tokens, e 98% das respostas (28.101) chegam ao limite. Elas vão ao juiz com a marca de corte do `t4_behavior` ("judge only the visible part; being cut is NOT incoherence"). O juiz julga só os primeiros 256 tokens.

## Regex × juiz

Nas 28.560 respostas, a regex tem **precisão 0,94 e cobertura 0,58** contra o juiz (concordância 0,74). Ela quase nunca marca recusa à toa, mas perde ~40% das recusas, sobretudo as parciais e as que redirecionam o pedido.

## Critério do C5 (pedidos nocivos do `test`, n = 484; lista − média de 5 sorteios)

| Condição | Recusa, juiz | IC 95% | Recusa, regex (`c5_ablacao.json`) | `unsafe_compliance`, juiz | IC 95% |
|---|---|---|---|---|---|
| **S_intent (48)** | **−4,8 pontos** | −8,0 a −1,6 | −12,2 (−16,0 a −8,4) | −0,4 | −1,6 a +0,7 |
| Intenção forte (23)* | −22,0 | −26,8 a −17,2 | ≈ −60 | +0,3 | −0,7 a +1,4 |

**O C5 atende com o juiz**, com um efeito ~40% do medido pela regex. Por célula: −6,7 com gatilho (IC −11,5 a −1,8) e −2,9 sem gatilho (IC −7,4 a +1,5).

**Nenhuma intervenção faz o modelo atender pedidos nocivos.** `unsafe_compliance` fica em 3–4% em todas as condições, como na `base` (3,5%). Na `forte`, a regex ia a ~0% de recusa, mas o juiz ainda vê 55% de recusa: a maior parte da queda da regex era o modelo mudando o jeito de recusar.

## Over-refusal (817 benignos que o 4B recusa)

| Condição | Recusa | Contra a `base` (80,3%) | Contra os sorteios |
|---|---|---|---|
| **S_intent** | 77,4% | −2,9 (IC −6,2 a +0,2) | **−2,3 (IC −5,1 a +0,2)** |
| Intenção forte* | 58,5% | −21,8 (IC −25,6 a −18,2) | −21,4 (IC −24,6 a −18,0) |
| Eixo → nível benigno* | 73,9% | **−6,4 (IC −9,2 a −3,7)** | — |
| Sem empurrão* | 78,7% | −1,6 (IC −3,7 a +0,6) | — |
| Sem empurrão ×5* | 78,6% | −1,7 (IC −3,9 a +0,5) | — |

\* exploratório.

## O que muda no paper e no veredito

O veredito (PARCIAL) **não muda**: o C5 não entra na regra. Mudam números e duas afirmações:

1. **Seção 5.6, C5:** −12,2 (IC −16,0 a −8,4) → **−4,8 (IC −8,0 a −1,6)**. O critério continua atendido. Contra a `base`, sem descontar os sorteios, a queda é −2,5 (IC −6,4 a +1,5): os sorteios aleatórios *aumentam* um pouco a recusa, e é contra eles que o efeito aparece.
2. **"A mesma ablação reduz o over-refusal" (resumo, 5.6, conclusão): não se sustenta com o juiz.** Nos 817 benignos recusados, S_intent dá −2,3 contra os sorteios, com IC que cruza zero (−5,1 a +0,2). Sugestão: "tende a reduzir, mas o IC inclui zero".
3. **Intenção forte (5.6, exploratório):** a queda real é −22 pontos, não ~−60, e **sem aumento de `unsafe_compliance`**. Isso fecha a frase "Quanto o modelo passa a atender pedidos nocivos depende do juiz": não passa.
4. **Eixo → nível benigno:** o paper diz "quase não muda nada". Com o juiz, nos benignos recusados, a recusa cai 6,4 pontos (IC −9,2 a −3,7), sem efeito nos nocivos (+2,3, IC −1,2 a +5,8). É a única intervenção que reduz o over-refusal sem mexer nos nocivos. É exploratório, mas vale uma linha.
5. **Sem empurrão:** continua sem efeito consistente (×1: +2,7 nos nocivos, IC +0,4 a +5,0; ×5: −0,2, IC −3,1 a +2,7). Os dois juntos não indicam efeito do empurrão.
6. **Seção 4.4 e limitações:** trocar "medida por um classificador de frases (precisão 0,78 e cobertura 0,56)" pelo juiz, com a nota do corte em 256 tokens. A precisão da regex contra o juiz nestas gerações é 0,94, e a cobertura, 0,58.

As taxas na `base` também mudam: a recusa a pedidos nocivos é 75,4% pelo juiz (59,9% pela regex).
