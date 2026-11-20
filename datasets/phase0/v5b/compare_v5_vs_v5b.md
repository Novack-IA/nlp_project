# Comparação `phase0_b200` (A) × `phase0_v5b` (B)

## Tamanho

| | A | B |
|---|---|---|
| sementes (P1) | 1500 | 2911 |
| grupos (P2) | 1440 | 2806 |
| válidos (P5) | 425 | 837 |
| export final (P8) | 425 | 500 |
| final por domínio | bias: 113, cyber: 116, violence: 104, privacy: 92 | bias: 125, cyber: 125, violence: 125, privacy: 125 |

## Taxa de validade por domínio (P5)

| domínio | A | B (todas) | B: sementes antigas | B: sementes novas |
|---|---|---|---|---|
| bias | 113/359 (31%) | 186/635 (29%) | 102/359 (28%) | 84/276 (30%) |
| cyber | 116/367 (32%) | 237/733 (32%) | 115/367 (31%) | 122/366 (33%) |
| privacy | 92/361 (25%) | 210/720 (29%) | 92/361 (25%) | 118/359 (33%) |
| violence | 104/353 (29%) | 204/718 (28%) | 105/353 (30%) | 99/365 (27%) |

Estabilidade dos grupos antigos: 397 válidos em A e em B; 28 só em A; 17 só em B (variação de rótulo entre rodadas).

## Qualidade do export final

| métrica | A | B | B: antigas | B: novas |
|---|---|---|---|---|
| força do rótulo: itens com consenso 3/3 | 0.956 | 0.954 | 0.957 | 0.951 |
| Q⁰ por retrieval (prompts reais) | 0.193 | 0.184 | 0.198 | 0.171 |
| cos médio Q⁺–Q⁰ (retrieval) | 0.649 | 0.646 | 0.644 | 0.647 |
| Q⁰ com hardness > 0 | 0.038 | 0.018 | 0.016 | 0.019 |
| HCR base (r_natural, Q⁺) | 0.005 | 0.008 | 0.012 | 0.004 |
| ORR base (r_natural, Q⁰) | 0.016 | 0.004 | 0.008 | 0.000 |
| severidade PKU | 2: 382, 3: 43 | 2: 454, 3: 46 | 2: 218, 3: 25 | 2: 236, 3: 21 |
| sub-buckets bias | lgbt: 13, race: 13, other: 87 | lgbt: 7, race: 25, other: 93 | lgbt: 7, race: 11, other: 49 | race: 14, other: 44 |

Taxa de `disputed` nas respostas (P4, todos os grupos): A = 0.031; B = 0.029; B antigas = 0.030; B novas = 0.028

Sobreposição dos exports finais: 232 grupos de A estão no final de B.

