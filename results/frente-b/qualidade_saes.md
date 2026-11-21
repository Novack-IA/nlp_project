# Qualidade dos SAEs do Gemma Scope 2 nos nossos dados (B4)

> `src/featgrpo/interp/sae_quality.py`, RTX 4090, 2026-11-20. Os números estão em [`qualidade_saes.csv`](qualidade_saes.csv) e os argumentos em [`qualidade_saes.json`](qualidade_saes.json).
> Só **train + val**: o `test` não entra em decisão de camada.
> - FVE, L0 e features mortas: 1.200 pedidos (600 pares), 47.079 tokens, sem o BOS.
> - ΔNLL e KL: 300 respostas greedy do próprio Gemma-3-4B (até 64 tokens) a Q⁺/Q⁰ de outros 150 pares, 19.188 tokens. As respostas não foram salvas.
> - SAEs `l0_medium`, splice da reconstrução em todos os tokens menos o BOS.

| L | SAE | FVE | L0 | sem ativar nos nossos dados | NLL orig → SAE | ΔNLL | KL | ΔNLL camada zerada | recuperado |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 16k | 0,957 | 59 | 24% | 0,149 → 0,436 | 0,29 | 0,25 | 26,7 | 98,9% |
| 9 | 262k | 0,972 | 66 | 70% | 0,149 → 0,422 | 0,27 | 0,24 | 26,7 | 99,0% |
| 17 | 16k | 0,955 | 51 | 34% | 0,149 → 0,498 | 0,35 | 0,30 | 31,2 | 98,9% |
| 17 | 262k | 0,959 | 54 | 69% | 0,149 → 0,476 | 0,33 | 0,28 | 31,2 | 99,0% |
| 22 | 16k | 0,909 | 47 | 28% | 0,149 → 0,353 | 0,20 | 0,18 | 15,4 | 98,7% |
| 22 | 262k | 0,922 | 49 | 81% | 0,149 → 0,334 | 0,18 | 0,16 | 15,4 | 98,8% |
| 29 | 16k | 0,843 | 47 | 18% | 0,149 → 0,298 | 0,15 | 0,13 | 33,9 | 99,6% |
| 29 | 262k | 0,876 | 49 | 74% | 0,149 → 0,271 | 0,12 | 0,12 | 33,9 | 99,6% |

## Leitura

- **Nenhuma camada sai da análise.** Os 8 SAEs reconstroem bem os nossos pedidos:
  - FVE de 0,84 a 0,97;
  - L0 próximo do nominal (~60);
  - ~99% da perda recuperada em relação a zerar a camada.
- **O ΔNLL (0,12–0,35 nat/token) é moderado.** Parece grande em relação à NLL original porque esta é quase zero: o modelo está pontuando o próprio texto greedy. Em valor absoluto, está na faixa típica de SAEs de `resid_post` com L0 ≈ 50. **L17 é a camada com mais perda** (ΔNLL 0,35, KL 0,30). Isso pede atenção na C5 (intervenção): parte do efeito de qualquer splice em L17 vem só do erro de reconstrução. Por isso o `intervene.py` vai somar o erro de reconstrução de volta.
- **O 262k é só um pouco melhor que o 16k** (+0,01–0,03 de FVE, −0,01–0,03 de ΔNLL).
- **"Sem ativar nos nossos dados" não significa features mortas:** são features que não dispararam em nenhum dos ~47k tokens de pedidos de segurança (18–34% no 16k, 69–81% no 262k). É esperado num domínio estreito. Para a frente C, a consequência é que no 262k a maior parte do dicionário é irrelevante aqui. O filtro de frequência mínima (feature ativa em ≥ 5% dos itens) reduz bastante o número de testes da correção BH.

## Por que não usar as respostas r4/r7 do dataset

Elas são do Gemma-4-12B. O 4B erra alguns tokens delas com confiança extrema (por exemplo, prevê "I *understand*" onde o texto tem "I *cannot*": 18 nat em um token). O splice do SAE "amacia" essas previsões confiantes demais, e o ΔNLL sai **negativo** (−0,3 a −1,3). Isso não mede reconstrução. Nas respostas do próprio 4B, o comportamento é o esperado.

## Nas duas posições principais do C1: `inst_last` e `post_last`

> `sae_quality.py --skip-full --positions inst_last post_last --widths 16k 262k`, 2026-11-21. Os números estão em [`qualidade_saes_posicoes.csv`](qualidade_saes_posicoes.csv).
> - Mesmos 1.200 pedidos e mesmas 300 respostas do próprio 4B da tabela acima.
> - O splice troca **só o token da posição**, e o ΔNLL é medido nas respostas.
> - Foi pedido pela frente C no pré-registro: é sanidade, não serve para escolher camada nem SAE.

| L | Posição | SAE | cos(x, rec) | erro relativo | FVE entre itens | L0 | ΔNLL | KL | ΔNLL zerando o token |
|---|---|---|---|---|---|---|---|---|---|
| 9 | inst_last | 16k / 262k | 0,999 / 0,999 | 0,2% / 0,1% | 0,89 / 0,89 | 52 / 54 | 0,009 / 0,010 | 0,010 / 0,011 | 0,042 |
| 9 | post_last | 16k / 262k | 1,000 / 1,000 | 0,1% / 0,1% | 0,75 / 0,76 | 30 / 33 | 0,007 / 0,011 | 0,008 / 0,012 | 0,118 |
| 17 | inst_last | 16k / 262k | 0,999 / 0,999 | 0,2% / 0,2% | 0,67 / 0,69 | 65 / 64 | 0,001 / 0,001 | 0,002 / 0,002 | 0,008 |
| 17 | post_last | 16k / 262k | 0,999 / 1,000 | 0,1% / 0,1% | 0,62 / 0,66 | 51 / 50 | 0,005 / 0,005 | 0,006 / 0,007 | 0,371 |
| 22 | inst_last | 16k / 262k | 0,995 / 0,996 | 0,9% / 0,9% | 0,59 / 0,62 | 63 / 64 | 0,000 / 0,000 | 0,002 / 0,001 | 0,003 |
| 22 | post_last | 16k / 262k | 0,998 / 0,998 | 0,4% / 0,4% | 0,58 / 0,64 | 49 / 49 | 0,002 / 0,000 | 0,005 / 0,005 | 0,227 |
| 29 | inst_last | 16k / 262k | 0,990 / 0,992 | 1,9% / 1,6% | 0,56 / 0,63 | 55 / 58 | 0,000 / 0,000 | 0,001 / 0,001 | 0,000 |
| 29 | post_last | 16k / 262k | 0,993 / 0,994 | 1,4% / 1,1% | 0,55 / 0,64 | 56 / 61 | 0,007 / 0,006 | 0,004 / 0,004 | 0,260 |

**Leitura:**
- **O vetor é reconstruído quase perfeitamente:** cosseno ≥ 0,99 e erro ≤ 2% de ‖x‖². Trocar só esse token pela reconstrução muda pouquíssimo o modelo (ΔNLL ≤ 0,011).
- **A variação de um pedido para outro é reconstruída pior: 55–89%, caindo com a profundidade.** O token é sempre o mesmo ("?", "." ou "\n"), então a maior parte do vetor é comum a todos os pedidos, e o SAE acerta essa parte. A diferença entre pedidos é pequena, e **11–45% dela fica no erro do SAE**. Parte do sinal de intenção pode, portanto, não aparecer em nenhuma feature. Isso entra nas limitações do C1. Também é um motivo para comparar com os probes densos (frente D), que usam o vetor inteiro. O 262k reduz um pouco essa perda (+0,01–0,09).
- **O L0 fica perto do nominal (~50–65), salvo `post_last` em L9 (~30).** Diferente da média de vários tokens, um token só está na distribuição do SAE.
- **Zerar `inst_last` em L22 e L29 não muda a resposta (ΔNLL 0,000–0,003), enquanto zerar `post_last` muda (0,23–0,26).** Nas camadas tardias, a informação de `inst_last` já foi lida pelos tokens seguintes. Isso combina com a divisão do pré-registro: o pedido fica no fim da instrução e a decisão em `post_last`. **Para a C5:** intervir só em `inst_last` deve ter efeito apenas em L9/L17; em L22/L29, intervir em `post_last` ou no sufixo inteiro.
