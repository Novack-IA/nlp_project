# SAE próprio (A5): Gemma-3-4B-IT, resid_post

| | |
|---|---|
| **Arquitetura** | BatchTopK, largura 16.384, k = 64. Salvo como JumpReLU com o limiar global aprendido, no formato SAELens (`cfg.json` + `sae_weights.safetensors`). O fator de normalização já está dobrado nos pesos, então o SAE recebe a ativação crua. |
| **Camada** | `resid_post` L = saída do bloco L = `hidden_states[L+1]`. Token BOS excluído. |
| **Treino** | 200M tokens, contexto de 1.024, lote de 4.096 tokens, Adam com lr 2e-4 (warmup de 1k passos e decaimento nos últimos 20%), perda auxiliar AuxK para features mortas |
| **Corpus** | UltraChat-200k (208k conversas) + PKU-SafeRLHF (132k pares pedido/resposta) + OR-Bench-80k (80k pedidos), **sem nenhum pedido da Fase 0** (5.612 excluídos) |
| **Código** | `src/featgrpo/interp/train_sae.py` (treino e avaliação) e `src/featgrpo/interp/prep_corpus.py` |
| **Pesos** | `/raid/user_gustavonovack/nlp_artifacts/sae_proprio/gemma-3-4b-it_L{17,22,29}_btk16k/sae/` (~336 MB cada; fora do git). Peça ao Gustavo para copiar para a pasta compartilhada. |

## Avaliação (`results/frente-a/sae/eval_L*.json`)

Os conjuntos de avaliação são:
- **UltraChat test**: 400 conversas, cerca de 353k tokens;
- **nossos pedidos**: 400 pedidos dos pares, splits `train` e `val`, cerca de 15k tokens.

O ΔNLL é a perda do modelo com o residual da camada substituído pela reconstrução, menos a perda original, medida em todos os tokens exceto o BOS.

| Camada | SAE | Conjunto | FVU | L0 | ΔNLL |
|---|---|---|---|---|---|
| 17 | **próprio** | nossos pedidos | **0,027** | 54 | +0,16 |
| 17 | Gemma Scope 2, 16k, L0 medium | nossos pedidos | 0,046 | 52 | +0,11 |
| 17 | **próprio** | UltraChat | **0,025** | 65 | +0,38 |
| 17 | Gemma Scope 2, 16k, L0 medium | UltraChat | 0,044 | 61 | −0,79 ⚠️ |
| 22 | **próprio** | nossos pedidos | **0,061** | 52 | **+0,21** |
| 22 | Gemma Scope 2, 16k, L0 medium | nossos pedidos | 0,095 | 47 | +0,55 |
| 22 | Gemma Scope 2, 16k, L0 small | nossos pedidos | 0,126 | 17 | +0,58 |
| 22 | **próprio** | UltraChat | **0,070** | 65 | +0,36 |
| 22 | Gemma Scope 2, 16k, L0 medium | UltraChat | 0,109 | 59 | −0,35 ⚠️ |
| 29 | **próprio** | nossos pedidos | **0,112** | 50 | **+0,46** |
| 29 | Gemma Scope 2, 16k, L0 medium | nossos pedidos | 0,163 | 48 | +0,69 |
| 29 | Gemma Scope 2, 16k, L0 small | nossos pedidos | 0,208 | 16 | +1,07 |
| 29 | **próprio** | UltraChat | **0,145** | 65 | +0,52 |
| 29 | Gemma Scope 2, 16k, L0 medium | UltraChat | 0,217 | 59 | −0,15 ⚠️ |

**Leitura**
- **Com L0 parecido, o SAE próprio reconstrói melhor** nas três camadas: o FVU fica entre 30% e 45% menor. A dificuldade cresce com a profundidade: o FVU fica em torno de 0,03 na camada 17, 0,06 na 22 e 0,11 na 29.
- **Nas camadas 22 e 29, nos nossos pedidos, ele também preserva melhor o comportamento do modelo:** ΔNLL +0,21 contra +0,55 na 22 e +0,46 contra +0,69 na 29. O corpus de treino inclui pedidos de segurança.
- **⚠️ Anomalia:** no UltraChat, que tem contextos longos, a reconstrução do Gemma Scope 2 *reduz* a perda do modelo. Isso não deveria acontecer e se repete nas três camadas. Hipóteses:
  1. tokens de norma muito alta no Gemma 3 (os "sinks" de atenção) que o SAE "conserta";
  2. diferença de pré-processamento esperada pelo Gemma Scope 2 (normalização, posição do hook, tokens de template).

  **Pendência para a frente B (B4):** checar a convenção de entrada do Gemma Scope 2 antes de usar o ΔNLL dele. A FVU dele está razoável, então a reconstrução em si funciona.
- **Limitações:**
  - "Fração morta no conjunto" conta só as features que não disparam naquele conjunto pequeno. Não é a fração morta global.
  - As avaliações usam no máximo 400 sequências.
