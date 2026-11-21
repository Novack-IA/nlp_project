# SAE próprio (A5): Gemma-3-4B-IT, resid_post

| | |
|---|---|
| **Arquitetura** | BatchTopK, largura 16.384, k = 64. Salvo como JumpReLU com o limiar global aprendido, no formato SAELens (`cfg.json` + `sae_weights.safetensors`). O fator de normalização já está dobrado nos pesos, então o SAE recebe a ativação crua. |
| **Camada** | `resid_post` L = saída do bloco L = `hidden_states[L+1]`. Token BOS excluído. |
| **Treino** | 200M tokens, contexto de 1.024, lote de 4.096 tokens, Adam com lr 2e-4 (warmup de 1k passos e decaimento nos últimos 20%), perda auxiliar AuxK para features mortas |
| **Corpus** | UltraChat-200k (208k conversas) + PKU-SafeRLHF (132k pares pedido/resposta) + OR-Bench-80k (80k pedidos), **sem nenhum pedido da Fase 0** (5.612 excluídos) |
| **Código** | `src/featgrpo/interp/train_sae.py` (treino e avaliação) e `src/featgrpo/interp/prep_corpus.py` |
| **Pesos** | Pasta compartilhada do time em `artifacts/sae_proprio/gemma-3-4b-it_L{17,22,29}_btk16k/` (~336 MB cada; fora do git). Original em `/raid/user_gustavonovack/nlp_artifacts/sae_proprio/.../sae/`. |

## Como carregar

Cada pasta tem dois arquivos de pesos com o **mesmo SAE** em duas convenções:

| Arquivo | Convenção | Para quem |
|---|---|---|
| `params.safetensors` | **Gemma Scope 2**: chaves `w_enc, b_enc, w_dec, b_dec, threshold`; `pre = x @ w_enc + b_enc` (o `b_dec` já está dobrado no `b_enc`) | `JumpReLUSAE` de `interp/common.py` (frentes B e C) |
| `sae_weights.safetensors` + `cfg.json` | **SAELens** (`jumprelu`, `apply_b_dec_to_input = true`): `pre = (x − b_dec) @ W_enc + b_enc` | `SAE.load_from_disk(pasta)` do SAELens |

⚠️ **Não carregue o `sae_weights.safetensors` no `JumpReLUSAE`.** As chaves batem depois do `lower()`, mas falta subtrair o `b_dec` e as features saem erradas. Use o `params.safetensors`.
- As duas versões dão a mesma pré-ativação até o arredondamento do fp32: erro relativo ≤ 3e-5 e o mesmo conjunto de features ativas.
- Nos dois casos, a entrada é o `resid_post` **cru** da camada L, `hidden_states[L+1]`: o fator de normalização já está dobrado.

```python
from safetensors.torch import load_file
from featgrpo.interp.common import JumpReLUSAE
p = load_file("artifacts/sae_proprio/gemma-3-4b-it_L22_btk16k/params.safetensors")
sae = JumpReLUSAE(p["w_enc"], p["b_enc"], p["w_dec"], p["b_dec"], p["threshold"]).to("cuda")
f = sae.encode(h)          # h = hidden_states[23][..., :] (L22), sem normalizar
```

| Camada | sha256 de `params.safetensors` |
|---|---|
| L17 | `137104d016ff710f839c7eead77bf5b7a51baf2118ad3778a3d13e35c24221d7` |
| L22 | `ec9feaba5921f31743a1dc604b37ec2d6657e174c2471d62abc4781226416d7c` |
| L29 | `ef861de04f0db19df194d7d27f8573885627b211da687c4d06936ddef9e7d692` |

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

  **Resolvido pela frente B (B4, `results/frente-b/qualidade_saes.md`).** O ΔNLL negativo aparece em texto *fora da política* do modelo. As respostas do UltraChat foram escritas por outro modelo, e o 4B erra alguns tokens delas com confiança extrema; o splice "amacia" essas previsões e a perda cai. Nas respostas do próprio 4B, o ΔNLL do Gemma Scope 2 é positivo e normal (+0,15 a +0,35). A convenção de entrada está certa. **O ΔNLL desta tabela no UltraChat não mede reconstrução**: para comparar os dois SAEs, use a FVU, ou o protocolo da B4 (pedido em `results/frente-b/PEDIDOS.md`).
- **Limitações:**
  - "Fração morta no conjunto" conta só as features que não disparam naquele conjunto pequeno. Não é a fração morta global.
  - As avaliações usam no máximo 400 sequências.
