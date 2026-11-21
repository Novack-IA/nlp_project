# Fatos do modelo (B-U1 / B-P1)

> Gerado por `src/featgrpo/interp/model_facts.py` na RTX 4090 em 2026-11-20. Os números brutos estão em [`fatos_modelo.json`](fatos_modelo.json).
> Ambiente: torch 2.6.0+cu124, transformers 5.23.0, Python 3.11, bf16, `attn_implementation="eager"`.

## Modelo

| | |
|---|---|
| Checkpoint | `google/gemma-3-4b-it` (`Gemma3ForConditionalGeneration`; usamos só o texto) |
| Camadas do LM | **34** (L0–L33) |
| d_model | **2560** |
| Caminho das camadas | `model.model.language_model.layers[i]` (com `model` = o `Gemma3ForConditionalGeneration`; use `common.get_layers(model)`) |
| Carregar | `AutoModelForImageTextToText.from_pretrained(..., dtype=torch.bfloat16)` |

## Convenção de camada (verificada)

`resid_post` da camada L = saída do bloco L = **`hidden_states[L+1]`**. O hook de saída do bloco L e `hidden_states[L+1]` dão **exatamente o mesmo tensor** (erro máximo 0,0 de L0 a L32).

**Exceção:** `hidden_states[34]` já tem a norm final aplicada e **não** é a saída do bloco 33. Isso não afeta as nossas camadas.

## Camadas escolhidas e SAEs (Gemma Scope 2, `resid_post`)

No repo `google/gemma-scope-2-4b-it`, todas as camadas têm 16k e 262k (L0 `small`/`big`). Só **L9, L17, L22 e L29** têm todas as larguras (16k, 65k, 262k, 1m) e os três L0 (`small` ≈ 20, `medium` ≈ 60, `big` ≈ 150). Essas são as 4 profundidades oficiais, e são as que usamos.

| L | Profundidade (L+1)/34 | SAE 16k (exploração) | SAE 262k (confirmação) |
|---|---|---|---|
| **9** | 0,29 | `layer_9_width_16k_l0_medium` | `layer_9_width_262k_l0_medium` |
| **17** | 0,53 | `layer_17_width_16k_l0_medium` | `layer_17_width_262k_l0_medium` |
| **22** | 0,68 | `layer_22_width_16k_l0_medium` | `layer_22_width_262k_l0_medium` |
| **29** | 0,88 | `layer_29_width_16k_l0_medium` | `layer_29_width_262k_l0_medium` |

- O nível de L0 se chama `small`/`medium`/`big`; **não existe `large`**, ao contrário do que diz o AGENTS.md §3. Fixamos **`l0_medium`** porque existe em 16k e 262k nas 4 camadas.
- Arquitetura: JumpReLU, `hf_hook_point = model.layers.{L}.output`, ou seja, a mesma saída de bloco acima. Parâmetros: `w_enc [2560, d_sae]`, `b_enc`, `threshold`, `w_dec [d_sae, 2560]`, `b_dec`, em float32. O SAE recebe o residual **cru**, sem normalização.
- Loader próprio em `common.load_sae`, que lê direto do `params.safetensors` sem depender do SAELens. Release equivalente no SAELens: **`gemma-scope-2-4b-it-res`** (conferido no SAELens 6.56.1; o AGENTS.md dizia `...-resid_post`, nome que não existe).
- Sanidade nos nossos pedidos (16 pedidos de train, todos os tokens menos o BOS):

  | L | Variância explicada | L0 |
  |---|---|---|
  | 9 | 0,96 | 60 |
  | 17 | 0,95 | 51 |
  | 22 | 0,90 | 48 |
  | 29 | 0,84 | 49 |

  As medidas completas, com ΔNLL, saem em `qualidade_saes.csv`.

## Chat template

```
<bos><start_of_turn>user
{pedido}<end_of_turn>
<start_of_turn>model
```

- Tokens do exemplo "How do I kill a Python process?": `<bos> <start_of_turn> user \n How ▁do ▁I ▁kill ▁a ▁Python ▁process ? <end_of_turn> \n <start_of_turn> model \n`.
- O template já insere o `<bos>`. Tokenize o texto renderizado com `add_special_tokens=False` para não duplicá-lo.
- O template faz `trim` do conteúdo.
- **Sem papel `system`:** um system prompt vira um prefixo do turno do usuário, separado por `\n\n` (`<start_of_turn>user\nSYS\n\n{pedido}`).

## Posições (definição fixada, B-P1)

| Posição | Definição |
|---|---|
| `inst_last` | o **último token do texto do usuário**, logo antes de `<end_of_turn>` (no exemplo, `?`). Índice: `inst_last_token_id`. É posição principal do C1 no pré-registro da frente C, junto com `post_last` |
| `post` | **média** dos 5 tokens `<end_of_turn>` `\n` `<start_of_turn>` `model` `\n` (ids 106, 107, 105, 4368, 107) |
| `post_last` | só o **último** desses tokens (o `\n` final), onde a geração começa |
| `mean` | média dos tokens do conteúdo do pedido (sem template) |
| `trig` | média dos tokens que cobrem o gatilho; nos pares, NaN quando não alinha (não aconteceu: 4.856/4.856 alinharam) |
| `slot_forma` / `slot_intencao` | 2×2: média dos tokens que cruzam `cells[c].spans.forma` / `.intencao` (de `render_cells`). Substituem `trig` no 2×2. Testado com o tokenizador real: "hack into" → `▁hack ▁into`; "log into" → `▁log ▁into` |

Alinhamento do gatilho, em ordem de tentativa:
1. **exato**: sem diferenciar maiúsculas e com limite de palavra, para que `kill` não case em `skill`;
2. **radical**: janela de palavras com radicais Porter iguais aos do gatilho;
3. **radical parcial**: palavras cujo radical está entre os do gatilho.

O índice registra `trigger_method`. Distribuição nos pares:

| | exato | radical | radical parcial |
|---|---|---|---|
| Q⁺ | 2.427 | 1 | 0 |
| Q⁰ | 2.043 | 129 | 256 |

## Formato do cache (mudanças em relação ao AGENTS.md §4)

- **Denso em `float32`, não `float16`.** O residual do Gemma 3 é enorme: norma média de ~11k em L9 e ~66k em L29, com componentes de até ±88.000 em L29 e ±295.000 no BOS. O máximo do float16 é 65.504, então o cache em fp16 saía com `inf`. Em float32 são ~50 MB por camada × posição nos pares.
- **Latentes:** `artifacts/acts/gemma-3-4b-it/resid_post/L{L:02d}/{conjunto}__{posicao}__{sae}.npz`, com `sae` ∈ {`16k`, `262k`} (`l0_medium`) ou `proprio`. É a convenção de `interp/cache_io.py` (scipy CSR, float32). As variantes por posição:
  - `{pos}`: o SAE aplicado ao vetor médio da posição, o mesmo do `.npy`;
  - `{pos}_tokmean` / `{pos}_tokmax`: o SAE aplicado a **cada token** da posição, seguido de média ou máximo, para `post`, `mean` e `trig`.

  **Atenção (frente C):** a média de muitos tokens sai da distribuição do SAE. O L0 mediano de `mean` é 6–9 e o de `post` é 14–20, contra ~50 por token. Para estatística por feature, as variantes `_tokmean`/`_tokmax` (e `post_last`, que é um token só) são as mais fiéis ao que o SAE "vê".
- **Índice** (`index/{conjunto}.jsonl`): `item_id, group_id, item, label, split, domain, subdomain, trigger, trigger_aligned, trigger_method, n_tokens, n_prompt_tokens`.
- **Lotes com padding à direita**, sem separar os itens de um `group_id`. Os tokens do prefixo comum às células de um grupo saem bit a bit iguais. O teste `test_slot_forma_has_zero_intent_effect` exige `slot_forma` idêntica entre células que só diferem na intenção (pedido da frente C). Com padding à esquerda (versão anterior), lote e item sozinho diferiam em ≤ 1,4% (L2 relativo), com cosseno ≥ 0,99996. É ruído do bf16; o teste usa L2 relativo < 2% e cosseno > 0,9995.
