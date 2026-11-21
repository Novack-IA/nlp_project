# Frente B: relatório dos testes e achados (até 2026-11-20)

> Escopo: o que foi rodado na RTX 4090 (`4090simulation`) para a frente B (B-U1, B-U2, B-U3, B4), o que cada teste mostrou e o que mudou por causa disso.
> Os detalhes estão em:
> - [`fatos_modelo.md`](fatos_modelo.md): modelo, camadas, template e posições;
> - [`qualidade_saes.md`](qualidade_saes.md): qualidade dos SAEs;
> - [`MANIFEST.json`](MANIFEST.json): hashes do cache.
>
> Nenhuma decisão usou o split `test`.

## 1. O que foi rodado

| Etapa | Script | Saída |
|---|---|---|
| Ambiente | conda `nlp-interp`, versões fixadas em `requirements-interp.txt` (torch 2.6.0, transformers 5.23.0) | |
| B-U1: fatos do modelo | `interp/model_facts.py` | `fatos_modelo.{json,md}` |
| B-U2: extrator + testes | `interp/extract.py`, `tests/test_extract.py` | suíte inteira: 62 testes passando na 4090 |
| B-U3: cache provisório | `extract.py --set pairs --widths 16k` | `artifacts/acts/gemma-3-4b-it/` (~760 MB denso + ~210 MB latentes) |
| B4: qualidade dos SAEs | `interp/sae_quality.py --widths 16k 262k` | `qualidade_saes.{csv,json,md}` |

## 2. Fatos do modelo (B-U1)

- **Gemma-3-4B-IT:** 34 camadas, d_model 2560, camadas em `model.model.language_model.layers`.
- **`resid_post` da camada L é `hidden_states[L+1]`**, com erro exatamente 0 de L0 a L32. A exceção é `hidden_states[34]`, que já tem a norm final e por isso não serve como saída do bloco 33.
- **SAEs do Gemma Scope 2:** só **L9, L17, L22 e L29** têm todas as larguras e todos os L0. Fixamos `l0_medium` (L0 ≈ 60) e as larguras 16k e 262k. O AGENTS.md citava um L0 `large`, que **não existe**: os níveis são `small`, `medium` e `big`.
- **Chat template:**
  - o template já põe o `<bos>`, então tokenizamos com `add_special_tokens=False`;
  - o template faz trim do conteúdo;
  - **não há papel `system`**: o texto de sistema vira prefixo do turno do usuário.
- **Posições:**
  - `post` é a média dos 5 tokens `<end_of_turn>\n<start_of_turn>model\n`;
  - `post_last` é o último desses tokens;
  - `mean` é a média sobre o conteúdo do pedido;
  - `trig` é a média sobre o gatilho;
  - `slot_forma` e `slot_intencao` são as posições do 2×2 em frase-molde.

## 3. Testes de sanidade do extrator (B-U2)

| Teste | Resultado |
|---|---|
| Hook na saída do bloco == `hidden_states[L+1]` | idênticos (`torch.equal`) |
| Reconstrução do SAE nos nossos pedidos | FVE 0,84–0,96 e L0 48–60 (16k) |
| Item sozinho × item em lote com padding à esquerda | L2 relativo ≤ 1,4%, cosseno ≥ 0,99996, tudo finito |
| Gatilho exato, sem diferenciar maiúsculas | ok |
| Limite de palavra (`kill` não casa em `skill`) | ok, depois de uma correção (ver §6) |
| Gatilho por radical Porter, multipalavra e parcial | ok |
| 2×2: `cells_items` e posições de slot, inclusive com espaço no início | ok. No tokenizador real, "hack into" → `▁hack ▁into` |
| Caminho dos latentes compatível com `cache_io.py` (Davi) | ok |

A diferença entre lote e item sozinho é ruído do bf16, não um bug. A tolerância do teste ficou em L2 relativo < 2% e cosseno > 0,9995.

## 4. Cache `pairs` (B-U3)

- **Tamanho:** 4.856 pedidos (2.428 pares Q⁺/Q⁰), 4 camadas, 3 posições, denso em float32 e latentes 16k (CSR).
- **Integridade:** dimensões corretas, nenhum NaN ou `inf`, índice alinhado com as linhas. Os sha256 estão no MANIFEST.
- **Gatilho alinhado em 4.856 de 4.856 pedidos.**

  | | exato | radical | radical parcial |
  |---|---|---|---|
  | Q⁺ | 2.427 | 1 | 0 |
  | Q⁰ | 2.043 | 129 | 256 |

  No Q⁰, o gatilho muitas vezes aparece flexionado ("killing", "hacked"), por isso o alinhamento por radical é necessário.
- **L0 mediano dos latentes 16k por posição** (o SAE aplicado ao vetor já agregado):

  | L | post | post_last | mean | trig |
  |---|---|---|---|---|
  | 9 | 17 | 30 | 8 | 38 |
  | 17 | 20 | 51 | 7 | 31 |
  | 22 | 16 | 48 | 6 | 27 |
  | 29 | 14 | 54 | 9 | 29 |

  **Achado:** a média de muitos tokens sai da distribuição do SAE e ativa poucas features (`mean` fica em 6–9, contra ~50 por token). Por isso o cache também tem `{pos}_tokmean` e `{pos}_tokmax`: o SAE é aplicado a cada token e depois se agrega. Para a frente C, essas variantes e o `post_last` são as mais fiéis.

## 5. Qualidade dos SAEs (B4)

**Amostra:** só train + val.
- FVE, L0 e features ativas: 1.200 pedidos, 47.079 tokens.
- ΔNLL e KL: 300 respostas greedy do próprio 4B, 19.188 tokens, NLL original 0,149. Essas respostas não foram salvas.

| L | SAE | FVE | L0 | sem ativar | ΔNLL | KL | ΔNLL camada zerada | recuperado |
|---|---|---|---|---|---|---|---|---|
| 9 | 16k | 0,957 | 59 | 24% | 0,29 | 0,25 | 26,7 | 98,9% |
| 9 | 262k | 0,972 | 66 | 70% | 0,27 | 0,24 | 26,7 | 99,0% |
| 17 | 16k | 0,955 | 51 | 34% | 0,35 | 0,30 | 31,2 | 98,9% |
| 17 | 262k | 0,959 | 54 | 69% | 0,33 | 0,28 | 31,2 | 99,0% |
| 22 | 16k | 0,909 | 47 | 28% | 0,20 | 0,18 | 15,4 | 98,7% |
| 22 | 262k | 0,922 | 49 | 81% | 0,18 | 0,16 | 15,4 | 98,8% |
| 29 | 16k | 0,843 | 47 | 18% | 0,15 | 0,13 | 33,9 | 99,6% |
| 29 | 262k | 0,876 | 49 | 74% | 0,12 | 0,12 | 33,9 | 99,6% |

Conclusões:
1. **Nenhuma camada é excluída.** Todos os SAEs recuperam ~99% da perda em relação a zerar a camada.
2. **L17 é a camada com mais perda** (ΔNLL 0,35). Na intervenção (C5), o erro de reconstrução vai ser somado de volta, para que o efeito medido seja o da feature e não o do SAE.
3. **O 262k é só um pouco melhor que o 16k** (+0,01–0,03 de FVE). O 16k basta para exploração.
4. **Muitas features nunca ativam nos nossos dados**: 18–34% no 16k e 69–81% no 262k. Isso é esperado num domínio estreito e não quer dizer que as features estão mortas. O filtro de frequência mínima reduz bastante o número de testes da correção BH.
5. **A FVE cai com a profundidade** (0,96 → 0,84), mas o ΔNLL também cai. As camadas tardias são mais difíceis de reconstruir e, ainda assim, menos sensíveis ao erro de reconstrução.

## 6. Problemas encontrados pelos testes e corrigidos

| Problema | Como apareceu | Correção |
|---|---|---|
| **Estouro do float16** | O residual chega a ±88k (±295k no BOS) e o máximo do fp16 é 65.504, então o cache saía com `inf` | Cache denso em **float32**, documentado no AGENTS.md §4 |
| Gatilho casava dentro de palavras | `kill` casava em `skill` | Regex com limite de palavra |
| `hidden_states[34]` ≠ saída do bloco 33 | Erro de 1,5e5 no `model_facts` | É a norm final. A verificação vai só até L32, com igualdade exata nas camadas escolhidas |
| Lote × item sozinho com relmax de 1,16% | Teste falhando | É ruído do bf16 (cosseno ≥ 0,99996). Tolerância por L2 relativo e cosseno |
| L0 baixo nos latentes agregados | Medianas de 6–20 | Variantes `_tokmean` e `_tokmax` |
| **ΔNLL negativo** (−0,3 a −1,3) | O SAE "melhorava" a NLL nas respostas r4/r7 | As respostas são do Gemma-4-12B, que o 4B erra com confiança extrema (até 18 nat num token). A medida passou a usar respostas do próprio 4B, mais KL |
| Bug em `data.response_items` (módulo compartilhado) | Quebrava porque `q_harm` é um objeto nos arquivos finais | Correção de 1 linha. **Precisa ser avisada ao grupo** |
| OOM na extração e no ΔNLL | Latentes densos na GPU; `log_softmax` sobre todos os logits | Latentes esparsos movidos para a CPU; `log_softmax` só nas posições de resposta |
| Convenção de caminho dos latentes | Diferente da do `cache_io.py` (Davi) | Adotei a do Davi, renomeei os arquivos e conferi os sha256 |

## 7. Mudanças pedidas pela frente C (pré-registro, 2026-11-20)

O pré-registro da frente C (branch `frente-c-features`) definiu as **posições principais do C1**:
- `inst_last`: o último token do pedido;
- `post_last`.

São as duas em que o token é o mesmo nas 4 células do 2×2.

| Mudança | Situação |
|---|---|
| Posição `inst_last` e `inst_last_token_id` no índice | Implementada. Teste puro passando |
| Lacunas pelo nome real da frente A (`slots.form`/`slots.intent`) | Implementada. Teste puro passando |
| Padding à direita, sem separar os itens de um mesmo grupo entre lotes, para que o prefixo comum saia bit a bit igual | Implementado. O teste de GPU `test_slot_forma_has_zero_intent_effect` (efeito de intenção exatamente 0 na `slot_forma`) **passa**. A suíte inteira tem 71 testes, todos passando |
| Qualidade do SAE em `inst_last` e `post_last` (`sae_quality.py --positions`) | Rodada (resumo abaixo e em `qualidade_saes.md`) |
| Release do SAELens é `gemma-scope-2-4b-it-res` | Conferido no SAELens 6.56.1 e corrigido no AGENTS.md |

**Cache `pairs` refeito em 2026-11-21:**
- tem `inst_last` e padding à direita;
- 65 arquivos, todos com sha256 no MANIFEST;
- gatilho alinhado em 4.856/4.856 pedidos.

O `inst_last` cai no mesmo token em 92% dos pedidos (id 236881) e em outro em 6% (id 236761). O L0 mediano dos latentes 16k:

| | L9 | L17 | L22 | L29 |
|---|---|---|---|---|
| `inst_last` | 50 | 64 | 61 | 54 |
| `post_last` | 30 | 51 | 48 | 54 |

**Qualidade do SAE nas duas posições:**
- o vetor é reconstruído quase perfeitamente (cosseno ≥ 0,99);
- só **55–89% da variação entre pedidos** é reconstruída, então parte do sinal pode ficar no erro do SAE, o que entra nas limitações;
- zerar `inst_last` em L22/L29 não muda a resposta, mas zerar `post_last` muda.

## 8. Limitações e pendências

- **O cache só cobre os pares.** A frente A subiu na master, em 2026-11-21, o `cells_2x2.jsonl`, os controles e o XSTest/OR-Bench; as próximas extrações são desses conjuntos. O extrator já suporta `--set cells2x2` com `slot_forma` e `slot_intencao`.
- **Latentes 262k ainda não foram extraídos** para nenhum conjunto. Por ora só os do 16k.
- **O ΔNLL mede respostas curtas** (≤ 64 tokens) e greedy. Basta para comparar camadas, mas não descreve gerações longas.
- **"Sem ativar" é uma medida em 47k tokens.** Não serve como estimativa de features mortas no pré-treino.
- **Próximo passo:** B5 / C5, com o `intervene.py` (x' = x − dec(f) + dec(f'), mais controles com features aleatórias e IC por bootstrap).
