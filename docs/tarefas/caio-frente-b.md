# Frente B: Caio (RTX 4090)

**Papel na resposta:** você é a **infraestrutura de medição**.
- Toda estatística de feature (frente C) e todo classificador (frente D) roda sobre o cache que você produz. Por isso a corretude do extrator é crítica: camada errada, posição errada ou desalinhamento de gatilho invalida tudo.
- Você também roda o **teste causal (C5)**.

Recursos: RTX 4090 (24 GB). O Gemma-3-4B-IT em bf16 ocupa ~9 GB; os SAEs de 16k a 262k cabem junto.

---

## B1. Setup e fatos do modelo (D1, entregar até o fim do dia)

- Ambiente da seção 7 do `AGENTS.md`. Fixe as versões que funcionarem em `requirements-interp.txt`.
- Carregar `google/gemma-3-4b-it` e descobrir o **caminho das camadas** e o **número de camadas**.
- Listar os SAEs disponíveis em `google/gemma-scope-2-4b-it` (`resid_post`): camadas, larguras e L0. Escolher as **4 camadas** a 25/50/65/85% que tenham **16k e 262k**.
- Escrever em `results/frente-b/fatos_modelo.md`:
  - número de camadas e d_model;
  - os índices L das 4 camadas e os `sae_id` exatos;
  - o mapeamento `resid_post L` ↔ `hidden_states[L+1]`;
  - o chat template exato;
  - os tokens da posição `post`.

**Avise o time no D1.** A frente A precisa da camada para o SAE próprio.

## B2. Extrator (`src/featgrpo/interp/extract.py`, D2)

**Entrada:** uma lista de itens `{item_id, text, trigger?, response?}`.

**Saída:** o formato da seção 4 do `AGENTS.md`.

- **Formatação:** pedido como turno do usuário, com o chat template, `add_generation_prompt=True`. Para respostas, use teacher forcing de pedido + resposta.
- **Posições:**
  - `post`: os tokens do sufixo de geração `<end_of_turn>\n<start_of_turn>model\n`. Salve a **média** desses tokens e também o **último**, em `post_last`.
  - `mean`: média dos tokens do conteúdo do pedido, sem os tokens de template.
  - `trig`: média dos tokens que cobrem o gatilho. Localize pelo `offset_mapping` do tokenizer: busca exata sem diferenciar maiúsculas; se falhar, por radical Porter palavra a palavra. Grave `trigger_aligned` no índice.
    - Nas células `*_notrig`, use o **termo neutro** como "gatilho". Isso mantém a posição comparável.
    - Nos controles `trig_only` e `neutral_only`, idem.
  - Para respostas, `resp_mean` e `resp_k8`, se a frente D pedir. Para a pergunta principal, bastam os pedidos.
- **Duas saídas por item:**
  1. **Denso:** `float16 [N, d_model]` por camada × posição.
  2. **Latentes do SAE:** codifique o denso de cada posição com o SAE da camada (16k e 262k). Salve **esparso** em `.npz` com `indices` e `values` no formato CSR. Uma opção a mais é guardar a ativação por token na posição `trig` (máximo sobre os tokens) além da média; a frente C decide qual usar.
- **Teste de sanidade obrigatório** (`tests/test_extract.py`):
  - `hidden_states[L+1]` é igual ao hook de saída do bloco L;
  - reconstruir com o SAE dá variância explicada > 0,6 nos nossos pedidos. Se der muito menos, provavelmente a camada ou a normalização estão erradas;
  - o mesmo item processado duas vezes, sozinho e em batch com padding, dá o mesmo vetor (tolerância de 1e-2). Atenção ao padding à esquerda e à máscara.
- **Lotes:** batch por comprimento, `torch.no_grad()`, bf16.

## B3. Caches (D2 provisório, D5–D6 completo)

| Cache | Conteúdo | Quando |
|---|---|---|
| `pairs` | 2.428 × {q_harm, q_border}, 4 camadas, posições `post`, `post_last`, `mean`, `trig`; denso + latentes 16k | **D2** (destrava C e D no modo provisório) |
| `cells2x2` | ~1.200 × 4 células (+ originais) | D5, quando A1 sair |
| `controls` | `trig_only` e `neutral_only` | D4–D5 |
| `xstest`, `orbench_hard` | itens externos | D5 |
| 262k | latentes 262k para `cells2x2` e `controls` nas 4 camadas | D6 |
| SAE próprio | latentes do SAE da frente A em `cells2x2` e `controls` (camada de A5) | D8 |

Publique na pasta compartilhada com a árvore `artifacts/acts/...` e um `MANIFEST.json` com SHA-256 dos arquivos, commit do código e versões. No git vai só o `MANIFEST.json`, em `results/frente-b/`.

## B4. Qualidade dos SAEs nos nossos dados (D4)

Para cada camada × SAE (16k, 262k e depois o próprio), sobre os pares e o 2×2:
- variância explicada;
- **ΔNLL**: perda do modelo na resposta natural com o residual substituído pela reconstrução, contra o original. Use os pares com resposta do `final`;
- L0 médio e fração de features que nunca ativam nos nossos dados.

**Saída:** `results/frente-b/qualidade_saes.csv` + um parágrafo de leitura. Se um SAE for ruim em alguma camada (ΔNLL alto), avise C e D: essa camada sai da análise ou entra com ressalva.

## B5. Intervenção e C5 (`src/featgrpo/interp/intervene.py`, D7–D10)

**Ferramenta:** hook na saída do bloco L que, em todos os tokens (pedido e geração):
- codifica com o SAE;
- zera (ablação) ou escala (steering, fator α) as features de uma lista;
- decodifica e **soma o erro de reconstrução de volta**: x' = x − dec(f) + dec(f'). Assim só as features escolhidas mudam.

**Experimentos**, com as listas S_intent e S_lex da frente C (camada escolhida por C):

| Experimento | Itens | Mede |
|---|---|---|
| Ablação de S_intent | `harm_trig`, `harm_notrig` (test) | Queda na taxa de recusa |
| Ablação de S_lex | `ben_trig` (test) e `hard_benign_4b` | Queda no over-refusal |
| Controle | as mesmas ablações com o mesmo número de features **aleatórias**, com distribuição de frequência casada (5 sorteios) | Linha de base |
| Efeito colateral | ablações em `ben_notrig` | Não deveria mudar nada |

- **Geração:** temperatura 0, máximo de 256 tokens.
- **Rotulagem das respostas intervindas:** o juiz da frente A. Junte as gerações em `results/frente-b/c5_geracoes.jsonl` e peça a rotulagem ao Gustavo. Para iterar rápido, use antes um classificador barato de recusa (regex de frases de recusa + WildGuard, se couber).
- **Estatística:** diferença de taxas com IC de 95% por bootstrap sobre grupos.
- **Saída:** `results/frente-b/c5_ablacao.json` e uma figura.

## Entregas e critérios de pronto

| Entrega | Pronto quando |
|---|---|
| `fatos_modelo.md` | Camadas, `sae_id` e template confirmados (D1) |
| `extract.py` + testes | Os 3 testes passam; cache `pairs` publicado (D2) |
| Caches completos + `MANIFEST.json` | Todos os conjuntos da tabela B3 |
| `qualidade_saes.csv` | 4 camadas × SAEs, com leitura |
| `intervene.py` + `c5_ablacao.json` | Experimentos com controle aleatório e IC |

**Branch:** `frente-b-ativacoes`.

**Interfaces:**
- Frederico e Davi leem seus arquivos. Mudança de formato só com aviso e atualização da seção 4 do `AGENTS.md`.
- Pedidos de outras frentes chegam em `results/frente-b/PEDIDOS.md`.
