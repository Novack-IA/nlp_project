# AGENTS.md: contexto para agentes de código (Claude Code, Codex, Cursor…)

Leia este arquivo inteiro antes de mexer no repositório. Ele vale para os quatro integrantes. A seção de cada frente diz o que é responsabilidade de quem.

## 1. O projeto em 30 segundos

**Pergunta de pesquisa (versão operacional, 2026-11-20):**

> **É possível identificar, nas features de um SAE, um sinal de intenção nociva que se distingue do sinal de forma lexical do pedido, usando pares mínimos nocivo↔benigno?**

**As tarefas de cada um estão em [`docs/tarefas/`](docs/tarefas/). Para começar agora, veja [`docs/tarefas/URGENTE.md`](docs/tarefas/URGENTE.md).** Para ler o dataset, use sempre `featgrpo.interp.data`. Comece por `00-visao-geral.md`: ele define as classes de feature (S_intent, S_lex), os critérios C1–C5 e a regra do veredito (`results/VEREDITO.md`). Depois leia o arquivo da sua frente.

> **2026-11-20, PR #1:** o 2×2 passou a ser feito por **frase-molde**, entraram 5 regras de análise e a auditoria do 2×2 ficou automática. O porquê está em [`docs/proposta/`](docs/proposta/README.md); o que vale está em `docs/tarefas/`.

Modelos alinhados recusam pedidos benignos que *soam* perigosos ("how do I **kill** a Python process?"). Esse é o *over-refusal*. A hipótese é que, internamente, o modelo representa duas coisas distintas:
- **nocividade real**: a intenção do pedido;
- **superfície**: palavras e tom alarmantes.

O over-refusal vem de o comportamento seguir a segunda. Testamos isso **só com interpretabilidade**: probes, diferenças de médias, features de Sparse Autoencoders (SAEs), ablação e steering no modelo **congelado**. Não há treino de modelo nem RL.

> O plano antigo (README-featcontroller.md, plano.md, "Paper 2 / controller GRPO") **saiu do escopo** em 2026-11-20. Não implemente nada de controller ou GRPO. O plano vigente está em [`docs/fase1-interpretabilidade.md`](docs/fase1-interpretabilidade.md), com hipóteses H1–H4, riscos e cronograma.

## 2. O dataset: `datasets/phase0/`

Construído na Fase 0. O protocolo está em `docs/relatorio_fase0.md`, e a ficha com o esquema em `datasets/phase0/README.md`.

Cada **grupo** (`group_id`) junta:
- `q_harm` (**Q⁺**): pedido nocivo, do PKU-SafeRLHF;
- `q_border` (**Q⁰**): pedido benigno que usa **a mesma expressão-gatilho** (`trigger`);
- em alguns arquivos, **8 respostas rotuladas** (`responses`, r0–r7):

| id | pedido | role | origem |
|---|---|---|---|
| r0 | Q⁺ | unsafe | PKU (Alpaca) |
| r1 | Q⁺ | safe_A | PKU, mesmo gerador |
| r2 / r5 | Q⁺ / Q⁰ | safe_helpful | Gemma-4-12B-it instruído |
| r3 / r6 | Q⁺ / Q⁰ | refusal | Gemma-4-12B-it instruído |
| r4 / r7 | Q⁺ / Q⁰ | natural | Gemma-4-12B-it **sem system prompt** (baseline de comportamento) |

| Arquivo | Linhas | Use para |
|---|---|---|
| `v5b/pairs_prompt_level.jsonl` | **2.428 pares** | **Análises de pedidos** (o principal). Campos: `group_id, domain, subdomain, trigger, q_harm, q_border, q_border_method, in_final, valid_full_group, split, natural_class_q_harm, natural_class_q_border` |
| `v5b/{train,val,test}.jsonl` | 351 / 76 / 73 grupos | **Análises de respostas**: grupos com todas as respostas de contraste validadas |
| `v5b/pool_groups.jsonl.gz` | 2.806 grupos | Tudo, com os rótulos de cada juiz (`labels.{wildguard,qwen3guard,judge,final}`) |
| `v5/…` | versão anterior | Só para comparação |

Domínios: `bias`, `cyber`, `violence`, `privacy`.

Contrastes de respostas:
- **conteúdo** = r0 − r1 (mesmo pedido, mesmo autor);
- **forma** = r0(Q⁺) − r5(Q⁰);
- **recusa** = r6 − r5.

**Regras dos dados:**
- **Splits são por grupo.** Q⁺ e Q⁰ do mesmo `group_id` ficam sempre no mesmo split. Use `split` do arquivo, ou `GroupKFold` por `group_id`.
- **O `test` não entra em nenhuma decisão:** escolha de camada, de feature, de limiar ou de hiperparâmetro. Use `train` + `val` para decidir e `test` uma vez, no fim.
- r0/r1 são texto do Alpaca, fora da distribuição do Gemma 3. Servem para *ler* representações, não como comportamento do modelo. As respostas naturais r4/r7 são do **Gemma-4-12B**. O comportamento do Gemma-3-4B vem da frente A (`behavior_gemma3_4b.jsonl`).
- O gatilho nem sempre aparece literalmente no Q⁰: às vezes casa só pelo radical ou foi confirmado pelo juiz (`shares_trigger_source`). Ao localizar tokens do gatilho, faça busca sem diferenciar maiúsculas e, se falhar, por radical (Porter). Registre os casos sem alinhamento; não os descarte em silêncio.
- **Ética:** o dataset contém pedidos e respostas nocivas (todas do PKU). **Não gere respostas nocivas.** Pedidos nocivos novos só como recombinação de trechos dos pedidos do PKU (a lacuna de intenção do 2×2), sem acrescentar detalhes operacionais. Não publique trechos em lugares públicos além deste repo; uso só para pesquisa.

## 3. Modelo e SAEs

- **Modelo:** `google/gemma-3-4b-it`. É gated no HF: cada pessoa aceita a licença e usa o próprio token de leitura em `.env` (`HF_TOKEN=...`; o `.env` está no `.gitignore`, nunca o commite).
  - É um checkpoint multimodal (`Gemma3ForConditionalGeneration`). Usamos só texto. O caminho das camadas depende da versão do `transformers`, algo como `model.language_model.layers[i]` ou `model.model.language_model.layers[i]`. Confira com `print(model)`.
  - Chat template: o Gemma 3 **não tem papel `system`**. Um system prompt vai no início do turno do usuário.
  - bf16 ocupa ~9 GB e cabe numa 4090 junto com SAEs.
- **SAEs prontos:** Gemma Scope 2, `google/gemma-scope-2-4b-it`.
  - Release no SAELens: **`gemma-scope-2-4b-it-res`** (conferido no SAELens 6.56.1; os pesos ficam na pasta `resid_post/` do repo HF), com `sae_id` do tipo `layer_{L}_width_{16k|65k|262k|1m}_l0_{small|medium|big}` (não existe `large`). Usamos `l0_medium`, a única variante com fichas no Neuronpedia.
  - Neuronpedia: `gemma-3-4b-it/{L}-gemmascope-2-res-16k`.
  - **Verifique a lista real de camadas e larguras** antes de fixar. As 4 profundidades com todas as larguras ficam em 25/50/65/85% das camadas; segundo o anúncio, as larguras 16k e 262k existem em todas.
- **Convenção de camada:** `resid_post` da camada L = saída do bloco L = `hidden_states[L+1]` no `output_hidden_states` do HF (`hidden_states[0]` são os embeddings). Errar isso desloca tudo em uma camada.
- **SAE próprio:** treinado pela frente A na h1003 (BatchTopK) para 1–2 camadas. É um controle para não depender de um único dicionário.

## 4. Formato do cache de ativações (contrato entre as frentes)

```
artifacts/acts/gemma-3-4b-it/resid_post/L{L:02d}/{conjunto}__{posicao}.npy          # denso, float32 [N, d_model]
artifacts/acts/gemma-3-4b-it/resid_post/L{L:02d}/{conjunto}__{posicao}__{sae}.npz   # latentes, scipy CSR float32 [N, d_sae]
artifacts/acts/gemma-3-4b-it/index/{conjunto}.jsonl                                  # linha i ↔ linha i do .npy/.npz
artifacts/acts/gemma-3-4b-it/MANIFEST.json                                           # sha256, commit, versões, sae_ids
```

- Leia com `featgrpo.interp.cache_io` (`load_dense`, `load_latents`, `load_index`).
- **Denso em float32, não float16:** o residual do Gemma 3 passa de 65.504 (máximo do fp16) a partir de ~L17; em fp16 saía `inf`. Detalhes em `results/frente-b/fatos_modelo.md`.
- **Camadas:** L9, L17, L22, L29. **`sae`:** `16k` ou `262k` (Gemma Scope 2, `l0_medium`) ou `proprio`.

- **`conjunto`:** `pairs` (2.428 × {q_harm, q_border}), `final_resp` (respostas dos 500 grupos), `cells2x2`, `xstest`, `orbench_hard`, …
- **`posicao`** (principais do C1, pelo pré-registro da frente C: `inst_last` e `post_last`):
  - `inst_last`: o último token do texto do usuário, logo antes de `<end_of_turn>`. No 2×2 é o mesmo token nas 4 células quando o molde termina em texto fixo; confira com `inst_last_token_id` no índice;
  - `trig`: média dos tokens do gatilho (pares originais e controles);
  - `slot_forma`: média dos tokens da lacuna de forma do 2×2 (o gatilho ou o termo neutro), a partir de `cells[*].slots.form` (formato da frente A) ou `cells[*].spans.forma`;
  - `slot_intencao`: média dos tokens da lacuna de intenção do 2×2, a partir de `cells[*].slots.intent` ou `cells[*].spans.intencao`;
  - `post`: **média** dos tokens do sufixo `<end_of_turn>\n<start_of_turn>model\n` (pós-instrução);
  - `post_last`: só o **último** token desse sufixo (como em Arditi et al.);
  - `mean`: média dos tokens do pedido;
  - `resp_mean`: média dos tokens da resposta, em teacher forcing;
  - `resp_k8`: média dos 8 primeiros tokens da resposta.
- **Variantes de latentes** (só nos `.npz`): `{posicao}` é o SAE aplicado ao vetor médio da posição. `{posicao}_tokmean` / `{posicao}_tokmax` é o SAE aplicado a cada token da posição, seguido de média ou máximo (para `post`, `mean`, `trig` e os slots). A média de muitos tokens sai da distribuição do SAE (L0 de `mean` ≈ 6–9 contra ≈ 50 por token); a frente C escolhe qual usar.
- **`index`** tem, por linha: `item_id, group_id, item` (`q_harm`/`q_border`, `r0`…`r7`, a célula do 2×2 `harm_trig`… ou `trig_only`/`neutral_only`), `label, has_trigger` (2×2 e controles), `split, domain, subdomain, trigger, trigger_aligned, trigger_method` (`span` nos controles, que trazem o span da peça), `template_id` (controles; o par é `group_id` + `template_id`, com o `group_id` do 2×2), `slots_aligned` (2×2), `inst_last_token_id`, `n_tokens` e `n_prompt_tokens`.
- **Lotes:** padding à direita, com os itens de um mesmo `group_id` sempre no mesmo lote. Os tokens do prefixo comum às células de um grupo saem bit a bit iguais (por exemplo, `slot_forma` quando ela vem antes da lacuna de intenção).
- Ativações por token (para exemplos de maior ativação) ficam com quem precisa delas (frente C). Não entram no cache compartilhado.
- O cache denso é pequeno (~50 MB por camada × posição para os pares; ~1 GB com latentes 16k). Vai para a pasta compartilhada do time com a mesma árvore. `artifacts/` **nunca** vai para o git.

## 5. Estrutura do repositório

```
datasets/phase0/          dataset (no git)
docs/                     relatório da Fase 0, plano da Fase 1, cluster
src/featgrpo/data/phase0/ pipeline que gerou o dataset (P0–P8): só a frente A mexe aqui
src/featgrpo/interp/      código novo da Fase 1 (crie aqui): extract.py, intervene.py, probes.py, features.py, ...
configs/                  YAML (configs/data/phase0*.yaml; novos em configs/interp/)
slurm/                    jobs da h1003 (só a frente A)
results/<frente>/         resultados pequenos versionados (csv/json/png) + README curto do que é cada arquivo
artifacts/                grandes, fora do git
notebooks/                exploração; nada essencial deve existir só aqui
```

## 6. Quem faz o quê

| Frente | Dono | Máquina | Escopo | Tarefas |
|---|---|---|---|---|
| A. Dados 2×2, controles, comportamento, SAE próprio | Gustavo | cluster h1003 (H100/B200) | células do 2×2 (caminho crítico), controles `trig_only`, XSTest/OR-Bench-Hard, comportamento do 4B, SAE próprio | [`gustavo-frente-a.md`](docs/tarefas/gustavo-frente-a.md) |
| B. Ativações, latentes, intervenção | Caio | RTX 4090 | `interp/extract.py` (cache da seção 4), qualidade dos SAEs, `interp/intervene.py`, C5 | [`caio-frente-b.md`](docs/tarefas/caio-frente-b.md) |
| C. Estatística por feature | Frederico | RTX 4090 | d_I, d_L e d_X por feature, classes S_intent/S_lex (C1), controles e autointerp (C3), robustez entre SAEs (C4a) | [`frederico-frente-c.md`](docs/tarefas/frederico-frente-c.md) |
| D. Baselines, probes, dissociação, veredito | Davi | CPU (sem cluster por ora) | TF-IDF, validador do 2×2, probes densos, dissociação dupla (C2), confusões (C4b/c), `VEREDITO.md` | [`davi-frente-d.md`](docs/tarefas/davi-frente-d.md) |

**Se você é o agente de um integrante,** trabalhe só na frente dele. Se uma tarefa exigir algo de outra frente (por exemplo, a D precisa de ativações novas), escreva o pedido em `results/<frente>/PEDIDOS.md` ou avise o integrante. Não implemente a parte do outro.

## 7. Ambiente local (4090 / CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 4090; CPU: sem --index-url
pip install "transformers>=4.50" accelerate safetensors sae-lens scikit-learn pandas numpy scipy statsmodels matplotlib nltk python-dotenv
```

- Fixe as versões que funcionarem em `requirements-interp.txt` e commite.
- Teste do SAELens: confira se a versão instalada conhece a release `gemma-scope-2-4b-it-res`. Se não conhecer, atualize. Em último caso, baixe os pesos com `huggingface_hub` e carregue à mão.
- Seeds fixas (`torch.manual_seed(0)`, `numpy`, `random`); registre a versão do modelo, do SAE e o commit do código em cada resultado.

## 8. Regras de trabalho

- **Uma branch por frente** (`frente-a-dados`, `frente-b-ativacoes`, `frente-c-features`, `frente-d-densa`). Commits pequenos, merge na `master` quando a etapa fecha. Não reescreva o histórico da `master`.
- **Não commite:** `.env`, tokens, `artifacts/`, `data/`, checkpoints, `.npy` ou `.pt`, notebooks com saídas gigantes.
- **Estatística:**
  - testes pareados por grupo;
  - correção de múltiplas comparações (Benjamini–Hochberg) ao testar features;
  - reporte tamanho de efeito e IC, não só p-valor;
  - compare sempre com o **baseline lexical**: o ponto do paper é o que a representação sabe *além das palavras*.
- **Resultados reprodutíveis:** cada número em `results/` sai de um script em `src/featgrpo/interp/` com os argumentos registrados no próprio JSON.
- **Cluster h1003:** só a frente A usa. As instruções estão na skill `.claude/skills/slurm-dgx-h100/` e em `docs/cluster-estrategias.md`. Nunca cancele jobs de outra pessoa e nunca apague dados de outra pessoa na raid.

## 9. Glossário rápido

- **Q⁺ / Q⁰:** pedido nocivo e pedido benigno de fronteira (*borderline*), com o mesmo gatilho.
- **HCR:** *harmful compliance rate*, o modelo atende a um pedido nocivo.
- **ORR:** *over-refusal rate*, o modelo recusa um pedido benigno.
- **resid_post:** o fluxo residual depois de um bloco do transformer.
- **SAE:** autoencoder esparso que decompõe a ativação em "features" interpretáveis. O d_j de uma feature j num contraste é a média da diferença pareada dividida pelo desvio-padrão dessa diferença.
- **Direção de recusa (Arditi et al., 2024):** a diferença de médias das ativações entre pedidos nocivos e benignos; removê-la desliga a recusa.
- **2×2:** intenção (nocivo/benigno) × forma lexical (com/sem gatilho). A frente A gera, por grupo, uma **frase-molde** com duas lacunas (`{FORMA}`, `{INTENCAO}`); as 4 células saem de preenchê-las. O validador automático da frente D confere a estrutura.
- **S_intent / S_lex:** features com efeito de intenção alto e lexical baixo, e vice-versa. Definição exata em `docs/tarefas/00-visao-geral.md`.
