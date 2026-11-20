# URGENTE: o que cada um faz agora (D0–D2), em paralelo

> Estas tarefas **não dependem de ninguém**. Todo mundo começa hoje. Os detalhes estão no arquivo da sua frente, e o porquê em `00-visao-geral.md`. Ao terminar um item, marque o checkbox **na sua branch** e mencione no grupo.

## 0. Todos (primeira hora)

- [ ] `git clone https://github.com/Novack-IA/nlp_project.git` e criar a sua branch: `frente-a-dados`, `frente-b-ativacoes`, `frente-c-features` ou `frente-d-densa`.
- [ ] Ler `AGENTS.md` e `docs/tarefas/00-visao-geral.md`. Se você usa um agente (Claude Code, Codex, Cursor), mande ele ler os dois e o arquivo da sua frente antes de qualquer coisa.
- [ ] Aceitar a licença do **Gemma 3** em https://huggingface.co/google/gemma-3-4b-it e criar um token de **leitura**. Coloque `HF_TOKEN=...` num `.env` na raiz; ele já está no `.gitignore`.
- [ ] Ambiente: seção 7 do `AGENTS.md`. Na 4090, com CUDA; no Davi, CPU basta.
- [ ] Rodar `PYTHONPATH=src python -m featgrpo.interp.data`. Deve imprimir 2.428 pares, 4.856 itens e "splits por grupo consistentes". Todo código novo lê o dataset por esse módulo.

## 1. Gustavo, frente A (h1003): caminho crítico

| # | Tarefa | Prazo | Destrava |
|---|---|---|---|
| A-U1 | Escrever os 4 prompts das células (`src/featgrpo/data/phase1/prompts/cell_*_v1.txt`) e o `same_intent_v1.txt` | D0 | tudo |
| A-U2 | **Rodada seca do 2×2 com 50 grupos** (gerador, guards, juiz, checagem de gatilho) e taxa de aprovação por célula | D1 | ajuste de prompt |
| A-U3 | Exportar `results/frente-a/amostra_2x2.csv` (60 grupos, D-U2 rotula) | D1 | auditoria do Davi |
| A-U4 | Controles `trig_only`/`neutral_only` por template: `datasets/phase1/controls_trig_only.jsonl` | D1 | B (cache de controles), C3 |
| A-U5 | Disparar a geração completa do 2×2, depois de A-U2 aprovado | D2 | B, C, D |
| A-U6 | Pedir ao admin acesso à h1003 para o Davi e contas para Caio e Frederico | D0 | — |

## 2. Caio, frente B (4090): medição

| # | Tarefa | Prazo | Destrava |
|---|---|---|---|
| B-U1 | **`results/frente-b/fatos_modelo.md`**: número de camadas, d_model, caminho das camadas, as 4 camadas a 25/50/65/85% com `sae_id` de 16k e 262k confirmados no HF, chat template, tokens `post`, mapeamento `resid_post L ↔ hidden_states[L+1]` | **D0–D1** | A5, C, D |
| B-U2 | `src/featgrpo/interp/extract.py`, usando `featgrpo.interp.data.prompt_items`, com os 3 testes de sanidade de B2 | D1–D2 | — |
| B-U3 | **Cache provisório `pairs`** (2.428 × 2, 4 camadas, posições `post`, `post_last`, `mean`, `trig`; denso + latentes 16k) na pasta compartilhada + `MANIFEST.json` | **D2** | C1 provisório, D3 |
| B-U4 | Criar a pasta compartilhada do time (Drive) com a árvore `artifacts/` e postar o link no grupo, **não no repo** | D0 | todos |

## 3. Frederico, frente C (4090): features

| # | Tarefa | Prazo | Destrava |
|---|---|---|---|
| C-U1 | **Commitar `results/frente-c/preregistro.md`** antes de ver qualquer dado | **D0** | credibilidade do resultado |
| C-U2 | `src/featgrpo/interp/feature_stats.py`: I_g, L_g, X_g, d, Wilcoxon e BH, classes. Testado com **dados sintéticos** (features plantadas de intenção, lexicais e mistas, para checar que o código as recupera) | D1 | C2 |
| C-U3 | Carregar um SAE do Gemma Scope 2 pelo SAELens e codificar 10 pedidos à mão (validação cruzada com B-U1) | D1 | — |
| C-U4 | Montar o autointerp: LLM local em 4 bits na 4090, prompt de descrição e de detecção, e a busca de rótulos no Neuronpedia | D2 | C3 |

## 4. Davi, frente D (CPU): baselines e régua

| # | Tarefa | Prazo | Destrava |
|---|---|---|---|
| D-U1 | **Baseline TF-IDF** (`src/featgrpo/interp/baselines.py`) nos 2.428 pares: AUROC em `val`/`test` com IC por bootstrap sobre grupos → `results/frente-d/baseline_lexical.json`. Só precisa do texto. | **D1** | C4 (régua) |
| D-U2 | Rotular `results/frente-a/amostra_2x2.csv` assim que o Gustavo exportar (intenção certa? gatilho presente/ausente? mesma intenção do original?) | D1–D2 | geração completa do 2×2 |
| D-U3 | `src/featgrpo/interp/probes.py` e `classifiers.py`, testados em **vetores aleatórios** com o formato da seção 4 do `AGENTS.md`, para rodar no D2 quando o cache chegar | D2 | D3, D4 |
| D-U4 | Esqueleto de `results/VEREDITO.md`, com a tabela C1–C5 vazia e as colunas valor, limiar, atende e fonte, e de `docs/paper/` | D2 | D7 |

## Pontos de sincronização

| Quando | O quê | Quem |
|---|---|---|
| **Fim do D0** | Pré-registro commitado (C-U1); pasta compartilhada criada (B-U4); prompts das células (A-U1) | Frederico, Caio, Gustavo |
| **Fim do D1** | `fatos_modelo.md` publicado (B-U1). Camadas fixadas para todos, inclusive para o SAE próprio | Caio → todos |
| **D1–D2** | Amostra do 2×2 → auditoria → ajuste dos prompts | Gustavo ↔ Davi |
| **Fim do D2** | Cache `pairs` publicado (B-U3) → C e D rodam o modo provisório. Geração completa do 2×2 disparada (A-U5) | Caio, Gustavo |

## Regras para não brigar no git

- Cada um mexe só nos **seus** arquivos (lista no arquivo da frente). O módulo compartilhado é `src/featgrpo/interp/data.py`: mudanças nele vão em PR pequeno avisado no grupo.
- Pedidos entre frentes vão em `results/frente-X/PEDIDOS.md` da frente que vai executar.
- Merge na `master` ao fechar cada tarefa. Antes, `git pull --rebase origin master`.
- Nada de `.env`, `artifacts/`, `.npy`, `.pt` ou checkpoints no git.
