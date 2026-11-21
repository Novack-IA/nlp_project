# results/frente-b: ativações, latentes e qualidade dos SAEs (Caio)

| Arquivo | O que é | Gerado por |
|---|---|---|
| `RELATORIO.md` | Resumo de tudo o que a frente B rodou, achou e corrigiu. **Comece por aqui** | à mão, a partir dos arquivos abaixo |
| `fatos_modelo.md` / `.json` | Camadas, `sae_id`, convenção `resid_post L == hidden_states[L+1]`, chat template, definição das posições e formato do cache | `interp/model_facts.py` |
| `qualidade_saes.md` | Leitura da qualidade dos SAEs (B4), em todos os tokens e nas posições `inst_last`/`post_last` | à mão, a partir dos CSV |
| `qualidade_saes.csv` | FVE, L0, features sem ativar, ΔNLL, KL e fração recuperada por camada × SAE | `interp/sae_quality.py` |
| `qualidade_saes_posicoes.csv` | As mesmas medidas só no token de `inst_last` e de `post_last` | `interp/sae_quality.py --skip-full --positions inst_last post_last` |
| `qualidade_saes.json` | Argumentos, commit e tamanhos de amostra de cada rodada do `sae_quality.py` | `interp/sae_quality.py` |
| `MANIFEST.json` | sha256 de cada arquivo do cache (`artifacts/acts/gemma-3-4b-it/`), commit, versões e `sae_id` | `interp/extract.py` (cópia do MANIFEST do cache) |
| `PEDIDOS.md` | Pedidos das outras frentes para a B e as respostas | à mão |

O cache em si (`.npy`/`.npz`) **não vai para o git**. Ele fica na pasta compartilhada, com a árvore da seção 4 do `AGENTS.md`. Leia com `featgrpo.interp.cache_io`.
