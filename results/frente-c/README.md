# Resultados da frente C (Frederico)

| Arquivo | O que é | Gerado por |
|---|---|---|
| `preregistro.md` | Pré-registro da frente C: posições, variante do SAE, escolhas na descoberta, pisos de poder, classes e critérios C1, C3 e C4a. Escrito antes de qualquer ativação. | à mão |
| `poder.json` | Base da seção 4 do pré-registro: contagens de grupos por split e domínio, projeções do 2×2 e simulação da regra de classes. Não usa ativações. | `PYTHONPATH=src python -m featgrpo.interp.sim_poder` |
| `feature_classes_L{L}_{pos}_{sae}.csv` | Uma linha por feature ativa (≥ 5% em alguma célula): d_I, d_L, d_X, ICs, p, q, classe e, na combinação principal depois de aberta a confirmação, os d no `test`. | `PYTHONPATH=src python -m featgrpo.interp.feature_stats --sae 16k` |
| `listas/S_{intent,lex}_L{L}_{pos}_{sae}.json` | Listas para as frentes B (C5) e D (C2), ordenadas por \|d\|, no formato que `classifiers.load_list` lê. | idem |
| `c1_c3.json` | Resumo por SAE: as 8 combinações na descoberta, a principal, o C1, a replicação, o top-20 e a análise por domínio. | idem |
| `provisorio_*.csv` / `provisorio_{sae}.json` | Modo provisório nos pares originais (diagnóstico, fora do veredito), com o controle de estilo. | `... feature_stats --modo provisorio` |
| `provisorio.md` | O mesmo, legível: resumo das 8 combinações e o top-20 de cada lado com o rótulo do Neuronpedia. | `... autointerp provisorio` |
| `robustez_<subconjunto>/` | O C1–C3 nos subconjuntos de robustez do QC (`qc_pass_lenient`, `todos`), na combinação principal. O principal é `qc_grammatical` (`DESVIOS.md`, 2026-11-21). | `... feature_stats --subconjunto <nome>` |
| `sanidade_sae.json` | Teste de sanidade do SAE (C-U3), só com pedidos escritos à mão e exemplos públicos do Neuronpedia: chat template, convenção de camada (resid_post L = `hidden_states[L+1]`, confere com o Neuronpedia com erro < 1%) e qualidade de reconstrução nas duas posições principais. | `python -m featgrpo.interp.sanidade_sae` numa 4090 (LAB150) |
| `PEDIDOS.md` | Pedidos de outras frentes para a frente C. | — |

Na posição, os arquivos usam o ID do cache: `inst_last` = fim_pedido, `post_last` = pre_resposta.

**Termos:** *descoberta* = grupos de `train` + `val` (onde as features são escolhidas); *confirmação* = grupos de `test` (aberta uma vez, no fim).
