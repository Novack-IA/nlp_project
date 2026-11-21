# Resultados da frente D (Davi)

| Arquivo | O que é | Gerado por |
|---|---|---|
| `baseline_lexical.json` | Baselines de texto (TF-IDF, só estilo, só gatilho) nos pares originais, no **2×2 real** (3 subconjuntos) e no XSTest | `PYTHONPATH=src python -m featgrpo.interp.baselines` |
| `validacao_2x2.json` | Validador da frente D sobre o `cells_2x2.jsonl` real (plano B): aprovados, motivos e a lista de grupos reprovados | `PYTHONPATH=src python -m featgrpo.data.phase1.validate_template datasets/phase1/cells_2x2.jsonl --report results/frente-d/validacao_2x2.json` |
| `probes.json` + `probes_auroc_por_camada.png` | *(quando o cache da frente B estiver baixado)* Probes densos | `PYTHONPATH=src python -m featgrpo.interp.probes` |
| `c2_c4.json` | *(quando existirem latentes do 2×2 e listas da frente C)* Matriz C2, C2(a/b/c), recortes C4b/c | `PYTHONPATH=src python -m featgrpo.interp.classifiers --layer L --pos P --sae 16k` |
| `PEDIDOS.md` | Pedidos de outras frentes para a frente D | — |

Testes: `tests/test_validate_template.py`, `tests/test_probes_classifiers.py` e `tests/test_baselines.py` (35 testes, dados sintéticos).

---

## 1. O 2×2 real (plano B) em uma leitura

Conjunto principal: `qc_grammatical` (1.723 grupos; `test` = 244), como no pré-registro da frente C. Os números abaixo são do `test`. Os do `qc_pass_lenient` e do conjunto completo dão o mesmo quadro (ver o JSON).

| Medida | AUROC | O que mostra |
|---|---|---|
| TF-IDF, intenção, **com** gatilho | 0,996 | Separa nocivo de benigno quase perfeitamente |
| TF-IDF, intenção, **sem** gatilho | 0,997 | **Tirar o gatilho não muda nada** para o TF-IDF: as outras palavras do pedido bastam |
| **Só estilo**, intenção (sem olhar o conteúdo) | **0,890** (IC 0,862–0,915) | No plano B, `harm_trig` e `ben_trig` são os pedidos originais. O contraste de intenção **herda a diferença de estilo PKU × gerador** |
| TF-IDF, presença do gatilho | 0,767 | A forma alarmante × neutra é variada demais para um classificador de palavras decorar |
| TF-IDF dos pares aplicado ao XSTest | 0,710 (IC 0,663–0,756) | Fora da distribuição, o classificador de palavras cai bastante |

**A metade TF-IDF do C2(b).** É quanto o TF-IDF cai em cada generalização. Onde ele cai ≤ 0,02, aquela generalização fica **indefinida**, sejam quais forem as features.

| Generalização | Queda do TF-IDF (`test`) | Queda (`val`) | Define o C2(b)? |
|---|---|---|---|
| Gatilho inédito | 0,003 (IC −0,002–0,009) | 0,003 | **não** |
| Leave-one-domain-out | 0,014 (IC 0,005–0,024) | 0,015 | **não** |
| XSTest | **0,281** (IC 0,234–0,329) | 0,282 | sim |

**Consequência:** com a regra atual ("atende se ≥ 2 das 3"), **o C2(b) não pode ser atendido**, porque só uma das três generalizações é definida. Isso vale em `val` e em `test` e não depende das features. A regra precisa ser revista em grupo (proposta no PR).

## 2. Validador sobre o 2×2 real

Depois do ajuste para o plano B: **1.794 de 1.825 grupos aprovados (98,3%)**. A regra antiga aprovava 10,7%, e a diferença vinha de três erros do validador apontados pelo Gustavo.

| Motivo | Grupos | O que é |
|---|---|---|
| `edicao_fora_da_forma` | 23 | **Erro real nos dados:** a troca foi feita no meio de uma palavra flexionada. Exemplos: "racist names" → "derogatory term**ss**", "cyberbullying" → "online abuse**ing**" |
| `tamanho` | 8 | A versão sem gatilho é > 20% e > 3 palavras mais longa que a com gatilho |
| `gatilho_presente` | 1 | A forma alarmante continua no texto sem gatilho |

**26 dos 31 grupos reprovados estão no `qc_grammatical`.** O juiz de gramática deixou passar as palavras quebradas. A lista está em `validacao_2x2.json` → `falhas`.

## 3. Pares originais (diagnóstico, não entra no veredito)

| Classificador (`test`, 347 pares) | AUROC | IC 95% |
|---|---|---|
| TF-IDF | 0,995 | 0,991–0,998 |
| Só estilo | 0,881 | 0,860–0,903 |
| Só gatilho | 0,500 | — |

Gatilho visto × inédito: 0,996 × 0,994. Por domínio: bias 0,989, cyber 0,995, privacy 0,999, violence 0,997.
