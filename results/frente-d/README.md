# Resultados da frente D (Davi)

| Arquivo | O que é | Gerado por |
|---|---|---|
| `baseline_lexical.json` | Baselines de texto (TF-IDF, só estilo, só gatilho) nos pares originais. As seções do 2×2 e do XSTest ficam "pendente" até a frente A entregar. | `PYTHONPATH=src python -m featgrpo.interp.baselines` |
| `PEDIDOS.md` | Pedidos de outras frentes para a frente D. | — |

## `baseline_lexical.json` em uma leitura

Os três classificadores são treinados só no `train`. O C é escolhido por validação cruzada agrupada por gatilho, e os ICs de 95% saem de bootstrap sobre grupos. Resultados no `test` (347 pares):

| Classificador | AUROC | IC 95% | O que mostra |
|---|---|---|---|
| TF-IDF (palavras + caracteres) | **0,995** | 0,991–0,998 | Só pelas palavras, nocivo e benigno se separam quase perfeitamente nos pares originais. |
| **Só estilo** (tamanho, pontuação, forma da pergunta) | **0,881** | 0,860–0,903 | Sem olhar o conteúdo, só o "jeito" do texto já acerta 88%. É a confusão PKU × gerador. |
| Só gatilho | 0,500 | — | O gatilho está nos dois lados (100% dos Q⁺ e 99,9% dos Q⁰), então sozinho não separa nada. Era o esperado. |

**Gatilho visto × inédito no treino** (133 e 214 pares do `test`): o TF-IDF dá 0,996 e 0,994. Nos pares originais, repetir o gatilho entre treino e teste **não infla** o baseline. Faz sentido: o gatilho aparece nos dois lados do par, então não serve para separá-los. O recorte continua necessário para as features e para o 2×2, onde o gatilho varia entre as células.

**Por domínio** (TF-IDF, `test`): bias 0,989, cyber 0,995, privacy 0,999, violence 0,997.

**Consequência para o projeto:**
- Nos pares originais, qualquer método (SAE, probe denso) vai ter número alto. Isso não prova nada sobre intenção, porque palavras e estilo já resolvem. Esses resultados são **diagnóstico**, não evidência.
- A régua de verdade é a queda do TF-IDF nas células do 2×2 (`cells_2x2.queda_auroc_tfidf`), calculada automaticamente quando `datasets/phase1/cells_2x2.jsonl` existir.
- O estilo sozinho chegar a 0,88 reforça a necessidade de o 2×2 reescrever as 4 versões com o mesmo gerador, ou a partir da frase-molde (proposta no PR #1).
