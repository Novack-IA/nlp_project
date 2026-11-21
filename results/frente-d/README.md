# Resultados da frente D (Davi)

| Arquivo | O que é | Gerado por |
|---|---|---|
| `baseline_lexical.json` | Baselines de texto (TF-IDF, só estilo, só gatilho) nos pares originais, no **2×2 real** (3 subconjuntos) e no XSTest | `PYTHONPATH=src python -m featgrpo.interp.baselines` |
| `validacao_2x2.json` | Validador da frente D sobre o `cells_2x2.jsonl` real (plano B): aprovados, motivos e a lista de grupos reprovados | `PYTHONPATH=src python -m featgrpo.data.phase1.validate_template datasets/phase1/cells_2x2.jsonl --report results/frente-d/validacao_2x2.json` |
| `probes.json` + `probes_auroc_por_camada.png` | Probes densos no 2×2 (3 subconjuntos) e nos pares | `PYTHONPATH=src python -m featgrpo.interp.probes` |
| `c2_c4_L29_post_last_{16k,262k}.json` | Matriz C2, C2(a/b/c), recortes C4b/c, na combinação principal da frente C | `PYTHONPATH=src python -m featgrpo.interp.classifiers --layer 29 --pos post_last --sae 16k --out …` |
| `c2b_controle_aleatorio.json` + `.py` | Controle do C2(b): features aleatórias e denso no XSTest | `PYTHONPATH=src python results/frente-d/c2b_controle_aleatorio.py 16k 262k` (listas em `artifacts/_downloads/listas_c`) |
| `c3_rotulos_davi.csv` | 2º avaliador da leitura cega do C3 (Claude; ver `DESVIOS.md`) | — |
| `PEDIDOS.md` | Pedidos de outras frentes para a frente D | — |

Testes: `tests/test_validate_template.py`, `tests/test_probes_classifiers.py` e `tests/test_baselines.py` (35 testes, dados sintéticos).

---

## 1. O 2×2 real (plano B) em uma leitura

Conjunto principal: `qc_grammatical` sem os 26 grupos reprovados pelo validador (**1.697 grupos**; `test` = 242), o mesmo da frente C (`DESVIOS.md`, 2026-11-21). Os números abaixo são do `test`. O `qc_grammatical` inteiro (1.723), o `qc_pass_lenient` e o conjunto completo dão o mesmo quadro, com diferença ≤ 0,001 (ver o JSON).

| Medida | AUROC | O que mostra |
|---|---|---|
| TF-IDF, intenção, **com** gatilho | 0,996 (IC 0,993–0,999) | Separa nocivo de benigno quase perfeitamente |
| TF-IDF, intenção, **sem** gatilho | 0,997 | **Tirar o gatilho não muda nada** para o TF-IDF: as outras palavras do pedido bastam |
| **Só estilo**, intenção (sem olhar o conteúdo) | **0,889** (IC 0,861–0,916) | No plano B, `harm_trig` e `ben_trig` são os pedidos originais. O contraste de intenção **herda a diferença de estilo PKU × gerador** |
| TF-IDF, presença do gatilho | 0,767 | A forma alarmante × neutra é variada demais para um classificador de palavras decorar |
| TF-IDF dos pares aplicado ao XSTest | 0,710 (IC 0,663–0,756) | Fora da distribuição, o classificador de palavras cai bastante |

**A metade TF-IDF do C2(b).** É quanto o TF-IDF cai em cada generalização. Onde ele cai ≤ 0,02, aquela generalização fica **indefinida**, sejam quais forem as features.

| Generalização | Queda do TF-IDF (`test`) | Queda (`val`) | Define o C2(b)? |
|---|---|---|---|
| Gatilho inédito | 0,003 (IC −0,002–0,008) | 0,003 | **não** |
| Leave-one-domain-out | 0,015 (IC 0,007–0,026) | 0,015 | **não** |
| XSTest | **0,280** (IC 0,236–0,330) | 0,282 | sim |

**Consequência:** com a regra atual ("atende se ≥ 2 das 3"), **o C2(b) não pode ser atendido**, porque só uma das três generalizações é definida. Isso vale em `val` e em `test` e não depende das features. **Revisão votada em 2026-11-21:** o C2(b) passa a ser decidido só pelo XSTest (Davi, Frederico e Caio a favor), antes de existir o cache do XSTest.

## 2. Validador sobre o 2×2 real

Depois do ajuste para o plano B: **1.794 de 1.825 grupos aprovados (98,3%)**. A regra antiga aprovava 10,7%, e a diferença vinha de três erros do validador apontados pelo Gustavo.

| Motivo | Grupos | O que é |
|---|---|---|
| `edicao_fora_da_forma` | 23 | **Erro real nos dados:** a troca foi feita no meio de uma palavra flexionada. Exemplos: "racist names" → "derogatory term**ss**", "cyberbullying" → "online abuse**ing**" |
| `tamanho` | 8 | A versão sem gatilho é > 20% e > 3 palavras mais longa que a com gatilho |
| `gatilho_presente` | 1 | A forma alarmante continua no texto sem gatilho |

**26 dos 31 grupos reprovados estão no `qc_grammatical`.** O juiz de gramática deixou passar as palavras quebradas. A lista está em `validacao_2x2.json` → `falhas`.

## 3. C2: classificadores só com as features da frente C (L29 · `post_last`)

Listas da frente C (`frente-c-features`, commit `b299ef4`): 48 S_intent no 16k e 60 no 262k; **S_lex vazia** nas duas. Conjunto principal (1.697 grupos), `test`, gatilho inédito. Arquivos: `c2_c4_L29_post_last_{16k,262k}.json` e `c2b_controle_aleatorio.json`.

| | 16k | 262k |
|---|---|---|
| **C2(a)**: S_intent (L1), intenção com / sem gatilho | 0,992 / 0,992 | 0,998 / 0,997 |
| Controle: features aleatórias do mesmo tamanho (média de 20) | 0,995 / 0,995 | 0,994 / 0,994 |
| Probe denso (teto) | 0,999 / 0,998 | 0,999 / 0,998 |
| S_intent detectando o gatilho | 0,54 | 0,53 |
| **C2(b)**: queda no XSTest, S_intent (TF-IDF cai 0,278) | 0,030, **razão 0,11** (IC 0,05–0,17) | 0,018, **razão 0,06** (IC 0,03–0,11) |
| Controle: mesma razão com features aleatórias (mediana de 20) | 0,11; **20/20 atendem** | 0,08; **20/20 atendem** |
| Controle: probe denso | razão 0,11 | — |
| C4(b): só Q⁰ *retrieved* (211 grupos), C2(a); C2(b) razão | 0,979 / 0,992; 0,17 (IC 0,08–0,27) | 0,969 / 0,973; −0,01 (IC −0,11–0,09) |
| C4(c): tamanho pareado (684 grupos), C2(a); C2(b) razão | 0,986 / 0,991; 0,18 (IC 0,11–0,27) | 0,999 / 0,996; 0,16 (IC 0,09–0,24) |

**Leitura:**
- **O C2 atende pela regra pré-registrada**, nos dois SAEs: S_intent separa nocivo de benigno com e sem gatilho, e no XSTest perde ~10 vezes menos que o TF-IDF.
- **Mas a S_intent não é melhor do que features sorteadas.** Dentro do 2×2, 48 features frequentes quaisquer acertam o mesmo (0,995). No XSTest, os 20 sorteios também perdem ~10 vezes menos que o TF-IDF, e o denso idem. Então a vantagem sobre o classificador de palavras é **da representação do modelo**, não da S_intent em particular.
- **O que é específico da S_intent** vem da frente C, não daqui: ela é selecionada por ter efeito de intenção e *nenhum* efeito do gatilho (C1), não reage ao gatilho solto (C3) e quase não detecta o gatilho (0,54 aqui). O C2 mostra que essa seleção não custa nada em classificação, mas não mostra que ela é necessária.
- **C2(c) indefinido**: sem S_lex, não há o lado lexical da dissociação.
- **C4(b/c)**: o C2(a) e o C2(b) se mantêm nos dois recortes, nos dois SAEs. Os 211 grupos do (b) são os do conjunto principal; o recorte inteiro tem 247. O C1 nos recortes é da frente C: (c) atende, (b) inconclusivo por falta de poder.
- O controle aleatório no XSTest foi acrescentado **depois** de ver o C2 (`DESVIOS.md`); a regra e o "atende" não mudam.

## 4. Probes densos no 2×2 (âncora densa do veredito)

Cache `cells2x2` da frente B (137/137 arquivos conferidos pelo sha256 do `MANIFEST.json`). Conjunto principal: `qc_grammatical` sem os 26 grupos reprovados pelo validador (1.697 grupos; `DESVIOS.md`, 2026-11-21). Regressão logística L2 no residual completo, C escolhido por GroupKFold por gatilho no `train`. AUROC no `test`, recorte **gatilho inédito** (157 grupos):

| Camada · posição | Intenção | Treina com gatilho → testa sem | Treina sem → testa com | Gatilho | Gatilho, treina no benigno → testa no nocivo |
|---|---|---|---|---|---|
| L9 · `post_last` | 0,999 | 0,998 | 0,999 | 0,69 | 0,66 |
| L17 · `post_last` | 0,999 | 0,999 | 1,000 | 0,64 | 0,60 |
| L22 · `post_last` | 0,999 | 0,999 | 0,999 | 0,62 | 0,61 |
| **L29 · `post_last`** (combinação principal da frente C) | **0,998** | **0,998** | **0,999** | **0,64** | **0,59** |
| L29 · `inst_last` | 0,988 | 0,987 | 0,989 | 0,61 | 0,59 |
| L29 · `mean` | 1,000 | 1,000 | 1,000 | 0,70 | 0,65 |
| L29 · `slot_forma` (no gatilho) | 0,971 | 0,966 | 0,968 | 0,83 | 0,80 |

As outras combinações estão no JSON; todas ficam entre 0,97 e 1,00 na intenção. `qc_grammatical` inteiro e o conjunto completo dão os mesmos números (diferença ≤ 0,001). `slot_intencao` não tem dados no plano B (toda NaN).

**Leitura:**
- **A intenção está no residual e não depende do gatilho.** Um probe treinado só nas células com gatilho acerta 0,998 nas células sem gatilho, com gatilhos que ele nunca viu. Isso vale em todas as camadas.
- **O gatilho quase não está no residual depois do pedido.** No último token, o probe de gatilho fica em 0,59–0,70. Só na própria posição do gatilho (`slot_forma`) ele chega a 0,83. Isso bate com a frente C não ter achado nenhuma S_lex.
- **Cuidado: o probe denso não vence as réguas de texto.** O TF-IDF também acerta 0,997 sem o gatilho, e o estilo sozinho acerta 0,89 (seção 1). No plano B, o contraste de intenção é entre pedidos diferentes, então 0,998 diz que a informação existe, não que ela é "intenção" e não palavras. Quem separa isso é o XSTest (C2(b)).
- **Cossenos entre direções:** intenção × recusa ≈ 0,99 em todas as camadas, mas é quase por construção: a direção de recusa vem dos pares (Q⁺ − Q⁰), e as células com gatilho são esses mesmos pares. O cosseno intenção × gatilho muda de sinal entre camadas e posições (−0,83 a 0,94), então não serve como medida; fica no JSON só como descrição.

## 5. Probes densos nos pares (diagnóstico, não entra no veredito)

Cache `pairs` da frente B (baixado da pasta compartilhada, 65/65 arquivos conferidos pelo sha256 do `MANIFEST.json`). Regressão logística L2 no residual completo, Q⁺ × Q⁰, C escolhido por GroupKFold por gatilho no `train`. AUROC no `test`:

| Camada | `inst_last` | `post_last` | `post` | `mean` | `trig` |
|---|---|---|---|---|---|
| L9 | 0,999 | 0,999 | 0,999 | 1,000 | 0,991 |
| L17 | 0,999 | 0,999 | 0,999 | 0,999 | 0,993 |
| L22 | 0,996 | 0,999 | 0,998 | 0,999 | 0,987 |
| L29 | 0,995 | 0,998 | 0,998 | 0,999 | 0,980 |

Com gatilho inédito, os números são os mesmos (0,976–1,000). **Leitura:** o residual separa Q⁺ de Q⁰ quase perfeitamente em todas as camadas e posições, tanto quanto o TF-IDF (0,995). Nos pares originais, isso não distingue intenção de palavras ou estilo, então serve só para confirmar que o cache e o código funcionam. A âncora densa do veredito sai do 2×2 (`cells2x2`), quando o cache dele existir.

## 6. Pares originais (diagnóstico, não entra no veredito)

| Classificador (`test`, 347 pares) | AUROC | IC 95% |
|---|---|---|
| TF-IDF | 0,995 | 0,991–0,998 |
| Só estilo | 0,881 | 0,860–0,903 |
| Só gatilho | 0,500 | — |

Gatilho visto × inédito: 0,996 × 0,994. Por domínio: bias 0,989, cyber 0,995, privacy 0,999, violence 0,997.
