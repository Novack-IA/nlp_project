# Pré-registro da frente C

> **Status:** versão 1, escrita antes de qualquer ativação ser extraída ou vista, com o adendo dos dados do 2×2 (2026-11-21), escrito antes do cache do 2×2. Mudanças posteriores vão para `results/DESVIOS.md`.
> **Autor:** Frederico (frente C) · **Base:** `docs/tarefas/00-visao-geral.md` e `docs/tarefas/frederico-frente-c.md` (commit 21d1fb4)

**Termos usados aqui:**
- **Descoberta:** os grupos com `split` = `train` ou `val`. É onde as features são calculadas, classificadas e escolhidas.
- **Confirmação:** os grupos com `split` = `test`. Fica fechada até o fim e é aberta uma vez, só para conferir as features escolhidas na descoberta.
- **Gatilho visto × inédito:** o mesmo gatilho pode aparecer em vários grupos, inclusive dos dois lados da separação. "residential address", por exemplo, está em 72 pares do `train` e em 11 do `test`. Um grupo da confirmação tem **gatilho visto** se o seu gatilho aparece em algum grupo de `train`, e **gatilho inédito** se não aparece. Antes de comparar, os gatilhos passam para minúsculas e são cortados até o radical (Porter, `normalize_trigger` de `validate_template.py`), então "Hacking into" conta como visto se "hack into" está no `train`.
  - **Por que importa:** uma feature pode "replicar" no gatilho visto só por reconhecer o mesmo vocabulário que viu na descoberta. No gatilho inédito, ela enfrenta palavras novas, e manter o efeito ali é evidência de intenção, não de vocabulário. Por isso o C1 usa o inédito (regra 1 do grupo, seção 5).
  - **Nos pares originais:** dos 347 pares do `test`, 133 têm gatilho visto e 214 têm gatilho inédito.

| Seção | Situação |
|---|---|
| 1. Posições lidas | Decidida em 2026-11-20 |
| 2. Variante de L0 do SAE | Decidida em 2026-11-20 |
| 3. Combinação principal, listas e top-20 | Decidida em 2026-11-20 |
| 4. Piso de poder | Decidida em 2026-11-20 |
| 5. Medidas, classes e critérios (do grupo) | Copiada; 2 detalhes em aberto decididos em 2026-11-20 |
| Adendo: dados do 2×2 | Escrito em 2026-11-21, antes do cache do 2×2 |

## 1. Posições lidas

**Decisão:** o C1, o C3 e o C4a são avaliados **só em duas posições principais**, as duas com o mesmo token nas 4 células do 2×2 e que enxergam o pedido inteiro. As outras posições são secundárias: aparecem num apêndice e não decidem nada.

| Rótulo no texto | ID no cache | Token lido | O que representa | Uso |
|---|---|---|---|---|
| `fim_pedido` | `inst_last` | último token do texto do usuário, logo antes de `<end_of_turn>` | o que o modelo entendeu do pedido (intenção) | **principal** |
| `pre_resposta` | `post_last` | último token de `<end_of_turn>\n<start_of_turn>model\n`, de onde sai a primeira palavra da resposta | o que o modelo vai fazer (decisão de recusar) | **principal** |
| `transicao` | `post` | média dos 5 tokens desse sufixo | mistura das duas | secundária |
| — | `slot_forma`, `slot_intencao`, `mean` | tokens das lacunas; média do pedido | efeitos triviais por construção (abaixo) | secundária, como sanidade |

Os rótulos são os nomes usados no texto, nas figuras e no paper. Os IDs do cache seguem a seção 4 do `AGENTS.md`, para não mudar o contrato com as frentes B e D.

**Por que as lacunas não decidem:**
- Na `slot_intencao`, as células nocivas e benignas têm palavras diferentes. Qualquer feature dessas palavras mostra efeito de intenção, o que faria o C1 passar sem o modelo precisar representar nocividade.
- Na `slot_forma`, as células com e sem gatilho têm palavras diferentes, então o efeito lexical é trivial. Se a lacuna de forma vem antes da de intenção no molde, a atenção causal impede que ela veja a intenção, e o efeito de intenção ali é exatamente zero.
- O `mean` inclui os tokens das lacunas e herda os dois problemas, diluídos.
- Nas duas posições principais, o token é idêntico nas 4 células. Uma diferença de ativação ali só pode vir do que o modelo computou sobre o pedido.

**Por que duas posições principais e não uma:** Zhao et al. (NeurIPS 2025, arXiv:2507.11878) mostram que a nocividade fica codificada no último token da instrução e a recusa no último token da sequência. A hipótese do projeto (o over-refusal segue a superfície) vive na diferença entre as duas.

**Regras que decorrem da decisão:**
1. **Mesmo token no `fim_pedido`.** Se o molde de um grupo termina numa lacuna, o último token muda entre as células. Esses grupos saem só da análise em `fim_pedido`, conferido por `inst_last_token_id` no `index`. O número de grupos excluídos é reportado.
   - **Revisão prevista:** se a exclusão passar de 20% dos grupos, ou deixar o `fim_pedido` abaixo do piso de poder (seção 4), esta regra é revista com o grupo. As opções são ler o `<end_of_turn>` (primeiro token do sufixo, igual nas 4 células e que também vê o pedido inteiro) ou pedir à frente A que os moldes terminem em texto fixo. A contagem sai só do texto tokenizado, sem nenhuma ativação, então a revisão acontece antes de vermos dados. A escolha vai para `results/DESVIOS.md`.
2. **Família do Benjamini–Hochberg.** A família é features × 4 camadas × 2 posições principais, corrigida separadamente para o efeito de intenção (q_I) e o lexical (q_L), e separadamente para cada SAE (16k, 262k). As posições secundárias têm família própria. Isso evita que os efeitos triviais das lacunas afrouxem o limiar das posições principais.
3. **Controles `trig_only` / `neutral_only` (C3)** são lidos nas mesmas duas posições, com a mesma regra do mesmo token.
4. **Sanidade (secundária):** na `slot_forma` devem existir features com efeito lexical forte. Se não existirem, o SAE não representa o gatilho, e isso é reportado antes de qualquer conclusão sobre S_lex. Também na `slot_forma`, quando ela vem antes da lacuna de intenção, o efeito de intenção tem de ser exatamente zero; se não for, há erro de alinhamento de tokens na extração.
5. **Previsão exploratória (não decide o veredito):** se o over-refusal segue a superfície, S_intent é mais forte em `fim_pedido`, e o efeito do gatilho (L) aparece mais em `pre_resposta` do que em `fim_pedido`.
6. **Sem system prompt.** Cada pedido entra como uma única mensagem do usuário no chat template do Gemma 3, `<bos><start_of_turn>user\n{texto}<end_of_turn>\n<start_of_turn>model\n`, sem system prompt nem outro texto. Qualquer acréscimo muda as posições e o que elas leem.

**Pedido associado:** extração de `inst_last` pela frente B (`results/frente-b/PEDIDOS.md`, 2026-11-20).

## 2. Variante de L0 do SAE

**Decisão:** todas as análises da frente C usam a variante **`medium`** do Gemma Scope 2, fixada antes de ver dados. As variantes `small` e `big` não são rodadas.

| Uso | Release no SAELens | `sae_id` | Neuronpedia |
|---|---|---|---|
| Principal (C1, C3) | `gemma-scope-2-4b-it-res` | `layer_{L}_width_16k_l0_medium` | `gemma-3-4b-it/{L}-gemmascope-2-res-16k` |
| Robustez (C4a) | `gemma-scope-2-4b-it-res` | `layer_{L}_width_262k_l0_medium` | `gemma-3-4b-it/{L}-gemmascope-2-res-262k` |

L ∈ {9, 17, 22, 29}. O L0 médio dessa variante é ~53 na camada 9 e ~60 nas outras (valores do SAELens 6.56.1). O nome da release no SAELens é `gemma-scope-2-4b-it-res`, não `gemma-scope-2-4b-it-resid_post` como diz a seção 3 do `AGENTS.md`.

**Por que `medium`:**
- É a única variante com fichas no Neuronpedia (explicação automática, exemplos de maior ativação, palavras promovidas), nas 4 camadas e nas duas larguras. O C3 usa essas fichas como ponto de partida da avaliação cega; sem elas, seria preciso gerar exemplos e explicações com um LLM local.
- Fica no meio da troca entre esparsidade e reconstrução. `small` (L0 ~20) reconstrói pior e arrisca perder no erro do SAE a informação que procuramos. `big` (L0 ~150) reconstrói melhor, mas tem features mais fragmentadas e não tem fichas.
- O critério de escolha não olha o efeito de intenção nem o lexical. Escolher a variante depois de ver qual dá mais S_intent seria justamente o "procurar até achar" que este documento existe para impedir.

**Regras que decorrem da decisão:**
1. **Qualidade do SAE é sanidade, não escolha.** A variância explicada e o ΔNLL medidos pela frente B nas duas posições principais são reportados por camada. Se uma camada sair ruim, isso entra nas limitações. Não se troca de variante depois.
2. **`small` e `big` não testadas** entram nas limitações do relatório.
3. **Opcional, só se sobrar tempo:** repetir o C4a no `layer_{L}_width_262k_l0_medium_seed_1`, o mesmo SAE treinado com outra semente, como um segundo dicionário. Não decide nada.

## 3. Combinação principal, listas e top-20

**Combinação** = um par (camada, posição principal): 4 camadas × 2 posições = 8. Cada camada tem o seu SAE, então cada combinação tem as suas próprias classes S_intent e S_lex.

**Decisão:** todas as escolhas abaixo usam **só a descoberta**. A confirmação é aberta uma vez, e só na combinação principal. Escolher a combinação olhando a confirmação daria 8 chances de "replicar" por acaso.

1. **Combinação principal.** É a combinação com mais features em S_intent na descoberta; em caso de empate, a de maior mediana de \|d_I\| entre as S_intent. O C1 é avaliado nela: \|S_intent\| ≥ 5 e, na confirmação com gatilho inédito, ≥ 70% dessas features com o mesmo sinal e \|d_I\| ≥ 0,3. As outras 7 combinações saem numa tabela da descoberta, só como descrição.
2. **Listas para as frentes B e D.** Todas as S_intent e S_lex da combinação principal, ordenadas por \|d\|, com d, IC 95%, q e sinal, em `results/frente-c/listas/`. As listas das outras 7 combinações vão junto, marcadas como secundárias. Quantas features usar no C2 e no C5 fica a cargo das frentes D e B.
3. **Top-20 do C3.** As 20 S_intent de maior \|d_I\| na combinação principal, **de qualquer sinal**, e as 20 S_lex de maior \|d_L\|. Reportamos quantas têm cada sinal (d_I positivo = mais ativa nos pedidos nocivos). Se uma classe tiver menos de 20 features, entram todas.
4. **Intenção geral × tema de um domínio (análise secundária).** Para cada S_intent, calculamos o d_I dentro de cada domínio (bias, cyber, privacy, violence) e contamos em quantos ela mantém o sinal com \|d_I\| ≥ 0,3. Uma feature é **geral** se isso vale em ≥ 3 domínios. Reportamos quantas S_intent são gerais e quantas são de um domínio só. Não muda o C1. Serve para dizer se o sinal achado é de intenção nociva ou de tema: uma feature que só acende nos pedidos nocivos de um domínio pode passar no limiar com os domínios juntos.

## 4. Piso de poder

**Decisão:** com poucos grupos, o IC de d fica largo demais e o teste de equivalência reprova até features que não reagem ao gatilho. O resultado seria "0 S_intent", lido como "NÃO", quando na verdade faltaram dados. Os pisos abaixo separam "não existe" de "não deu para medir".

1. **Classificação (descoberta): ≥ 260 grupos**, contados por posição principal depois das exclusões do `fim_pedido`. Abaixo disso, a classificação naquela posição sai como **"inconclusiva por falta de poder"**, nunca como "NÃO", e o relatório informa o n.
2. **Replicação (confirmação com gatilho inédito): ≥ 50 grupos.** Abaixo disso, a replicação do C1 usa a confirmação inteira (gatilho visto + inédito) e é marcada como "baixo poder". Esta regra alternativa vale desde já, para não ser um desvio decidido depois de ver os números.
3. **Análise por domínio (seção 3, item 4): ≥ 50 grupos por domínio na descoberta.** Abaixo disso, o domínio aparece como "dados insuficientes", e não como "a feature não se mantém nesse domínio".
4. **Sensibilidade (secundária):** reportamos também quantas features seriam S_intent pela regra anterior ao PR #1 (\|d_I\| ≥ 0,5, q_I < 0,01 e \|d_L\| < 0,2 na estimativa pontual, sem exigir o IC). Se a contagem mudar muito entre as duas regras, o texto diz isso. O C1 continua usando a regra oficial.

**De onde vêm os números** (`results/frente-c/poder.json`, gerado por `src/featgrpo/interp/sim_poder.py`, sem nenhum dado de ativação):
- **260** é o n em que uma feature que não reage ao gatilho (d_L = 0) passa no teste de equivalência com ~80% de chance: a meia-largura do IC de d é ~1,96/√n ≈ 0,12, e o d_L estimado precisa ficar a menos de 0,08 do zero.
- **Simulação da regra completa** (300 repetições por caso; ativações esparsas do tipo ReLU; q < 0,01 do BH aproximado por p < 10⁻⁶):

| Feature plantada | n = 150 | 260 | 400 | 550 | 680 |
|---|---|---|---|---|---|
| Intenção pura (d_I 0,6; d_L 0) | 34% | 75% | 96% | 99% | 100% |
| Intenção forte (d_I 1,0; d_L 0) | 33% | 81% | 98% | 100% | 100% |
| Intenção com vazamento leve (d_I 0,6; d_L 0,1) | 22% | 34% | 49% | 69% | 77% |
| Lexical (d_I 0; d_L 0,8) | 0% | 0% | 0% | 0% | 0% |
| Tema de um domínio (25% dos grupos; d_I 0,55) | 17% | 68% | 88% | 96% | 97% |
| Nula | 0% | 0% | 0% | 0% | 0% |

  Leitura: no n esperado a regra recupera quase todas as features de intenção e nunca classifica uma lexical ou nula como S_intent. Ela é rígida de propósito com features que vazam um pouco do gatilho, o que motiva a análise de sensibilidade (item 4). Features de tema passam tão fácil quanto as de intenção, o que motiva a análise por domínio (seção 3, item 4).
- **Grupos esperados**, projetando o split e os domínios reais dos 2.428 pares (supõe rendimento igual em todos os splits e domínios):

| Fração dos pares que vira grupo do 2×2 | Descoberta (com 20% perdidos no `fim_pedido`) | Confirmação com gatilho inédito | Menor domínio na descoberta |
|---|---|---|---|
| 20% (~486 grupos) | 333 | 43 | 99 |
| 33% (~801, meta da frente A) | 549 | 71 | 163 |
| 50% (~1.214) | 832 | 107 | 247 |

  Só o piso da replicação (item 2) dispara, e só no cenário de 20%.

## 5. Medidas, classes e critérios (do grupo)

Copiado de `docs/tarefas/00-visao-geral.md` (commit 21d1fb4), com as decisões das seções 1 a 4 aplicadas. Dois detalhes que o documento do grupo deixa em aberto foram decididos aqui, em 2026-11-20: o que conta como "maioria" no controle do C3 e qual combinação usar no C4a.

**Efeitos por feature**, em cada combinação, com a(h, t) = ativação da feature na célula (intenção h ∈ {+, 0}, gatilho t ∈ {1, 0}) de um grupo g:
- intenção: I_g = ½[a(+,1) + a(+,0) − a(0,1) − a(0,0)]
- lexical: L_g = ½[a(+,1) + a(0,1) − a(+,0) − a(0,0)]
- interação: X_g = ½[a(+,1) − a(+,0) − a(0,1) + a(0,0)]
- tamanho de efeito pareado: d = média dos efeitos entre os grupos ÷ desvio-padrão deles, para I, L e X.

**Testes:**
- p-valor de Wilcoxon pareado, com o método de Pratt para os zeros;
- q de Benjamini–Hochberg na família definida na seção 1, com q < 0,01;
- IC 95% de d por bootstrap sobre grupos, com 1.000 sorteios e semente fixa (0).

**Features raras:** só entram features ativas (> 0) em ≥ 5% dos itens de alguma célula, na descoberta.

**Classes:**

| Classe | Critério |
|---|---|
| S_intent | \|d_I\| ≥ 0,5, q_I < 0,01 e IC 95% de d_L inteiro dentro de (−0,2; 0,2) |
| S_lex | \|d_L\| ≥ 0,5, q_L < 0,01 e IC 95% de d_I inteiro dentro de (−0,2; 0,2) |
| S_mixed | \|d_I\| ≥ 0,5, \|d_L\| ≥ 0,5, q_I < 0,01 e q_L < 0,01 |
| resto | irrelevante |

**Regras de análise do grupo (PR #1):**
1. Todo número da confirmação sai em duas partes, gatilho visto e gatilho inédito. Os critérios usam o inédito, salvo o piso da seção 4.
2. "Sem efeito" exige o IC 95% de d dentro de ±0,2 (teste de equivalência, na tabela acima).
3. O C2 é relativo ao TF-IDF onde ele sofre. ~~Gatilho inédito, leave-one-domain-out e XSTest; atende com ≥ 2 das 3 (frente D, emenda 3 do Gustavo).~~ **Atualizado em 2026-11-21** (proposta do Davi no PR #8, voto a favor no adendo 3; `DESVIOS.md`): o C2(b) é **só o XSTest**. S_intent, treinado no 2×2, perde no XSTest no máximo metade da AUROC que o TF-IDF perde (razão ≤ 0,5 e limite superior do IC 95% da razão < 1). Gatilho inédito e leave-one-domain-out viram descrição, porque neles o TF-IDF cai ≤ 0,02 e a régua fica indefinida.
4. Os pares originais são diagnóstico e não entram no veredito.
5. C1 e C2 saem também por domínio e por tipo de gatilho (`trigger_types.jsonl` da frente A), como análise secundária.

**Critérios da frente C:**
- **C1. Existência:** na combinação principal (seção 3), \|S_intent\| ≥ 5, e na confirmação com gatilho inédito ≥ 70% dessas features mantêm o sinal com \|d_I\| ≥ 0,3.
- **C3. Não é lexical:**
  1. *Controle.* Para cada S_intent da combinação principal, o d de a(`trig_only`) − a(`neutral_only`), pareado por frase de controle, na mesma posição. Atende se **mais da metade** das S_intent tiver o IC 95% desse d dentro de ±0,2. Para S_lex espera-se o contrário, o que mostra que o controle funciona.
  2. *Avaliação cega.* Uma ficha para cada feature do top-20 (seção 3), com a explicação e os exemplos do Neuronpedia e os 10 itens do 2×2 em que a feature mais acende, **só da descoberta**. As fichas vão em ordem aleatória (semente fixa), sem dizer a classe. Frederico e Davi rotulam cada uma como conceitual (intenção, dano, ação perigosa) ou lexical (palavra ou expressão específica). Atende se ≥ 60% das fichas de S_intent forem conceituais para os dois. Reportamos o κ de Cohen entre os dois.
- **C4a. Outro dicionário:** no 262k `medium`, na **mesma combinação** escolhida no 16k, com as mesmas regras e família de BH própria. Atende se o C1 valer ali. O pareamento entre dicionários (para cada S_intent do 16k, a feature do 262k com maior cosseno entre vetores do decoder, e se ela também é S_intent) é descritivo, porque uma feature do 16k costuma se dividir em várias no 262k (Bricken et al., 2023).

**Modo provisório (pares originais):** roda o mesmo código nas mesmas duas posições, com intenção = a(Q⁺) − a(Q⁰). Serve para depurar e não entra no veredito. O `provisorio.md` abre avisando que, nos pares, só o estilo do texto já separa nocivo de benigno com AUROC 0,88 (`results/frente-d/baseline_lexical.json`).

**Reprodutibilidade:** cada número em `results/frente-c/` sai de um script em `src/featgrpo/interp/`, com os argumentos, as sementes, o modelo, a release e o `sae_id` do SAE, a versão do SAELens e o commit registrados no próprio JSON.

## Adendo: dados do 2×2

Escrito em 2026-11-21, antes de qualquer ativação do 2×2. Tudo foi contado só no texto tokenizado; o modo provisório nos pares originais já tinha rodado, como diagnóstico.

**Desenho: plano B.** A frase-molde aproveitou 14% e 18% dos grupos nas rodadas secas, abaixo do corte de 40%, então vale o plano B: troca só do trecho do gatilho. Ele aproveitou 82% na rodada seca e 75% na geração completa (`results/frente-a/RELATORIO_noite.md`; `DESVIOS.md`).

**As células com gatilho são os pedidos originais.** `harm_trig` é o Q⁺ do PKU e `ben_trig` é o Q⁰ original (gerado em 86% dos grupos, *retrieved* em 14%). As versões sem gatilho trocam só o trecho do gatilho, e o resto de cada linha é idêntico. Consequência: o contraste de intenção herda a diferença de fonte entre Q⁺ e Q⁰ (regra 4 da seção 5). Por isso o controle de estilo entrou como análise secundária (`DESVIOS.md`, 2026-11-21).

**Conjunto principal: `qc_grammatical`** (`DESVIOS.md`, 2026-11-21), os grupos em que as duas versões sem gatilho são gramaticais.

| Subconjunto | Grupos | train / val / test | Descoberta (no `fim_pedido`) | Confirmação com gatilho inédito | Papel |
|---|---|---|---|---|---|
| `qc_grammatical` | 1.723 | 1.219 / 260 / 244 | 1.479 (1.306) | 156 | **principal** |
| `qc_pass_lenient` | 604 | 435 / 85 / 84 | 520 (461) | 51 | robustez |
| todos | 1.825 | 1.297 / 271 / 257 | 1.568 (1.375) | 165 | robustez |
| `qc_pass` | 148 | 101 / 24 / 23 | 125 (114) | 14 | só descrição (abaixo dos pisos) |

- **`fim_pedido`:** no conjunto principal, 205 grupos (11,9%) têm o último token diferente entre as células e saem só dessa posição. Fica abaixo dos 20% da revisão prevista (seção 1, regra 1), então a regra se mantém.
- **Pisos de poder (seção 4):** atendidos no conjunto principal e no `qc_pass_lenient`.
- **Domínios** (conjunto principal): privacy 439, bias 436, cyber 434, violence 414. **Tipos de gatilho:** ação 800, marcador de intenção 482, superfície 441. **Q⁰ *retrieved*:** 224 grupos.
- **Controles:** 3 frases por grupo (5.475 pares de frases no conjunto completo, 4.434 distintos), filtrados pelo mesmo subconjunto do 2×2.

## Adendo 2: caminho (b) e IC do controle do C3

Escrito em 2026-11-21, antes de qualquer cache do 2×2 (original ou em estilo comum). O cache dos controles já tinha chegado (hashes conferidos), mas não foi analisado. As duas decisões estão no `DESVIOS.md` com a mesma data.

**1. Qual 2×2 é o principal.** O caminho (b) da frente A reescreve Q⁺ e Q⁰ juntos num estilo comum, antes da mesma troca do gatilho (`cells_2x2_style.jsonl`, mesmos `group_id`). Ele vira o conjunto principal do C1–C3 se valerem **as três** condições:

| Condição | Limiar | Medida |
|---|---|---|
| Tamanho | ≥ 1.200 grupos no `qc_grammatical` do (b) | contagem no arquivo da frente A |
| Estilo | AUROC só de estilo ≤ 0,70 (hoje 0,887) | `python -m featgrpo.data.phase1.style_gap`, no `qc_grammatical` do (b) |
| Prazo | cache do (b) com a frente C até **2026-11-22, 12h** (Brasília) | data de entrega do zip, com hashes conferidos |

- As três usam só o texto e a data de entrega, nunca ativações. Por isso dá para saber qual conjunto é o principal antes de olhar qualquer resultado.
- **Se valerem:** o `qc_grammatical` do (b) é o principal. A combinação principal, as listas, o C1, o C3 e a confirmação saem dele. O `qc_grammatical` do 2×2 atual vira robustez, na combinação principal do (b), como os outros subconjuntos de robustez. As listas que as frentes B e D tiverem recebido do 2×2 atual são substituídas pelas do (b).
- **Se alguma falhar:** nada muda. O 2×2 atual continua principal, e o (b), se chegar, entra como robustez na combinação principal do 2×2 atual.
- **C4a:** usa o mesmo conjunto do C1. Se o cache 262k do (b) não existir, o C4a fica no 2×2 atual, e isso é declarado.
- **Controle de estilo** (análise secundária): roda nos dois conjuntos. No (b), ele mede quanto do efeito de intenção restava do estilo.
- **Confirmação:** nenhum `test` de 2×2 é aberto antes de o conjunto principal estar definido. Até lá, o 2×2 atual roda só na descoberta. Depois, a confirmação é aberta uma vez, no principal, como na seção 3.

**2. IC do controle `trig_only` (C3, item 1).** O d continua sendo o dos pares de frases (seção 5). Muda só o IC 95%: ele passa a vir de um bootstrap **por grupo do 2×2**, em que cada sorteio pega um grupo inteiro com as suas frases (1.000 reamostras, semente 0). As 3 frases de um grupo usam o mesmo gatilho e o mesmo termo neutro, então não são independentes. Sortear frases soltas estreitaria o IC (até √3 vezes, quando as frases são quase iguais) e facilitaria passar no critério de equivalência. O bootstrap por frase, a versão original, é reportado como sensibilidade. O critério continua o mesmo: atende se mais da metade das S_intent tiver o IC dentro de ±0,2.

## Adendo 3: conjunto principal final

Escrito em 2026-11-21, com o cache do 2×2 já recebido e conferido, mas **antes de calcular qualquer efeito**. Os números abaixo usam só o texto e o índice do cache.

**Principal: `qc_grammatical AND validator_pass`** (`DESVIOS.md`, 2026-11-21). O `validator_pass` é a marca do validador estrutural da frente D, anotada pela frente A no `cells_2x2.jsonl` (PR #11). Até esse PR entrar no `master`, rodamos com uma cópia do arquivo anotado (`--cells-jsonl`), cujo sha256 fica gravado no `c1_c3.json`.

| Subconjunto | Grupos | Descoberta (`fim_pedido` / `pre_resposta`) | `test` com gatilho inédito (`fim_pedido` / `pre_resposta`) | Papel |
|---|---|---|---|---|
| `qc_grammatical` + validador | 1.697 | 1.288 / 1.455 | 137 / 155 | **principal** |
| `qc_grammatical` sem validador | 1.723 | 1.306 / 1.479 | 138 / 156 | robustez |
| `qc_pass_lenient` + validador | 593 | 451 / 509 | 42 / 51 | robustez |
| todos | 1.825 | 1.375 / 1.568 | 145 / 165 | robustez |

- **`fim_pedido`:** no principal, 199 grupos (11,7%) têm o último token diferente entre as células e saem só dessa posição. Fica abaixo dos 20% (seção 1, regra 1).
- **Pisos (seção 4):** atendidos no principal. Por domínio: privacy 434, bias 428, cyber 427, violence 408. No `qc_pass_lenient`, o `test` com gatilho inédito no `fim_pedido` fica abaixo de 50, então ali vale o `test` inteiro (seção 3).
- **Caminho (b):** não virou o principal, porque reprovou em tamanho e estilo (`DESVIOS.md`). Entra como robustez.
- **C2(b):** a frente C vota a favor da proposta do Davi (PR #8), em que o C2(b) passa a ser só o XSTest e o gatilho inédito e o domínio inédito ficam como descrição. A troca no texto da seção 5 e a entrada no `DESVIOS.md` ficam com a frente D, depois da votação do grupo. Feito em 2026-11-21, a pedido do Davi: seção 5, regra 3.
