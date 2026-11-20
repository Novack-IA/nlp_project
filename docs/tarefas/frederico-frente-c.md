# Frente C: Frederico (RTX 4090)

**Papel na resposta:** você produz o **núcleo da evidência**:
- quais features do SAE carregam **intenção** (S_intent) e quais carregam **forma lexical** (S_lex): critério **C1**;
- se as de intenção não são apenas palavras: critério **C3**;
- se o resultado sobrevive a outro dicionário: parte **C4a**.

Defina os limiares em `train` + `val` e só então rode o `test`, uma vez.

Recursos: RTX 4090 para codificar com os SAEs e rodar o autointerp com um LLM local. A estatística roda em CPU.

---

## C0. Pré-registro (D1)

Antes de ver qualquer dado, commite `results/frente-c/preregistro.md` com:
- limiares das classes (copiados de `00-visao-geral.md`);
- teste estatístico (Wilcoxon pareado + Benjamini–Hochberg, q < 0,01);
- quais camadas, posições e SAEs entram;
- o critério de replicação no `test`.

Mudanças posteriores vão em `results/DESVIOS.md` com data e motivo. Isso protege o resultado contra "procurar até achar".

## C1. Estatística por feature: modo provisório (D3, com o cache `pairs` de B)

Enquanto o 2×2 não sai, use os 2.428 pares (`split` = `train`/`val`):
- **intenção provisória:** d de a(Q⁺) − a(Q⁰), pareado por grupo, por feature × camada × posição;
- liste as features com o maior \|d\| e as **compartilhadas**: alta ativação nos dois lados do par na posição `trig`, as candidatas a lexicais.

Isso serve para depurar o código e ter uma primeira ideia, **não para o veredito**. Escreva `results/frente-c/provisorio.md` com o top-20 de cada lado e os rótulos do Neuronpedia (`gemma-3-4b-it/{L}-gemmascope-2-res-16k/{índice}`).

## C2. Estatística no 2×2 (D6–D8)

**Código:** `src/featgrpo/interp/feature_stats.py`.

1. Para cada grupo completo do 2×2, cada feature j, camada L e posição p (`post`, `mean`, `trig`):
   - calcule I_g, L_g e X_g (fórmulas em `00-visao-geral.md`);
   - calcule d_I, d_L e d_X;
   - calcule p-valores de Wilcoxon e q por BH sobre **todas** as features × camadas × posições do SAE em uso.
2. Classifique em `S_intent`, `S_lex`, `S_mixed` e irrelevante, com os limiares pré-registrados.
   - **Atenção a features raras:** exija que a feature ative (> 0) em pelo menos 5% dos itens de alguma célula. Senão, d fica instável.
3. **Replicação no `test`:** para cada feature de S_intent e S_lex escolhida em `train`+`val`, recalcule d_I e d_L no `test`. Reporte a fração com mesmo sinal e \|d\| ≥ 0,3.
4. **Saídas:**
   - `results/frente-c/feature_classes_L{L}_{p}_{sae}.csv`, com colunas `feature, d_I, d_L, d_X, q_I, q_L, classe, d_I_test, d_L_test, freq_ativacao`;
   - `results/frente-c/c1_c3.json`, com o resumo por camada (tamanhos das classes, replicação) e a melhor camada × posição para S_intent;
   - as listas finais para B e D: `results/frente-c/listas/S_intent_L{L}_{p}_{sae}.json` e idem para `S_lex`.
5. **Figura:** um scatter d_I × d_L de todas as features, por camada, com as classes coloridas. É **a figura central do paper**: dois "braços" separados (intenção no eixo x, lexical no eixo y) são a resposta visual.

## C3. As features de intenção não são lexicais (D8–D10)

1. **Controle `trig_only`:** para cada feature de S_intent, calcule o d de a(`trig_only`) − a(`neutral_only`). Critério: \|d\| < 0,2 para a maioria. Para S_lex, esperamos o contrário, o que confirma que o controle funciona.
2. **Exemplos de maior ativação** para o top-20 de S_intent e de S_lex:
   - os 20 trechos com maior ativação num corpus de referência (~2M tokens de chat genérico + pedidos de segurança, **sem** o nosso dataset), com a posição destacada;
   - os 10 itens do nosso 2×2 com maior ativação, com a célula.
   - Use o extrator da frente B no modo por token.
3. **Autointerp:**
   - um LLM local na 4090 (por exemplo, Qwen3-8B ou Gemma-3-12B-IT em 4 bits) lê os exemplos e escreve uma descrição de uma frase;
   - pontuação de detecção: o LLM prevê ativação em 10 exemplos ativos + 10 inativos novos; reporte a acurácia;
   - inclua também o rótulo do Neuronpedia, quando houver.
4. **Avaliação humana às cegas:**
   - gere `results/frente-c/fichas_cegas.csv`, com descrição + exemplos, **sem** dizer se a feature é de S_intent ou S_lex, em ordem aleatória;
   - você e o Davi rotulam cada uma como **conceitual** (intenção, dano, ação perigosa) ou **lexical** (palavra ou expressão específica);
   - critério C3: ≥ 60% das fichas de S_intent rotuladas conceituais por ambos. Reporte o κ entre vocês dois.
5. **Saída:** `results/frente-c/fichas_top20.md`, com a ficha de cada feature (descrição, exemplos, d_I, d_L, rótulos), mais as métricas C3 em `c1_c3.json`.

## C4a. Outro dicionário (D9–D10)

Repita C2 (no `train`+`val` e no `test`) com:
- os latentes **262k** de B;
- os latentes do **SAE próprio** da frente A, na camada dele.

Reporte se C1 continua valendo. Compare os dicionários:
- para cada feature de S_intent no 16k, a feature do outro SAE com maior cosseno entre vetores do decoder;
- quantas têm par com cosseno > 0,7 e se esse par também cai em S_intent.

**Saída:** a seção "robustez" em `c1_c3.json` e uma tabela no `fichas_top20.md`.

## Entregas e critérios de pronto

| Entrega | Pronto quando |
|---|---|
| `preregistro.md` | Commitado no D1, antes dos dados |
| `provisorio.md` | Top-20 dos pares com rótulos |
| `feature_classes_*.csv`, listas, scatter | Todas as camadas × posições do 16k; replicação no `test` reportada |
| `fichas_top20.md` + avaliação cega | Autointerp com pontuação + κ humano |
| Robustez (262k, SAE próprio) | C1 reavaliado nos dois |

**Branch:** `frente-c-features`.

**Interfaces:**
- Você consome o cache de B (seção 4 do `AGENTS.md`).
- Você entrega listas para B (C5) e D (C2). Avise quando as listas finais estiverem prontas.
- Pedidos de extração vão em `results/frente-b/PEDIDOS.md`.
