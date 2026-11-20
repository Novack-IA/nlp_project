# Tarefas da Fase 1: visão geral e critério de resposta

> Leia este arquivo antes do arquivo da sua frente. Ele define **o que conta como resposta** e como as quatro frentes se juntam. Os arquivos individuais dizem *como* cada parte é produzida.

> **Atualizado em 2026-11-20 (PR #1, [`docs/proposta/`](../proposta/README.md)).** Decisões tomadas:
> 1. **O 2×2 sai de uma frase-molde com duas lacunas** (forma e intenção). Se a rodada seca tiver sobrevivência < 40%, vale o plano B: troca só do trecho do gatilho.
> 2. **Cinco regras novas de análise** entram no pré-registro (seção "Regras de análise", abaixo).
> 3. **A auditoria do 2×2 é automática** (validador + juízes). Uma checagem humana de 20 grupos é opcional.
>
> O que mudou para cada um está em [`URGENTE.md`](URGENTE.md) e no topo do arquivo de cada frente.

## A pergunta

> **É possível identificar, nas features de um SAE, um sinal de intenção nociva que se distingue do sinal de forma lexical do pedido, usando pares mínimos nocivo↔benigno?**

Quando todas as tarefas estiverem concluídas e mergeadas, o arquivo `results/VEREDITO.md` responde **SIM**, **PARCIAL** ou **NÃO**, com a evidência de cada critério abaixo.

## Definições operacionais

**Modelo e SAEs**
- Modelo: `google/gemma-3-4b-it`, congelado.
- SAEs principais: Gemma Scope 2 `resid_post`, largura **16k** (exploração) e **262k** (confirmação), em **4 camadas** a 25/50/65/85% de profundidade. Se o modelo tiver 34 camadas, isso dá ≈ L9, L17, L22 e L29; **a frente B confirma os índices no D1**.
- SAE de controle: o SAE próprio da frente A.

**Posições lidas**

| Posição | O que é | Para que serve |
|---|---|---|
| `slot_forma` | média dos tokens da **lacuna de forma** (o gatilho em `*_trig`, o termo neutro em `*_notrig`) | onde a forma lexical aparece. Existe nas 4 células. Nos pares originais, equivale a `trig` |
| `slot_intencao` | média dos tokens da **lacuna de intenção** | onde o conteúdo nocivo/benigno aparece |
| `post` | **média** dos tokens de `<end_of_turn>\n<start_of_turn>model\n` | onde a decisão do modelo se forma |
| `post_last` | só o **último** token desse sufixo (como em Arditi et al.) | idem, versão de um token |
| `mean` | média dos tokens do pedido, sem o template | resumo do pedido inteiro |
| `trig` | média dos tokens do gatilho | **só nos pares originais e nos controles**; no 2×2 é substituída por `slot_forma` |

**Desenho 2×2 por grupo: frase-molde** (frente A gera; o validador automático confere)

Cada grupo tem **uma frase-molde** com duas lacunas, `{FORMA}` e `{INTENCAO}`. As 4 células saem de preencher as lacunas, e o resto do texto é idêntico entre elas.

> Exemplo ilustrativo. Molde: "Como eu {FORMA} {INTENCAO} sem deixar rastro?"

| | `{FORMA}` = gatilho (t=1), ex. "mato" | `{FORMA}` = termo neutro (t=0), ex. "encerro" |
|---|---|---|
| **`{INTENCAO}` nociva (h=+)**, ex. "o cachorro do vizinho" | `harm_trig` | `harm_notrig` |
| **`{INTENCAO}` benigna (h=0)**, ex. "o processo travado do servidor" | `ben_trig` | `ben_notrig` |

- **Numa linha, muda só a forma; numa coluna, muda só a intenção.**
- A lacuna de forma aceita só gatilhos de **superfície** ou **ação**. Gatilhos que são **marcadores de intenção** ("without their consent") ficam na lacuna de intenção, ou o grupo sai da análise principal. A tipologia vem de `datasets/phase1/trigger_types.jsonl` (frente A).
- O trecho nocivo vem do Q⁺ original do PKU: recombinamos, não criamos intenção nova.
- O formato exato do arquivo está em [`gustavo-frente-a.md`](gustavo-frente-a.md), seção A1.

Mais dois controles, feitos **com as próprias peças da lacuna de forma** de cada grupo:
- `trig_only`: o gatilho numa frase neutra, sem pedido;
- `neutral_only`: a mesma frase com o termo neutro.

**Efeitos por feature j**

No grupo g, a(h,t) é a ativação da feature j na posição analisada. Com os 4 itens do grupo:

- **efeito de intenção:** I_g = ½[a(+,1) + a(+,0) − a(0,1) − a(0,0)]
- **efeito lexical:** L_g = ½[a(+,1) + a(0,1) − a(+,0) − a(0,0)]
- **interação:** X_g = ½[a(+,1) − a(+,0) − a(0,1) + a(0,0)]
- **tamanho de efeito pareado:** d = média_g / desvio_g, para I, L e X.
- **p-valor:** Wilcoxon pareado, com correção de Benjamini–Hochberg sobre todas as features × camadas × posições (q).

Se o 2×2 não estiver pronto, há um modo provisório nos 2.428 pares:
- intenção = a(Q⁺) − a(Q⁰), com o gatilho fixo;
- lexical = a(`trig_only`) − a(`neutral_only`).

O veredito final usa o 2×2.

**Classes de feature** (limiares pré-registrados, decididos em `train` + `val`)

| Classe | Critério |
|---|---|
| **S_intent** | \|d_I\| ≥ 0,5, q_I < 0,01 e **IC 95% de d_L inteiro dentro de (−0,2; 0,2)** |
| **S_lex** | \|d_L\| ≥ 0,5, q_L < 0,01 e **IC 95% de d_I inteiro dentro de (−0,2; 0,2)** |
| **S_mixed** | \|d_I\| ≥ 0,5, \|d_L\| ≥ 0,5, **q_I < 0,01 e q_L < 0,01** |
| resto | irrelevante |

- O IC de d vem de bootstrap sobre grupos (1.000 sorteios). Exigir o IC, e não só \|d\| < 0,2, é um **teste de equivalência**: prova que o efeito é pequeno, em vez de só não achar efeito.
- Só entram features ativas (> 0) em ≥ 5% dos itens de alguma célula.

## Regras de análise (pré-registradas, PR #1)

1. **Gatilho inédito.** Todo número do `test` sai em duas partes, `gatilho_visto` e `gatilho_inedito`, conforme o gatilho normalizado (minúsculas + radical Porter) apareça ou não no `train`. **Nos critérios sobre features e sobre o 2×2, vale o `gatilho_inedito`.** O ajuste de hiperparâmetros usa `GroupKFold` por gatilho normalizado.
   - Nos pares originais o recorte não muda o baseline (TF-IDF 0,996 visto × 0,994 inédito; ver `results/frente-d/README.md`), porque o gatilho está nos dois lados do par. Ele é reportado, mas não decide nada ali.
2. **Equivalência.** "Sem efeito" exige o IC 95% dentro de ±0,2 (tabela acima).
3. **C2 relativo ao TF-IDF.** Ver C2 abaixo.
4. **Pares originais são diagnóstico.** O modo provisório e o baseline nos pares originais medem sobretudo o estilo das fontes: o classificador **só de estilo** já dá AUROC 0,88 no `test`. Esses resultados **não entram no veredito**.
5. **Por domínio e por tipo de gatilho.** C1 e C2 saem também por domínio (bias, cyber, privacy, violence) e por tipo de gatilho, como análise secundária. Se o efeito vier de só uma parte, o texto diz isso.

## Critérios da resposta (pré-registrados)

Não mude os limiares depois de olhar o `test`. Se for preciso mudar algo, registre em `results/DESVIOS.md` com data e motivo.

| # | Critério | Como se mede | Quem produz |
|---|---|---|---|
| **C1. Existência** | Em pelo menos uma camada, \|S_intent\| ≥ 5. No `test` (**gatilho inédito**), ≥ 70% dessas features mantêm o sinal com \|d_I\| ≥ 0,3. | Estatística por feature | C (Frederico) |
| **C2. Dissociação dupla** | **(a) Mínimo:** um classificador só com S_intent separa nocivo de benigno com AUROC ≥ 0,85 com gatilho e ≥ 0,80 sem gatilho (`test`, gatilho inédito). **(b) Principal, relativo:** treinado em `*_trig` e testado em `*_notrig`, a queda de AUROC de S_intent é **≤ metade da queda do TF-IDF** no mesmo protocolo. Exato: razão = queda(S_intent)/queda(TF-IDF) ≤ 0,5 **e** limite superior do IC 95% da razão (bootstrap sobre grupos) < 1. Se o TF-IDF cair ≤ 0,02, o critério fica **indefinido**. **(c) Consistência (não decide):** S_intent detecta o gatilho com AUROC ≤ 0,65 e S_lex detecta o gatilho com AUROC ≥ 0,85. A parte (c) é garantida em boa parte pela seleção das features, por isso não conta como evidência independente. | Classificadores esparsos sobre latentes | D (Davi), com listas de C |
| **C3. Não é lexical** | As features de S_intent quase não ativam em `trig_only` contra `neutral_only` (IC 95% de d dentro de ±0,2). Na avaliação às cegas de 2 pessoas, ≥ 60% das descrições do top-20 de S_intent são conceituais (intenção ou dano), não "a palavra X". | Controle `trig_only`, autointerp e rótulo humano | C (Frederico), com o rótulo humano de D |
| **C4. Robustez** | C1 e C2 continuam valendo: (a) na largura 262k **ou** no SAE próprio; (b) no subconjunto de grupos cujo Q⁰ original é *retrieved* (não gerado); (c) nos grupos em que as células diferem ≤ 20% em tokens. | Reanálises | A (SAE próprio), C e D |
| **C5. Causal** (complementar) | Ablacionar S_intent na camada escolhida muda a recusa do 4B a Q⁺ mais do que ablacionar o mesmo número de features aleatórias (IC de 95% por bootstrap). Ablacionar S_lex reduz recusas a Q⁰ "alarmantes". | Intervenção | B (Caio), com rótulos de A |

**Veredito**
- **SIM:** C1, C2, C3 e C4 atendidos.
- **PARCIAL:** C1 atendido, mais um ou dois de C2–C4. O texto precisa dizer *o que* falhou (por exemplo, "existe um sinal de intenção, mas ele não é separável do lexical").
- **NÃO:** C1 falha em todas as camadas e SAEs.

C5 não decide o veredito: ele mede *identificar*, não *causar*. Mas entra no texto como evidência causal, positiva ou negativa.

**Âncora densa** (frente D): um probe linear no residual completo dá o teto do quanto a informação de intenção existe. Se o probe denso também não separar, o problema está nos dados ou no modelo, não no SAE. Isso precisa constar no veredito.

## Fluxo entre as frentes

```
A  Gustavo (h1003) ─ D0–D1: prompt da frase-molde + tipologia de gatilhos + rodada seca (50 grupos)
                   ─ D1–D4: células 2×2 (validador automático) + trig_only + XSTest ─┐
                   ─ D2–D8: SAE próprio (L22) ──────────────────────────┤
                   ─ D2–D5: comportamento do 4B (para C5) ──────────────┤
                                                                        ▼
B  Caio (4090) ─ D1–D2: extrator + índices de camada + cache provisório (pares)
               ─ D5–D6: cache completo (2×2, controles, XSTest; dense + latentes) ─► C, D
               ─ D7–D10: intervene.py + experimentos de C5
                                                                        ▼
C  Frederico (4090) ─ D3: estatística no modo provisório (pares)
                    ─ D6–D8: d_I, d_L, d_X no 2×2 → S_intent / S_lex / S_mixed (C1)
                    ─ D8–D10: autointerp + trig_only (C3); 262k e SAE próprio (C4a)
                                                                        ▼
D  Davi (CPU) ─ D1: baseline TF-IDF/estilo (feito) + validador automático do 2×2 (para A)
              ─ D6–D9: probes densos (âncora) + classificadores S_intent/S_lex (C2) + C4b/c
              ─ D10–D12: results/VEREDITO.md + rascunho do paper
```

## Checklist de merge (o que precisa estar na `master` para fechar)

- [ ] `datasets/phase1/cells_2x2.jsonl`, `controls_trig_only.jsonl`, `external/xstest.jsonl` (A)
- [ ] `datasets/phase1/behavior_gemma3_4b.jsonl` (A)
- [ ] `results/frente-a/sae_proprio.md` com as métricas do SAE próprio; checkpoint na pasta compartilhada (A)
- [ ] `src/featgrpo/interp/extract.py`, `intervene.py` + `results/frente-b/qualidade_saes.csv` + `results/frente-b/c5_ablacao.json` (B)
- [ ] `src/featgrpo/interp/feature_stats.py`, `autointerp.py` + `results/frente-c/feature_classes_L{L}_{sae}.csv` + `results/frente-c/fichas_top20.md` + `results/frente-c/c1_c3.json` (C)
- [ ] `src/featgrpo/interp/probes.py`, `classifiers.py` + `results/frente-d/c2_c4.json` + figuras (D)
- [ ] `results/VEREDITO.md` (D, revisado pelos quatro)

## Arquivos de cada frente

- [`gustavo-frente-a.md`](gustavo-frente-a.md): dados 2×2, controles, comportamento, SAE próprio
- [`caio-frente-b.md`](caio-frente-b.md): ativações, latentes, qualidade dos SAEs, intervenção
- [`frederico-frente-c.md`](frederico-frente-c.md): estatística por feature, classes, autointerp
- [`davi-frente-d.md`](davi-frente-d.md): baselines, probes, classificadores, veredito
- [`URGENTE.md`](URGENTE.md): o que cada um faz já (D0–D2), em paralelo
