# Frente D: Davi (CPU, sem cluster por ora)

**Papel na resposta:** você faz o teste de **dissociação dupla (C2)**: as features de intenção servem para intenção e não para forma, e as lexicais o contrário.
- Também cuida das checagens de confusão (**C4b/c**) e dos baselines que dão escala aos números (TF-IDF e probe denso).
- Por fim, **escreve o veredito**.

Tudo aqui roda em CPU: os caches da frente B são pequenos. Um vetor denso é 2.560 floats; os latentes vêm esparsos.

Se você recuperar o acesso à h1003, avise o time: o SAE próprio (A5) pode passar para você.

---

## D1. Baseline lexical (D1–D3, só texto, sem dependências)

**Código:** `src/featgrpo/interp/baselines.py`.

1. **TF-IDF + regressão logística** (n-gramas de palavra 1–2 e de caractere 3–5) para nocivo × benigno:
   - treino nos 2.428 pares originais (`train`), avaliação em `val` e `test`;
   - CV agrupada por `group_id` para hiperparâmetros.
2. **Generalização lexical:** o mesmo classificador aplicado às células do 2×2 quando A1 sair (`harm_trig`, `harm_notrig`, `ben_trig`, `ben_notrig`) e ao XSTest. Espera-se que o TF-IDF caia bastante nas células sem gatilho. Essa queda é a régua de C4.
3. **"Só gatilho":** um classificador que vê **apenas** a presença do gatilho. Deve dar AUROC ≈ 0,5 nos pares, porque os dois lados têm o gatilho. Isso documenta o desenho.
4. **Saída:** `results/frente-d/baseline_lexical.json` com AUROC e acurácia por conjunto e célula, com IC por bootstrap sobre grupos.

## D2. Auditoria humana do 2×2 (D4, quando A1 exportar a amostra)

Rotule `results/frente-a/amostra_2x2.csv` (60 grupos × 4 células). Para cada célula:
- (a) a intenção está certa (nocivo/benigno)?
- (b) o gatilho está presente ou ausente como deveria?
- (c) a intenção é a mesma do original?

Devolva com as concordâncias por célula. Abaixo de 85% numa célula, o Gustavo revisa o prompt. Isso é a garantia de que o 2×2 mede o que diz medir.

## D3. Âncora densa: probes no residual (D3–D7)

**Código:** `src/featgrpo/interp/probes.py`. Use o cache denso de B: `pairs` no D2, `cells2x2` no D5.

1. Por camada (as 4) e posição (`post`, `post_last`, `mean`, `trig`), treine uma regressão logística L2 para **intenção** e outra para **gatilho presente** no 2×2.
2. **Generalização cruzada** (o ponto principal):

   | Treina em | Testa em | Mede |
   |---|---|---|
   | `harm_trig` × `ben_trig` | `harm_notrig` × `ben_notrig` | A intenção sobrevive à remoção do gatilho? |
   | `harm_notrig` × `ben_notrig` | `harm_trig` × `ben_trig` | A volta |
   | pares originais | XSTest (`contrast_*` e geral) | Fora da distribuição |
   | gatilho: `ben_trig` × `ben_notrig` | `harm_trig` × `harm_notrig` | O sinal lexical independe da intenção? |

3. **Direções:** a diferença de médias da intenção (d_int) e a do gatilho (d_lex) por camada. Reporte o cosseno entre elas. Perto de 0 indica que a forma e a intenção ocupam direções distintas. Reporte também o cosseno com a direção de recusa (média de Q⁺ menos média de Q⁰, à la Arditi).
4. **Saída:**
   - `results/frente-d/probes.json`;
   - a figura "AUROC por camada", com linhas para dentro da célula e generalização cruzada, e o TF-IDF tracejado.

**Leitura:** se o probe denso não generalizar entre células, o SAE também não vai separar. Registre isso: é a âncora do veredito.

## D4. Dissociação dupla com features (C2, D8–D9)

**Código:** `src/featgrpo/interp/classifiers.py`. Use as listas S_intent e S_lex da frente C e os latentes de B.

1. **Classificadores restritos** (regressão logística L1 ou média simples das features padronizadas; reporte os dois) usando **apenas** as features de S_intent ou **apenas** as de S_lex. Escolha camada e posição pela melhor de C em `train`+`val`.
2. **Matriz 2×2 de AUROC no `test`** (o resultado de C2):

   | | Tarefa: intenção | Tarefa: gatilho presente |
   |---|---|---|
   | **S_intent** | alta (≥ 0,85 com gatilho; ≥ 0,80 sem) | baixa (≤ 0,65) |
   | **S_lex** | baixa (≤ 0,65) | alta (≥ 0,85) |

   Inclua dois controles na mesma tabela:
   - um conjunto **aleatório** de features com o mesmo tamanho (média de 20 sorteios);
   - o **probe denso** de D3 como teto.
3. **Generalização:** S_intent treinado no 2×2 e aplicado ao XSTest.
4. **Saída:** `results/frente-d/c2_c4.json` e o heatmap da matriz.

## D5. Confusões (C4b/c, D9)

Refaça D4.2 e a métrica de C1 (com o código da frente C) em três recortes:
- **(b)** só grupos cujo Q⁰ original é *retrieved* do OR-Bench (`q_border_method`), não gerado;
- **(c)** grupos pareados por tamanho: diferença de tokens entre células ≤ 20%;
- **(d)** C4, parte final: a queda de AUROC de S_intent entre `*_trig` e `*_notrig` comparada com a queda do TF-IDF (D1.2). S_intent deve cair menos.

Sem esses recortes, um revisor dirá que achamos "features de estilo do PKU". A reescrita no estilo comum (A1) e esses recortes respondem a isso.

## D6. Avaliação cega das fichas (C3, D9–D10)

Rotule `results/frente-c/fichas_cegas.csv` junto com o Frederico, sem ver a classe. Mais detalhes em `frederico-frente-c.md`, C3.4.

## D7. Veredito e escrita (D10–D12)

1. **`results/VEREDITO.md`:**
   - a pergunta;
   - uma tabela C1–C5 com o número, o limiar, **atende/não atende** e o link para o arquivo de origem;
   - o veredito (SIM, PARCIAL ou NÃO, regra em `00-visao-geral.md`);
   - um parágrafo de interpretação;
   - limitações;
   - os desvios do pré-registro.

   Os quatro revisam antes do merge.
2. **Esqueleto do paper** em `docs/paper/` (Markdown ou LaTeX):
   - introdução, related work (reaproveite o slide e o relatório da Fase 0);
   - dataset (Fase 0 + 2×2);
   - métodos (posições, SAEs, estatística);
   - resultados (scatter d_I × d_L da frente C, matriz de C2, probes);
   - limitações.

   Cada frente escreve a própria seção de métodos; você integra.

## Entregas e critérios de pronto

| Entrega | Pronto quando |
|---|---|
| `baseline_lexical.json` | Pares + 2×2 + XSTest, com IC |
| Auditoria do 2×2 | Concordância por célula devolvida ao Gustavo |
| `probes.json` + figura | 4 camadas × posições, generalização cruzada |
| `c2_c4.json` + heatmap | Matriz 2×2 no `test` com controles aleatório e denso; recortes b, c, d |
| `VEREDITO.md` | Revisado pelos quatro |

**Branch:** `frente-d-densa`.

**Interfaces:**
- Você consome o cache de B e as listas de C.
- Se precisar de outra posição ou conjunto extraído, peça em `results/frente-b/PEDIDOS.md`.
- Use sempre o `split` dos arquivos e nunca ajuste nada olhando o `test`.
