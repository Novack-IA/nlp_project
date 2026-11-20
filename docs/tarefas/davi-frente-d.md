# Frente D: Davi (CPU, sem cluster por ora)

**Papel na resposta:** você faz o teste de **dissociação dupla (C2)**: as features de intenção servem para intenção e não para forma, e as lexicais o contrário.
- Também cuida das checagens de confusão (**C4b/c**) e dos baselines que dão escala aos números (TF-IDF e probe denso).
- Por fim, **escreve o veredito**.

Tudo aqui roda em CPU: os caches da frente B são pequenos. Um vetor denso é 2.560 floats; os latentes vêm esparsos.

Se você recuperar o acesso à h1003, avise o time: o SAE próprio (A5) pode passar para você.

> **O que mudou em 2026-11-20 (PR #1):**
> - O baseline ganhou o classificador só de estilo e o recorte gatilho visto × inédito (D1, **feito**).
> - A auditoria humana virou o **validador automático** (D2).
> - O C2 passa a ser **relativo ao TF-IDF** (D4).
> - O veredito ganha uma seção pronta para o caso PARCIAL (D7).

---

## D1. Baseline lexical (feito em 2026-11-20; 2×2 e XSTest quando A entregar)

**Código:** `src/featgrpo/interp/baselines.py`. **Resultado:** `results/frente-d/baseline_lexical.json`, com a leitura em `results/frente-d/README.md`.

1. **TF-IDF + regressão logística** (n-gramas de palavra 1–2 e de caractere 3–5) para nocivo × benigno:
   - treino nos 2.428 pares originais (`train`), avaliação em `val` e `test`;
   - CV agrupada **por gatilho normalizado** para hiperparâmetros;
   - `test` em gatilho visto × inédito e por domínio.
2. **Só estilo:** atributos que não olham o conteúdo (tamanho, pontuação, forma da pergunta). Mede a confusão de fonte PKU × gerador. Deu AUROC 0,88 no `test`.
3. **Generalização lexical:** o mesmo classificador aplicado às células do 2×2 quando A1 sair e ao XSTest. Espera-se que o TF-IDF caia bastante nas células sem gatilho. **Essa queda é a régua do C2(b).** O script já calcula `cells_2x2.queda_auroc_tfidf` quando o arquivo existir.
4. **"Só gatilho":** vê **apenas** a presença do gatilho. Deu AUROC 0,5 nos pares, como o desenho prevê.

## D2. Validador automático do 2×2 (D1, substitui a auditoria humana)

**Código:** `src/featgrpo/data/phase1/validate_template.py`, usado pela frente A na geração.

- `render_cells(template, slots)`: monta as 4 células e devolve as posições das lacunas (`spans`).
- `validate_group(group)`: confere, para cada grupo,
  - (a) estrutura: o molde tem `{FORMA}` e `{INTENCAO}` uma vez cada, e cada `text` é o molde preenchido;
  - (b) gatilho: `*_trig` contém o gatilho e `*_notrig` não (literal e radical Porter); o termo neutro não contém o gatilho;
  - (c) diff: entre células da mesma linha só muda a forma, e entre células da mesma coluna só muda a intenção;
  - (d) tamanho: as células diferem ≤ 20% em palavras;
  - (e) `spans` batem com o texto.
- A intenção certa e a mesma intenção ficam com os juízes da frente A.
- *(Opcional, D-P6)* checagem humana de 20 grupos do 2×2 final, registrada em `results/frente-d/checagem_humana.md`.

## D3. Âncora densa: probes no residual (D3–D7)

**Código:** `src/featgrpo/interp/probes.py`. Use o cache denso de B: `pairs` no D2, `cells2x2` no D5.

1. Por camada (as 4) e posição (`post`, `post_last`, `mean`, `slot_forma`, `slot_intencao`; `trig` nos pares originais), treine uma regressão logística L2 para **intenção** e outra para **gatilho presente** no 2×2. O `test` sai em gatilho visto × inédito.
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
2. **Matriz 2×2 de AUROC no `test`, gatilho inédito** (o resultado de C2):

   | | Tarefa: intenção | Tarefa: gatilho presente |
   |---|---|---|
   | **S_intent** | alta (≥ 0,85 com gatilho; ≥ 0,80 sem): C2(a) | baixa (≤ 0,65): consistência |
   | **S_lex** | baixa (≤ 0,65): consistência | alta (≥ 0,85): consistência |

   Inclua três controles na mesma tabela:
   - um conjunto **aleatório** de features com o mesmo tamanho (média de 20 sorteios);
   - o **probe denso** de D3 como teto;
   - o **TF-IDF** de D1 como régua.
3. **C2(b), o critério principal, relativo:** treine em `harm_trig` × `ben_trig` e teste em `harm_notrig` × `ben_notrig`. Compare a queda de AUROC de S_intent com a do TF-IDF no mesmo protocolo. **Atende** se razão = queda(S_intent)/queda(TF-IDF) ≤ 0,5 **e** o limite superior do IC 95% da razão (bootstrap sobre grupos) < 1, ou seja, S_intent cai menos que o TF-IDF com confiança. Implementado em `classifiers.relative_drop`. Se o TF-IDF não cair (queda ≤ 0,02), o critério fica **indefinido** e isso vai para o veredito: o 2×2 não removeu o sinal lexical.
4. **Generalização:** S_intent treinado no 2×2 e aplicado ao XSTest.
5. **Por domínio e por tipo de gatilho** (análise secundária).
6. **Saída:** `results/frente-d/c2_c4.json` e o heatmap da matriz.

## D5. Confusões (C4b/c, D9)

Refaça D4.2, D4.3 e a métrica de C1 (com o código da frente C) em dois recortes:
- **(b)** só grupos cujo Q⁰ original é *retrieved* do OR-Bench (`orig.q_border_method`), não gerado;
- **(c)** grupos em que as células diferem ≤ 20% em tokens.

Sem esses recortes, um revisor dirá que achamos "features de estilo do PKU". A frase-molde (A1) e esses recortes respondem a isso.

## D6. Avaliação cega das fichas (C3, D9–D10)

Rotule `results/frente-c/fichas_cegas.csv` junto com o Frederico, sem ver a classe. Mais detalhes em `frederico-frente-c.md`, C3.4.

## D7. Veredito e escrita (D10–D12)

1. **`results/VEREDITO.md`:**
   - a pergunta;
   - uma tabela C1–C5 com o número, o limiar, **atende/não atende** e o link para o arquivo de origem;
   - o veredito (SIM, PARCIAL ou NÃO, regra em `00-visao-geral.md`);
   - um parágrafo de interpretação;
   - limitações;
   - os desvios do pré-registro;
   - uma seção pronta para o caso **PARCIAL**: qual critério falhou, em qual recorte, e o que a âncora densa diz (limitação do SAE ou do modelo).

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
| `baseline_lexical.json` | Pares (**feito**) + 2×2 + XSTest, com IC |
| `validate_template.py` + testes | Usado pela frente A na rodada seca (D1) |
| `probes.json` + figura | 4 camadas × posições, generalização cruzada |
| `c2_c4.json` + heatmap | Matriz 2×2 no `test` (gatilho inédito) com controles aleatório, denso e TF-IDF; C2(b) relativo; recortes b e c |
| `VEREDITO.md` | Revisado pelos quatro |

**Branch:** `frente-d-densa`.

**Interfaces:**
- Você consome o cache de B e as listas de C.
- Se precisar de outra posição ou conjunto extraído, peça em `results/frente-b/PEDIDOS.md`.
- Use sempre o `split` dos arquivos e nunca ajuste nada olhando o `test`.
