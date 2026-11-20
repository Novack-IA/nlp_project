# Intenção ou palavras? Separando o sinal de nocividade da forma lexical nas features de um SAE

> **Rascunho.** Esqueleto de 2026-11-20. As seções marcadas com _(a escrever: nome)_ têm dono em [`README.md`](README.md). Os números entram de `results/VEREDITO.md`.

## Resumo

_(a escrever no fim: Davi.)_ Quatro ou cinco frases: o problema (over-refusal), a pergunta, o método (pares mínimos em 2×2 por frase-molde, SAEs do Gemma Scope 2 no Gemma-3-4B-IT, critérios pré-registrados), o resultado (veredito e o número principal) e o que ele implica.

## 1. Introdução

_(a escrever: Davi.)_ Roteiro:

1. **O problema.** Modelos alinhados recusam pedidos benignos que *soam* perigosos ("how do I kill a Python process?"). Isso é o *over-refusal*, e ele convive com o atendimento a pedidos realmente nocivos. As duas falhas sugerem que o modelo reage, ao menos em parte, à superfície do pedido.
2. **A hipótese.** Internamente, o modelo representa duas coisas distintas: a **intenção** do pedido e a sua **forma lexical**. O over-refusal acontece quando o comportamento segue a segunda.
3. **A pergunta operacional.** É possível identificar, nas features de um SAE, um sinal de intenção nociva que se distingue do sinal de forma lexical, usando pares mínimos nocivo↔benigno?
4. **Por que é difícil.** Em pares comuns, nocivo e benigno também diferem em estilo e fonte. No nosso dataset, um classificador que só olha o estilo do texto separa os dois com AUROC 0,88. <!-- fonte: results/frente-d/baseline_lexical.json --> Um resultado "positivo" pode, então, ser só estilo.
5. **Contribuições:**
   - um desenho 2×2 por **frase-molde**, em que só a lacuna de forma ou só a lacuna de intenção muda entre as células;
   - critérios de resposta **pré-registrados** (C1–C5), incluindo um critério relativo a um baseline lexical;
   - a resposta (SIM, PARCIAL ou NÃO), com âncora densa e teste causal.

## 2. Trabalhos relacionados

_(a escrever: Davi.)_ Três blocos. Reaproveitar o relatório da Fase 0 (§3, §5.5, §13).

- **SAEs e interpretabilidade.** Superposição (Elhage et al., 2022); SAEs como dicionários de features (Bricken et al., 2023; Cunningham et al., 2023; Templeton et al., 2024); variantes TopK, JumpReLU e BatchTopK (Gao et al., 2024; Rajamanoharan et al., 2024; Bussmann et al., 2024); avaliação e estabilidade (Karvonen et al., 2025; Paulo & Belrose, 2025, que motiva o SAE de controle). Gemma Scope, SAEs JumpReLU abertos para o Gemma 2 (Lieberum et al., 2024), e Gemma Scope 2, a versão para toda a família Gemma 3, incluindo os modelos de chat (Google DeepMind, 2025).
- **Recusa como direção.** A recusa é mediada por uma direção no residual (Arditi et al., 2024); steering de recusa com SAEs (O'Brien et al., 2025). Nossa pergunta complementa essas: em vez de *onde está a recusa*, *de que ela depende*, da intenção ou da forma.
- **Over-refusal e datasets.** XSTest (Röttger et al., 2024), OR-Bench (Cui et al., 2025), FalseReject (Zhang et al., 2025) e PKU-SafeRLHF (Ji et al., 2025). Nenhum combina pares mínimos, controle de estilo e um fator lexical manipulado de forma independente da intenção.
- **Autointerp.** Um LLM escreve a descrição de um neurônio e a explicação é pontuada pela capacidade de prever a ativação (Bills et al., 2023). Paulo et al. (2025) escalam isso para milhões de features de SAE, com pontuações mais baratas como a de detecção, que é a que usamos no C3.

## 3. Dados

_(a escrever: Gustavo.)_

### 3.1 Pares da Fase 0

2.428 pares Q⁺/Q⁰ com o mesmo gatilho, em 4 domínios (bias, cyber, privacy, violence), com splits por grupo (1.710 / 371 / 347). <!-- fonte: datasets/phase0/README.md --> O protocolo de geração e rotulagem está no relatório da Fase 0.

**Por que não bastam:** Q⁺ vem do PKU e 83% dos Q⁰ foram gerados. O TF-IDF separa os pares com AUROC 0,995 e o estilo sozinho com 0,88. <!-- fonte: results/frente-d/baseline_lexical.json -->

### 3.2 O 2×2 por frase-molde

- Uma frase com duas lacunas, `{FORMA}` e `{INTENCAO}`; as quatro células saem de preenchê-las (Figura 1).
- A tipologia dos gatilhos (superfície, ação, marcador de intenção) e por que marcadores de intenção ficam fora da lacuna de forma.
- Validação: validador automático (estrutura, gatilho, diff, tamanho) e três juízes (intenção). Rodada seca: sobrevivência e decisão molde × plano B. <!-- fonte: results/frente-a/rodada_seca.md -->
- Números finais: grupos, por domínio e por tipo de gatilho. Controles `trig_only` e `neutral_only`. XSTest e OR-Bench-Hard como conjuntos externos.

## 4. Métodos

### 4.1 Modelo, SAEs e extração

_(a escrever: Caio.)_ Gemma-3-4B-IT congelado. Gemma Scope 2 `resid_post`, 16k e 262k, em 4 camadas (25/50/65/85%); SAE próprio BatchTopK como controle. Convenção `resid_post L = hidden_states[L+1]`. Posições: `post`, `post_last`, `mean`, `slot_forma`, `slot_intencao`. Qualidade dos SAEs nos nossos dados (variância explicada, ΔNLL, L0).

### 4.2 Estatística por feature

_(a escrever: Frederico.)_ Efeitos I, L e X por grupo; d pareado com IC por bootstrap; Wilcoxon e Benjamini–Hochberg. Classes S_intent, S_lex e S_mixed com teste de equivalência. Replicação no `test` com gatilho inédito. Autointerp e avaliação cega.

### 4.3 Baselines, probes e dissociação dupla

_(a escrever: Davi.)_

- **Baselines de texto:** TF-IDF (palavras 1–2 e caracteres 3–5) com regressão logística; um classificador só de estilo; um "só gatilho". C por validação cruzada agrupada por gatilho, só no `train`; IC por bootstrap sobre grupos.
- **Âncora densa:** regressão logística L2 no residual, por camada e posição; generalização cruzada com → sem gatilho e a volta; gatilho treinado no benigno e testado no nocivo; cossenos entre as direções de intenção, gatilho e recusa.
- **C2:** classificadores restritos (L1 e média padronizada) só com S_intent ou só com S_lex; controles aleatório (20 sorteios) e denso. Critério principal relativo: em três generalizações em que um classificador de palavras falha (gatilho inédito, domínio inédito e XSTest), a queda de AUROC de S_intent é no máximo metade da queda do TF-IDF (limite superior do IC 95% da razão < 1), em pelo menos 2 das 3.
- **Recortes C4:** Q⁰ *retrieved* e células de tamanho pareado.

### 4.4 Intervenção

_(a escrever: Caio.)_ Ablação de features com o erro de reconstrução somado de volta; controle com features aleatórias de frequência casada; rotulagem das respostas pelo juiz.

### 4.5 Pré-registro

Os limiares, os testes e as regras de análise foram commitados antes de qualquer dado de ativação (`results/frente-c/preregistro.md`). Os desvios estão em `results/DESVIOS.md`.

## 5. Resultados

_(a escrever: todos, integrado pelo Davi, a partir de `results/VEREDITO.md`.)_ Ordem sugerida:

1. **Existe um sinal de intenção? (C1).** Figura 2 (scatter d_I × d_L). Tamanho das classes por camada; replicação.
2. **Ele se separa do lexical? (C2).** Figura 3 (matriz). Critério relativo ao TF-IDF.
3. **Não é só uma palavra? (C3).** Controles `trig_only` e fichas.
4. **É robusto? (C4).** Outro dicionário; recortes.
5. **Âncora densa.** Figura 4. O que o residual sabe e o SAE não.
6. **Ele causa a recusa? (C5).** Figura 5.
7. **Veredito.**

## 6. Limitações

_(a escrever: Davi, revisado por todos.)_ Partir da seção "Limitações" de `results/VEREDITO.md`.

## 7. Ética

O dataset contém pedidos e respostas nocivas, todos do PKU-SafeRLHF. Não geramos respostas nocivas. Os pedidos nocivos do 2×2 são recombinações de trechos dos pedidos do PKU, sem detalhes operacionais novos. Os dados são usados só para pesquisa e não são reproduzidos fora do repositório do projeto.

## Referências

_Do relatório da Fase 0 (§13), mais as quatro novas (Bills, Google DeepMind, Lieberum, Paulo et al. 2025), conferidas na fonte em 2026-11-20._

- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.*
- Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., Sutskever, I., Leike, J., Wu, J. & Saunders, W. (2023). *Language Models Can Explain Neurons in Language Models.* OpenAI. https://openai.com/index/language-models-can-explain-neurons-in-language-models/
- Bricken et al. (2023). *Towards Monosemanticity.* Transformer Circuits.
- Bussmann et al. (2024). *BatchTopK Sparse Autoencoders.*
- Cui et al. (2025). *OR-Bench.*
- Cunningham et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* arXiv:2309.08600.
- Elhage et al. (2022). *Toy Models of Superposition.* arXiv:2209.10652.
- Gao et al. (2024). *Scaling and Evaluating Sparse Autoencoders.*
- Google DeepMind (2025, 19 dez.). *Gemma Scope 2: Helping the AI Safety Community Deepen Understanding of Complex Language Model Behavior.* https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/ · pesos e documentação: https://ai.google.dev/gemma/docs/gemma_scope
- Han et al. (2024). *WildGuard.*
- Ji et al. (2025). *PKU-SafeRLHF.* ACL.
- Karvonen et al. (2025). *SAEBench.*
- Lieberum, T. et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.* BlackboxNLP 2024, p. 278–300. arXiv:2408.05147.
- O'Brien et al. (2025). *Steering Language Model Refusal with Sparse Autoencoders.*
- Paulo, G., Mallen, A., Juang, C. & Belrose, N. (2025). *Automatically Interpreting Millions of Features in Large Language Models.* ICML 2025. arXiv:2410.13928.
- Paulo & Belrose (2025). *Sparse Autoencoders Trained on the Same Data Learn Different Features.*
- Qwen Team (2025). *Qwen3Guard.*
- Rajamanoharan et al. (2024). *JumpReLU Sparse Autoencoders.*
- Röttger et al. (2024). *XSTest.* NAACL.
- Templeton et al. (2024). *Scaling Monosemanticity.* Transformer Circuits.
- Zhang et al. (2025). *FalseReject.* arXiv:2505.08054.
