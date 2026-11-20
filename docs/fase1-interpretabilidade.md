# Fase 1: interpretabilidade (escopo reduzido)

> Decisão de 2026-11-20, por recomendação dos orientadores: **o controller sai do escopo**. O trabalho passa a ser só de interpretabilidade, com o modelo congelado.

## Pergunta de pesquisa

**As representações internas de um LLM separam o que é nocivo de como o pedido soa, sem retreinar o modelo?**

Ela se desdobra em quatro hipóteses testáveis:

| | Hipótese | Teste principal |
|---|---|---|
| H1 | Existe uma representação de **nocividade**: ela distingue Q⁺ de Q⁰ mesmo com o gatilho lexical fixo. | Probes e diferença de médias por camada em Q⁺ × Q⁰; features de SAE com efeito pareado alto. |
| H2 | Existe uma representação de **superfície** (o gatilho "alarmante"): ela é ativada igualmente por Q⁺ e Q⁰. | Features ativas nos dois lados do par, no token do gatilho; elas também disparam com o gatilho fora de contexto. |
| H3 | As duas são **separáveis**: a direção de nocividade generaliza quando a superfície muda, e a de superfície não prevê nocividade. | Generalização cruzada para o desenho 2×2 (abaixo), para o XSTest e para o OR-Bench-Hard. |
| H4 | O **comportamento** acompanha essas representações: os over-refusals do modelo coincidem com superfície ativa e nocividade baixa. Intervir nas features muda o comportamento sem retreinar. | Correlação com as recusas naturais; ablação ou steering das features no modelo congelado. |

## Modelo: Gemma-3-4B-IT + Gemma Scope 2

- **SAEs prontos no próprio modelo de chat.** O [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-4b-it) tem SAEs de `resid_post`, `attn_out` e `mlp_out` para o **Gemma-3-4B-IT**:
  - 4 profundidades (25%, 50%, 65% e 85%) com larguras de 16k a 1M;
  - segundo o anúncio, também SAEs de 16k e 262k em todas as camadas.
- **Ferramentas prontas.** Carrega no SAELens (release `gemma-scope-2-4b-it-resid_post`, por exemplo `layer_12_width_16k_l0_small`). Há rótulos de features no [Neuronpedia](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k).
- **SAE próprio é viável.** d_model ≈ 2,5k. Um BatchTopK de 16k treinado em ~200M tokens leva algumas horas em uma GPU. Serve de controle: os resultados não podem depender de um único dicionário.
- **Mais fraco que o Gemma-4-12B.** O 4B deve errar mais no nosso dataset (o 12B errou menos de 1%), o que dá casos de comportamento para H4.
- **Leve.** Roda em 1 GPU com folga, então cada membro trabalha em paralelo.
- **Alternativas, se algo travar:**
  - Gemma-2-2B-IT com Gemma Scope 1. É a ferramenta mais madura, mas os SAEs foram treinados no modelo base.
  - Llama-3.1-8B-Instruct com Llama Scope (base) e Goodfire (uma camada).

> **A verificar no dia 1:** a lista exata de camadas e larguras disponível para o 4B-IT no HF, e o suporte do SAELens/TransformerLens ao Gemma 3. Se o TransformerLens não suportar, use ganchos do HF ou o nnsight.

## O dataset basta?

Sim, com folga para as análises de pedidos, e de forma suficiente para as de respostas.

| Nível | Conjunto | Tamanho | Poder estatístico (teste pareado, Bonferroni sobre 262k features, 80% de poder) |
|---|---|---|---|
| **Pedidos** | `pairs_prompt_level` | **2.428 pares** (4.856 pedidos); 1.710 / 371 / 347 | treino: detecta efeito d ≳ 0,15; teste: d ≳ 0,33 |
| **Respostas** | `final` | 500 grupos (pool de 837) | d ≳ 0,27 |

Os 2.428 pares exigem só que os **pedidos** sejam válidos (Q⁺ nocivo, Q⁰ benigno, mesmo gatilho). A maior parte dos 1.969 grupos descartados na v5b caiu por causa de respostas (a `safe_A` do PKU falhou em 1.726), o que não afeta a análise de pedidos.

### Representações de pedidos e de respostas

As duas são possíveis:

- **Pedidos.** Aplicar o chat template e ler o residual em três pontos:
  1. no(s) **token(s) do gatilho**: superfície;
  2. nos **tokens pós-instrução** (`<end_of_turn>\n<start_of_turn>model\n`): onde a decisão de recusar se forma, como em Arditi et al.;
  3. na média do pedido.
- **Respostas.** Teacher forcing de pedido + resposta, lendo o residual nos tokens da resposta (média e primeiros k tokens). Contrastes:
  - conteúdo = r0 − r1;
  - forma = r0(Q⁺) − r5(Q⁰);
  - recusa = r6 − r5.
- **Ressalvas:**
  - r0 e r1 são texto do Alpaca, fora da distribuição do Gemma 3. Servem para *leitura* das representações, não como comportamento do modelo.
  - As respostas naturais (r4, r7) foram geradas pelo Gemma-4-12B e precisam ser **regeradas com o Gemma-3-4B** para H4.
  - O Gemma 3 não tem papel `system`: o system prompt vai no início do turno do usuário.

## Pedidos mais difíceis com um modelo mais fraco

1. **Trocar de modelo já ajuda.** A dificuldade é relativa ao modelo. O primeiro passo é medir o ORR e o HCR do Gemma-3-4B nos 2.428 pares. No P6, os modelos de 7–8B ainda recusaram só ~1% dos Q⁰ válidos. Se o 4B também quase não errar, os itens abaixo resolvem.
2. **Conjuntos externos difíceis.** Já foram excluídos do nosso dataset no P0, então servem de teste fora da distribuição:
   - **XSTest**: 250 seguros e 200 inseguros, com contrastes do mesmo tipo;
   - **OR-Bench-Hard-1K**: benignos que a maioria dos modelos recusa;
   - **FalseReject**.
3. **Mineração adversarial barata.** Gerar variações de Q⁰ que *soem* mais alarmantes (mais gatilhos, tom mais duro) mantendo a intenção benigna. O **4B filtra**: só fica o que ele recusa. O **juiz de 70B** confirma que o pedido continua benigno, rodando só nos sobreviventes. O modelo barato faz a peneira em massa, e o caro roda pouco.
4. **Desenho 2×2: nocividade × superfície.** Necessário para H3.

   | | com gatilho | sem gatilho |
   |---|---|---|
   | **nocivo** | Q⁺ (temos) | Q⁺ parafraseado com eufemismos (gerar) |
   | **benigno** | Q⁰ (temos) | Q⁰ com o gatilho trocado por termo neutro (gerar) |

   As células novas passam pelo mesmo P4 (guards + juiz) para confirmar o rótulo de intenção.

O pipeline da Fase 0 já é parametrizado pelo modelo-alvo (`models.subject` no YAML). Gerar e rotular com o Gemma-3-4B é uma config nova: não exige código novo.

## Divisão em 4 frentes

Cada frente tem um dono, entregáveis com formato fixo e só uma dependência forte: o cache de ativações, disponível no dia 2.

| Frente | Dono | O que faz | Entrega |
|---|---|---|---|
| **A. Dados e comportamento** | Gustavo (h1003) | Config `phase1_gemma3_4b.yaml`: respostas naturais do 4B aos 2.428 pares, rotuladas com P4. Formatar XSTest, OR-Bench-Hard-1K e FalseReject. Gerar e validar as células do 2×2. Mineração adversarial de Q⁰. | `behavior_gemma3_4b.jsonl`, `cells_2x2.jsonl`, `external/*.jsonl`, tabela HCR/ORR por domínio |
| **B. Ativações e SAE próprio** | Davi (h1003) | Extrator único de ativações (camadas × posições: gatilho, pós-instrução, média, resposta), com cache em disco. Avaliar o Gemma Scope 2 nos nossos dados: variância explicada, ΔNLL ao substituir a ativação pela reconstrução, L0, features mortas. Treinar o SAE próprio (BatchTopK, 1–2 camadas). Script de intervenção: ablação e steering de features. | `acts/<modelo>/<camada>/<split>.safetensors` + índice, checkpoint do SAE, tabela de qualidade dos SAEs, `intervene.py` |
| **C. Análise densa** | Caio | Probes lineares por camada (nocivo × benigno) com CV agrupada por `group_id`. **Baseline lexical** (TF-IDF + regressão logística) para mostrar o que a representação sabe além das palavras. Direções por diferença de médias. Generalização cruzada (pares → 2×2, XSTest, OR-Bench-Hard). Cosseno com a direção de recusa. Escolher as camadas para o SAE. | curvas por camada, matriz de generalização, camadas escolhidas |
| **D. Features do SAE** | Frederico | Efeito pareado d_j por feature em cada contraste (pedidos e respostas), com FDR. Classificar features em nocividade, superfície/gatilho, recusa ou mistas. Rótulos do Neuronpedia + autointerp com LLM nos exemplos de maior ativação. Comparar Gemma Scope 2 × SAE próprio (casamento por cosseno dos decoders). Com B, rodar o teste causal de H4. | ranking de features por contraste, fichas das top-k, resultados de ablação |

### Cronograma sugerido

| Dias | O que acontece |
|---|---|
| D1–D2 | **B** entrega um cache pequeno (500 grupos, 4 camadas). Para não ficarem parados: **C** monta o baseline lexical e o código dos probes; **D** monta o autointerp e o acesso ao Neuronpedia; **A** sobe o 4B e começa a rotular. |
| D3–D7 | Análises completas nos 2.428 pares. **A** entrega o 2×2 e os conjuntos externos. **B** treina o SAE próprio. |
| D8–D10 | Testes de H3 (generalização) e H4 (comportamento e intervenção). Comparação entre os dois SAEs. |
| D11+ | Figuras e escrita. Cada frente escreve sua seção de métodos e resultados. |

### Acesso ao cluster

Só Gustavo e Davi têm conta na h1003. Caio e Frederico trabalham sobre o **cache de ativações** (CPU ou GPU pequena basta para probes e estatística) e sobre o 4B, que cabe numa GPU de 24 GB. Vale pedir contas ao admin ou usar o cluster Spark (laudite).

## Riscos

- **Confusões entre Q⁺ e Q⁰ além da nocividade:** fonte (PKU vs OR-Bench/gerado), tamanho e tópico. Mitigação:
  - controlar o tamanho;
  - analisar separadamente pares *retrieved* e *generated*;
  - usar o 2×2 e o baseline lexical.
- **SAEs do Gemma Scope 2 ruins no nosso domínio:** o SAE próprio é o plano B, e o ΔNLL avisa cedo.
- **O 4B errar pouco (H4 sem casos):** mineração adversarial, OR-Bench-Hard, ou trocar para o Gemma-3-1B-IT. O Gemma Scope 2 cobre todos os tamanhos do Gemma 3; confirmar se há SAEs da versão IT do 1B.
