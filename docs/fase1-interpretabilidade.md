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

Os recursos de cada um definem a divisão:
- **Gustavo:** acesso normal à h1003 (H100/B200, juiz de 70B).
- **Caio e Frederico:** uma RTX 4090 cada (24 GB). O Gemma-3-4B em bf16 ocupa ~9 GB, e os SAEs de 16k a 262k cabem junto.
- **Davi:** temporariamente **sem GPU de cluster**. Fica com o que roda em CPU: texto, vetores densos já extraídos e a escrita.

| Frente | Dono | Recurso | O que faz | Entrega |
|---|---|---|---|---|
| **A. Dados, comportamento e SAE próprio** | Gustavo | h1003 | Tudo que precisa de modelo grande ou de muitas horas de GPU: respostas do 4B aos 2.428 pares e aos conjuntos externos, rotulagem com o juiz de 70B (P4), geração e validação das células do 2×2, mineração adversarial de Q⁰, treino do SAE próprio (BatchTopK, 1–2 camadas) | `behavior_gemma3_4b.jsonl`, `cells_2x2.jsonl`, `external/*.jsonl`, checkpoint do SAE |
| **B. Ativações e intervenção** | Caio | 4090 | Extrator único de ativações (camadas × posições) e o cache denso, publicado na pasta compartilhada. Qualidade do Gemma Scope 2 nos nossos dados: variância explicada, ΔNLL, L0, features mortas. `intervene.py` para ablação e steering de features por hooks. Roda os testes causais de H4 junto com C. | `artifacts/acts/...` + `index.jsonl`, tabela de qualidade dos SAEs, `intervene.py` |
| **C. Features do SAE** | Frederico | 4090 | Codifica as ativações com o Gemma Scope 2 (16k, depois 262k). Efeito pareado d_j por feature em cada contraste, com FDR. Classifica features em nocividade, superfície, recusa ou mistas. Gera os exemplos de maior ativação e o autointerp (LLM local na 4090, mais os rótulos do Neuronpedia). Compara com o SAE próprio da frente A. | ranking de features por contraste, fichas das top-k, resultados de ablação (com B) |
| **D. Análise densa, baseline e escrita** | Davi | CPU | **Baseline lexical** (TF-IDF + regressão logística), que só precisa de texto e começa no D1. Especificação e prompts das células do 2×2, que a frente A roda. Sobre o cache denso de B (~25 MB por camada × posição): probes por camada com CV agrupada, generalização cruzada (pares → 2×2, XSTest, OR-Bench-Hard), direções por diferença de médias e cosseno com a direção de recusa, correlação com o comportamento (H4, com os rótulos de A). Coordena o texto final: esqueleto, related work e figuras. | curvas por camada, matriz de generalização, rascunho do paper |

Se o Davi recuperar o acesso à h1003, ele assume o treino do SAE próprio, e a frente A fica só com dados e comportamento.

### Fluxo e dependências

```
            texto (datasets/phase0) ──► D: baseline lexical, especificação do 2×2
                                         │
A (h1003): 4B + juiz 70B ──► comportamento, 2×2, externos ──► B, C, D
                                         │
B (4090): extrator ──► cache denso ──► pasta compartilhada ──► D (probes)
       └─► extrator ──► C (4090): latentes do SAE, features
B + C: intervenção (H4)       A: SAE próprio ──► C (comparação)
```

### Cronograma sugerido

| Dias | O que acontece |
|---|---|
| D1–D2 | **B**: extrator e cache pequeno (500 grupos finais, 4 camadas). **C**: SAELens + Gemma Scope 2 e o pipeline de autointerp. **D**: baseline lexical e especificação do 2×2. **A**: config do 4B, respostas e rotulagem. |
| D3–D7 | **B**: cache dos 2.428 pares e das respostas. **C**: ranking de features. **D**: probes e camadas. **A**: entrega o 2×2 e os conjuntos externos, e treina o SAE próprio. |
| D8–D10 | H3 (generalização: D e C) e H4 (comportamento: D; intervenção: B e C). Comparação entre os dois SAEs (C). |
| D11+ | Figuras e escrita, coordenadas por D. Cada frente escreve sua seção. |

### Onde fica cada coisa

- **Dataset:** `datasets/phase0/`, no git.
- **Artefatos grandes** (ativações, latentes, checkpoints): `artifacts/`, fora do git. Ficam numa pasta compartilhada do time (Drive) com a mesma estrutura.
- **Resultados pequenos** (CSV, JSON, PNG): `results/<frente>/`, no git.
- **Contexto para os agentes de cada um:** [`AGENTS.md`](../AGENTS.md).

## Riscos

- **Confusões entre Q⁺ e Q⁰ além da nocividade:** fonte (PKU vs OR-Bench/gerado), tamanho e tópico. Mitigação:
  - controlar o tamanho;
  - analisar separadamente pares *retrieved* e *generated*;
  - usar o 2×2 e o baseline lexical.
- **SAEs do Gemma Scope 2 ruins no nosso domínio:** o SAE próprio é o plano B, e o ΔNLL avisa cedo.
- **O 4B errar pouco (H4 sem casos):** mineração adversarial, OR-Bench-Hard, ou trocar para o Gemma-3-1B-IT. O Gemma Scope 2 cobre todos os tamanhos do Gemma 3; confirmar se há SAEs da versão IT do 1B.
