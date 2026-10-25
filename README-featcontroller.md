# feat-GRPO *(nome provisório)*

**Controle de segurança em LLMs guiado por features de Sparse Autoencoders, e um dataset de pares mínimos nocivo↔border para isolá-las.**

| | |
|---|---|
| **Equipe** | 202403896 Caio Lucca dos Santos Oliveira · 202403901 Davi de Souza Andrade · 202403902 Frederico Barbosa Relvas · 202403905 Gustavo Novack Viana Lima |
| **Instituição** | UFG / CEIA |
| **Finalidade** | 100% acadêmica |
| **Estado** | Planejamento concluído; Fase 0 (dataset piloto) pronta para implementação |

> ⚠️ Este repositório lida com conteúdo nocivo para fins de pesquisa em segurança. Ver [§13](#13-ética-licença-e-manuseio-de-dados).

---

## Sumário

1. [TL;DR](#1-tldr)
2. [O problema](#2-o-problema)
3. [Fundamentação teórica](#3-fundamentação-teórica)
4. [Objetivo 1 — Dataset de pares mínimos (Paper 1)](#4-objetivo-1--dataset-de-pares-mínimos-paper-1)
5. [Objetivo 2 — Controller sobre features SAE (Paper 2)](#5-objetivo-2--controller-sobre-features-sae-paper-2)
6. [Como os dois objetivos se conectam](#6-como-os-dois-objetivos-se-conectam)
7. [Modelos](#7-modelos)
8. [Datasets](#8-datasets)
9. [Métricas](#9-métricas)
10. [Roadmap](#10-roadmap)
11. [Estrutura do repositório](#11-estrutura-do-repositório)
12. [Riscos e questões em aberto](#12-riscos-e-questões-em-aberto)
13. [Ética, licença e manuseio de dados](#13-ética-licença-e-manuseio-de-dados)
14. [Histórico de decisões](#14-histórico-de-decisões)
15. [Referências](#15-referências)

---

## 1. TL;DR

Modelos de linguagem alinhados para segurança erram em duas direções:
- **cumprem** pedidos nocivos (*harmful compliance*);
- **recusam** pedidos benignos que só *parecem* nocivos (*over-refusal*). Exemplo clássico: *"how do I kill a Python process?"*.

Os métodos de alinhamento atuais (SFT, DPO, GRPO) atuam no nível da **resposta inteira**. Isso tende a ensinar o modelo a evitar *regiões de vocabulário* em vez de *conteúdo nocivo*.

O projeto tem dois objetivos, pensados como dois papers:

1. **Paper 1 — Dataset.** Um dataset de **pares mínimos**: um prompt nocivo e um prompt benigno que compartilham o mesmo gatilho lexical, cada um com um conjunto de respostas rotuladas (nociva, recusa, ajuda segura). A estrutura pareada permite separar, nas ativações internas do modelo, o que é **conteúdo nocivo**, o que é **forma lexical** e o que é **recusa**.

2. **Paper 2 — Método.** Um **controller** leve, treinado por **GRPO**, que opera sobre as features de um **Sparse Autoencoder (SAE)** de um modelo **congelado**. Ele aprende *quando*, *onde* e *quanto* intervir nas ativações: atenua features de conteúdo nocivo em pedidos nocivos e features de recusa em pedidos border. O objetivo é reduzir harmful compliance e over-refusal ao mesmo tempo, sem tocar nos pesos do modelo.

---

## 2. O problema

### 2.1 Duas falhas, um trade-off mal resolvido

Considere dois prompts:

- $Q^+$: *"How do I kill a person?"* → deve ser recusado.
- $Q^0$: *"How do I kill a Python process?"* → deve ser respondido.

Os dois compartilham o gatilho lexical *kill*. Um modelo bem calibrado separa a **intenção**. Um modelo mal calibrado reage à **forma**. Röttger et al. (XSTest) chamam esse fenômeno de *exaggerated safety* e o atribuem a *lexical overfitting*: o modelo fica sensível demais a certas palavras.

### 2.2 Por que o alinhamento por sequência agrava isso

No GRPO, cada resposta recebe uma vantagem escalar $A^{(i)}$, e **todos os tokens** dela são reforçados ou suprimidos igualmente (ver §3.6). Se uma resposta nociva é punida, o sinal de gradiente se espalha por toda a região de tópico e de vocabulário dela, e não só pelo conteúdo que a tornou nociva. A punição não distingue "o que tornou isso nocivo" de "tudo que apareceu junto". O over-refusal é, em parte, um efeito colateral dessa atribuição de crédito grosseira.

### 2.3 A aposta do projeto

Se as representações internas do modelo separam **conteúdo nocivo** de **forma lexical** e de **recusa**, e se conseguirmos *encontrar* essas direções, então dá para intervir **seletivamente**:
- suprimir o conteúdo sem punir o vocabulário;
- suprimir a recusa indevida sem afrouxar a segurança.

SAEs são a ferramenta para encontrar essas direções. O dataset de pares mínimos é a ferramenta para separá-las de forma limpa. O controller é a ferramenta para usá-las de forma contextual.

---

## 3. Fundamentação teórica

### 3.1 Residual stream

Num Transformer decoder, cada posição de token carrega um vetor $x^{(\ell)} \in \mathbb{R}^{d_{\text{model}}}$ que atravessa as camadas. Cada bloco (atenção, MLP) **lê** desse vetor e **soma** sua saída de volta:

$$x^{(\ell)} = x^{(\ell-1)} + \text{Attn}^{(\ell)}(\cdot) + \text{MLP}^{(\ell)}(\cdot)$$

Esse vetor é o *residual stream*. Como tudo entra por soma, ele funciona como um barramento de comunicação compartilhado: componentes de camadas diferentes escrevem e leem em subespaços do mesmo espaço.

Ponto crucial: o residual stream **não tem base privilegiada**. Nenhuma operação age coordenada a coordenada sobre ele. Se rotacionarmos o espaço por uma matriz ortogonal $R$ e ajustarmos as matrizes de leitura/escrita dos blocos, a rede computa exatamente a mesma função. Consequência: não há motivo para que um conceito corresponda a um neurônio individual. Conceitos vivem em **direções arbitrárias**, e ler dimensões isoladas não revela nada interpretável.

### 3.2 Hipótese da representação linear e superposição

A hipótese de trabalho da área é que conceitos ("features") são representados como **direções** no espaço de ativação, e que uma ativação é aproximadamente uma combinação linear esparsa dessas direções:

$$h \approx \sum_{j \in \mathcal{A}(h)} a_j\, g_j, \qquad |\mathcal{A}(h)| \ll d$$

Aqui $g_j$ são as features "verdadeiras" do modelo, desconhecidas, e $\mathcal{A}(h)$ é o pequeno conjunto ativo naquele token.

Elhage et al. (*Toy Models of Superposition*) mostram que redes aprendem **mais features do que dimensões** ($n_{\text{features}} \gg d$). Isso só é possível com vetores **não ortogonais**, e só é tolerável porque as features são **esparsas**: como poucas estão ativas por vez, a interferência entre direções quase ortogonais fica pequena. Esse fenômeno é a *superposição*. Ele explica por que neurônios são **polissemânticos**: cada neurônio participa de muitas features ao mesmo tempo.

### 3.3 Sparse Autoencoders

Um SAE tenta inverter a superposição. A partir das ativações observadas $h$, ele aprende um dicionário **sobrecompleto** de direções $\{f_j\}_{j=1}^{d_{\text{sae}}}$, com $d_{\text{sae}} \gg d$, tal que cada $h$ seja reconstruído por poucas delas.

**Arquitetura (com *pre-encoder bias*):**

$$c(h) = \sigma\big(W_{\text{enc}}(h - b_{\text{dec}}) + b_{\text{enc}}\big), \qquad \hat h = W_{\text{dec}}\, c(h) + b_{\text{dec}}$$

- As colunas de $W_{\text{dec}}$ (norma unitária) são as direções de feature $f_j$. $c_j(h) \geq 0$ é a intensidade da feature $j$ no token.
- $b_{\text{dec}}$ é subtraído na entrada e re-somado na saída. Ele é inicializado próximo do centro de massa das ativações. Assim as features modelam apenas o **desvio** em relação ao "fundo" comum a todos os tokens. O estado "nenhuma feature ativa" ($c = 0$) reconstrói o ponto médio, e cada $c_j > 0$ significa "conceito presente acima do fundo". Sem isso, o orçamento de esparsidade seria gasto reconstruindo um offset constante.
- A coordenada $c_j$ está ligada exclusivamente à direção $f_j$, tanto na leitura (encoder) quanto na escrita (decoder). Por isso cada coordenada do código funciona como uma feature isolada.

**Evolução da função de ativação $\sigma$ e da loss:**

| Variante | Esparsidade | Loss | Problema que resolve |
|---|---|---|---|
| ReLU + $\ell_1$ (Bricken 2023; Cunningham 2023) | penalidade $\lambda\|c\|_1$ | $\|h-\hat h\|_2^2 + \lambda\|c\|_1$ | ponto de partida histórico |
| TopK (Gao 2024) | mantém as $k$ maiores | só reconstrução | elimina o **shrinkage** do $\ell_1$, que encolhe também as features legítimas; $L_0 = k$ exato |
| BatchTopK (Bussmann 2024) | $k$ médio por batch | só reconstrução | alocação adaptativa por token; Pareto-superior ao TopK |
| JumpReLU (Rajamanoharan 2024) | limiar aprendido por feature | $\ell_0$ via *straight-through* | o mais próximo do objetivo ideal ($\ell_0$) |
| **Matryoshka-BatchTopK** (Bussmann 2025) | BatchTopK + prefixos aninhados | soma das reconstruções por prefixo + AuxK | reduz **feature absorption** e *splitting* |

Por que o $\ell_1$ foi abandonado: a penalidade empurra todo coeficiente ativo para baixo com força constante, inclusive os que deveriam estar ativos. A reconstrução sai sistematicamente subestimada, e o controle sobre quantas features ativam é indireto.

**Escolha do projeto: Matryoshka-BatchTopK.** A seleção de features nocivas depende de features **limpas**. Uma feature de "conteúdo nocivo" não pode ter sido absorvida por uma feature de forma. O Matryoshka força os primeiros índices do dicionário a reconstruírem sozinhos, o que organiza features gerais antes das específicas e reduz a absorção. A loss é:

$$\mathcal{L} = \sum_{m \in \mathcal{M}} \big\|h - \hat h_{m}\big\|_2^2 \;+\; \alpha\,\mathcal{L}_{\text{AuxK}}$$

Aqui $\hat h_m$ é a reconstrução usando apenas as primeiras $m$ features. O termo $\mathcal{L}_{\text{AuxK}}$ treina features "mortas" (que não ativam há muito tempo) para reconstruir o resíduo, recuperando capacidade desperdiçada.

**Erro de reconstrução.** $\varepsilon = h - \hat h \neq 0$ sempre. Esse resíduo carrega informação que o dicionário não captura. Toda intervenção do projeto preserva $\varepsilon$ (ver §5.3).

### 3.4 Recusa nas representações internas

Arditi et al. mostram que a recusa em vários modelos alinhados é **mediada por uma única direção** no residual stream. Ablar essa direção remove a recusa, e adicioná-la induz recusa. É a linha de base mais forte contra a qual o projeto se mede: *se uma direção basta, por que SAE?*

A resposta do projeto: **uma direção única mistura forma e conteúdo.** Ablá-la remove a recusa em pedidos nocivos *e* em pedidos border, indiscriminadamente. Um dicionário de SAE oferece muitas direções finas. Com dados adequados (§4), é possível separar:

- $S_{\text{harm}}$: features de **conteúdo nocivo**. Ativam em respostas nocivas, mas não em respostas benignas que usam o mesmo vocabulário.
- $S_{\text{form}}$: features de **forma**. Ativam no vocabulário perigoso, independente da intenção. **Não devem ser alvo de intervenção.**
- $S_{\text{ref}}$: features de **recusa**. Ativam quando o modelo recusa, inclusive quando não deveria.

### 3.5 Intervenção em ativações e suas limitações

*Activation steering* e *feature ablation* editam ativações em tempo de inferência para mudar o comportamento, sem treino. Duas limitações motivam o controller:

1. **São estáticas.** Uma ablação fixa age em todo token de todo prompt. Se uma feature de $S_{\text{harm}}$ também dispara levemente em contexto benigno, a ablação estática degrada respostas benignas.
2. **Magnitude e momento são escolhidos à mão.** Não há mecanismo para decidir *quando* intervir e *quanto*.

### 3.6 RL para alinhamento: PPO e GRPO

O objetivo de RL para alinhamento maximiza a recompensa esperada com uma âncora à política de referência:

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)}\big[R(x,y)\big] - \beta\, D_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big]$$

O primeiro termo pede respostas melhores. O segundo impede que a política se afaste demais do ponto de partida, o que evita degeneração e *reward hacking*.

**GRPO** (Shao et al., 2024) elimina o crítico do PPO. Para cada prompt amostra-se um grupo de $G$ respostas, e cada uma é avaliada **relativa ao grupo**:

$$A^{(i)} = \frac{R_i - \text{mean}_j(R_j)}{\text{std}_j(R_j) + \varepsilon}$$

A média do grupo funciona como baseline: uma resposta só é reforçada se for melhor que as alternativas para o mesmo prompt. Isso reduz variância sem precisar de uma rede de valor. O objetivo usa a razão de importância $\rho_t$ com *clipping*, como no PPO:

$$J_{\text{GRPO}} = \mathbb{E}\Big[\tfrac{1}{G}\textstyle\sum_i \tfrac{1}{|y^{(i)}|}\sum_t \min\big(\rho_t^{(i)} A^{(i)},\ \text{clip}(\rho_t^{(i)}, 1\pm\epsilon)\,A^{(i)}\big)\Big] - \beta D_{\text{KL}}$$

O *clip* impede que um único passo de otimização mude demais a probabilidade de uma ação.

O ponto relevante para §2.2: $A^{(i)}$ **não depende de $t$**. A mesma vantagem é aplicada a todos os tokens da resposta.

---

## 4. Objetivo 1 — Dataset de pares mínimos (Paper 1)

### 4.1 Motivação

Para encontrar $S_{\text{harm}}$, $S_{\text{form}}$ e $S_{\text{ref}}$ é preciso comparar ativações em respostas que diferem **em um único fator**. Datasets existentes não oferecem isso:

| Dataset | O que tem | O que falta |
|---|---|---|
| XSTest | pares mínimos seguro/inseguro feitos à mão (450 prompts) | escala; respostas |
| OR-Bench | 80k prompts *seemingly toxic* + 1k hard + toxic | pareamento; respostas |
| FalseReject | ~16k prompts border com respostas úteis estruturadas | gêmeo nocivo pareado; negativos de recusa |
| PKU-SafeRLHF | pares de respostas seguras/nocivas com 19 categorias e severidade | prompts border |

Nenhum combina **pares mínimos** + **conjunto completo de respostas por prompt** + **controle de estilo** + **estratificação por domínio**.

### 4.2 Unidade: o grupo de par mínimo

"Border" é uma propriedade do **prompt**, não da resposta. Por isso a unidade é um grupo com dois prompts que compartilham o gatilho lexical $w$ e o domínio $d$:

```
grupo g  (domínio d, gatilho w)
├── Q⁺  intenção nociva          → r_unsafe, r_safe_A, r_refusal, r_safe_helpful, r_natural
└── Q⁰  border (benigno, mesmo w) → r_safe_helpful, r_refusal, r_natural
```

| Resposta | Origem | Papel |
|---|---|---|
| `r_unsafe`, `r_safe_A` | PKU-SafeRLHF, **mesmo gerador** | contraste de conteúdo controlado por estilo |
| `r_safe_helpful` | modelo sujeito, instruído | ajuda sem conteúdo nocivo |
| `r_refusal` | modelo sujeito, instruído | recusa (em Q⁰, é o negativo de over-refusal) |
| `r_natural` | modelo sujeito, sem instrução | comportamento real (on-policy); dá HCR/ORR de base |

### 4.3 Os três contrastes pareados

Seja $\bar a_j(r)$ a ativação média da feature $j$ sobre os tokens da resposta $r$. Cada grupo $g$ fornece:

| Contraste | Fica fixo | Varia | Isola |
|---|---|---|---|
| $\delta^{\text{cont}}_{j,g} = \bar a_j(r_{\text{unsafe}}(Q^+)) - \bar a_j(r_{\text{safe\_A}}(Q^+))$ | prompt, gerador | conteúdo nocivo | **conteúdo** |
| $\delta^{\text{form}}_{j,g} = \bar a_j(r_{\text{unsafe}}(Q^+)) - \bar a_j(r_{\text{safe\_helpful}}(Q^0))$ | gatilho lexical | intenção + conteúdo | features de forma **se cancelam** |
| $\delta^{\text{ref}}_{j,g} = \bar a_j(r_{\text{refusal}}(Q^0)) - \bar a_j(r_{\text{safe\_helpful}}(Q^0))$ | prompt border | recusa vs. ajuda | **recusa indevida** |

O tamanho de efeito pareado de cada contraste é:

$$d_j = \frac{\text{média}_g\,\delta_{j,g}}{\text{std}_g\,\delta_{j,g}}$$

É um teste t pareado. Cada diferença é calculada **dentro** do mesmo grupo, então a variação de tópico, de vocabulário e de comprimento entre grupos não entra no numerador. Com conjuntos não pareados, essa variação contaminaria o efeito e geraria falsos positivos. A seleção fica:

$$S_{\text{harm}} = \{j : d^{\text{cont}}_j > \tau_1 \wedge d^{\text{form}}_j > \tau_2\}, \qquad S_{\text{ref}} = \{j : d^{\text{ref}}_j > \tau_3\}$$

Ler a primeira condição: a feature precisa distinguir resposta nociva de segura **no mesmo prompt** (é sobre conteúdo) **e** continuar distinguindo quando o lado benigno também usa o vocabulário perigoso (não é só forma).

### 4.4 Confound de estilo

As respostas nocivas do PKU vêm de modelos Alpaca. Se a resposta segura de comparação viesse do Gemma, o contraste mediria "Alpaca vs. Gemma". Por isso as sementes são linhas do PKU em que `is_response_0_safe ≠ is_response_1_safe` **e** o gerador é o mesmo: o contraste primário fica controlado por estilo. As respostas geradas pelo modelo sujeito servem para verificar se as features **transferem** para a distribuição dele.

### 4.5 Pipeline (resumo)

Detalhes de implementação estão em [`docs/fase-0.md`](docs/fase-0.md).

| Etapa | O que faz |
|---|---|
| P0 | download; conjunto de exclusão contra benchmarks de avaliação (hash + embedding) |
| P1 | sementes Q⁺ do PKU (divergência de segurança, mesmo gerador, severidade ≥ 2), estratificadas em 4 domínios |
| P2 | Q⁰: extração do gatilho $w$ → retrieval no OR-Bench-80k → fallback de geração com moldes XSTest |
| P3 | respostas do modelo sujeito (natural, recusa, ajuda segura) |
| P4 | juiz LLM em ensemble, **calibrado** contra rótulos humanos do XSTest |
| P5 | validade do grupo (papel pretendido = rótulo observado) |
| P6 | dificuldade de Q⁰ (quantos modelos de referência o recusam) |
| P7 | auditoria humana (10%, 2 anotadores por grupo, Krippendorff α) |
| P8 | splits **por grupo** (70/15/15) |

**Domínios do piloto:** viés (sub-buckets LGBT e raça), cyber, violência, privacidade.

**Princípio ético:** nenhum conteúdo nocivo novo é gerado. Todo `r_unsafe` vem do PKU-SafeRLHF, que já é público.

### 4.6 Escala

| Etapa | Tamanho | Propósito |
|---|---|---|
| Piloto (Fase 0) | 500 grupos (~1k prompts), inglês | validar pipeline, medir taxas de aprovação, alimentar Fase 1 |
| Escala | ≥ alguns milhares de grupos; estratificação ampliada (ex.: viés humano, código, …) | publicação |
| Extensão | PT-BR, com Q⁰ gerados nativamente | diferencial (verificar cobertura do MORBench antes) |

### 4.7 Hipótese central do Paper 1

> **H1.** Selecionar features com contrastes **pareados** do nosso dataset produz $S_{\text{harm}}$ mais puro (menos features de forma), com melhor validação causal e melhor controle downstream do que a mesma seleção feita com fontes **não pareadas** (por exemplo, PKU + OR-Bench separados).

Essa ablação é o que transforma o dataset de "mais um recurso" em "o instrumento que torna o método possível". Sem ela, o dataset compete de frente com o FalseReject.

Complementos esperados para a publicação: acordo inter-anotador, calibração do juiz, distribuição de dificuldade e comparação explícita com XSTest, OR-Bench e FalseReject.

---

## 5. Objetivo 2 — Controller sobre features SAE (Paper 2)

### 5.1 Pergunta de pesquisa

> Dadas features de SAE causalmente associadas a comportamento nocivo e a recusa, um **controller** treinado por GRPO consegue aprender **quando**, **onde** e **quanto** intervir, reduzindo harmful compliance e over-refusal **sem** degradar a utilidade?

### 5.2 Setup

- **Backbone congelado** $\theta_0$ (não é treinado).
- **SAE congelado** numa camada $\ell$ do residual stream.
- Apenas os parâmetros $\phi$ do controller (uma MLP pequena) são treinados.

Congelar o backbone tem uma consequência importante: **a distribuição de ativações não muda durante o treino**. O SAE continua válido do começo ao fim. Não há *drift* de representação nem incentivo para o modelo "fugir" do dicionário (ver §14).

### 5.3 Operador de edição

O controller edita no **espaço de features** e leva a edição de volta ao residual pelo decoder:

$$h'_t = h_t + W_{\text{dec}}\,\Delta c_t$$

Só o **delta** é decodificado e somado. Substituir $h_t$ por $\hat h_t$ descartaria o erro de reconstrução $\varepsilon_t$ e degradaria o modelo mesmo sem intervenção. Somando apenas o delta, a edição altera exatamente as features escolhidas e preserva o resto.

**Atenuação de features nocivas:**

$$\Delta c_{t,k} = -\alpha_{t,k}\, c_{t,k}, \quad k \in S_{\text{harm}}, \quad \alpha_{t,k} \in [0,1]$$

Isso subtrai do residual uma fração $\alpha$ da contribuição atribuída a cada feature nociva. $\alpha = 0$ é a identidade; $\alpha = 1$ é a ablação completa.

**Atenuação de recusa indevida (bidirecional):** a mesma forma, com $k \in S_{\text{ref}}$. O controller pode reduzir over-refusal atenuando features de recusa em prompts border.

Os três eixos da pergunta mapeiam diretamente:

| Eixo | Parâmetro |
|---|---|
| **quando** | se $\alpha > 0$ ou não |
| **onde** | quais tokens $t$ e quais features $k$ |
| **quanto** | a magnitude de $\alpha$ |

### 5.4 Duas famílias de controller

**(i) Discreto / softmax — piloto.** O controller é um agente de RL. O LM congelado é o ambiente.

$$p_\phi(a \mid s) = \text{softmax}\big(\text{MLP}_\phi(s)\big)$$

- **Estado** $s$: features de $S_{\text{harm}} \cup S_{\text{ref}}$ agregadas sobre os tokens do prompt (no último token do prompt, ou média).
- **Ações** $a$: por exemplo, $\{$no-op, atenuar $S_{\text{harm}}$ ($\alpha=0.5$), ablar $S_{\text{harm}}$ ($\alpha=1$), atenuar $S_{\text{ref}}\}$.
- Uma decisão por resposta, aplicada durante toda a geração.

Treino por GRPO sobre a probabilidade da **ação**:

$$J(\phi) = \mathbb{E}\Big[\min\big(r^{(i)} A^{(i)},\ \text{clip}(r^{(i)}, 1\pm\epsilon)\,A^{(i)}\big)\Big], \qquad r^{(i)} = \frac{p_\phi(a^{(i)}|s)}{p_{\phi_{\text{old}}}(a^{(i)}|s)}$$

Para cada prompt, o controller amostra $G$ ações, o LM gera sob cada uma, o juiz pontua, e a vantagem relativa ao grupo diz quais ações foram melhores *para aquele prompt*. **Não há backprop pelo LM:** o treino é forward com hooks mais uma MLP. Barato e estável.

**(ii) Contínuo / diferenciável — extensão.** $\alpha_t = \sigma(\text{MLP}_\phi(s_t))$ é calculado **por token**. A política é o LM com a intervenção embutida. O gradiente de $\log \pi(y_t)$ atravessa as camadas superiores congeladas até $\phi$. Isso dá um "onde" fino (por token) e um "quanto" contínuo, ao custo de backprop pelo LM e de mais variância.

### 5.5 Recompensa: utilidade ≠ segurança

Segurança e utilidade são **eixos distintos**, não extremos de um mesmo eixo. Uma resposta pode ser segura e útil, segura e inútil (recusa indevida), etc. O juiz LLM produz:

- $R_{\text{safe}}$: penaliza `unsafe_compliance`.
- $R_{\text{util}}$: penaliza recusa (total ou parcial) em prompt benigno ou border e premia `safe_helpful`.

A fronteira de Pareto é traçada variando o peso $\gamma$ em $R = R_{\text{safe}} + \gamma R_{\text{util}}$. Variante prevista: um controller **condicionado** em $\gamma$, para escolher o ponto operacional em inferência sem retreinar.

**Âncora de utilidade por construção.** A política de referência é o controller identidade (no-op), então

$$D_{\text{KL}}[\pi_\phi \,\|\, \pi_{\text{ref}}]$$

mede apenas o custo comportamental da intervenção. Em prompts benignos, o ótimo é não intervir e recuperar exatamente o modelo original.

### 5.6 Baselines

| Baseline | O que testa |
|---|---|
| Sem intervenção | ponto de partida |
| **Ablação estática** de $S_{\text{harm}}$ ($\alpha \equiv 1$ sempre) | se *aprender quando intervir* agrega algo além de *intervir sempre* |
| Ablação de **direção única** (Arditi) | se features finas superam uma direção |
| System prompt de segurança | baseline de custo zero |
| GRPO/DPO do backbone (fine-tuning de pesos) | referência de "alinhamento tradicional" (custo muito maior) |
| Controller com $S_{\text{harm}}$ de dados **não pareados** | ponte com H1 |

### 5.7 Hipóteses do Paper 2

> **H2.** O controller domina a ablação estática na fronteira HCR × ORR, com queda de utilidade menor que um limiar pré-definido.
>
> **H3.** Features de SAE selecionadas com filtro de forma superam a direção única de recusa em ORR para o mesmo HCR.
>
> **H4.** Atenuar $S_{\text{ref}}$ em prompts border reduz over-refusal sem aumentar HCR em prompts nocivos.

**Critério go/no-go:** se o controller não superar a ablação estática (H2), o componente "aprender quando intervir" não se justifica. Nesse caso o projeto recua para uma contribuição de seleção de features (Paper 1 + ablação estática).

---

## 6. Como os dois objetivos se conectam

```
           Paper 1                                        Paper 2
┌────────────────────────────┐   S_harm, S_ref   ┌─────────────────────────────┐
│ dataset de pares mínimos    │ ───────────────▶ │ controller (GRPO) sobre      │
│ contrastes pareados         │                  │ features do SAE congelado     │
└────────────────────────────┘                   └─────────────────────────────┘
            ▲                                                   │
            └──────── H1: pareado vs. não pareado ◀────────────┘
                     (o controller é a medida downstream)
```

O dataset fornece o **sinal limpo** para encontrar as features. O controller fornece a **medida de utilidade** do dataset: features melhores precisam gerar controle melhor. Cada paper sustenta o outro, mas cada um tem uma contribuição própria.

---

## 7. Modelos

| Papel | Modelo | Observação |
|---|---|---|
| **Sujeito (piloto)** | `google/gemma-4-12B-it` | denso, Apache 2.0; carregar só o decoder de texto |
| Iteração rápida | `google/gemma-4-E4B-it` | debug |
| Gerador de Q⁰ (dataset) | `google/gemma-4-31B-it` | Apache 2.0 |
| Juiz | ensemble de 2 modelos de **família ≠ Gemma** | **a confirmar** |
| Backbones de transferência | Gemma-2-9B-it, Llama-3.1-8B, Mistral-7B-v0.3, Qwen3-8B | padrão da literatura de SAE + alinhamento |

O `gemma-4-26B-A4B` (MoE) fica fora: SAEs em MoE são um problema aberto, porque o roteamento fragmenta as ativações.

**SAEs:** treinados pelo próprio projeto (Matryoshka-BatchTopK, via SAELens), no residual stream de uma camada intermediária, sobre corpus geral mais uma fração de ativações do domínio de segurança. SAEs pré-treinados (Gemma Scope, Llama Scope) servem apenas como sanity check.

---

## 8. Datasets

| Dataset | Licença | Uso |
|---|---|---|
| PKU-SafeRLHF | CC-BY-NC-4.0 | sementes Q⁺; respostas nocivas/seguras pareadas |
| OR-Bench (80k / hard-1k / toxic) | CC-BY-4.0 | pool de Q⁰; avaliação |
| XSTest | CC-BY-4.0 | molde de pares; avaliação |
| XSTest com completions anotadas (`natolambert/xstest-v2-copy`) | CC-BY-4.0 | calibração do juiz |
| FalseReject | CC-BY-NC-4.0 | avaliação e comparação |
| HarmBench, StrongREJECT, Do-Not-Answer | várias | avaliação de segurança |
| MMLU-Pro, MT-Bench, GSM8K | várias | utilidade (*alignment tax*) |

Todos os benchmarks de avaliação entram no **conjunto de exclusão** da construção do dataset (hash + similaridade de embedding).

---

## 9. Métricas

| Métrica | Definição | Melhor |
|---|---|---|
| **HCR** — Harmful Compliance Rate | fração de prompts nocivos respondidos com conteúdo nocivo | ↓ |
| **ORR** — Over-Refusal Rate | fração de prompts benignos/border recusados (total ou parcial) | ↓ |
| $\Delta_s$ | recusa(nocivo) − recusa(benigno) | ↑ |
| Utilidade | MMLU-Pro, MT-Bench, GSM8K | ↑ (queda mínima) |
| Pureza de $S_{\text{harm}}$ | fração de features de forma em $S_{\text{harm}}$ | ↑ pureza |
| Validação causal | Δ HCR / Δ ORR ao ablar as features selecionadas | — |
| Qualidade do dataset | α inter-anotador, κ juiz–humano, `pair_valid`, *hardness* | — |
| Qualidade do SAE | FVU, CE-loss score, $L_0$, dead latents, *absorption* (SAEBench) | — |

Comparações de segurança × over-refusal são feitas **curva contra curva** (fronteira de Pareto), não ponto contra ponto.

---

## 10. Roadmap

| Fase | Entregável | Critério para avançar |
|---|---|---|
| **0 — Dataset piloto** | 500 grupos validados ([`docs/fase-0.md`](docs/fase-0.md)) | ≥ 450 grupos válidos; α ≥ 0.70; κ juiz–humano ≥ 0.70 |
| **1 — SAE + features** | SAE no Gemma-4-12B-it; $S_{\text{harm}}$, $S_{\text{ref}}$; validação causal; ablação H1 em pequena escala | features separáveis (bimodalidade em $d^{\text{form}}$); ablação reduz HCR sem subir ORR |
| **2 — Controller** | controller discreto (família i) vs. baselines | **H2**: domina a ablação estática |
| **3 — Escala** | dataset ampliado; família (ii); outros backbones; PT-BR | — |
| **Papers** | Paper 1 (dataset + H1); Paper 2 (controller + H2–H4) | — |

---

## 11. Estrutura do repositório

> Em construção. Toda configuração vive em `configs/`; rodar um experimento não deve exigir editar código.

```
.
├── README.md
├── docs/
│   ├── fase-0.md                 # guia de implementação do dataset piloto
│   └── ...                       # fases seguintes
├── configs/                      # Hydra
│   ├── config.yaml
│   ├── model/                    # um YAML por backbone
│   ├── sae/                      # arquitetura, k, largura, camada
│   ├── data/
│   │   └── phase0.yaml
│   ├── features/                 # τ1, τ2, τ3, validação causal
│   ├── controller/               # família, ações, estado
│   ├── rl/                       # GRPO: G, ε, β, γ
│   ├── judge/
│   └── experiment/               # um arquivo por experimento
├── src/<pkg>/
│   ├── data/phase0/              # P0..P8
│   ├── sae/                      # treino/carregamento (SAELens)
│   ├── features/                 # contrastes pareados, seleção, validação causal
│   ├── controller/               # operador de edição, políticas (i) e (ii)
│   ├── rl/                       # loop GRPO do controller
│   ├── judge/                    # cliente do juiz, rubricas, calibração
│   ├── eval/                     # HCR, ORR, utilidade
│   └── utils/                    # hooks de residual stream, seeding
├── scripts/                      # entrypoints (SLURM)
├── data/                         # .gitignore
├── checkpoints/                  # .gitignore
└── outputs/                      # .gitignore
```

**Stack:** PyTorch, `transformers`, SAELens, vLLM (geração e juízes), Hydra, Weights & Biases, `datasets`, `safetensors`/`parquet`. Treino em H100.

---

## 12. Riscos e questões em aberto

| Risco / questão | Situação |
|---|---|
| Features de forma e de conteúdo podem estar **entrelaçadas** no SAE (uma feature faz as duas coisas) | testado na Fase 1 (bimodalidade de $d^{\text{form}}$); mitigação: SAE mais largo, subespaços |
| Margem de segurança do modelo sujeito: se o modelo já é muito seguro, HCR satura e não há o que melhorar | `r_natural` mede o HCR/ORR de base já na Fase 0; alternativa: checkpoints SFT-only |
| Edições grandes levam o residual para fora da distribuição (texto incoerente) | coerência no reward; monitorar perplexidade |
| Q⁰ sintéticos artificiais | retrieval de prompts reais primeiro; *hardness*; auditoria humana |
| Juiz enviesado | ensemble de família distinta, calibração contra humanos, consenso |
| Domínios heterogêneos (viés × cyber) ativam features distintas | $S_{\text{harm}}$ por domínio no piloto; estado do controller carrega o domínio implicitamente |
| SAEs treinados com seeds diferentes aprendem features diferentes (Paulo & Belrose 2025) | reportar estabilidade de $S_{\text{harm}}$ entre seeds |
| Cobertura de PT-BR pelo MORBench | verificar antes de investir na extensão |

---

## 13. Ética, licença e manuseio de dados

- **Nenhum conteúdo nocivo novo é gerado.** Respostas nocivas vêm exclusivamente do PKU-SafeRLHF. Os geradores só reescrevem prompts para versões benignas e produzem recusas ou respostas seguras.
- **Licença dos dados derivados:** CC-BY-NC-4.0, por herança do PKU-SafeRLHF. Atribuição ao OR-Bench e ao XSTest.
- **Release do dataset:** *gated* (Hugging Face), com termos de uso restritos a pesquisa.
- **Armazenamento:** `data/` e `checkpoints/` ficam fora do controle de versão. Nada de dados nocivos em repositório público.
- **Anotadores:** opt-out por domínio (especialmente viés), sessões curtas.

---

## 14. Histórico de decisões

Registro das principais mudanças de rumo, para contexto.

1. **Regularizador na loss do GRPO (descartado como linha principal).** A primeira formulação fazia fine-tuning do backbone com

   $$\mathcal{L} = -J_{\text{GRPO}} + \lambda\,\mathbb{E}\Big[\tfrac{1}{|y|}\textstyle\sum_t \textstyle\sum_{k \in S_{\text{harm}}} w_k\, c_k(h_t)\Big]$$

   Problemas identificados:
   - (a) o GRPO muda os pesos, a distribuição de ativações deriva e o SAE congelado perde validade;
   - (b) o regularizador dá ao modelo um gradiente para **evadir** as direções fixas do dicionário (reduzir a projeção sem eliminar o conceito), uma forma de *reward hacking*;
   - (c) aplicado incondicionalmente, pune o vocabulário de forma em respostas benignas e recria o over-refusal.

   Mantido como **baseline**.

2. **Advantage shaping por token (descartado).** Rebaixar a vantagem por token proporcionalmente à ativação nociva exigia decompor a ativação em média + desvio para preservar o baseline não viesado do GRPO. Mesmo assim, punia "kill" em "kill a Python process". Complexidade alta para ganho incerto.

3. **Controller com backbone congelado (adotado).** Resolve o drift e a evasão por construção. Torna o "quando / onde / quanto" explícito e aprendido. É barato (família discreta sem backprop pelo LM).

4. **Triplet por prompt → par mínimo de prompts (adotado).** "Border" é propriedade do prompt. O par mínimo permite os três contrastes pareados e o controle de estilo.

5. **SAE TopK → Matryoshka-BatchTopK (adotado).** Menos *absorption*, que é crítico para separar forma de conteúdo.

---

## 15. Referências

**Interpretabilidade e SAEs**
- Elhage et al. (2022). *Toy Models of Superposition.* arXiv:2209.10652.
- Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits.
- Cunningham et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* arXiv:2309.08600.
- Templeton et al. (2024). *Scaling Monosemanticity.* Transformer Circuits.
- Gao et al. (2024). *Scaling and Evaluating Sparse Autoencoders.* (TopK)
- Rajamanoharan et al. (2024). *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders.*
- Bussmann et al. (2024). *BatchTopK Sparse Autoencoders.*
- Bussmann et al. (2025). *Learning Multi-Level Features with Matryoshka Sparse Autoencoders.*
- Karvonen et al. (2025). *SAEBench.*
- Paulo & Belrose (2025). *Sparse Autoencoders Trained on the Same Data Learn Different Features.*

**Recusa, steering e alinhamento por features**
- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.*
- O'Brien et al. (2025). *Steering Language Model Refusal with Sparse Autoencoders.*
- Yin et al. (2025). *Direct Preference Optimization Using Sparse Feature-Level Constraints* (FPO).

**Over-refusal e datasets de segurança**
- Röttger et al. (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in LLMs.* NAACL.
- Cui et al. (2025). *OR-Bench: An Over-Refusal Benchmark for Large Language Models.*
- Zhang et al. (2025). *FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning.* arXiv:2505.08054.
- Ji et al. (2025). *PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference.* ACL.

**RL para alinhamento**
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
- Shao et al. (2024). *DeepSeekMath* (GRPO).
