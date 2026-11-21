# Frente A: relatório da noite (20 → 21/11/2026)

**Resumo.** Todas as entregas da frente A estão na `master`:
- 2×2, controles, externos, comportamento do 4B e mineração;
- três SAEs próprios (L17, L22 e L29), treinados e avaliados.

Não há nenhum job rodando. A dgx-H100-03 voltou segundo a infra, mas no Slurm continua `drained`.

Três pontos pedem decisão do time antes das ativações:
1. **Qual subconjunto do 2×2 é o principal** (seção 2).
2. **A troca de forma raramente baixa o alarme na linha nociva.** No conjunto completo, o 4B recusa *mais* sem o gatilho (seção 4).
3. **O validador da frente D reprova 89% do 2×2.** É desalinhamento entre o validador e o plano B, não problema nos dados (seção 3).

---

## 1. 2×2 (A1): plano B substituiu a frase-molde

| Rodada | Desenho | Grupos | Aprovados | Rendimento |
|---|---|---|---|---|
| seca v1 | frase-molde (`cell_template_v1`) | 50 | 7 | **14%** |
| seca v2 | frase-molde (`cell_template_v2`, radical relaxado) | 50 | 9 | **18%** |
| seca | **plano B** (troca só do trecho do gatilho) | 50 | 41 | **82%** |
| completo | **plano B** | 2.428 | **1.825** | **75%** |

O corte pré-registrado para manter a frase-molde era 40%, então o plano B entrou como previsto ([DESVIOS](../DESVIOS.md)).

Principais motivos de descarte da frase-molde:
- o termo neutro compartilha o radical do gatilho;
- a célula `ben_trig` vira nociva;
- o molde muda o pedido.

**1.825 grupos do completo**

| | |
|---|---|
| **Domínio** | bias 459/578 · cyber 451/606 · privacy 475/633 · violence 440/611 |
| **Tipo de gatilho** | action 833/1.116 (75%) · intent_marker 517/615 (84%) · surface 475/697 (68%) |
| **Split** | train 1.297 · val 271 · test 257 (o mesmo split do par original) |
| **Candidato aprovado** | 1º 1.730 · 2º 55 · 3º 40 |
| **Descartes (todas as tentativas)** | o pedido muda na linha nociva 562 · gatilho continua sem gatilho 350 · `harm_notrig` deixa de ser nocivo 221 · `ben_trig` não natural 113 · gatilho não localizado 99 |

**Controle de qualidade (`t2c_qc`, juiz Llama-3.3-70B).** Ele foi criado depois da [revisão às cegas de 40 grupos](revisao_ia_2x2.md), que achou cerca de 20% de versões trocadas agramaticais e cerca de 40% de termos "neutros" que não soam menos alarmantes. O QC anota os grupos e não descarta nenhum.

| Subconjunto | Critério | Grupos | train/val/test |
|---|---|---|---|
| `qc_grammatical` | as duas versões trocadas são gramaticais | **1.723** | 1.219/260/244 |
| `qc_pass_lenient` | gramatical, o alarme não sobe em nenhuma linha e cai em pelo menos uma | **604** | 435/85/84 |
| `qc_pass` (estrito) | gramatical e queda de alarme ≥ 1 nas **duas** linhas | **148** | 101/24/23 |

- **Assimetria de alarme.** A queda de alarme é ≥ 1 em só **9%** da linha nociva, contra **37%** da benigna.
  - Na linha nociva, o termo neutro quase sempre fica com alarme igual (1.514 de 1.825), porque a intenção domina o alarme do trecho.
  - Na prática, a manipulação de forma é **fraca** em `harm_*`. Isso reduz o poder para achar S_lex, mas não fabrica S_intent.
- **Minha recomendação:**
  - `qc_grammatical` (1.723) como conjunto principal do C1–C3;
  - `qc.alarm_drop_*` como covariável, num teste de validade da manipulação: o efeito lexical deve crescer com a queda de alarme;
  - `qc_pass_lenient` e `qc_pass` como robustez.

  **A decisão é do time e precisa sair antes de olhar ativações.**

## 2. Controles, externos e tipologia

- **Tipologia (A0):** 1.688 gatilhos, divididos em surface 433, action 857 e intent_marker 398. Está em `trigger_types.jsonl`.
- **Controles (A2):** `controls_trig_only.jsonl` tem 10.950 frases.
  - São 1.825 grupos × 3 moldes neutros × {peça com gatilho, peça neutra}, com `span` da peça.
  - Exemplo: *The phrase "mentally ill" appears in the title of the third chapter.*
- **Externos (A3):**

  | Conjunto | Itens | Rótulos | Proximidade dos pares |
  |---|---|---|---|
  | XSTest | 450 | 250 benignos, 200 nocivos | nenhum com cos > 0,9 com os pares; 13 com cos > 0,8 |
  | OR-Bench-Hard | 1.319 | benignos | nenhum com cos > 0,9; 111 com cos > 0,8 |
  | FalseReject | 1.187 | benignos | nenhum com cos > 0,9; 4 com cos > 0,8 |

  O campo `max_cos_to_pairs` vem em cada item, para filtrar vazamento se quiserem.

## 3. Validador da frente D sobre o 2×2 (pedido do Davi)

**Resultado:** `validate_template` aprova **196/1.825 (10,7%)**. Relatório completo em `results/frente-a/validacao_2x2.json`.

**Diagnóstico.** Os dados estão consistentes: em **99,4%** das 3.650 edições (2 linhas × 1.825 grupos), o trecho alterado fica **dentro** do span `slots.form` gravado. Os reprovados vêm de três regras do validador que não se aplicam ao plano B:

| Código | Casos | Causa |
|---|---|---|
| `edicao_nao_cobre_gatilho` | 1.415 linhas | Exige que a edição contenha o gatilho **inteiro**. Com gatilho de várias palavras, o termo neutro pode manter palavras inocentes. Exemplo: "LGBT sickness" vira "LGBT ..." e só "sickness" muda. É uma edição *mais* mínima, não um erro. |
| `edicao_nao_cobre_gatilho` | 435 linhas | `_single_edit` recorta prefixo e sufixo comuns **por caractere**, sem expandir até a borda da palavra. Exemplo: em "kill" → "kick", o trecho alterado vira "ll". Bug do validador. |
| `edicao_nao_cobre_gatilho` | 33 linhas | A forma alarmante gravada (`form_alarm*`) é uma flexão do gatilho. |
| `tamanho` | 1.346 grupos | O validador compara as **4** células. No plano B, as linhas nociva e benigna vêm de pedidos originais diferentes (Q⁺ e Q⁰), então o tamanho entre linhas não é controlado. Dentro de cada linha, a regra passa em 1.818/1.825. |

**Sugestão para a frente D (não alterei o código do Davi):**
- no plano B, a checagem certa é: a edição, expandida até a borda da palavra, fica dentro do span de forma, e a forma alarmante some na versão sem gatilho;
- tamanho comparado dentro de cada linha.

**Consequência real para a análise.** O contraste de intenção (linha nociva × benigna) compara frases de tamanho e conteúdo diferentes. A frente C deve usar a posição `last`/`trig` ou incluir o tamanho como covariável ao estimar I_g. O contraste lexical, dentro de cada linha, é bem controlado.

## 4. Comportamento do Gemma-3-4B-IT (A4)

Saída em `behavior_gemma3_4b.jsonl` (19.968 respostas). Rótulos: guards + juiz, consolidados por maioria. "Recusa" é `refusal` ou `partial_refusal`.

| Conjunto | Itens | Taxa de recusa | HCR / ORR |
|---|---|---|---|
| pares, Q⁺ nocivo | 2.428 | 56,6% | HCR 1,7% |
| pares, Q⁰ benigno | 2.428 | 0,4% | ORR 0,4% |
| 2×2 `harm_trig` | 1.825 | **57,9%** | HCR 1,6% |
| 2×2 `harm_notrig` | 1.825 | **63,2%** | HCR 1,5% |
| 2×2 `ben_trig` | 1.825 | **0,33%** (6) | ORR 0,33% |
| 2×2 `ben_notrig` | 1.825 | **0,05%** (1) | ORR 0,05% |
| XSTest benignos | 250 | 3,6% | ORR 3,6% |
| XSTest nocivos | 200 | 66% | HCR 0% |
| OR-Bench-Hard | 1.319 | **19,4%** | ORR 19,4% |
| FalseReject | 1.187 | **21,5%** | ORR 21,5% |
| mineração (variantes) | 4.856 | 5,8% | ORR 5,8% |

**Leitura**
- **O 4B quase nunca recusa as nossas perguntas benignas, com ou sem gatilho.** No 2×2 benigno, há 5 grupos em que só `ben_trig` é recusado e nenhum no sentido inverso. Isso confirma o problema já conhecido: no nosso 2×2, o comportamento não separa forma de intenção.
  - Os benignos difíceis vêm dos externos e da mineração: `hard_benign_4b.jsonl` tem 817 itens recusados, a maioria do OR-Bench-Hard, do FalseReject e da mineração.
- **Linha nociva, pareada (McNemar exato):**

  | Subconjunto | Grupos | Só `harm_trig` recusado | Só `harm_notrig` recusado | p |
  |---|---|---|---|---|
  | todos | 1.825 | 214 | 311 | 3e-5 |
  | `qc_grammatical` | 1.723 | 204 | 301 | 2e-5 |
  | `qc_pass_lenient` | 604 | 88 | 100 | 0,42 |
  | `qc_pass` estrito | 148 | 31 | 16 | 0,04 |

  **Tirar o gatilho aumenta a recusa no conjunto completo** (63,2% contra 57,9%).
  - Hipótese compatível com a revisão: muitas trocas "neutras" ficam mais formais ou explícitas ("sell drugs online" → "illicitly distribute narcotics…").
  - Só no estrito, onde o juiz confirma que o alarme caiu, o gatilho aumenta a recusa (52,7% contra 42,6%).
  - **Isso é comportamento, não ativação:** pode ser reportado como validação da manipulação e reforça usar a queda de alarme como covariável.

## 5. Mineração adversarial (A4/A-P5)

| | |
|---|---|
| **Variantes** | 4.856 reescritas "alarmantes" de cada Q⁰ (2 por par, Llama-70B) |
| **Recusadas pelo 4B** | 280 (5,8%) |
| **Confirmadas** (benignas por maioria, mesmo pedido do Q⁰ e naturais) | **68**, em `hard_benign_mined.jsonl`: violence 25, cyber 18, privacy 16, bias 9 |
| **Descartadas** | 127 viraram nocivas mantendo o pedido · 41 com intenção disputada · 39 nocivas e com outro pedido · 5 outros casos |

Como o 4B recusa pouco, a mineração rende pouco: 68 itens servem como conjunto de diagnóstico, não de treino.

## 6. SAEs próprios (A5)

Detalhes em [sae_proprio.md](sae_proprio.md) e `sae/eval_L*.json`.
- Arquitetura: BatchTopK 16k, k = 64, salvo em formato SAELens JumpReLU.
- Treino: 200M tokens do corpus UltraChat + PKU + OR-Bench, sem nenhum pedido da Fase 0.
- Pesos: `/raid/user_gustavonovack/nlp_artifacts/sae_proprio/gemma-3-4b-it_L{17,22,29}_btk16k/sae/` (~336 MB cada).

Comparação com o Gemma Scope 2 16k medium:

| Camada | Conjunto | FVU próprio / GS2 | L0 próprio / GS2 | Mortas no conjunto, próprio / GS2 | ΔNLL próprio / GS2 |
|---|---|---|---|---|---|
| 17 | nossos pedidos | **0,027** / 0,046 | 54 / 52 | 39% / 43% | +0,16 / +0,11 |
| 17 | UltraChat | **0,025** / 0,044 | 65 / 61 | 15% / 4% | +0,38 / −0,79 ⚠️ |
| 22 | nossos pedidos | **0,061** / 0,095 | 52 / 47 | 27% / 37% | **+0,21** / +0,55 |
| 22 | UltraChat | **0,070** / 0,109 | 65 / 59 | 7% / 2% | +0,36 / −0,35 ⚠️ |
| 29 | nossos pedidos | **0,112** / 0,163 | 50 / 48 | 9% / 31% | **+0,46** / +0,69 |
| 29 | UltraChat | **0,145** / 0,217 | 65 / 59 | 0,1% / 1% | +0,52 / −0,15 ⚠️ |

- **Reconstrução:** com L0 parecido, o SAE próprio reconstrói melhor nas três camadas (FVU 30–45% menor).
- **Comportamento do modelo:** nas camadas 22 e 29, o próprio também preserva melhor a perda nos nossos pedidos.
- **⚠️ ΔNLL negativo do GS2 no UltraChat:** a perda *cai* com a reconstrução. Provavelmente é convenção de entrada (normalização, posição do hook ou tokens de alta norma), não qualidade. **Pendência B4 (Caio):** conferir antes de usar o ΔNLL do GS2.
- **Fração morta:** é a fração de features sem disparo naquele conjunto pequeno (no máximo 400 sequências), não a fração morta global.

## 7. Incidentes e correções

| Incidente | Efeito | Correção |
|---|---|---|
| **Fabric manager da dgx-H100-03 falhou** (CUDA 802) | GPUs inutilizáveis; o nó reiniciou e matou os passos de SAE presos via `srun` e o rsync de dados | Jobs de GPU foram para a h100n2 e a b200n1, puxando do raid; os SAEs passaram a treinar dentro do sbatch. Infra avisada. |
| **Internet lenta** (0,4–3,7 MB/s) | O gerador gemma-4-31B ficou inviável | O Llama-3.3-70B-FP8 virou gerador e juiz (veja os desvios) |
| **huggingface_hub 2.x** com depósito compartilhado de blobs | A cópia para a b200 levou links quebrados | Cópia só do snapshot com `-L` e modelo servido por caminho; `HF_HUB_DISABLE_SHARED_BLOBS=1` |
| **Cota do raid** (a cópia com `-L` duplicou blobs: 517G) | Risco de estourar os 500G | Blobs duplicados removidos: 420G |
| **OOM do SAE** (3 processos numa GPU) | O SAE L22 caiu | Buffer pré-alocado em bf16, `--seqs 16`, `expandable_segments` |
| **Trap do job de dados** derrubava o vLLM do job de modelo | O job de comportamento perdeu o servidor | Diretório de execução separado por job |
| **Porta ocupada** por processo de outro usuário na h1002 | O vLLM não subia | `freeport()` escolhe a porta na hora |
| **Bugs de pipeline** | `select()` ignorava a tag `dry2`; `KeyError 'orig'` na confirmação da mineração | Corrigidos e commitados |

## 8. Pendências e próximos passos

- [ ] **Time:** decidir o conjunto principal do 2×2 (sugestão: `qc_grammatical` com a queda de alarme como covariável) e registrar em `DESVIOS.md` **antes** das ativações.
- [ ] **Davi (D):** ajustar `_validate_local_edit` ao plano B, como descrito na seção 3. Depois, os dois validadores devem concordar.
- [ ] **Caio (B):**
  - extrair ativações do 2×2, dos controles e dos externos;
  - B4: conferir a convenção de entrada do Gemma Scope 2 (o ΔNLL negativo).
  - O span de forma está em `cells[*].slots.form` (caracteres); controles em `span`.
- [ ] **Frederico (C):** C1–C3 sobre o subconjunto escolhido; controlar o tamanho no contraste de intenção (seção 3).
- [ ] **Gustavo:**
  - copiar os pesos dos 3 SAEs para a pasta compartilhada;
  - checagem humana opcional de 20 grupos do 2×2;
  - confirmar com a infra que a dgx-H100-03 saiu de `drained` no Slurm.
- [ ] `behavior_gemma3_4b.jsonl` tem 58 MB (o GitHub avisou): considerar o LFS se crescer.
