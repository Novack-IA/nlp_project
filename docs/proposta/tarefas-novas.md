# O que muda para cada um

> Vale **só se a proposta for aprovada**. Até lá, siga `docs/tarefas/URGENTE.md`.
> As tarefas novas têm o prefixo **P** (de "proposta"): `A-P1`, `B-P1`, etc. Quando uma substitui uma tarefa antiga, isso está indicado.

## Visão rápida

| Quem | Quanto muda | Resumo |
|---|---|---|
| **Gustavo (A)** | Muito | Troca os 4 prompts de reescrita por 1 prompt de frase-molde e ganha um validador automático. |
| **Caio (B)** | Pouco | Fixa a definição de `post`/`post_last` e extrai duas posições novas (as lacunas). |
| **Frederico (C)** | Médio | Pré-registro com as 5 regras novas; teste de equivalência na classificação das features. |
| **Davi (D)** | Médio | Baseline com gatilho visto/inédito e classificador de estilo; C2 relativo ao TF-IDF; validação automática no lugar da auditoria. |

---

## Gustavo — frente A

**O que muda:** em vez de reescrever cada versão do 2×2 separadamente, o gerador escreve uma frase-molde com duas lacunas, e as 4 versões saem dela. A validação fica quase toda automática.

| # | Tarefa | Substitui | Prazo |
|---|---|---|---|
| A-P1 | Prompt único `src/featgrpo/data/phase1/prompts/cell_template_v1.txt`. Ele recebe o par original (Q⁺, Q⁰, gatilho) e devolve, em JSON: `molde`, `forma_trig`, `forma_neutra`, `intencao_nociva`, `intencao_benigna`. | A-U1 (4 prompts de célula) | D0 |
| A-P2 | **Tipologia dos gatilhos:** o juiz de 70B classifica cada um dos 1.688 gatilhos em `superficie`, `acao` ou `marcador_intencao`. Saída: `datasets/phase1/trigger_types.jsonl`. | nova | D1 |
| A-P3 | **Rodada seca com 50 grupos**, usando o molde e o validador automático. Reportar a taxa de sobrevivência por domínio e por tipo de gatilho. **Abaixo de ~40%, vamos para o plano B** (troca só do trecho do gatilho). | A-U2 | D1 |
| A-P4 | Controles `trig_only` e `neutral_only` feitos com as peças `forma_trig` e `forma_neutra` de cada grupo. | A-U4 (muda a fonte) | D1 |
| A-P5 | Se o Gemma-3-4B recusar **menos de 5%** dos `ben_trig` (A4), começar a mineração adversarial. | nova (gatilho de decisão) | até D7 |

**O validador automático** (o Davi escreve, você roda junto com a geração). Para cada grupo, ele checa:
- as 4 versões são o molde com as lacunas preenchidas, e o diff entre elas é só o trecho da lacuna;
- `*_trig` contém o gatilho (literal ou radical) e `*_notrig` não contém;
- a diferença de tamanho entre as versões é ≤ 20% em tokens.

Além dele, os 3 juízes de sempre conferem a intenção, e o `same_intent` confere a equivalência com o original. Um grupo sai se qualquer checagem falhar, e o motivo fica registrado.

**Novo formato de `cells_2x2.jsonl`:** os campos atuais, mais `molde`, `slots` (as 4 peças, com posição em caracteres) e `trigger_type`.

**Deixa de existir:** a geração de `harm_notrig` por paráfrase livre, exceto no plano B.

---

## Caio — frente B

**O que muda:** pouco. A extração ganha duas posições, e uma definição fica fixada.

| # | Tarefa | Substitui | Prazo |
|---|---|---|---|
| B-P1 | No `fatos_modelo.md` (B-U1), fixar: `post` = média dos tokens de `<end_of_turn>\n<start_of_turn>model\n`; `post_last` = só o último token. Atualizar a seção 4 do `AGENTS.md`. | complementa B-U1 | D1 |
| B-P2 | No extrator, as posições `slot_forma` (média dos tokens da lacuna de forma) e `slot_intencao` (idem para a de intenção), usando os offsets de `slots`. No 2×2, `slot_forma` substitui `trig`. | complementa B2 | D5 |

**Não muda:** o cache provisório dos pares originais (B-U3), as camadas e o formato `.npy`/`index`.

---

## Frederico — frente C

**O que muda:** o pré-registro inclui as regras novas, e a classificação das features fica mais rigorosa.

| # | Tarefa | Substitui | Prazo |
|---|---|---|---|
| C-P1 | Pré-registro (C-U1) com as **5 regras novas** do README: teste com gatilho inédito, teste de equivalência, C2 relativo ao TF-IDF, pares originais só como diagnóstico, resultados por domínio. A descrição dos dados entra como **adendo no D1**, depois da rodada seca. | complementa C-U1 | D0 (+ adendo D1) |
| C-P2 | Em `feature_stats.py`: S_intent exige que o **IC de 95% de d_L** fique entre −0,2 e 0,2 (bootstrap sobre grupos). S_mixed passa a exigir q < 0,01 nos dois efeitos. | ajusta C-U2 | D1 |
| C-P3 | Toda saída do `test` em duas partes: `gatilho_visto` e `gatilho_inedito`. O critério de replicação (C1) usa o inédito. | ajusta C2.3 | D6–D8 |
| C-P4 | Resultados de C1 também por domínio e por tipo de gatilho (`trigger_types.jsonl`). | nova | D6–D8 |
| C-P5 | Escrever em `provisorio.md` que o modo provisório é diagnóstico e não entra no veredito. | ajusta C1 | D3 |

---

## Davi — frente D

**O que muda:** o baseline mede também a confusão de estilo e o vazamento de gatilho; o C2 vira relativo; a auditoria humana vira validação automática.

| # | Tarefa | Substitui | Prazo |
|---|---|---|---|
| D-P1 | `baselines.py`: TF-IDF com o `test` em gatilho visto e inédito; `GroupKFold` por gatilho no ajuste; **classificador só de estilo** (tamanho, pontuação, tipo de pergunta) para medir a confusão PKU × gerador. | ajusta D-U1 | D1 |
| D-P2 | **Validador automático** do 2×2 (`src/featgrpo/data/phase1/validate_template.py`), com testes, para o Gustavo rodar na geração. | D-U2 (auditoria humana) | D1 |
| D-P3 | `classifiers.py` com o **C2 relativo**: queda de AUROC de S_intent ≤ metade da queda do TF-IDF, com IC por bootstrap. A parte "S_intent não detecta o gatilho" vira checagem de consistência. | ajusta D-U3 / D4 | D2 |
| D-P4 | `VEREDITO.md` com a tabela C1–C5 revisada e uma seção pronta para o caso "PARCIAL". | ajusta D-U4 | D2 |
| D-P5 | Registrar no `results/DESVIOS.md`: auditoria automática, C2 relativo, teste de equivalência e a nova regra de ética. | nova | após a aprovação |
| D-P6 | *(Opcional)* checagem humana de 20 grupos do 2×2 final, cerca de 30 minutos. | — | quando houver tempo |

---

## Todos

| # | Tarefa | Prazo |
|---|---|---|
| T-P1 | Aprovar ou rejeitar a proposta no grupo (os três pontos de "O que precisamos decidir" no README). | D0 |
| T-P2 | Trocar no `AGENTS.md` a regra "não gere conteúdo nocivo novo" por "não gerar respostas nocivas; pedidos nocivos só como recombinação de trechos do PKU". | após a aprovação |
| T-P3 | Depois da aprovação, um PR pequeno atualiza `00-visao-geral.md`, `URGENTE.md` e os arquivos de cada frente com o que foi aprovado. | D1 |
