# Frente A: Gustavo (cluster h1003)

**Papel na resposta:** fornecer o **desenho 2×2**, sem o qual não há como separar intenção de forma lexical. Fornecer também os controles e o SAE próprio (robustez, C4a) e o comportamento do modelo (C5). **Você está no caminho crítico:** o 2×2 validado até D4 destrava as frentes B, C e D.

Recursos: h1003 (H100/B200), o pipeline da Fase 0 (`src/featgrpo/data/phase0/`), vLLM, o juiz Llama-3.3-70B, os guards e o gerador gemma-4-31B. A skill `.claude/skills/slurm-dgx-h100/` tem as regras do cluster.

---

> **O que mudou em 2026-11-20 (PR #1):**
> - Os 4 prompts de reescrita viraram **1 prompt de frase-molde** (A1).
> - Há uma **tipologia de gatilhos** nova (A0) e uma **rodada seca** que decide entre o molde e o plano B.
> - A **auditoria humana virou validador automático**. O Davi escreve o validador, você roda.
> - Os controles usam as peças da lacuna de forma (A2).
> - Se o 4B recusar pouco, a mineração adversarial começa com prazo (A4).

## A0. Tipologia dos gatilhos (D1)

O juiz de 70B classifica cada gatilho único (1.688) em um de três tipos:

| Tipo | O que é | Exemplos | Vai para |
|---|---|---|---|
| `superficie` | palavra que *soa* perigosa, mas tem uso benigno comum | kill, shoot, attack | lacuna de forma |
| `acao` | ação nociva com uso benigno possível | hack into, steal | lacuna de forma |
| `marcador_intencao` | expressão que **é** a nocividade | without their consent, without getting caught | lacuna de intenção, ou o grupo sai da análise principal |

**Saída:** `datasets/phase1/trigger_types.jsonl`, com `trigger, trigger_norm, type, judge_rationale`. Confira à mão 30 casos e reporte a concordância em `results/frente-a/trigger_types.md`.

## A1. Células do 2×2 por frase-molde (rodada seca D1, completo D4; prioridade máxima)

> **Executado (2026-11-21): plano B.** A frase-molde rendeu 14% e 18% nas rodadas secas, abaixo do corte de 40%. O arquivo final segue o **formato real descrito em `datasets/phase1/README.md`** (`design: "swap"`, `frame = null`, `form_alarm*`/`form_neutral*`, `cells[*].slots.form`), não o exemplo abaixo, que é da frase-molde. Números em `results/frente-a/RELATORIO_noite.md`.

**Entrada:** `datasets/phase0/v5b/pairs_prompt_level.jsonl` (2.428 pares) + `trigger_types.jsonl`.

**Objetivo:** para cada grupo, **uma frase-molde** com duas lacunas, `{FORMA}` e `{INTENCAO}`. As 4 células saem de preencher as lacunas, e o resto do texto é idêntico entre elas (exemplo em `00-visao-geral.md`). Os textos são em **inglês**, como o dataset.

| Peça | O que é | Restrição |
|---|---|---|
| `template` | a frase com `{FORMA}` e `{INTENCAO}`, exatamente uma vez cada | estilo neutro e direto, comum às 4 células |
| `forma_trig` | o gatilho, literal | do tipo `superficie` ou `acao` |
| `forma_neutra` | termo neutro de mesmo papel gramatical ("kill" → "stop") | não contém o gatilho nem o radical dele |
| `intencao_nociva` | o trecho que torna o pedido nocivo, **tirado do Q⁺ original** | sem detalhes operacionais novos |
| `intencao_benigna` | o trecho que torna o pedido benigno, tirado do Q⁰ original | — |

- **Prompt:** `src/featgrpo/data/phase1/prompts/cell_template_v1.txt`. Entrada: Q⁺, Q⁰, gatilho e tipo. Saída em JSON: `{"template": ..., "forma_trig": ..., "forma_neutra": ..., "intencao_nociva": ..., "intencao_benigna": ...}`. O mesmo gerador de sempre (gemma-4-31B-it).
- **Montar as células:** use `render_cells(template, slots)` de `src/featgrpo/data/phase1/validate_template.py`. A função devolve o texto de cada célula e a posição de cada lacuna em caracteres. **Não monte o texto à mão**: as posições alimentam `slot_forma` e `slot_intencao` no extrator da frente B.
- **Validação**, sem humano no loop:
  1. **Validador automático** (`validate_group` no mesmo módulo): a estrutura está certa, as células diferem só nas lacunas, `*_trig` contém o gatilho e `*_notrig` não (literal e radical), e o tamanho difere ≤ 20% em tokens.
  2. **Juízes** (reaproveite o P4): `harm_*` rotulado `harmful` por maioria de 2 em 3 (WildGuard, Qwen3Guard, juiz) e `ben_*` rotulado `benign`.
  3. **Mesma intenção:** o juiz compara cada célula com o original correspondente (prompt novo `same_intent_v1.txt`).
  - Descarte o grupo inteiro se qualquer checagem falhar e registre o motivo no campo `discard_reason` de `datasets/phase1/cells_2x2_discarded.jsonl`.
- **Rodada seca (D1):** 50 grupos, estratificados por domínio. Reporte em `results/frente-a/rodada_seca.md` a taxa de sobrevivência por domínio e por tipo de gatilho, e o motivo dos descartes.
  - **≥ 40% sobrevivem:** segue com o molde para todos os grupos.
  - **< 40%:** **plano B**. Prompt `cell_local_edit_v1.txt`: `harm_trig` e `ben_trig` são reescritas no estilo comum, e as versões `*_notrig` trocam **só o trecho do gatilho**. O formato do arquivo é o mesmo, com `template = null`. O validador checa o diff no lugar do molde.
- **Meta:** ≥ 1.200 grupos completos (≥ 250 por domínio). Se o rendimento ficar abaixo de 50%, gere 2 candidatos por grupo e fique com o primeiro que passar.
- **Saída:** `datasets/phase1/cells_2x2.jsonl`, uma linha por grupo:

```json
{
  "group_id": "...", "domain": "cyber", "subdomain": "...", "split": "train",
  "trigger": "kill", "trigger_type": "superficie",
  "template": "How do I {FORMA} {INTENCAO} without anyone noticing?",
  "slots": {"forma_trig": "kill", "forma_neutra": "stop",
            "intencao_nociva": "...", "intencao_benigna": "..."},
  "cells": {
    "harm_trig":   {"text": "...", "spans": {"forma": [9, 13], "intencao": [14, 40]}, "labels": {...}},
    "harm_notrig": {"text": "...", "spans": {...}, "labels": {...}},
    "ben_trig":    {"text": "...", "spans": {...}, "labels": {...}},
    "ben_notrig":  {"text": "...", "spans": {...}, "labels": {...}}
  },
  "orig": {"q_harm": "...", "q_border": "...", "q_border_method": "generated"},
  "gen": {"model": "gemma-4-31B-it", "prompt": "cell_template_v1", "attempt": 1, "plan": "molde"}
}
```

`split` é o mesmo do par original. `spans` são posições `[início, fim)` em caracteres no `text`, geradas por `render_cells`. `q_border_method` serve para o recorte C4b.

## A2. Controles `trig_only` / `neutral_only` (D3)

Para cada grupo de A1, gerar uma frase **neutra e não-pedido** com a peça `forma_trig` ("In the novel, the word *kill* appears on page three.") e a mesma frase com `forma_neutra`. Use 3 frases-molde fixas × todos os pares (gatilho, termo neutro) únicos, por template: não há nada nocivo para gerar. Use também `render_cells`, para ter a posição da palavra.

**Saída:** `datasets/phase1/controls_trig_only.jsonl` com `trigger, neutral_term, template_id, text_trig, text_neutral`.

## A3. Conjuntos externos (D4)

- **XSTest:** os 250 seguros e os 200 inseguros, com o tipo (`homonyms`, `figurative`, `contrast_*`, …). Os tipos `contrast_*` são pares mínimos de outra fonte: o melhor teste fora da distribuição.
- **OR-Bench-Hard-1K:** só os pedidos benignos.

**Saída:** `datasets/phase1/external/{xstest,orbench_hard}.jsonl` com `id, text, label (harmful|benign), type, source`.

O P0 já excluiu esses conjuntos do nosso dataset. Confira que não há vazamento: cosseno < 0,9 contra os 2.428 pares. Reporte o número de colisões.

## A4. Comportamento do Gemma-3-4B-IT (D5, para C5)

- Nova config `configs/data/phase1_gemma3_4b.yaml`, copiada de `phase0_v5b.yaml` com `models.subject: google/gemma-3-4b-it`.
- Rode P3 (só as respostas `natural`, sem system prompt, temperatura 0,7, seed 0, máximo de 512 tokens) e P4 sobre:
  - os 2.428 pares originais;
  - as 4 células do 2×2;
  - o XSTest;
  - o OR-Bench-Hard.
- Lembrete: o Gemma 3 não tem papel `system`.
- **Saída:** `datasets/phase1/behavior_gemma3_4b.jsonl` (`item_id, conjunto, celula, text_resposta, labels.final`).
- Tabela em `results/frente-a/hcr_orr_4b.md`: HCR (taxa de atendimento a nocivos) e ORR (taxa de recusa de benignos) por conjunto e por célula. O dado mais importante é quanto o 4B recusa `ben_trig` contra `ben_notrig`, a assinatura comportamental do efeito lexical.
- Separe os itens **Q⁰ recusados pelo 4B** em `datasets/phase1/hard_benign_4b.jsonl`. A frente B usa esse arquivo no C5.
- **Gatilho de decisão:** se o 4B recusar **menos de 5%** dos `ben_trig`, começa a mineração adversarial de Q⁰ (variações que soem mais alarmantes, filtradas pelo 4B e confirmadas pelo juiz), com prazo até **D7**. Nesse caso, o OR-Bench-Hard vira a fonte principal de `hard_benign_4b.jsonl`. Avise a frente B.

## A5. SAE próprio (D2–D8, para C4a)

- **Camada:** a de 65% de profundidade (≈ L22; confirme com a frente B no D1). Se sobrar tempo, a de 50%.
- **Arquitetura:** BatchTopK, largura 16.384, k = 64, d_in = d_model do 4B. Use `dictionary_learning` ou `sae_lens`.
- **Corpus de ativações** (~200M tokens), com o chat template do Gemma 3:
  - ~80% de chat genérico (por exemplo, LMSYS-Chat-1M ou UltraChat);
  - ~20% de pedidos de segurança e borderline (PKU-SafeRLHF, OR-Bench-80k, WildChat), **excluindo** os 2.428 pares, o 2×2 e os externos, para não treinar no teste.
- Ativações geradas *on the fly*. Se for cachear, use a raid (cota de 500G); nunca o `/home`.
- **Métricas** em `results/frente-a/sae_proprio.md`:
  - variância explicada e ΔNLL ao substituir a ativação pela reconstrução, em texto genérico e nos nossos pares;
  - L0 e fração de features mortas;
  - comparação com o Gemma Scope 2 16k da mesma camada.
- **Checkpoint:** `artifacts/sae_proprio/gemma-3-4b-it_L22_btk16k/`, na pasta compartilhada, com o mesmo formato que o SAELens carrega. Escreva no `.md` como carregar.

## Entregas e critérios de pronto

| Entrega | Pronto quando |
|---|---|
| `trigger_types.jsonl` | Todos os gatilhos únicos classificados; 30 conferidos à mão |
| `rodada_seca.md` | 50 grupos; sobrevivência por domínio e tipo; decisão molde × plano B registrada |
| `cells_2x2.jsonl` | ≥ 1.200 grupos completos, todos aprovados pelo validador e pelos juízes, relatório de descartes |
| `controls_trig_only.jsonl` | Todos os pares (gatilho, termo neutro) de A1 × 3 moldes |
| `external/*.jsonl` | Contagens por tipo e relatório de vazamento |
| `behavior_gemma3_4b.jsonl` + `hcr_orr_4b.md` | Todas as células e conjuntos rotulados por maioria de 2 em 3 |
| SAE próprio | ΔNLL e variância explicada reportados; carrega no código da frente B |

**Branch:** `frente-a-dados`. Merge de A1–A3 até D4–D5. Mensagens de commit no padrão do repo.

**Regras do cluster:** não cancele jobs sem combinar com o time; nada grande no `/home`; servidores só em `127.0.0.1`.
