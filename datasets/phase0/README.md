---
license: cc-by-nc-4.0
language:
- en
pretty_name: "feat-GRPO Fase 0: pares mínimos nocivo × borderline"
tags:
- safety
- over-refusal
- interpretability
- minimal-pairs
size_categories:
- 1K<n<10K
extra_gated_prompt: "Este dataset contém pedidos nocivos e respostas nocivas (vindas do PKU-SafeRLHF). Uso restrito a pesquisa em segurança e interpretabilidade, sem redistribuição."
extra_gated_fields:
  Afiliação: text
  Uso pretendido: text
configs:
- config_name: final
  default: true
  data_files:
  - split: train
    path: v5b/train.jsonl
  - split: validation
    path: v5b/val.jsonl
  - split: test
    path: v5b/test.jsonl
- config_name: pairs_prompt_level
  data_files:
  - split: all
    path: v5b/pairs_prompt_level.jsonl
- config_name: pool
  data_files:
  - split: all
    path: v5b/pool_groups.jsonl.gz
- config_name: final_v5
  data_files:
  - split: train
    path: v5/train.jsonl
  - split: validation
    path: v5/val.jsonl
  - split: test
    path: v5/test.jsonl
---

# Pares mínimos nocivo × borderline (Fase 0, v5b)

Cada **grupo** junta um pedido nocivo **Q⁺** (PKU-SafeRLHF) e um pedido benigno **Q⁰** que usa a **mesma expressão-gatilho** (por exemplo, "kill" em "how to kill a Python process"), mais 8 respostas rotuladas. O dataset foi feito para separar *o que é nocivo* de *como o pedido soa* nas representações internas de um LLM.

O protocolo completo está no repositório, em `docs/relatorio_fase0.md`.

> Esta pasta (`datasets/phase0/`) é a cópia de trabalho do time. Ela foi gerada por `PYTHONPATH=src python3 -m featgrpo.data.phase0.export_release` a partir de `data/phase0_v5b` e `data/phase0_b200` no h1003. Os pools estão comprimidos (`.jsonl.gz`). O cabeçalho YAML permite subir a pasta inteira para o Hugging Face mais tarde, sem mudanças.

## Como carregar

```python
import pandas as pd
final_train = pd.read_json("datasets/phase0/v5b/train.jsonl", lines=True)          # 351 grupos
pairs = pd.read_json("datasets/phase0/v5b/pairs_prompt_level.jsonl", lines=True)    # 2.428 pares
pool = pd.read_json("datasets/phase0/v5b/pool_groups.jsonl.gz", lines=True)         # 2.806 grupos
# ou: datasets.load_dataset("json", data_files="datasets/phase0/v5b/pairs_prompt_level.jsonl")
```

**Splits por grupo:** nunca coloque Q⁺ e Q⁰ do mesmo `group_id` em splits diferentes. O split de teste não entra em nenhuma escolha de camada, feature ou limiar.

## Configurações

| config | linhas | o que é |
|---|---|---|
| `final` (padrão) | 500 grupos (351/76/73) | 125 por domínio (bias, cyber, violence, privacy). Pedidos **e** respostas de contraste validados. |
| `pairs_prompt_level` | 2.428 pares | Todos os grupos com Q⁺ nocivo, Q⁰ benigno e gatilho compartilhado, **sem exigir respostas válidas**. Use para análises só de pedidos. Tem `split` (o do `final` quando o grupo está lá; senão, hash do `group_id`, 70/15/15) e a classe da resposta natural do Gemma-4-12B a Q⁺ e Q⁰. |
| `pool` | 2.806 grupos | Tudo o que saiu do P2 com os rótulos de todos os juízes, válido ou não (`valid`, `invalid_reason`). |
| `final_v5` | 425 válidos | A versão anterior (1.500 sementes), mantida para comparação. |

## As 8 respostas de cada grupo (`responses`)

| id | pedido | `role` | origem |
|---|---|---|---|
| r0 | Q⁺ | `unsafe` | PKU-SafeRLHF (Alpaca) |
| r1 | Q⁺ | `safe_A` | PKU-SafeRLHF, mesmo gerador da r0 |
| r2 | Q⁺ | `safe_helpful` | Gemma-4-12B-it, system prompt `safe_helpful_v1` |
| r3 | Q⁺ | `refusal` | Gemma-4-12B-it, `refusal_v1` |
| r4 | Q⁺ | `natural` | Gemma-4-12B-it, sem system prompt (baseline) |
| r5 | Q⁰ | `safe_helpful` | Gemma-4-12B-it, `safe_helpful_v1` |
| r6 | Q⁰ | `refusal` | Gemma-4-12B-it, `refusal_v1` |
| r7 | Q⁰ | `natural` | Gemma-4-12B-it, sem system prompt (baseline) |

Cada resposta tem `labels.{wildguard, qwen3guard, judge, final}`. O juiz é o Llama-3.3-70B-Instruct-FP8, e `final` é a maioria de 2 em 3 (ou `disputed`).

Os contrastes usados na análise são:
- conteúdo = r0 − r1;
- forma = r0 − r5;
- recusa = r6 − r5.

## Qualidade

- **Calibração contra o XSTest (κ):** WildGuard 0,88, Qwen3Guard 0,84, juiz 0,79.
- **Rótulos decisivos:** 95% com os 3 juízes de acordo.
- **Teste-reteste:** 97,6% dos rótulos iguais ao rotular tudo de novo.
- **Revisão cega por IA** (Claude Opus 5.5, `v5b/audit/ai_review_opus55.json`): 98,3% de concordância, κ 0,969.
- **Auditoria humana:** ainda não foi feita. Por isso o manifest marca `provisional: true`.

## Ética

Este dataset não traz conteúdo nocivo novo: toda resposta `unsafe` vem do PKU-SafeRLHF. Uso restrito a pesquisa, sob a licença CC-BY-NC-4.0.
