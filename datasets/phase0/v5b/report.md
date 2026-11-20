# Relatório da Fase 0 — `phase0-v5b-h100`

Gerado em 2026-11-08 21:07. **PROVISÓRIO: sem auditoria humana (P7).**

## 1. Funil por etapa e domínio

| domínio | elegíveis P1 | amostrados P1 | grupos P2 | válidos P5 | final P8 |
|---|---|---|---|---|---|
| bias | 663 | 661 | 635 | 186 | 125 |
| cyber | 1520 | 750 | 733 | 237 | 125 |
| violence | 1058 | 750 | 718 | 204 | 125 |
| privacy | 806 | 750 | 720 | 210 | 125 |

Motivos de invalidade (P5): `{"q_harm:intent": 146, "safe_A": 1726, "refusal": 421, "q_border:intent": 233, "unsafe": 131, "q_border:shares_trigger": 4, "safe_helpful": 66}`

## 2. Retrieval × geração (P2)

```json
{
 "bias": {
  "retrieved": 121,
  "generated": 514
 },
 "cyber": {
  "retrieved": 240,
  "generated": 493
 },
 "violence": {
  "retrieved": 130,
  "generated": 588
 },
 "privacy": {
  "retrieved": 123,
  "generated": 597
 }
}
```

## 3. Calibração (κ contra XSTest)

- **wildguard**: κ = 0.878 (binary_refusal, n=2250)
- **qwen3guard**: κ = 0.836 (binary_refusal, n=2250)
- **judge** (RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic): κ = 0.793 (3class, n=2250)

## 4. Acordo entre juízes

- itens `disputed` no P4: 819
- grupos com erro do juiz: {}
- consolidação: `majority`; validade: `contrast`

## 5. Auditoria

```json
pendente
```

## 6. HCR e ORR de base do sujeito (r_natural)

| domínio | HCR (Q⁺) | ORR (Q⁰) | n |
|---|---|---|---|
| bias | 0.011 | 0.000 | 186 |
| cyber | 0.000 | 0.004 | 237 |
| violence | 0.010 | 0.005 | 204 |
| privacy | 0.005 | 0.024 | 210 |
| all | 0.006 | 0.008 | 837 |

## 7. Distribuição de hardness

```json
{
 "bias": {
  "0.0": 185,
  "0.25": 1
 },
 "cyber": {
  "0.0": 236,
  "0.25": 1
 },
 "violence": {
  "0.0": 201,
  "0.25": 3
 },
 "privacy": {
  "0.0": 196,
  "0.25": 12,
  "0.5": 2
 }
}
```

## 8. Sub-buckets de bias

Meta: {'lgbt': 30, 'race': 30}; obtido (final): {'lgbt': 7, 'race': 25, 'other': 93}

## 9. Chamadas e tokens

```json
{
 "p3": {
  "vllm:google/gemma-4-12B-it:calls": 8193,
  "vllm:google/gemma-4-12B-it:prompt_tokens": 632436,
  "vllm:google/gemma-4-12B-it:completion_tokens": 1950145
 },
 "p4": {
  "judge": {
   "calls": 1366,
   "prompt_tokens": 2941276,
   "completion_tokens": 345378
  },
  "qwen3guard": {
   "calls": 13623,
   "prompt_tokens": 7608178,
   "completion_tokens": 174910
  },
  "wildguard": {
   "calls": 13625,
   "prompt_tokens": 4447945,
   "completion_tokens": 283440
  }
 },
 "calib": {
  "judge": {
   "calls": 225,
   "prompt_tokens": 461584,
   "completion_tokens": 46829
  },
  "qwen3guard": {
   "calls": 2240,
   "prompt_tokens": 1215657,
   "completion_tokens": 29489
  },
  "wildguard": {
   "calls": 2240,
   "prompt_tokens": 636737,
   "completion_tokens": 44800
  }
 }
}
```
