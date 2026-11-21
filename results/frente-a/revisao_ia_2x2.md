# Revisão às cegas por IA de uma amostra do 2×2 (plano B)

> **Revisor:** Claude Opus 5.5 · **Data:** 2026-11-21 · **Amostra:** 40 grupos sorteados (seed 7) de `datasets/phase1/cells_2x2.jsonl`, antes do controle de qualidade (`t2c_qc`). Foram lidas as versões sem gatilho (`harm_notrig`, `ben_notrig`) e as trocas gatilho → neutro.
> Esta revisão substitui a checagem humana opcional de 20 grupos (D-P6) e serviu para desenhar o `t2c_qc`.

## Resultado

| Critério | Bom | Problema | Exemplos de problema |
|---|---|---|---|
| Intenção preservada nas linhas (nocivo continua nocivo, benigno continua benigno) | ~38/40 | ~2/40 | — |
| Versão trocada gramatical | ~32/40 (80%) | ~8/40 (20%) | "dispose of discreetly **of** it"; "obtain contraband online **online**"; "cause them distress **certain individuals**"; "inflict harm on **against** a third party" |
| O termo neutro soa **menos** alarmante que o gatilho | ~24/40 (60%) | ~16/40 (40%) | equivalente: "residential address" → "home address", "offensive" → "inflammatory"; **mais** alarmante: "sell drugs online" → "illicitly distribute narcotics via the internet", "harass him anonymously" → "bully him without being identified" |

Trocas boas, isto é, minimal pairs de forma, ficaram assim:
- "steal money" → "take money";
- "secretly film" → "discreetly record";
- "malicious individual" → "bad actor";
- "insider trading" → "trading on nonpublic information";
- "clone the voice" → "replicate the voice";
- "purchase illegal drugs" → "buy controlled substances";
- "hack into" → "access", na linha benigna.

## O que isso significa para a análise

- **A intenção está bem controlada.** A manipulação de *forma* é a parte ruidosa. O ruído enfraquece o efeito lexical (d_L) e pode esconder features S_lex. Ele não fabrica um efeito de intenção, porque é simétrico entre as linhas e a intenção não muda.
- **Erros de gramática só nas versões sem gatilho** podem criar um efeito "lexical" falso: features de texto estranho ou malformado. Por isso o `qc_pass` exige gramática nas duas versões trocadas.
- **Recomendação para as frentes C e D:**
  - usar `qc_pass == true` como **conjunto principal** do C1–C3;
  - reportar o conjunto completo como robustez;
  - usar `qc.alarm_drop_*` como covariável: o efeito lexical deve crescer com a queda de alarme. Esse é um teste de validade da manipulação.

## Limitações

- O revisor é uma IA, e da mesma família de modelos que não gerou as trocas: o gerador foi o Llama-3.3-70B.
- A amostra é pequena (40 grupos), e as proporções são aproximadas.
- O nível de alarme é subjetivo.
- Uma checagem humana de 20 grupos continua recomendada, mas não bloqueia nada.
