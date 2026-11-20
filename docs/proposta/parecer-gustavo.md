# Parecer sobre a proposta da frase-molde

> **Autor:** Gustavo (frente A) · **Data:** 2026-11-20
> **Voto:** **aprovar**, com as 6 emendas abaixo. A frente A já implementou a frase-molde com estas emendas (branch `frente-a-dados`) e disparou a rodada seca.

Conferi no dataset os números de `criticas.md`, e todos batem:
- 119 de 347 pares de teste têm gatilho visto no treino;
- 2.016 dos 2.428 Q⁰ foram gerados;
- a mediana é de 23 palavras no Q⁺ contra 31 no Q⁰;
- há 1.688 gatilhos únicos.

As críticas 1–3 e 5–7 são problemas reais do desenho anterior.

## Emendas

1. **A forma neutra é um quase-sinônimo, não outra ação.** O exemplo do README ("mato/encerro o cachorro") muda o pedido. O par certo é kill → terminate/get rid of, hack into → access, steal → take. A validação compara a mesma intenção **dentro de cada linha** (com × sem gatilho), além de comparar com o original.
2. **A parte comum do molde é neutra.** Sigilo, falta de consentimento e evasão ("sem deixar rastro") vão para a lacuna de intenção. Isso está no prompt e é checado pelos 3 juízes nas versões benignas.
3. **O C2 relativo ao TF-IDF não mede nada no molde.** As palavras da intenção estão na lacuna e não mudam entre "com" e "sem gatilho", então o TF-IDF quase não cai. Proposta: comparar com o TF-IDF onde ele sofre:
   - gatilho inédito;
   - leave-one-domain-out;
   - XSTest.

   A régua passa a ser "S_intent perde no máximo metade do que o TF-IDF perde **nessas generalizações**".
4. **A meta fecha com o plano B.** São 3 moldes candidatos por grupo (o k-ésimo só é tentado se os anteriores falharem). A meta passa a **≥ 800 grupos, com ≥ 150 por domínio**. Com ~560 grupos de treino, o IC de d fica em ±0,08, e a equivalência ±0,2 continua viável.
5. **Gatilhos que são marcadores de intenção.** O juiz classifica os 1.688 gatilhos (`trigger_types.jsonl`). Nos `intent_marker`, o gatilho vai para a lacuna de intenção e a de forma recebe outra palavra alarmante do Q⁺. Os resultados saem também por tipo de gatilho.
6. **Risco novo: S_intent virar feature de tópico.** No molde, a intenção mora nas palavras da lacuna. Mitigação:
   - exigir **consistência entre grupos**: a fração de grupos com I_g > 0 é reportada junto com d_I;
   - reportar por domínio;
   - fazer uma revisão às cegas por IA de uma amostra, como na Fase 0, além da checagem humana opcional de 20 grupos.

## Ajuste no validador

A regra "≤ 20% de diferença de tamanho" sozinha reprova o próprio exemplo canônico. "How do I kill my neighbor's dog…" (10 palavras) contra "How do I get rid of a frozen process…" (13) dá 30%. Implementei: razão ≤ 1,2 **ou** diferença absoluta ≤ 3 palavras. Isso vai para `results/DESVIOS.md`.

## Fatos do modelo (adiantando o B-U1 do Caio)

Conferido no Hugging Face:
- `google/gemma-3-4b-it`: **34 camadas**, d_model = **2560**.
- `google/gemma-scope-2-4b-it/resid_post`: camadas **9, 17, 22 e 29**, larguras 16k, 65k, 262k e 1M, L0 `small`, `medium` e **`big`** (não "large").
- `resid_post_all/`: **as 34 camadas**, 16k e 262k, L0 `small` e `big`.
- Arquivos de cada SAE: `config.json`, `params.safetensors` e `examples.safetensors`.
