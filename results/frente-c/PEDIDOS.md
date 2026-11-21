# Pedidos para a frente C

Formato: `- [ ] AAAA-MM-DD · quem pede · o que precisa · para quê`

- [ ] 2026-11-21 · Davi · O conjunto principal `qc_grammatical` tem **26 grupos com células quebradas**, como "derogatory termss" e "online abuseing" (lista em `results/frente-d/validacao_2x2.json` → `falhas`). Pedi à frente A para marcá-los. Se você concordar, tire esses grupos no adendo do pré-registro **antes do cache do 2×2** · senão eles entram como ruído no efeito lexical L_g.
- [x] 2026-11-21 · Davi · **O C2(b) como está não pode ser atendido** (`results/frente-d/README.md`, seção 1). No 2×2 real, o TF-IDF quase não cai com gatilho inédito (0,003) nem com domínio inédito (0,014), só no XSTest (0,28). O mesmo vale em `val`. Com "≥ 2 de 3", o critério falha por construção. Proposta em discussão no PR da frente D; se aprovada, a linha do C2 no seu pré-registro muda (registrada em `DESVIOS.md`, decidida com `val` e antes de qualquer feature) · para o C2 continuar medindo alguma coisa.

