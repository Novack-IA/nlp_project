Fase 0 — Dataset piloto (1k Q).

Esquema acima, estratificado de forma leve por intent × 3–4 domínios.
Respostas geradas por um modelo forte, rótulos validados pelo juiz, amostra revisada por humanos.

Fase 1 — SAE + features (Gemma-4 text-only).

Treinar o SAE na camada $\ell$.
Selecionar $S_{\text{harm}}$ e $S_{\text{ref}}$ pelo contraste pareado.
Validação causal com ablação estática.

Fase 2 — Controller (família i). Comparar o controller GRPO contra:

sem intervenção;
ablação estática ($\alpha \equiv 1$ sempre);
ablação de direção única (Arditi).