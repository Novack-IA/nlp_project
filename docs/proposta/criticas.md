# Críticas ao planejamento da Fase 1

> Base: `docs/tarefas/` (versão de 2026-11-20) e contagens simples em `datasets/phase0/v5b/pairs_prompt_level.jsonl` (2.428 pares), só de gatilhos, splits e tamanhos.
> A solução proposta para quase todos os pontos está no [`README.md`](README.md) desta pasta.

Cada ponto segue o mesmo formato: **o problema em uma frase**, **um exemplo**, **por que importa** e **o que propomos**.

| # | Ponto | Gravidade | Resolvido pela frase-molde? |
|---|---|---|---|
| 1 | Muitos gatilhos são a própria intenção | Sério | Sim |
| 2 | As versões "sem gatilho" são feitas de jeitos diferentes | Sério | Sim |
| 3 | A posição `trig` não existe nas versões sem gatilho | Sério | Sim |
| 4 | Parte do C2 passa "de graça" | Sério | Não: regra nova 3 |
| 5 | O mesmo gatilho aparece no treino e no teste | Sério | Não: regra nova 1 |
| 6 | Nos pares originais, nocivo e benigno têm "cara" diferente | Médio | Sim, e regra nova 4 |
| 7 | "Efeito pequeno" não é o mesmo que "sem efeito" | Médio | Não: regra nova 2 |
| 8 | A auditoria humana tem pouca gente e pouca amostra | Médio | Sim: validação automática |
| 9 | A posição `post` está definida de dois jeitos | Médio | Não: definição do Caio |
| 10 | O resultado pode valer só para alguns temas | Médio | Não: regra nova 5 |
| 11 | A regra de ética contradiz o desenho | Menor | Sim |
| 12 | O teste causal (C5) pode ficar sem casos | Menor | Não: gatilho de decisão |
| 13 | O "SIM" é muito difícil de alcançar | Menor | Não: texto preparado |

---

## Problemas sérios

### 1. Muitos "gatilhos" não são só palavras: são a própria intenção

**Em uma frase:** o projeto supõe que o gatilho é só uma palavra que *assusta*, mas muitos gatilhos são justamente o que torna o pedido nocivo.

**Exemplo.** Entre os gatilhos mais comuns:

| Gatilho | Pares | O que é |
|---|---|---|
| residential address | 95 | assunto |
| home address | 65 | assunto |
| hack into | 50 | ação |
| without their consent | 28 | **intenção** |
| without getting caught | 18 | **intenção** |
| steal | 17 | ação |
| without their permission | 17 | **intenção** |

"Sem o consentimento dela" não é uma palavra que soa mal: é o motivo de o pedido ser nocivo.

**Por que importa.** Para criar a versão "nociva sem gatilho", o gerador vai trocar "sem o consentimento dela" por algo como "sem que ela fique sabendo", outra frase com o mesmo papel. Uma feature que acende nessa frase vai parecer "de intenção", mas pode só reconhecer sinônimos. O controle atual (o gatilho solto numa frase neutra) testa apenas a palavra original e não pega isso.

**O que propomos.** Na frase-molde, a lacuna de forma só aceita palavras que assustam; marcadores de intenção vão para a lacuna de intenção. Também reportamos os resultados por tipo de gatilho.

### 2. As duas versões "sem gatilho" são feitas de jeitos muito diferentes

**Em uma frase:** na versão benigna trocamos uma palavra; na nociva reescrevemos a frase toda.

**Exemplo.**
- Benigno: "Como **mato** um processo travado no Linux?" vira "Como **encerro** um processo travado no Linux?". Muda uma palavra.
- Nocivo: o pedido inteiro é parafraseado com eufemismos. Muda quase tudo.

**Por que importa.** Parte da diferença "com × sem gatilho" na linha nociva vem só de a frase ter sido reescrita. Isso infla o efeito lexical e pode criar uma "interação" que é artefato de geração.

**O que propomos.** Na frase-molde, entre "com" e "sem gatilho" muda só a lacuna de forma, nas duas linhas.

### 3. Em uma das posições lidas, a conta não fecha

**Em uma frase:** a posição `trig` (média dos tokens do gatilho) não existe nas versões sem gatilho.

**Por que importa.** A conta dos efeitos usa as quatro versões de cada grupo. Se uma delas não tem tokens de gatilho, o efeito nessa posição fica indefinido.

**O que propomos.** Na frase-molde, a lacuna de forma existe nas quatro versões. Lemos a lacuna, seja ela "mato" ou "encerro".

### 4. Parte do teste principal (C2) passa "de graça"

**Em uma frase:** metade do C2 é garantida pela forma como escolhemos as features, e os limiares podem ser batidos até por um classificador de palavras.

**Exemplo.** O C2 pede que as features de intenção *não* detectem o gatilho. Mas nós já as escolhemos exigindo que elas não reajam ao gatilho. É como selecionar só pessoas baixas e depois "descobrir" que o grupo é baixo.

Além disso, o C2 pede AUROC acima de 0,85. Um TF-IDF simples pode passar disso, porque o pedido nocivo tem outras palavras de dano além do gatilho. Se até ele passa, passar no C2 não prova nada sobre o SAE.

**O que propomos.** A parte "garantida" vira checagem de consistência. O critério passa a ser relativo: ao tirar o gatilho, as features de intenção perdem no máximo metade da nota que o TF-IDF perde.

### 5. O mesmo gatilho aparece no treino e no teste

**Em uma frase:** separamos treino e teste por grupo, mas o mesmo gatilho se repete em grupos diferentes.

**Exemplo.** "residential address" aparece em 95 pares, espalhados entre treino e teste. No total, **119 dos 347 pares do teste (34%)** têm um gatilho que já apareceu no treino.

**Por que importa.** Um modelo que só decorou "residential address" acerta esses pares sem entender nada. O teste fica otimista.

**O que propomos.** Reportar o teste em duas partes, gatilho visto e gatilho inédito; a conclusão principal usa o inédito. Não precisa refazer os splits.

**Atualização (baseline rodado).** Nos pares originais, o recorte **não mudou** o TF-IDF (0,996 visto × 0,994 inédito). Faz sentido: o gatilho está nos dois lados do par, então decorá-lo não ajuda a separar. O ponto continua valendo para as features e para o 2×2, onde o gatilho varia entre as células, mas pesa menos do que este texto sugeria.

---

## Problemas médios

### 6. Nos pares originais, nocivo e benigno têm "cara" diferente

**Em uma frase:** nos pares originais dá para separar nocivo de benigno só pelo estilo do texto.

**Exemplo.** Os Q⁺ vêm todos do PKU; dos Q⁰, 83% foram escritos por um gerador (2.016 de 2.428). Os Q⁰ também são mais longos: 31 palavras contra 23, na mediana.

**O que propomos.** Os resultados nos pares originais viram diagnóstico, não evidência. O baseline ganha um classificador só de estilo (tamanho, pontuação), para medir o tamanho do problema. A frase-molde elimina isso no 2×2.

### 7. "Efeito pequeno" não é o mesmo que "sem efeito"

**Em uma frase:** hoje basta o efeito lexical medido ser menor que 0,2 para dizer que a feature "não reage ao gatilho", mas um 0,15 com muita incerteza pode ser, de verdade, 0,4.

**O que propomos.** Exigir que a margem de erro inteira (IC de 95%) fique entre −0,2 e 0,2. Exigir significância também na classe "mista". Manter o filtro de features raras (ativas em ≥ 5% dos itens).

### 8. A auditoria humana tem pouca gente e pouca amostra

**Em uma frase:** uma pessoa só olhando 60 grupos não basta para aprovar ou reprovar um prompt.

**Por que importa.** Com 60 grupos, uma concordância medida de 85% pode ser qualquer coisa entre 76% e 92%.

**O que propomos.** Com a frase-molde, quase toda a auditoria vira checagem automática (ver o README). Uma checagem humana pequena fica opcional.

### 9. A posição `post` está definida de dois jeitos

**Em uma frase:** o `AGENTS.md` e o `URGENTE.md` descrevem `post` de formas diferentes.

**O que propomos.** O Caio fixa no `fatos_modelo.md`: `post` = média dos tokens que fecham a fala do usuário e abrem a do modelo; `post_last` = só o último desses tokens.

### 10. O resultado pode valer só para alguns temas

**Em uma frase:** "palavra que assusta" faz sentido em violência e cibersegurança, mas bem menos em viés e privacidade.

**O que propomos.** Reportar também por domínio. Os quatro têm tamanhos parecidos: privacy 633, violence 611, cyber 606, bias 578.

---

## Pontos menores

**11. Ética.** O `AGENTS.md` diz "não gere conteúdo nocivo novo", mas as versões nocivas do 2×2 são pedidos nocivos gerados. *Proposta:* a regra passa a ser "não gerar respostas nocivas; pedidos nocivos só como recombinação de trechos do PKU".

**12. C5 sem casos.** Na Fase 0, modelos de 7–8B recusaram ~1% dos benignos. Se o Gemma-3-4B também recusar pouco, não há recusas para "desligar". *Proposta:* se o 4B recusar menos de 5% dos benignos com gatilho, começa a busca por pedidos mais difíceis, com dono e prazo. O OR-Bench-Hard vira a fonte principal de recusas.

**13. "SIM" difícil.** O resultado mais provável é "PARCIAL". *Proposta:* manter a regra e preparar o texto do paper para esse caso desde já.
