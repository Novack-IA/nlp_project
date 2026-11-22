# Rascunho do paper

**Arquivo principal:** [`paper.md`](paper.md), em Markdown. Converter para LaTeX só no fim, quando o texto estiver estável.

## Quem escreve o quê

| Seção | Quem escreve | Fonte principal |
|---|---|---|
| 1. Introdução | Davi | `docs/fase1-interpretabilidade.md`, relatório da Fase 0 §2 |
| 2. Trabalhos relacionados | Davi | relatório da Fase 0 §3, §5.5 e §13 |
| 3. Dados (Fase 0 + 2×2) | Gustavo | `docs/relatorio_fase0.md`, `results/frente-a/` |
| 4.1 Modelo, SAEs e extração | Caio | `results/frente-b/fatos_modelo.md`, `qualidade_saes.csv` |
| 4.2 Estatística por feature | Frederico | `results/frente-c/preregistro.md` |
| 4.3 Probes, baselines e C2 | Davi | `docs/tarefas/davi-frente-d.md` |
| 4.4 Intervenção (C5) | Caio | `src/featgrpo/interp/intervene.py` |
| 5. Resultados | todos, integrado pelo Davi | `results/VEREDITO.md` e os JSON de cada frente |
| 6. Limitações e trabalhos futuros | Davi, revisado por todos | `results/VEREDITO.md`, `results/DESVIOS.md` |
| 7. Conclusão | a definir no grupo | `results/VEREDITO.md`; achados exploratórios em `results/frente-c/descoberta.md` |
| Uso responsável dos dados (fim da seção 3) | Gustavo, revisado por todos | antiga seção de ética |

## Regras

- **Cada número no texto aponta para o arquivo de onde saiu**, num comentário `<!-- fonte: results/... -->` logo depois. Assim, se um JSON mudar, dá para achar o número.
- Não cite trechos do dataset que sejam pedidos nocivos. Use os exemplos ilustrativos do `docs/tarefas/00-visao-geral.md` ou descreva sem citar.
- Figuras ficam em `docs/paper/figuras/`, copiadas de `results/` pelo script que as gera, nunca editadas à mão.
- Referências marcadas com **(a confirmar)** ainda não foram checadas na fonte. Confira antes de submeter.

## Figuras planejadas

| # | Figura | Quem gera | Arquivo de origem |
|---|---|---|---|
| 1 | O desenho 2×2 por frase-molde (diagrama) | Davi | à mão, a partir do exemplo do `00-visao-geral.md` |
| 2 | **Scatter d_I × d_L** de todas as features, por camada, com as classes coloridas. É a figura central | Frederico | `results/frente-c/` |
| 3 | Matriz C2 (heatmap): S_intent, S_lex, aleatório, denso × intenção com/sem gatilho, gatilho | Davi | `docs/paper/figuras/fig3_c2_matriz.png`, gerada por `results/frente-d/c2_secundaria.py` |
| 4 | AUROC do probe denso por camada, com o TF-IDF tracejado | Davi | `docs/paper/figuras/fig4_probes.png`, gerada por `results/frente-d/c2_secundaria.py` |
| 5 | Efeito causal da ablação (C5) | Caio | `results/frente-b/c5_ablacao.json` |
| 6 | Queda no XSTest: S_intent × features sorteadas × denso × TF-IDF (C2b) | Davi | `docs/paper/figuras/fig6_c2b_xstest.png`, gerada por `results/frente-d/c2_secundaria.py` |
