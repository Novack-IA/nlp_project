# Rascunho do paper

**Arquivo principal:** [`paper.md`](paper.md), em Markdown. Converter para LaTeX só no fim, quando o texto estiver estável.

## Quem escreve o quê

| Seção | Quem escreve | Fonte principal |
|---|---|---|
| 1. Introdução | Davi | `docs/fase1-interpretabilidade.md`, relatório da Fase 0 §2 |
| 2. Trabalhos relacionados | Davi | relatório da Fase 0 §3, §5.5 e §13 |
| 3. Dados (Fase 0 + 2×2) | Gustavo | `docs/relatorio_fase0.md`, `results/frente-a/` |
| 4.1 O modelo, o que lemos e o que o SAE faz | Caio | `results/frente-b/fatos_modelo.md`, `qualidade_saes.csv` |
| 4.2 Como medimos cada feature | Frederico | `results/frente-c/preregistro.md` |
| 4.3 Classificadores (réguas de texto, probes, XSTest) | Davi | `docs/tarefas/davi-frente-d.md` |
| 4.4 Intervenção | Caio | `src/featgrpo/interp/intervene.py` |
| 5. Resultados, em três blocos e o veredito | todos, integrado pelo Davi | `results/VEREDITO.md` e os JSON de cada frente |
| 6. Limitações e trabalhos futuros | Davi, revisado por todos | `results/VEREDITO.md`, `results/DESVIOS.md` |
| 7. Conclusão | a definir no grupo | `results/VEREDITO.md`; achados exploratórios em `results/frente-c/descoberta.md` |
| Uso responsável (fim da seção 6) | Gustavo, revisado por todos | antiga seção de ética |
| Apêndices A (critérios e pré-registro), B (implementação) e C (secundárias e exploratórias) | todos | `results/VEREDITO.md`, `results/DESVIOS.md`, os JSON de cada frente |

## Regras

- **Cada número no texto aponta para o arquivo de onde saiu**, num comentário `<!-- fonte: results/... -->` logo depois. Assim, se um JSON mudar, dá para achar o número.
- Não cite trechos do dataset que sejam pedidos nocivos. Use os exemplos ilustrativos do `docs/tarefas/00-visao-geral.md` ou descreva sem citar.
- Figuras ficam em `docs/paper/figuras/`, copiadas de `results/` pelo script que as gera, nunca editadas à mão.
- Referências marcadas com **(a confirmar)** ainda não foram checadas na fonte. Confira antes de submeter.
- **Estrutura (revisão de 2026-11-22, as 32 decisões aprovadas pelo grupo):** os códigos C1 a C5 e o pré-registro ficam no Apêndice A; o corpo fala das perguntas em palavras. Análises secundárias e exploratórias ficam no Apêndice C, com no máximo uma frase no corpo. Sem dois-pontos no meio das frases (só em rótulos de lista e antes de listas). Números em tabelas e figuras; o texto explica.

## Figuras

| # | Figura | Quem gera | Arquivo |
|---|---|---|---|
| 1 | O caminho da análise: tokens, 34 camadas, 8 pontos de leitura, SAE e os três testes (diagrama) | Frederico | `figuras/fig1_pipeline.png`, desenhado à mão em `figuras/fonte/fig1_pipeline.html` (o comando para gerar o PNG está no arquivo) |
| 2 | Efeito de intenção × efeito da palavra na combinação principal, um painel | Frederico | `figuras/fig2_dI_dL.{png,pdf}`, gerada por `python -m featgrpo.interp.figuras --sae 16k` |
| 3 | AUROC dos probes no vetor inteiro por camada, com o TF-IDF tracejado | Davi | `figuras/fig3_probes.png`, gerada por `results/frente-d/c2_secundaria.py` |
| 4 | Queda da AUROC no XSTest, em barras (TF-IDF, features de intenção, sorteadas, probe denso) | Frederico, com os números da frente D | `figuras/fig4_xstest.{png,pdf}`, gerada por `featgrpo.interp.figuras` a partir de `results/frente-d/c2b_controle_aleatorio.json` |
| 5 | Intervenção, em barras: features de intenção × sorteadas, nos nocivos e nos benignos recusados | Frederico, com os números da frente B | `figuras/fig5_c5.{png,pdf}`, gerada por `featgrpo.interp.figuras` a partir de `results/frente-b/c5_ablacao.json` |
| 6 (apêndice) | Efeito de intenção × efeito da palavra nos 8 painéis (camada × posição) | Frederico | `figuras/fig6_dI_dL_paineis.{png,pdf}`, gerada por `featgrpo.interp.figuras` |
| 7 (apêndice) | Intervenção completa, com as exploratórias | Caio | `figuras/fig7_c5_completo.png`, cópia de `results/frente-b/fig_c5_ablacao.png` |

A matriz do C2 (antiga Figura 3) virou a Tabela 4, só com o SAE de 16k.
