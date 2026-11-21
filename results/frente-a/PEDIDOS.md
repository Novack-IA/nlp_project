# Pedidos para a frente A

Formato: `- [ ] AAAA-MM-DD · quem pede · o que precisa · para quê`

- [x] 2026-11-20 · Davi · Ao mergear a `frente-a-dados`, atualizar a seção A1 de `docs/tarefas/gustavo-frente-a.md` para o **formato real** que o `t2_cells.py` grava (`frame` com `{FORM}`/`{INTENT}`, `form_alarm`, `form_neutral`, `intent_harmful`, `intent_benign`, `cells[*].slots.{form,intent}`, tipos `surface`/`action`/`intent_marker`) · hoje o documento descreve outros nomes (`template`, `slots`, `spans`, `superficie`…) e quem ler vai implementar errado. O validador da frente D já aceita os dois formatos.
  - **Feito (2026-11-21, Gustavo):** o formato real é o do plano B (`design: "swap"`, `frame = null`, `form_alarm*`/`form_neutral*`, `cells[*].slots.form`), documentado em `datasets/phase1/README.md`. A seção A1 aponta para lá.
- [x] 2026-11-20 · Davi · Atenção a conflitos no merge: a `frente-a-dados` saiu da `master` **antes** do PR #2. Três arquivos vão conflitar: `src/featgrpo/data/phase1/__init__.py` (os dois criaram), `results/DESVIOS.md` (os dois acrescentaram linhas no fim) e, talvez, `AGENTS.md`. Faça `git pull --rebase origin master` e **mantenha as linhas dos dois lados** · para não apagar as decisões do PR #2 (posições do cache, regra de ética, desvios).
  - **Feito:** merges com a `master` resolvidos mantendo os dois lados (`DESVIOS.md`).
- [x] 2026-11-20 · Davi · Rodar também o validador da frente D sobre a saída do `t2_cells` (segunda checagem, independente): `PYTHONPATH=src python -m featgrpo.data.phase1.validate_template datasets/phase1/cells_2x2.jsonl --report results/frente-a/validacao_2x2.json` · se ele reprovar algo que o `structural_check` aprovou, me avise. Um dos dois está errado.
  - **Feito:** aprova 196/1.825 (10,7%); relatório em `results/frente-a/validacao_2x2.json`. O diagnóstico está em `RELATORIO_noite.md`, seção 3. Os dados estão consistentes: 99,4% das edições ficam dentro de `slots.form`. Os erros vêm de regras do validador que não servem ao plano B:
    - edição tem de conter o gatilho inteiro;
    - recorte por caractere sem expandir até a borda da palavra;
    - tamanho comparado entre as linhas nociva e benigna, que no plano B vêm de pedidos diferentes.

    **Pedido de volta para a frente D:** ajustar `_validate_local_edit`.

