# Fase 0: notas de execução das versões v5 e v5b (cluster H100/B200, 2026-11-08)

Este documento registra **o que mudou** em relação à especificação `phase0-v4` (`fase-0.md`, executada em outubro no cluster DGX Spark), **por quê**, e os **problemas conhecidos** da implementação atual.

## 1. Contexto

O código da v4 vivia só no cluster Spark (alias `laudite`), que ficou inacessível. Ele também nunca chegou ao git, porque o `.gitignore` tinha `data/` e isso ignorava `src/featgrpo/data/` (corrigido para `/data/`). Por isso o pipeline foi **reimplementado** a partir de três fontes:
- a especificação (`fase-0.md`);
- os testes (`tests/test_phase0.py`, a especificação executável das funções puras);
- os logs da v4 (`slurm-logs/`).

## 2. Mudanças de protocolo

| Item | v4 (Spark) | v5 / v5b | Motivo |
|---|---|---|---|
| Juiz geral | Claude Sonnet 5.5 via Claude Code CLI (assinatura) | **Llama-3.3-70B-Instruct FP8** local (`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`) | contingência §12 do plano; sem cota, sem dependência externa, zero bloqueios por salvaguarda (a v4 teve ~136 grupos de *cyber* bloqueados) |
| Calibração do juiz | κ = 0,82 (judge_v2) | κ = **0,793** | passou do limiar de 0,70 sem iterar o prompt |
| Saída do juiz | JSON livre + parse | **JSON schema** (structured outputs do vLLM) | elimina erros de parse |
| Prompt de sub-bucket | lista JSON | objeto `{"items": [...]}` | o schema exige um objeto na raiz |
| Consolidação | maioria 2/3 (decisão de 08/10) | igual | — |
| Validade | só os itens dos contrastes (decisão de 08/10) | igual | — |
| Sementes | 1.500 (375/domínio) | v5: 1.500; **v5b: 2.911** (750/domínio; bias 661) | v5 deu 425 válidos (< 450) → contingência §12 "dobrar candidatos" |
| Oversample de sub-bucket | 3× | v5b: 6× | priorizar `race`/`lgbt` na expansão |
| Regeneração do P5 | sujeito no ar durante o P5 | **pré-geração** (seeds 1 e 2 de todos os itens regeneráveis) na fase de geração; o P5 lê do cache | com 1–2 GPUs, o sujeito e o juiz não cabem juntos; o resultado é idêntico (mesmas chaves de cache) |
| P6 generate | só grupos válidos | **todos** os Q⁰ (os rótulos e a dificuldade continuam só nos válidos) | rodar em paralelo com o P4 |
| Revisão (P7) | humana, 2 anotadores | **revisão às cegas por IA** (Claude Opus 5.5), provisória | prazo; ver `docs/relatorio_fase0.md` §9.4 |

## 3. Infraestrutura

- **Cluster:** DGX com Slurm. Partições `h100n2` (8×H100), `h100n3` (3×H100 no Slurm), `b200n1` (8×B200) e `h100n1`. QoS `onejob`: **2 jobs rodando por usuário**, sem limite de GPUs por job.
- **Storage:** o `/raid` é **local de cada nó**; só o dgx-H100-03 tem `/raid/user_gustavonovack`. O `/home` (NFS) é compartilhado, mas pequeno (~250 GB livres para todos).
  - Solução atual: código e dados no `~/nlp_project`; imagem apptainer e venv do vLLM em `~/nlp_stage` (~19,5 GB);
  - pesos (~250 GB) baixados no `/tmp/$USER/nlp` de cada nó, onde **persistem** entre jobs e precisam de limpeza manual.
- **Servidores:** vLLM 0.30 (venv do projeto feat-grpo, extraído no `/tmp`) dentro de `pytorch-2.8.0-cu128-devel.sif`, sempre em `127.0.0.1`. Abrir portas em `0.0.0.0` é bloqueado pela política de segurança, então **jobs não conversam entre si pela rede**, só por arquivos no `/home`.
- **Venv do driver:** a imagem define `PIP_CONSTRAINT=/etc/pip/constraint.txt` (`fsspec==2025.2.0`), o que quebra o `datasets` (`Invalid pattern: '**'`). O sbatch instala o venv do driver com `env -u PIP_CONSTRAINT` e recria o venv quando o `requirements-phase0.txt` muda.
- **Layouts de GPU** (`slurm/run_phase0.sbatch`):
  - **2×H100:** juiz TP=2 (0,62), Qwen3Guard na GPU0 (0,28), WildGuard e embedder na GPU1; 31B (0,92) → referências (3×0,30) na GPU0; 12B (0,85) na GPU1.
  - **1×B200:** tudo na GPU0; juiz 0,50, guards 0,14 e 0,13, 31B 0,45, 12B 0,30, referências 0,14.

## 4. Rodadas executadas

| Job | Nó | Versão / diretório | Resultado |
|---|---|---|---|
| 36611 | B200 | v5 / `data/phase0_b200` | 1.440 grupos, **425 válidos** |
| 36612 | 2×H100 | v5 / `data/phase0_h100n2` | 1.440 grupos, 414 válidos: **rotulagem independente** (teste-reteste: 97,6% de concordância) |
| 36615 | B200 | **v5b** / `data/phase0_v5b` | 2.806 grupos, **837 válidos**, export de 500 |

A comparação entre v5 e v5b está em `data/phase0_v5b/compare_phase0_b200_vs_phase0_v5b.md`. A expansão não degrada nenhuma métrica de qualidade.

## 5. Problemas conhecidos

1. **Falso positivo do gatilho por radical:** `shares_trigger` compara conjuntos de radicais de Porter, e *personal* e *person* viram ambos `person`. Correção sugerida: exigir a sequência contígua de radicais e a concordância do juiz.
2. **O P8 não prioriza sub-buckets**, então `lgbt` caiu para 7 no export da v5b. Também não prioriza grupos de versões anteriores. Correção: ordem de prioridade (auditados → sub-bucket → versão anterior → sorteio).
3. **A rotulagem varia por hardware:** FP8 em B200 e em H100 diverge em ~2,4% dos rótulos. Fixe o nó da rotulagem oficial.
4. **Os relatórios de chamadas por etapa contam só as chamadas reais** (sem hits de cache). Em rodadas que reusam cache, os números de tokens ficam subestimados.
5. **Arquivos `__pycache__` estavam versionados** (`src/featgrpo/__pycache__`, `tests/__pycache__`). Devem ser removidos do índice: `git rm -r --cached '**/__pycache__'`.
6. Os scripts `slurm/*.sbatch` da v4 (Spark) continuam no repo como referência, mas **não funcionam** neste cluster: usam enroot, imagem NGC e o caminho `/beegfs`.

## 6. Arquivos que precisam de limpeza manual (só quando decidido)

- `~/nlp_stage/` (19,5 GB, `/home`): imagem e venv do vLLM usados no staging.
- `/tmp/$USER/nlp/` em dgx-B200-1 e dgx-H100-02: ~250 GB de pesos em cada um.
- Destino de longo prazo dos dados: `/raid/user_gustavonovack/nlp_project/data/` no dgx-H100-03.
