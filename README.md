# feat-GRPO: dataset de pares mínimos e controle de segurança guiado por features de SAE

Projeto acadêmico (UFG / CEIA) sobre segurança em LLMs. Tem dois objetivos:

1. **Paper 1, dataset de pares mínimos.** Cada grupo junta um pedido nocivo (Q⁺) e um pedido benigno com o mesmo gatilho lexical (Q⁰), mais respostas rotuladas. A estrutura permite isolar, nas ativações do modelo, as features de *conteúdo nocivo*, de *forma lexical* e de *recusa* por meio de contrastes pareados.
2. **Paper 2, controller.** Uma MLP pequena, treinada por GRPO, decide quando, onde e quanto atenuar essas features de um Sparse Autoencoder num modelo **congelado** (Gemma-4-12B-it). O objetivo é reduzir ao mesmo tempo o *harmful compliance* e o *over-refusal*.

| Documento | Conteúdo |
|---|---|
| [`docs/relatorio_fase0.md`](docs/relatorio_fase0.md) | **Comece por aqui.** Relatório autocontido: contexto, fundamentos, protocolo, resultados e limitações. |
| [`README-featcontroller.md`](README-featcontroller.md) | Plano de pesquisa completo (teoria, hipóteses H1–H4, modelos, métricas, roadmap, histórico de decisões). |
| [`fase-0.md`](fase-0.md) | Especificação prescritiva da Fase 0 (`phase0-v4`): cada etapa, limiar e critério de aceitação. |
| [`docs/cluster-estrategias.md`](docs/cluster-estrategias.md) | Como usar o cluster: acesso ao /raid do h1003 de outros nós, pré-aquecimento, paralelismo, pedidos ao admin. |
| [`docs/phase0-v5-execucao.md`](docs/phase0-v5-execucao.md) | O que mudou da v4 (Spark) para a v5/v5b (cluster H100/B200), problemas conhecidos e decisões. |
| [`plano.md`](plano.md) | Roteiro resumido das fases 0 a 2. |

## Estado atual (2026-11-08)

**Fase 0 concluída em versão provisória**, `phase0-v5b-h100`, sem auditoria humana:

- 2.911 sementes viraram 2.806 grupos, com **837 válidos**; o export tem **500 grupos** (125 por domínio: bias, cyber, violence, privacy) e splits 351/76/73;
- juízes calibrados contra humanos (XSTest): κ = 0,79 (Llama-3.3-70B), 0,84 (Qwen3Guard) e 0,88 (WildGuard);
- revisão às cegas por IA (Claude Opus 5.5, 50 grupos): concordância de 98,3%, κ = 0,97 e `pair_valid` de 94%;
- **pendências:** auditoria humana (P7) e Fase 1 (treino do SAE e seleção de features).

## Estrutura do repositório

```
.
├── README.md                     # este arquivo
├── README-featcontroller.md      # plano de pesquisa
├── fase-0.md                     # especificação da Fase 0 (v4)
├── plano.md
├── docs/
│   ├── relatorio_fase0.md        # relatório completo da Fase 0 (v5b)
│   └── phase0-v5-execucao.md     # notas de execução da v5/v5b no cluster H100
├── configs/data/
│   ├── phase0.yaml               # phase0-v5-h100  (1.500 sementes)
│   └── phase0_v5b.yaml           # phase0-v5b-h100 (2.911 sementes; versão atual)
├── src/featgrpo/data/phase0/     # pipeline P0 → P8
│   ├── common.py                 # config, normalização, cache em disco, cliente vLLM, JSONL
│   ├── download_models.py        # baixa os pesos para o HF_HUB_CACHE (marcadores .ok_*)
│   ├── p0_download.py            # fontes + conjunto de exclusão (hash + embedding)
│   ├── p1_seeds.py               # sementes Q⁺ do PKU-SafeRLHF (--extend-from para expandir)
│   ├── p2_border.py              # Q⁰: gatilho → retrieval OR-Bench-80k → geração
│   ├── p3_responses.py           # respostas do sujeito (natural / recusa / ajuda segura)
│   ├── p4_label.py               # calibração, guards, juiz geral, consolidação
│   ├── guards.py                 # formatos e parsers do WildGuard e do Qwen3Guard
│   ├── judge.py                  # juiz geral local (vLLM, saída JSON por schema)
│   ├── claude_cli.py             # juiz da v4 (Claude Code CLI); não usado na v5
│   ├── p5_validate.py            # validade do grupo + regeneração (--pregen)
│   ├── p6_hardness.py            # dificuldade de Q⁰ com 4 modelos de referência
│   ├── p7_audit.py               # export dos CSVs de auditoria, métricas, aplicação
│   ├── p8_finalize.py            # seleção 125/domínio, splits, export, manifest, report.md
│   ├── compare_versions.py       # compara duas rodadas (ex.: v5 × v5b)
│   ├── metrics.py                # Cohen κ, Krippendorff α
│   └── prompts/                  # templates (trigger, border_gen, judge_v1/v2, calib, subbucket, ...)
├── tests/test_phase0.py          # 28 testes das funções puras (sem GPU)
├── slurm/
│   ├── run_phase0.sbatch         # ★ pipeline completo P0→P8 num job (2×H100 ou 1×B200)
│   └── *.sbatch (demais)         # jobs da v4 no cluster DGX Spark (legado, referência)
├── slurm-logs/                   # logs dos jobs (v4 Spark e v5 H100)
└── data/                         # fora do git (ver abaixo)
```

## Como rodar

O cluster e suas particularidades estão descritos em `.claude/skills/slurm-dgx-h100/SKILL.md`. Em resumo:
- tudo roda via Slurm a partir do `~/nlp_project` (no `/home`, compartilhado entre os nós);
- os pesos e o venv do vLLM ficam em staging no `/tmp` do nó, porque o `/raid` é local de cada nó.

```bash
# pré-requisitos (uma vez): ~/nlp_stage com a imagem apptainer e o venv do vLLM; .env com HF_TOKEN
# rodada completa na B200 (1 GPU de 180 GB)
PHASE0_DATA=data/phase0_X PHASE0_CACHE=data/phase0_cache PHASE0_RAW=data/phase0_raw \
  sbatch -p b200n1 --gres=gpu:1 --export=ALL slurm/run_phase0.sbatch

# mesma coisa em 2×H100
... sbatch -p h100n2 --gres=gpu:2 --export=ALL slurm/run_phase0.sbatch

# versão expandida (v5b): mantém as sementes de uma rodada e completa até 750 por domínio
PHASE0_CONFIG=configs/data/phase0_v5b.yaml PHASE0_DATA=data/phase0_v5b \
P1_ARGS="--extend-from $HOME/nlp_project/data/phase0_b200/seeds.jsonl" ... sbatch ... slurm/run_phase0.sbatch

# rodada seca
PHASE0_DATA=data/phase0_dry P1_ARGS='--per-domain 20' ... sbatch ...

# comparar duas versões
PYTHONPATH=src python3 -m featgrpo.data.phase0.compare_versions data/phase0_b200 data/phase0_v5b
```

Variáveis de ambiente do pipeline:

| Variável | Padrão | Para quê |
|---|---|---|
| `PHASE0_DATA` | `data/phase0` | diretório da rodada (saídas de P0–P8) |
| `PHASE0_CONFIG` | `configs/data/phase0.yaml` | config da versão |
| `PHASE0_CACHE` | `$PHASE0_DATA/cache` | cache de chamadas (compartilhável entre rodadas) |
| `PHASE0_RAW` | `$PHASE0_DATA/raw` | fontes brutas do P0 (compartilháveis) |
| `P1_ARGS` | vazio | `--per-domain N`, `--extend-from seeds.jsonl` |

O job é **retomável**: cada etapa pula se a sua saída já existe, e o cache evita repetir chamadas. Para refazer uma etapa, apague a saída dela e as das etapas seguintes.

Testes (sem GPU): `PYTHONPATH=src python -m pytest -q tests/`. O sbatch roda os testes antes de subir qualquer modelo.

## Dados (`data/`, fora do git)

| Diretório | Conteúdo |
|---|---|
| `data/phase0_v5b/` | **versão atual**: `final/{train,val,test}.jsonl`, `manifest.json`, `report.md`, `compare_*.md`, `audit/` (CSVs para anotadores e `ai_review_opus55.json`), `groups_p*.jsonl` e relatórios por etapa |
| `data/phase0_b200/` | v5 (1.500 sementes, 425 válidos) |
| `data/phase0_h100n2/` | segunda rotulagem independente da v5 (teste-reteste) |
| `data/phase0_cache/` | cache de todas as chamadas a modelos (chave = sha256 de modelo, template e entrada) |
| `data/phase0_raw/` | fontes do Hugging Face em JSONL |

Esquema de um registro final: veja a §6.11 de `docs/relatorio_fase0.md`.

## Ética e licença

Nenhum conteúdo nocivo novo é gerado: todo `r_unsafe` vem do PKU-SafeRLHF. Os dados derivados ficam sob **CC-BY-NC-4.0**, com release *gated*. `data/` e `.env` nunca vão para o git.

## Equipe

Caio Lucca dos Santos Oliveira · Davi de Souza Andrade · Frederico Barbosa Relvas · Gustavo Novack Viana Lima
