---
name: slurm-dgx-h100
description: Como executar coisas no cluster DGX H100/B200 da UFG (alias ssh h1003 = dgx-H100-03) — partições, limites, storage local por nó, staging, apptainer + vLLM, armadilhas conhecidas. Use antes de submeter, acompanhar ou dimensionar jobs, ou mexer em storage neste cluster.
---

# Cluster DGX H100/B200 (Slurm)

O acesso é por `ssh h1003`, que é o próprio nó **dgx-H100-03** (x86_64, 224 CPUs, 2 TB de RAM). Não é um container de login: é um nó de computação, mas a GPU deve ser pedida **sempre via Slurm**.

## Partições e limites

| Partição | Nó | GPUs no Slurm |
|---|---|---|
| `h100n1` | dgx-H100-01 | 3×H100 (frequentemente fora do ar) |
| `h100n2` | dgx-H100-02 | 8×H100 80 GB |
| `h100n3` | dgx-H100-03 | **3**×H100 (o nó tem 8; 5 ficam fora do Slurm) |
| `b200n1` | dgx-B200-1 | 8×B200 180 GB |

- **QoS `onejob`: 2 jobs rodando por usuário.** Os demais esperam na fila, e **isso é normal**. Não existe limite de GPUs por job: se precisa de N GPUs, peça `--gres=gpu:N` e deixe esperar.
- Para checar a ocupação: `scontrol show node <nó> | grep AllocTRES` e `squeue -p <partição>`.
- **Nunca cancele job do usuário sem pedido explícito.**

## Storage

- `/raid` é **disco local de cada nó**. Só o dgx-H100-03 tem `/raid/user_gustavonovack`, com imagens, venvs, caches HF e os dados de longo prazo. Nos outros nós o `/raid` não é gravável para nós.
- `/home` (NFS, `master:/home`) é compartilhado entre todos os nós, mas **pequeno** (~250 GB livres para todos). Use só como ponte e nunca para pesos. **Não mover nem apagar nada do /home sem ordem direta do usuário.**
- `/tmp` de cada nó fica no disco do sistema (~1,8 TB, compartilhado com o SO e outros usuários). O Slurm não o limpa ao fim do job. Serve como staging de pesos em h100n2/b200n1 (`/tmp/$USER/nlp`). Limpe só quando o usuário mandar.

## Ambiente

- Imagem: `/raid/user_gustavonovack/images/pytorch-2.8.0-cu128-devel.sif` (Apptainer; `apptainer exec --nv`).
- vLLM 0.30 / transformers 5.17 / torch 2.13: venv em `/raid/user_gustavonovack/feat-grpo/out/judge_serve_venv`. O shebang aponta para `/w/out/judge_serve_venv`, então monte com `-B <dir>:/w`.
- **`PIP_CONSTRAINT=/etc/pip/constraint.txt` na imagem** fixa `fsspec==2025.2.0` e outros pacotes. Instale venvs novos com `env -u PIP_CONSTRAINT pip install ...`.
- `export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs` e `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- Há internet nos nós (HF a ~30 MB/s por nó quando dois nós baixam ao mesmo tempo).

## Usar o /raid do h1003 a partir de outros nós

- **Funciona via `rsync` sobre SSH** (testado na B200 e na h100n2): `source slurm/lib/raid.sh`, depois `raid_pull <remoto> <local>`, `raid_push <local> <remoto>`, `raid_ls` e `raid_quota`.
- **Rede entre nós = 1 GbE** (só a `eno3`; InfiniBand e as portas de storage estão DOWN): ~60 MB/s. Use para dados e resultados, não para centenas de GB de pesos.
- **Cota no `/raid` do h1003: 500 GB soft / 600 GB hard** (em 09/11: 460 GB usados). Confira com `raid_quota` antes de gravar.
- Pré-aqueça nós com `slurm/stage_node.sbatch` (só CPU) e encadeie o job de GPU com `--dependency=afterok:<id>`.
- Previsão de início: `sbatch --test-only -p b200n1,h100n2 --gres=gpu:N ...`. Várias partições num `-p` faz o job começar onde liberar primeiro.
- Na partição disputada (h100n2), **até job só de CPU fica pendente por `Priority`** atrás de jobs de GPU.
- Os admins criam `/raid/<user>` por nó com `/cm/shared/scripts/cria_pasta_raid.sh` (cota de 500/600 GB por nó). Para ter pasta própria na b200n1 ou na h100n2, peça a eles.
- `sshfs` e `rclone` **não existem** no cluster (`/dev/fuse` e `fusermount3` existem). Instalar um binário externo é bloqueado pelo classificador de segurança: peça autorização ao usuário antes.
- `/cm/shared` (NFS do head node) é só leitura; Pyxis/enroot inativos (sem `srun --container-image`); `/raid/dat` no `/etc/exports` é dado de outro usuário, e o servidor NFS está desligado.
- Detalhes e pedidos ao admin: `docs/cluster-estrategias.md`.

## Rede

Servidores (vLLM etc.) **só em `127.0.0.1`**: abrir em `0.0.0.0` é bloqueado pela política de segurança. Consequência: dois jobs não conversam por HTTP; troque dados por arquivos no `/home`.

## Padrão de job do nlp_project

`slurm/run_phase0.sbatch` é a referência. Ele faz:
1. staging idempotente no `/tmp`;
2. venv do driver e testes;
3. downloads em background;
4. `serve()` e `stop()` de servidores vLLM por fração de memória;
5. etapas que pulam quando a saída existe.

Os logs ficam em `~/nlp_project/slurm-logs/<jobid>/`, um por servidor e por etapa.
