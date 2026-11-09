# Cluster DGX (H100/B200): como disparar jobs e usar o storage de forma limpa

Exploração feita em 2026-11-09. Os números foram medidos com jobs de teste reais.

## 1. Como o cluster é por dentro (o que limita tudo)

| Fato | Medido / fonte | Consequência |
|---|---|---|
| `/raid` é **disco local de cada nó**; só o dgx-H100-03 tem `/raid/user_gustavonovack` | `df`, `ls /raid` em cada nó | outros nós não veem nosso `/raid` |
| **Cota no `/raid` do h1003: 460 GB usados de 500 GB (soft) / 600 GB (hard)** | `quota -s` | sobram ~40 GB antes do soft; não cabem ~250 GB de pesos lá |
| Os admins criam `/raid/<user>` por nó com `/cm/shared/scripts/cria_pasta_raid.sh` (cota de 500/600 GB por nó) | o próprio script | pedir a pasta na b200n1 e na h100n2 custa 1 comando ao admin |
| **Rede entre nós: só a `eno3`, 1 GbE** (~117 MB/s de teto, compartilhado) | `/sys/class/net/*/speed` | todo tráfego entre nós (SSH, NFS do `/home`, HF) disputa 1 Gb |
| **12 portas InfiniBand ConnectX (até 400 Gb/s) e 2 de storage (`ens6f*`) estão DOWN** | `/sys/class/infiniband/*/ports/*/state` | a infra para 100–400× mais banda existe, só não está ligada |
| SSH entre nós funciona (chave `id_ecdsa` no `/home`; `sshd` permite `group_u_dgx-h100-3`) | job de teste na B200 e na h100n2 | dá para ler e gravar o `/raid` do h1003 a partir de qualquer nó |
| Vazão SSH → `/raid` do h1003: **56–67 MiB/s** | 4 GiB lidos para `/dev/null` | 250 GB de pesos ≈ 70 min; dados e resultados (MB–GB): segundos a minutos |
| `/tmp` de cada nó: disco do sistema (~1,8 TB), **não é limpo pelo Slurm** (`TmpFS=/tmp`, `job_container/none`) | `scontrol show config` | serve de cache persistente por nó, mas é compartilhado e exige limpeza manual |
| `/cm/shared` (NFS do head node, 325 GB livres) | `df` | só leitura para usuários |
| `/home` (NFS, ~228 GB livres para todos) | `df` | só código e arquivos pequenos de coordenação |
| `sshfs` e `rclone` **não instalados**; `/dev/fuse` (rw) e `fusermount3` existem nos nós | `which`, `ls -l /dev/fuse` | uma montagem FUSE do `/raid` do h1003 é viável se instalarmos um cliente (§3.5) |
| Pyxis/enroot inativos (`srun` sem `--container-image`) | `srun --help` | containers via `apptainer exec` |
| QoS `onejob`: **2 jobs rodando por usuário**, sem limite de GPU por job; preempção por QoS (`REQUEUE`) | `sacctmgr` | paralelismo = poucos jobs grandes, não muitos pequenos |

## 2. Resposta à pergunta principal: dá para usar o `/raid` do h1003 estando na B200 ou na h100n2?

**Sim, via `rsync` sobre SSH**, que é o mesmo método que os admins usam (`/cm/shared/scripts/copia_dados_everest.sh`). Testado nos dois sentidos, a partir das duas máquinas.

- `slurm/lib/raid.sh` traz:
  - `raid_pull <remoto> <local>` e `raid_push <local> <remoto>`;
  - `raid_ls` e `raid_quota`;
  - uma única conexão SSH reaproveitada (ControlMaster), retomada de transferências parciais;
  - funciona igual no próprio dgx-H100-03, onde vira cópia local.
- Padrão recomendado: **o `/raid` do h1003 é a "casa" dos dados**:
  - o job puxa as entradas no início e devolve as saídas no fim;
  - o `/home` guarda só código e arquivos pequenos;
  - o `/tmp` do nó é só cache.

```bash
source slurm/lib/raid.sh
raid_pull nlp_project/data/phase0_v5b/final/ /tmp/$USER/work/final/      # entrada
# ... job ...
raid_push /tmp/$USER/work/out/ nlp_project/data/phase1/run_X/              # saída
```

**Limites:**
- 1 GbE (~60 MB/s): ótimo para dados, resultados, checkpoints de SAE e caches de chamadas; ruim para centenas de GB de pesos;
- a cota de 40 GB livres no `/raid` do h1003 é o gargalo real de espaço;
- não é uma montagem transparente: é preciso puxar e empurrar explicitamente (a montagem está na §3.5).

## 3. Estratégias para paralelizar e usar o cluster de forma limpa

### 3.1 Pré-aquecer o nó num job **só de CPU** antes do job de GPU
`slurm/stage_node.sbatch` (testado na B200 em ~1 min, com tudo já presente):
- puxa a imagem e o venv do vLLM **direto do `/raid` do h1003**, o que dispensa o `~/nlp_stage` no `/home`;
- monta o venv do driver e roda os testes;
- baixa os pesos para o `/tmp` do nó.

O job de GPU só começa quando o nó está pronto:

```bash
J=$(sbatch --parsable -p b200n1 slurm/stage_node.sbatch)
sbatch -p b200n1 --gres=gpu:1 --dependency=afterok:$J --export=ALL slurm/run_phase0.sbatch
```

Na rodada de 08/11, os downloads aconteceram **com GPUs reservadas**: ~1h de 1 B200 + 2 H100 ociosas.

Ressalva: em partição muito disputada (a h100n2 hoje), até um job só de CPU espera por `Priority`.

### 3.2 Submeter em várias partições de uma vez
```bash
sbatch -p b200n1,h100n2,h100n3 --gres=gpu:N ...   # começa onde liberar primeiro
sbatch --test-only -p b200n1,h100n2 --gres=gpu:2 ...  # previsão de início, sem submeter
squeue -u $USER --start                              # previsão dos pendentes
```
- O `run_phase0.sbatch` já detecta o layout (1×B200 ou 2×H100) pelo número de GPUs.
- Para N GPUs iguais nos dois tipos de nó, o script precisa decidir as frações pela memória da GPU. Fica como ajuste futuro: ler `nvidia-smi --query-gpu=memory.total`.
- Em 09/11, um pedido de 1 GPU começaria **imediatamente** na B200; 2 ou mais GPUs, só por volta de 14/11.

### 3.3 Paralelismo **dentro** do job (o limite é de 2 jobs, não de GPUs)
- Peça as GPUs de que a etapa precisa num job só e rode vários servidores ou processos em paralelo, como o `run_phase0.sbatch` já faz.
- Para tarefas independentes dentro de uma alocação, use `srun --exact -n1 --gres=gpu:1 ... &` (*job steps*). Eles não contam no limite de 2 jobs. É o padrão do Slurm 23.11, mas ainda não foi testado neste cluster.
- Job arrays contam no limite: no máximo 2 elementos rodando ao mesmo tempo.

### 3.4 Cache por nó, com convenção e limpeza explícita
- Layout fixo: `/tmp/$USER/nlp/{hf, w/out/judge_serve_venv, driver_venv, *.sif}`. Os marcadores `.ok_<modelo>` tornam os downloads idempotentes.
- Hoje há ~222 GB na B200 e ~250 GB na h100n2. **Só limpe quando o usuário mandar.** Use um job só de CPU no nó com `rm -rf /tmp/$USER/nlp`.

### 3.5 Montagem transparente do `/raid` do h1003 (FUSE), se aprovada
- Com um cliente FUSE no `/home` (por exemplo `rclone`, binário único, backend SFTP sobre o mesmo SSH), um job em qualquer nó faria:
  `rclone mount h1003:/raid/$USER /tmp/$USER/raid_h1003 --vfs-cache-mode full --daemon`
- Isso dá leitura e escrita transparentes, com cache local em disco.
- **Não foi instalado:** baixar binários externos exige autorização explícita.
- A banda continua limitada a 1 GbE, então é conveniência, não velocidade.

## 4. Pedidos ao admin (do mais barato ao mais impactante)

1. **Criar `/raid/user_gustavonovack` na dgx-B200-1 e na dgx-H100-02** com `cria_pasta_raid.sh user user_gustavonovack` (500/600 GB cada). Os pesos ficariam permanentes em cada nó, fora do `/tmp` compartilhado.
2. **Cota no `/raid` do h1003** (460/500 GB usados), ou combinar uma limpeza do que é antigo: feat-grpo, sft_factory, Laudite, `pesos.tar.gz` etc.
3. **Instalar `sshfs`** nos nós (`apt install sshfs`). Montagem FUSE padrão, sem binário de usuário.
4. **Ativar a rede de alta velocidade entre os nós** (InfiniBand ou as portas `ens6f*`). Hoje está tudo DOWN e o cluster vive em 1 GbE. Com IB, ler o `/raid` do h1003 de outro nó (NFS/RDMA) seria quase tão rápido quanto disco local, e `/home` e downloads deixariam de disputar a mesma banda.
5. (Opcional) Um cache compartilhado de modelos do Hugging Face num volume NFS. Com o item 4, um único download serviria a todos os nós.

## 5. Arquivos de teste deixados pela exploração
- `/raid/user_gustavonovack/.raidtest/` (2 arquivos de 50 bytes) e `~/slurm-check/` (scripts e logs de teste): podem ser apagados.
- O job 36639 (`stage` na h100n2) ficou na fila; é idempotente e só confere ou baixa o que falta.
