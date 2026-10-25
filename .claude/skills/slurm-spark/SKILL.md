---
name: slurm-spark
description: Como executar QUALQUER coisa neste repositório — o ambiente é um container de login sem recursos, e toda execução vai para as DGX Spark (aarch64) via Slurm. Use antes de rodar python, testes, scripts de avaliação, pip/venv, treinamento, inferência, compilação ou qualquer comando que não seja git/leitura de arquivo; e para submeter, acompanhar, cancelar ou dimensionar jobs (sbatch, squeue, sacct, scancel), preparar ambientes e containers, ou cuidar do espaço em disco do cluster.
---

# Slurm nas DGX Spark

Você está no **container de login** (`hostname` = `login-*`, x86_64). Ele não
tem recurso computacional para usuário: **nada roda aqui**. Toda execução é
um job do Slurm numa DGX Spark (aarch64, Ubuntu 24.04, 1 GPU GB10, 20 CPUs,
~111 GB de memória *compartilhada entre CPU e GPU*).

## O que pode e o que não pode no login

**Pode** — gerência leve: `git`, ler/editar arquivo, `ls`/`cat`/`grep`/`find`/
`sed` pontual, `sbatch`/`squeue`/`sacct`/`scancel`/`sinfo`/`scontrol`, `du -sh`.

**Não pode**, nem "só um script rápido": `python`/`python3` (inclusive
`python3 -c` e heredoc para ler JSON), `pip`, criar venv, `pytest`, notebooks,
`node`, compilação, `jq`/`awk` sobre arquivo grande, qualquer coisa com CPU ou
memória significativa. Para inspecionar um JSON, use `grep`/`head` ou mande um
job. Na dúvida, é job.

Além da regra: login é x86_64 e o nó é aarch64 — binário ou venv criado aqui
não roda lá.

## Como executar: `.sbatch` tradicional, escrito à mão

**Política deste repositório:** todo job é um arquivo `.sbatch` explícito,
legível de cima a baixo, versionado em `slurm/` na raiz do repo. Nada de
gerador de `.sbatch` nem de script wrapper que esconda as diretivas: o usuário
audita cada job lendo o arquivo. Um `.sbatch` por tarefa reutilizável
(ex.: `slurm/s0_env.sbatch`, `slurm/p3_responses.sbatch`); parâmetros
variáveis entram por `--export` ou argumentos, nunca reescrevendo o arquivo.

Modelo (copie e ajuste; mantenha a ordem e os comentários):

```bash
#!/bin/bash
# <o que este job faz, em uma ou duas linhas>
#SBATCH --job-name=<nome>
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45-ephem   # ou hpc-spark-45 (principal)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=24G                        # o que o job usa + ~30%; máx. 97G (RealMemory=100000M)
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1                     # só se usar GPU
#SBATCH --chdir=/beegfs/home/user_gustavonovack/nlp
#SBATCH -o /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.out
#SBATCH -e /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.err
set -uo pipefail
echo "[job $SLURM_JOB_ID] $(hostname) $(uname -m) inicio=$(date -Is)" >&2

# Vigia de memória: o Slurm daqui NÃO aplica --mem. Cancela o job se o
# consumo do nó (queda do MemAvailable, inclui GPU) passar do pedido.
LIMITE_KB=$((24 * 1024 * 1024))          # = --mem
A0=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
( while sleep 5; do
    A=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if (( A0 - A > LIMITE_KB )); then
      echo "[vigia] $(( (A0 - A) / 1024 ))M > --mem; cancelando" >&2
      scancel "$SLURM_JOB_ID"; exit
    fi
  done ) &
trap 'kill $! 2>/dev/null' EXIT

# ---- trabalho ----
...
```

Submissão e acompanhamento:

```bash
sbatch slurm/X.sbatch                    # imprime o JOBID
sbatch --wait slurm/X.sbatch             # bloqueia até o fim
sacct -j JOBID -o JobID%-14,State%-14,Elapsed,ReqMem,MaxRSS,ExitCode
```

Logs em `slurm-logs/NOME-JOBID.{out,err}` na raiz do repo (fora do git): **um `.out`
e um `.err` por job, e nada mais**. Subprocessos (ex.: `vllm serve`) escrevem
neles com prefixo (`[vllm modelo]`), nunca em arquivo próprio. Python com
`PYTHONUNBUFFERED=1` para acompanhar ao vivo.

**Espera** — `sbatch --wait` bloqueia até o fim. O Bash tem teto de 10 min:
- job curto (< ~8 min): `sbatch --wait` direto, com `timeout: 600000`;
- job longo: `sbatch --wait` **com `run_in_background: true`** — você é
  notificado quando termina, sem polling. Não faça loop de `squeue` com `sleep`.
- job de horas/dias: `sbatch` sem `--wait`, informe o JOBID ao usuário e
  consulte com `sacct -j JOBID` quando ele pedir.

Ao terminar, mostre ao usuário o fim do `.out`/`.err` e o `sacct`
(`MaxRSS` × `ReqMem`) para dimensionar o próximo.

## Regras de todo job

- `#SBATCH --account=default` — obrigatório.
- **`--mem` e `--time` sempre explícitos.** O default da partição é memória
  ILIMITADA e tempo = máximo (7 dias).
- **`--mem` não é aplicado pelo Slurm daqui** (medido: 2 GB usados com 512M
  pedidos, job COMPLETED). Nada impede um job de consumir a memória
  compartilhada e travar o nó. Todo `.sbatch` tem o **vigia** do modelo acima, que cancela o job
  quando o consumo passa do `--mem` — por isso peça o que o job realmente usa,
  com folga de ~30%. Sem vigia, nenhum job é submetido. Como o nó é
  exclusivo, jobs de vLLM podem usar o nó quase todo: `--mem=97G` (máximo do
  Slurm) e vigia em 112G (de ~117 GiB livres); `--gpu-memory-utilization` é
  fração dos 121.7 GiB da GB10 (memória unificada com a CPU).
- **Caminhos sempre em `/beegfs/home/...`**, nunca `/home/...` (lento pela
  rede ou falha).
- **`--chdir` com caminho absoluto.** `%u` NÃO é expandido em `--chdir` neste
  cluster (o job cai em `/tmp` em silêncio — o exemplo do manual está
  errado). Em `-o`/`-e`, `%u` funciona.
- GPU: só com `--gres=gpu:1` quando o código usa GPU. (Ela fica visível mesmo sem
  `--gres`, porque o nó é exclusivo — mas pedir é o que declara o uso.)

## Dimensionamento: o cluster é minúsculo

São **3 nós**, e **todo job ocupa um nó inteiro** (`OverSubscribe=EXCLUSIVE`
— um job de 1 CPU recebe as 20). Cada job, por menor que seja, tira 1/3 do
cluster de todos os usuários.

**Limite por usuário (QOS `normal`): 1 job rodando e 2 submetidos** (pendentes
contam). Com um job longo em execução, o próximo fica `PD
(QOSMaxJobsPerUserLimit)` até ele acabar, e um terceiro `sbatch` é recusado.
Por isso trabalho longo que precisa de muitos processos (ex.: o orquestrador
da evolução) roda TUDO dentro de um job, como subprocessos — nunca um job por
tarefa. Antes de submeter, `squeue -u $USER`: se já há um job longo rodando,
avise o usuário que o novo vai esperar. Então:

- **Agrupe**: rode testes + avaliação + conferência num job só
  (passos em sequência no mesmo `.sbatch`), não em três.
- **Não submeta job para descobrir algo que dá pra ler** (código, logs, `sacct`).
- `--time` curto e realista — é ele que libera o nó para os outros; job que
  termina antes libera na hora, mas o agendador planeja pelo pedido.
- Memória: comece pequeno; use o `MaxRSS` impresso ao fim para ajustar. O
  `MaxRSS` é amostrado — em job de segundos pode vir subestimado.
- Nunca deixe job esquecido: `squeue -u $USER` ao terminar uma tarefa;
  `scancel JOBID` no que não serve mais.

## Filas

| | `hpc-spark-45-ephem` (`-p ephem`) | `hpc-spark-45` (`-p principal`) |
|---|---|---|
| uso | testes, exploração, trabalho não crítico | o que precisa terminar |
| tempo máx. | 12 h | 7 dias |
| preempção | **CANCELADO** quando a principal precisa do nó, com só **30 s** de aviso | não é interrompido |

Na efêmera, salve resultado intermediário com frequência (checkpoint) — 30 s
não bastam para salvar nada grande. Resultado que importa: fila principal.

## Ambiente no nó

- `python3` = 3.12, `pip3`, `git`, `java`, `nvidia-smi`, `enroot`. **Sem `uv`,
  sem `module`, sem apptainer.** Internet OK (PyPI acessível).
- Venv: crie **por job**, no diretório do projeto:
  `$J -m 8G -t 00:20:00 -- 'python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt'`.
  Rode sempre `.venv/bin/python`, nunca `python3` do sistema.
- Pacote com extensão nativa sem wheel `aarch64` (ex.: dependências do
  medspacy) compila no nó — tem `gcc`/`g++`, mas **não tem os headers do
  Python**. Sem root, baixe-os uma vez, num job:
  `cd /tmp && apt-get download libpython3.12-dev python3.12-dev && for d in *python3.12-dev*.deb; do dpkg -x $d ~/opt/pyheaders-aarch64; done`
  e instale com
  `H=~/opt/pyheaders-aarch64/usr/include; CFLAGS="-I$H -I$H/python3.12" CPPFLAGS="$CFLAGS" .venv/bin/pip install ...`
  (os dois `-I` são necessários: `pyconfig.h` inclui `aarch64-linux-gnu/python3.12/pyconfig.h`).
  Já feito: os headers estão em `~/opt/pyheaders-aarch64/`.
- Claude Code aarch64 (para agentes em job): `~/opt/claude-aarch64/.local/bin/claude`,
  com `DISABLE_AUTOUPDATER=1`. Usa as credenciais de `~/.claude`.
- `/tmp` do nó é NVMe local (~3 TB livres): bom para scratch rápido dentro de
  um job, mas **não sobrevive** ao job e não é visto do login. Limpe o que
  criar e copie o resultado para `/beegfs/home/...` antes de sair.
- Containers: imagem **aarch64** (`enroot import -a aarch64 ...`), em
  `.sqsh` no BeeGFS, com `--container-image=/beegfs/home/$USER/images/X.sqsh`
  e `--container-mounts=/beegfs/home/$USER:/beegfs/home/$USER`. Ainda não
  testado neste repositório — valide com job curto antes de usar.

## Armazenamento

- **500 GB por usuário** (não há cota automática — ninguém vai te barrar,
  vão te remover). Confira: `du -sh /beegfs/home/$USER`.
- **Sem backup**, e falha de hardware perde tudo. Código vai para o remoto
  (`git push`); resultado importante sai do cluster. Avise o usuário quando
  gerar algo que só existe aqui.
- Limpe: `~/.cache/pip`, venvs abandonadas, checkpoints intermediários,
  imagens `.sqsh` sem uso, `slurm-logs/` antigos.
- Conta inativa > 60 dias pode ser apagada.

## Ambiente experimental: reporte o estranho

A equipe pediu relato de todo erro ou comportamento inesperado. Quando algo
fugir do documentado, **diga ao usuário** com o JOBID e o trecho do log. Já
observado (2026-09-23):

1. `%u` em `--chdir` não é expandido → job roda em `/tmp`.
2. `--mem` não é aplicado (sem limite de memória por cgroup).
3. GPU visível no job sem `--gres=gpu:1`.
4. No login, `$HOME` já é `/beegfs/home/$USER` (o manual diz `/home/$USER`;
   os dois caminhos apontam para o mesmo lugar).
