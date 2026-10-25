#!/bin/bash
# P0 (passo S2) — baixa as fontes da §4 e constrói o conjunto de exclusão E
# (hashes + embeddings Qwen3-Embedding-0.6B na GPU).
#SBATCH --job-name=p0
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45-ephem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/beegfs/home/user_gustavonovack/nlp
#SBATCH -o /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.out
#SBATCH -e /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.err
set -uo pipefail
echo "[job $SLURM_JOB_ID] $(hostname) $(uname -m) inicio=$(date -Is)" >&2

# Vigia de memória: o Slurm daqui NÃO aplica --mem. Cancela o job se o
# consumo do nó (queda do MemAvailable, inclui GPU) passar do pedido.
LIMITE_KB=$((32 * 1024 * 1024))          # = --mem
A0=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
( while sleep 5; do
    A=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if (( A0 - A > LIMITE_KB )); then
      echo "[vigia] $(( (A0 - A) / 1024 ))M > --mem; cancelando" >&2
      scancel "$SLURM_JOB_ID"; exit
    fi
  done ) &
trap 'kill $! 2>/dev/null; rm -rf /tmp/$USER-enroot' EXIT

# ---- container: imagem vLLM NGC, repo e HF_HOME montados ----
export ENROOT_CACHE_PATH=/tmp/$USER-enroot/cache ENROOT_DATA_PATH=/tmp/$USER-enroot/data ENROOT_TEMP_PATH=/tmp/$USER-enroot/tmp
enroot create -n pipe /beegfs/home/$USER/images/vllm-26.09-py3.sqsh
set -a; source .env; set +a
RUN="enroot start -m /beegfs/home/$USER:/beegfs/home/$USER \
  -e HOME=/beegfs/home/$USER -e PYTHONUNBUFFERED=1 -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/beegfs/home/$USER/.cache/huggingface -e HF_HUB_DISABLE_XET=1 \
  -e PYTHONPATH=/beegfs/home/$USER/nlp/src pipe"

# ---- trabalho ----
$RUN bash -c 'cd /beegfs/home/$USER/nlp && .venv/bin/python -m featgrpo.data.phase0.p0_download'
echo "fim=$(date -Is)"
