#!/bin/bash
# Setup do ambiente Python do pipeline (dentro da imagem vLLM): cria o venv
# .venv com o que falta à imagem (requirements-phase0.txt) e roda os testes.
#SBATCH --job-name=setup
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45-ephem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/beegfs/home/user_gustavonovack/nlp
#SBATCH -o /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.out
#SBATCH -e /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.err
set -uo pipefail
echo "[job $SLURM_JOB_ID] $(hostname) $(uname -m) inicio=$(date -Is)" >&2

# Vigia de memória: o Slurm daqui NÃO aplica --mem. Cancela o job se o
# consumo do nó (queda do MemAvailable, inclui GPU) passar do pedido.
LIMITE_KB=$((16 * 1024 * 1024))          # = --mem
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
# venv com --system-site-packages: herda torch/vllm/numpy da imagem e só
# instala o que falta. O python do venv é o da imagem: só roda dentro do container.
$RUN bash -c 'cd /beegfs/home/$USER/nlp && [ -x .venv/bin/python ] || python3 -m venv --system-site-packages .venv'
$RUN bash -c 'cd /beegfs/home/$USER/nlp && .venv/bin/pip install -q -r requirements-phase0.txt'
$RUN bash -c 'cd /beegfs/home/$USER/nlp && .venv/bin/python -c "
import datasets, nltk, pytest, torch, vllm, numpy
print(\"datasets\", datasets.__version__, \"nltk\", nltk.__version__, \"torch\", torch.__version__, \"vllm\", vllm.__version__, \"numpy\", numpy.__version__)
"'
$RUN bash -c "cd /beegfs/home/$USER/nlp && .venv/bin/pip freeze > slurm-logs/pip-freeze-$SLURM_JOB_ID.txt"

echo "== testes unitários"
$RUN bash -c 'cd /beegfs/home/$USER/nlp && .venv/bin/python -m pytest -q tests/' || echo "TESTES FALHARAM"
echo "fim=$(date -Is)"
