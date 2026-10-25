#!/bin/bash
# P4.1 — reanálise da calibração só a partir do cache (sem GPU, sem chamadas novas):
# recalcula κ e grava calibration_items_<prompt>.jsonl e a matriz de confusão do juiz.
#   sbatch --export=ALL,JUDGE_PROMPT=judge_calib_v1 slurm/p4_calib_offline.sbatch
#SBATCH --job-name=p4-calib-offline
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45-ephem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/beegfs/home/user_gustavonovack/nlp
#SBATCH -o /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.out
#SBATCH -e /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.err
set -uo pipefail
echo "[job $SLURM_JOB_ID] $(hostname) $(uname -m) inicio=$(date -Is) PHASE0_DATA=${PHASE0_DATA:-data/phase0}" >&2

# Vigia de memória: o Slurm daqui NÃO aplica --mem. Cancela o job se o
# consumo do nó (queda do MemAvailable, inclui GPU) passar do pedido.
LIMITE_KB=$((4 * 1024 * 1024))          # = --mem
A0=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
( while sleep 5; do
    A=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if (( A0 - A > LIMITE_KB )); then
      echo "[vigia] $(( (A0 - A) / 1024 ))M > --mem; cancelando" >&2
      scancel "$SLURM_JOB_ID"; exit
    fi
  done ) &
trap 'kill $(jobs -p) 2>/dev/null; pkill -u $USER -f "vllm serve"; sleep 5; rm -rf /tmp/$USER-enroot' EXIT

# ---- container: imagem vLLM NGC, repo e HF_HOME montados ----
export ENROOT_CACHE_PATH=/tmp/$USER-enroot/cache ENROOT_DATA_PATH=/tmp/$USER-enroot/data ENROOT_TEMP_PATH=/tmp/$USER-enroot/tmp
enroot create -n pipe /beegfs/home/$USER/images/vllm-26.09-py3.sqsh
set -a; source .env; set +a
RUN="enroot start -m /beegfs/home/$USER:/beegfs/home/$USER \
  -e HOME=/beegfs/home/$USER -e PYTHONUNBUFFERED=1 -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/beegfs/home/$USER/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/beegfs/home/$USER/nlp/src -e PHASE0_DATA=${PHASE0_DATA:-data/phase0} pipe"
PY="$RUN /beegfs/home/$USER/nlp/.venv/bin/python -m featgrpo.data.phase0"

# ---- trabalho ----
$RUN bash -c 'cd /beegfs/home/$USER/nlp && .venv/bin/python -m pytest -q tests/' || exit 1
$PY.p4_label calibrate --offline --qg-url http://127.0.0.1:1 --wg-url http://127.0.0.1:1 --judge-prompt "${JUDGE_PROMPT:-judge_calib_v1}"
echo "fim=$(date -Is)"
