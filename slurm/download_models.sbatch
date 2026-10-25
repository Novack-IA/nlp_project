#!/bin/bash
# Baixa os pesos de todos os modelos da §5 para o HF_HOME no BeeGFS
# (só safetensors + configs/tokenizer). Retomável: rodar de novo continua.
#SBATCH --job-name=download-models
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --time=10:00:00
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

# ---- container (mesma imagem do S0) ----
IMG=/beegfs/home/$USER/images/vllm-26.09-py3.sqsh
export ENROOT_CACHE_PATH=/tmp/$USER-enroot/cache ENROOT_DATA_PATH=/tmp/$USER-enroot/data ENROOT_TEMP_PATH=/tmp/$USER-enroot/tmp
enroot create -n dl "$IMG"
set -a; source .env; set +a

# ---- trabalho ----
enroot start -m /beegfs/home/$USER:/beegfs/home/$USER \
  -e HF_TOKEN="$HF_TOKEN" -e HF_HOME=/beegfs/home/$USER/.cache/huggingface \
  -e HF_XET_HIGH_PERFORMANCE=1 dl python3 - <<'PY'
from huggingface_hub import snapshot_download
MODELS = [  # ordem: o que destrava o S1/S3 primeiro
    "google/gemma-4-12B-it", "google/gemma-4-31B-it",
    "Qwen/Qwen3Guard-Gen-8B", "allenai/wildguard",
    "Qwen/Qwen3-Embedding-0.6B",
    "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
for m in MODELS:
    path = snapshot_download(
        m, max_workers=8,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"],
        ignore_patterns=["original/*", "consolidated*"],
    )
    print("OK", m, path, flush=True)
PY
echo "fim=$(date -Is)"
du -sh /beegfs/home/$USER/.cache/huggingface/hub/models--* 
