#!/bin/bash
# S0 — validação do ambiente: enroot + imagem vLLM NGC, GPU no container,
# acesso HF a partir do nó e Claude Code CLI headless.
#SBATCH --job-name=s0-env
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45-ephem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/beegfs/home/user_gustavonovack/nlp
#SBATCH -o /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.out
#SBATCH -e /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.err
set -uo pipefail
echo "[job $SLURM_JOB_ID] $(hostname) $(uname -m) inicio=$(date -Is)" >&2

# Vigia de memória: o Slurm daqui NÃO aplica --mem. Cancela o job se o
# consumo do nó (queda do MemAvailable, inclui GPU) passar do pedido.
LIMITE_KB=$((40 * 1024 * 1024))          # = --mem
A0=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
( while sleep 5; do
    A=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if (( A0 - A > LIMITE_KB )); then
      echo "[vigia] $(( (A0 - A) / 1024 ))M > --mem; cancelando" >&2
      scancel "$SLURM_JOB_ID"; exit
    fi
  done ) &
trap 'kill $! 2>/dev/null; rm -rf /tmp/$USER-enroot /tmp/$USER-judge-sandbox' EXIT

# ---- trabalho ----
TAG=26.09-py3
IMG=/beegfs/home/$USER/images/vllm-$TAG.sqsh
export ENROOT_CACHE_PATH=/tmp/$USER-enroot/cache
export ENROOT_DATA_PATH=/tmp/$USER-enroot/data
export ENROOT_TEMP_PATH=/tmp/$USER-enroot/tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"
ok=1

echo "== host"
uname -m
nvidia-smi --query-gpu=name,driver_version --format=csv
enroot version

echo "== import da imagem (pula se o .sqsh já existe)"
if [[ ! -s $IMG ]]; then
  enroot import -a aarch64 -o "$IMG" "docker://nvcr.io#nvidia/vllm:$TAG" || { echo "FALHA import"; ok=0; }
fi
ls -lh "$IMG"

echo "== container: torch/vllm/GPU"
enroot create -n s0vllm "$IMG"
enroot start -m /beegfs/home/$USER:/beegfs/home/$USER s0vllm python3 -c "
import torch, vllm, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability(0))
print('vllm', vllm.__version__, 'transformers', transformers.__version__)
" || { echo "FALHA container"; ok=0; }

echo "== HF a partir do nó (token do .env)"
set -a; source .env; set +a
curl -s -o /dev/null -w '%{http_code} llama gated\n' -I -L -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json

echo "== Claude Code CLI"
unset ANTHROPIC_API_KEY
export DISABLE_AUTOUPDATER=1
mkdir -p /tmp/$USER-judge-sandbox && cd /tmp/$USER-judge-sandbox
~/opt/claude-aarch64/.local/bin/claude --version
~/opt/claude-aarch64/.local/bin/claude -p 'Responda apenas: OK' --model claude-sonnet-5-5 --output-format json \
  | grep -oE '"(result|num_turns|is_error|subtype)":[^,]*|"claude-[a-z0-9-]+"' | sort -u \
  || { echo "FALHA claude"; ok=0; }

if (( ok )); then echo "S0 OK"; else echo "S0 FALHOU"; exit 1; fi
