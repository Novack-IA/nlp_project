#!/bin/bash
# Repasse de pesos já baixados (cache HF do /tmp da b200n1, da Fase 0) para o /raid do h1003
# (nlp_artifacts/hf_cache) pela rede interna,
# quando a internet do cluster está lenta. Só CPU. -L: o huggingface_hub 2.x guarda os pesos em hf/blobs/
# (depósito compartilhado) e as pastas models--* só têm links; -L copia os arquivos de verdade.
#   TARGET_HOST=dgx-H100-02 MODELS="..." sbatch -p b200n1 slurm/relay_models.sbatch
# Também: baixa os MODELS_NET (internet) e os empurra para o /raid do h1003; copia o venv do driver do
# TARGET_HOST para o /tmp do h1003 (mesmo caminho, funciona dentro da imagem).
#SBATCH --job-name=relay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/user_gustavonovack/nlp_project
#SBATCH -o /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.out
#SBATCH -e /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.err
set -uo pipefail
say() { echo "[$(date +%H:%M:%S)] $*"; }
SRC=/tmp/$USER/nlp/hf
TARGET_HOST=${TARGET_HOST:-dgx-H100-02}
MODELS=${MODELS:-"Qwen/Qwen3-Embedding-0.6B RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic Qwen/Qwen3Guard-Gen-8B allenai/wildguard google/gemma-4-31B-it"}
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
source /home/user_gustavonovack/nlp_project/slurm/lib/raid.sh
say "nó $(hostname -s); cache local: $(du -sh $SRC 2>/dev/null | cut -f1)"
ls -a $SRC | head -40

# 1. pesos da Fase 0 → TARGET_HOST:/tmp/$USER/nlp/hf (em paralelo por modelo)
pids=()
for m in $MODELS; do
  d=models--${m//\//--}; mk=.ok_${m//\//--}
  if [ ! -f $SRC/$mk ] || [ ! -d $SRC/$d ]; then say "AUSENTE na origem: $m"; continue; fi
  ( raid_push $SRC/$d nlp_artifacts/hf_cache/ -L && raid_push $SRC/$mk nlp_artifacts/hf_cache/ \
      && say "ok $m → raid" || say "FALHOU $m" ) &
  pids+=($!)
done

for p in "${pids[@]}"; do wait $p; done
say "fim do repasse"
