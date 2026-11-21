#!/bin/bash
# Frente A ajudando a B: extração de ativações (interp/extract.py) na dgx-H100-03, com GS2 16k + SAE próprio.
#   sbatch slurm/run_extract_frente_a.sbatch [conjuntos...]   (default: cells2x2 controls)
# Saída: /raid/.../nlp_artifacts/acts_frente_a/gemma-3-4b-it (mesma árvore e MANIFEST do contrato do AGENTS.md)
#SBATCH --job-name=p1-extract
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/user_gustavonovack/nlp_project
#SBATCH -o /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.out
#SBATCH -e /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.err
set -uo pipefail
say() { echo "[$(date +%H:%M:%S)] $*" >&2; }
R=/raid/user_gustavonovack; REPO=/home/user_gustavonovack/nlp_project
IMG=$R/images/pytorch-2.8.0-cu128-devel.sif; VPY=$R/feat-grpo/out/judge_serve_venv/bin/python
OUT=$R/nlp_artifacts/acts_frente_a/gemma-3-4b-it; mkdir -p $OUT
set -a; source $REPO/.env; set +a
export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs TRITON_CACHE_DIR=$R/nlp_stage1/cache/triton
export HF_HUB_CACHE=$R/nlp_artifacts/hf_cache HF_HUB_DISABLE_SHARED_BLOBS=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
       PYTHONPATH=$R/venvs/interp_pkgs:$REPO/src PYTHONUNBUFFERED=1 SAE_PROPRIO_DIR=$R/nlp_artifacts/sae_proprio \
       TOKENIZERS_PARALLELISM=false
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(cells2x2 controls)
say "job $SLURM_JOB_ID em $(hostname) GPU=$CUDA_VISIBLE_DEVICES conjuntos=${SETS[*]}"
for s in "${SETS[@]}"; do
  say "== $s"
  apptainer exec --nv -B $R -B /home/$USER $IMG $VPY -m featgrpo.interp.extract --set $s --widths 16k proprio \
    --batch-size 32 --out-root $OUT || { say "falhou: $s"; exit 1; }
done
say "fim"
