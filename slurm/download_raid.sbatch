#!/bin/bash
# Baixa pesos (internet) direto para o /raid do h1003 (nlp_artifacts/hf_cache), um processo por modelo,
# e prepara o corpus do SAE. Só CPU, na h100n3 (dgx-H100-03).
#   MODELS="a b c" sbatch slurm/download_raid.sbatch
#SBATCH --job-name=dl-raid
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/user_gustavonovack/nlp_project
#SBATCH -o /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.out
#SBATCH -e /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.err
set -uo pipefail
say() { echo "[$(date +%H:%M:%S)] $*"; }
REPO=/home/user_gustavonovack/nlp_project
R=/raid/user_gustavonovack
HFC=$R/nlp_artifacts/hf_cache; mkdir -p $HFC $R/nlp_artifacts/sae_corpus $R/nlp_stage1/tmp
IMG=$R/images/pytorch-2.8.0-cu128-devel.sif
DV=/tmp/$USER/nlp/driver_venv
MODELS=${MODELS:-"google/gemma-3-4b-it Qwen/Qwen3Guard-Gen-8B allenai/wildguard RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"}
set -a; source $REPO/.env; set +a
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export HF_HUB_CACHE=$HFC HF_HOME=$R/nlp_stage1/hf_home HF_DATASETS_CACHE=$R/nlp_stage1/hf_datasets \
       TMPDIR=$R/nlp_stage1/tmp HF_XET_HIGH_PERFORMANCE=1 PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1
CRUN() { apptainer exec -B $R -B /home/$USER "$IMG" "$@"; }
mon() { while true; do say "raid hf_cache: $(du -sh $HFC 2>/dev/null | cut -f1)"; sleep 120; done; }
mon & MON=$!
pids=()
for m in $MODELS; do
  ( CRUN $DV/bin/python -m featgrpo.data.phase0.download_models $m && say "ok $m" || say "FALHOU $m" ) > $REPO/slurm-logs/dl-$SLURM_JOB_ID-${m//\//--}.log 2>&1 &
  pids+=($!)
done
[ "${CORPUS:-0}" = 1 ] && ( CRUN $DV/bin/python -m featgrpo.interp.prep_corpus --out $R/nlp_artifacts/sae_corpus && say "corpus ok" ) > $REPO/slurm-logs/dl-$SLURM_JOB_ID-corpus.log 2>&1 &
[ "${CORPUS:-0}" = 1 ] && pids+=($!)
for p in "${pids[@]}"; do wait $p; done
kill $MON 2>/dev/null
say "fim: $(du -sh $HFC | cut -f1)"; ls -a $HFC
