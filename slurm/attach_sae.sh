#!/bin/bash
# Treina um SAE extra DENTRO do job de modelo já alocado (fora da h1003: /tmp/$USER/nlp).
#   srun --jobid=<J> --overlap -n1 bash slurm/attach_sae.sh <camada> [args extras do train_sae]
set -uo pipefail
REPO=/home/user_gustavonovack/nlp_project; ST=/tmp/$USER/nlp; L=$1; shift
OUT=$ST/artifacts/sae_proprio/gemma-3-4b-it_L${L}_btk16k; mkdir -p $OUT
set -a; source $REPO/.env; set +a
export HF_HUB_CACHE=$ST/hf HF_HOME=$ST/hf_home TMPDIR=$ST/tmp PYTHONUNBUFFERED=1 PYTHONPATH=$REPO/src \
       HF_HUB_OFFLINE=1 SAE_MODEL_PATH=$ST/models/google/gemma-3-4b-it PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
LOG=$REPO/slurm-logs/$SLURM_JOB_ID/sae_L$L.log
apptainer exec --nv -B $ST/w:/w -B /home/$USER $ST/pytorch-2.8.0-cu128-devel.sif /w/out/judge_serve_venv/bin/python \
  -m featgrpo.interp.train_sae train --layer $L --out $OUT --corpus $ST/artifacts/sae_corpus "$@" >> $LOG 2>&1 \
  && apptainer exec --nv -B $ST/w:/w -B /home/$USER $ST/pytorch-2.8.0-cu128-devel.sif /w/out/judge_serve_venv/bin/python \
  -m featgrpo.interp.train_sae eval --layer $L --out $OUT --corpus $ST/artifacts/sae_corpus \
  --ref-gemmascope resid_post/layer_${L}_width_16k_l0_medium,resid_post/layer_${L}_width_16k_l0_small >> $REPO/slurm-logs/$SLURM_JOB_ID/sae_eval_L$L.log 2>&1
source $REPO/slurm/lib/raid.sh; raid_push $OUT nlp_artifacts/sae_proprio/ && echo "SAE L$L pronto e no raid" >> $LOG
