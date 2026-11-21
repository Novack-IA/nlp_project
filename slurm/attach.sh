#!/bin/bash
# Roda um passo extra DENTRO de um job já alocado, usando os servidores vLLM dele (127.0.0.1).
#   srun --jobid=<J> --overlap -n1 bash slurm/attach.sh <modulo> [args...]
# Ex.: srun --jobid=37697 --overlap -n1 bash slurm/attach.sh featgrpo.data.phase1.t5_mining variants --gen-url http://127.0.0.1:36974
set -uo pipefail
REPO=/home/user_gustavonovack/nlp_project
if [ -d /raid/user_gustavonovack ] && [ "$(hostname -s)" = dgx-H100-03 ]; then
  R=/raid/user_gustavonovack; ST=$R/nlp_stage1; IMG=$R/images/pytorch-2.8.0-cu128-devel.sif; BIND="-B $R/feat-grpo:/w -B $R"
else
  ST=/tmp/$USER/nlp; IMG=$ST/pytorch-2.8.0-cu128-devel.sif; BIND="-B $ST/w:/w"
fi
set -a; source $REPO/.env; set +a
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export HF_HUB_CACHE=$ST/hf HF_HOME=$ST/hf_home TMPDIR=$ST/tmp PYTHONUNBUFFERED=1 PYTHONPATH=$REPO/src \
       PHASE0_CONFIG=configs/data/phase0_v5b.yaml PHASE0_CACHE=data/phase0_cache PHASE0_RAW=data/phase0_raw \
       PHASE0_DATA=data/phase0_v5b PHASE1_DATA=data/phase1
cd $REPO
DV=$ST/driver_venv; [ -x /tmp/$USER/nlp/driver_venv/bin/python ] && DV=/tmp/$USER/nlp/driver_venv   # o do job (requirements em dia)
exec apptainer exec --nv $BIND -B /home/$USER "$IMG" $DV/bin/python -m "$@"
