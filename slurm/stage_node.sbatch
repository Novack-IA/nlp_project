#!/bin/bash
# Pré-aquece o /tmp/$USER/nlp de um nó (b200n1, h100n2, …) SEM ocupar GPU:
#   1. imagem apptainer e venv do vLLM: puxados direto do /raid do dgx-H100-03 (rsync/SSH; ver slurm/lib/raid.sh)
#   2. venv do driver (pipeline) + testes
#   3. pesos dos modelos para o HF_HUB_CACHE local do nó (marcadores .ok_*)
# Idempotente: o que já existe é pulado. Depois, o job de GPU só sobe os servidores.
#
# Uso:  J=$(sbatch --parsable -p b200n1 slurm/stage_node.sbatch)
#       sbatch -p b200n1 --gres=gpu:1 --dependency=afterok:$J --export=ALL slurm/run_phase0.sbatch
#       MODELS="Qwen/Qwen3-Embedding-0.6B google/gemma-4-12B-it" sbatch -p h100n2 slurm/stage_node.sbatch
#SBATCH --job-name=stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/user_gustavonovack/nlp_project
#SBATCH -o /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.out
#SBATCH -e /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.err
set -uo pipefail
say() { echo "[$(date +%H:%M:%S)] $*"; }
REPO=/home/user_gustavonovack/nlp_project
ST=/tmp/$USER/nlp; mkdir -p $ST/w/out $ST/hf $ST/tmp
source $REPO/slurm/lib/raid.sh
say "nó $(hostname -s); /tmp livre: $(df -h /tmp | awk 'NR==2{print $4}')"

# 1. imagem e venv do vLLM vindos do /raid do h1003 (substitui o ~/nlp_stage no /home)
t=$SECONDS; raid_pull images/pytorch-2.8.0-cu128-devel.sif $ST/ --size-only && say "imagem ok ($((SECONDS-t))s)" || exit 1
t=$SECONDS; raid_pull feat-grpo/out/judge_serve_venv $ST/w/out/ && say "venv vLLM ok ($((SECONDS-t))s)" || exit 1
IMG=$ST/pytorch-2.8.0-cu128-devel.sif

# 2. venv do driver (isolado do PIP_CONSTRAINT da imagem) + testes
set -a; source $REPO/.env; set +a
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export HF_HUB_CACHE=$ST/hf HF_HOME=$ST/hf_home TMPDIR=$ST/tmp PYTHONPATH=$REPO/src HF_XET_HIGH_PERFORMANCE=1
CRUN() { apptainer exec -B $ST/w:/w -B /home/$USER "$IMG" "$@"; }
DV=$ST/driver_venv; REQ_MD5=$(md5sum $REPO/requirements-phase0.txt | cut -c1-12)
if [ ! -f $DV/.req_$REQ_MD5 ]; then
  say "criando venv do driver"; rm -rf $DV
  CRUN python3 -m venv $DV && CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -U pip && \
  CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -r $REPO/requirements-phase0.txt && touch $DV/.req_$REQ_MD5 || exit 1
fi
CRUN $DV/bin/python -m pytest -q $REPO/tests | tail -1

# 3. pesos (ordem de uso no pipeline); MODELS sobrescreve a lista
MODELS=${MODELS:-"Qwen/Qwen3-Embedding-0.6B RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic Qwen/Qwen3Guard-Gen-8B allenai/wildguard google/gemma-4-31B-it google/gemma-4-12B-it meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3"}
t=$SECONDS; CRUN $DV/bin/python -m featgrpo.data.phase0.download_models $MODELS || exit 1
say "pesos ok ($((SECONDS-t))s); cache do nó: $(du -sh $ST/hf | cut -f1); /tmp livre: $(df -h /tmp | awk 'NR==2{print $4}')"
