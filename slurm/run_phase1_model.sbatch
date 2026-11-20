#!/bin/bash
# Fase 1, frente A — modelo (3×H100 na h100n3 = dgx-H100-03, que tem o /raid local).
#   GPU0: vLLM do Gemma-3-4B-IT (comportamento natural) + SAE da camada 29
#   GPU1: SAE da camada 22 (principal)      GPU2: SAE da camada 17
# Comportamento: pares e externos já; células quando o job de dados escrever CELLS_DONE.
# Retomável (checkpoints do SAE; arquivos de comportamento pulam se existem).
#SBATCH --job-name=p1-model
#SBATCH --partition=h100n3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=30:00:00
#SBATCH --chdir=/home/user_gustavonovack/nlp_project
#SBATCH -o /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.out
#SBATCH -e /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.err
set -uo pipefail
ts() { date +%H:%M:%S; }
say() { echo "[$(ts)] $*"; echo "[$(ts)] $*" >&2; }
say "job $SLURM_JOB_ID em $(hostname), GPUs=$CUDA_VISIBLE_DEVICES"

REPO=/home/user_gustavonovack/nlp_project
R=/raid/user_gustavonovack
ST=$R/nlp_stage1; mkdir -p $ST/hf $ST/hf_home $ST/cache $ST/tmp $ST/run
IMG=$R/images/pytorch-2.8.0-cu128-devel.sif
WBIND=$R/feat-grpo                                   # contém out/judge_serve_venv
SAE=$R/nlp_artifacts/sae_proprio; CORPUS=$R/nlp_artifacts/sae_corpus; mkdir -p $SAE $CORPUS
D1=data/phase1; D=$REPO/$D1; mkdir -p $D
LOGD=$REPO/slurm-logs/$SLURM_JOB_ID; mkdir -p $LOGD
read -r -a GPU <<< "${CUDA_VISIBLE_DEVICES//,/ }"
[ ${#GPU[@]} -ge 3 ] || { say "precisa de 3 GPUs"; exit 1; }
P_SUBJ=$((21000 + (SLURM_JOB_ID % 2000) * 10))
U() { echo "http://127.0.0.1:$1"; }
SUBJ=google/gemma-3-4b-it

trap 'say "saindo"; for f in $ST/run/pg_*; do [ -f "$f" ] && kill -TERM -- -$(cat $f) 2>/dev/null; done; sleep 5; for f in $ST/run/pg_*; do [ -f "$f" ] && kill -KILL -- -$(cat $f) 2>/dev/null; rm -f $f; done; jobs -p | xargs -r kill 2>/dev/null' EXIT

set -a; source $REPO/.env; set +a
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
export HF_HUB_CACHE=$ST/hf HF_HOME=$ST/hf_home HF_DATASETS_CACHE=$ST/hf_datasets XDG_CACHE_HOME=$ST/cache \
       TMPDIR=$ST/tmp VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_XET_HIGH_PERFORMANCE=1 \
       PYTHONUNBUFFERED=1 PYTHONPATH=$REPO/src OMP_NUM_THREADS=8 \
       VLLM_CACHE_ROOT=$ST/cache/vllm TRITON_CACHE_DIR=$ST/cache/triton \
       PHASE0_CONFIG=configs/data/phase0_v5b.yaml PHASE0_CACHE=data/phase0_cache PHASE0_RAW=data/phase0_raw \
       PHASE1_DATA=$D1
CRUN() { apptainer exec --nv -B $WBIND:/w -B /home/$USER -B $R "$IMG" "$@"; }
VPY=/w/out/judge_serve_venv/bin/python
DV=$ST/driver_venv
REQ_MD5=$(md5sum $REPO/requirements-phase0.txt | cut -c1-12)
if [ ! -f $DV/.req_$REQ_MD5 ]; then
  say "criando venv do driver"; rm -rf $DV
  CRUN python3 -m venv $DV && CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -U pip && \
  CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -r $REPO/requirements-phase0.txt && touch $DV/.req_$REQ_MD5 || exit 1
fi

# pesos do 4B + corpus (em paralelo)
CRUN $DV/bin/python -m featgrpo.data.phase0.download_models $SUBJ > $LOGD/download.log 2>&1 || { say "download falhou"; exit 1; }
( CRUN $DV/bin/python -m featgrpo.interp.prep_corpus --out $CORPUS ) > $LOGD/corpus.log 2>&1 &
CORP=$!

# GPU0: vLLM do 4B (fração menor, para dividir com o SAE L29)
CUDA_VISIBLE_DEVICES=${GPU[0]} HF_HUB_OFFLINE=1 setsid apptainer exec --nv -B $WBIND:/w -B /home/$USER -B $R "$IMG" \
  /w/out/judge_serve_venv/bin/vllm serve $SUBJ --host 127.0.0.1 --port $P_SUBJ --gpu-memory-utilization 0.40 \
  --max-model-len 4096 --max-num-seqs 128 --limit-mm-per-prompt '{"image": 0}' > $LOGD/vllm_subj.log 2>&1 &
echo $! > $ST/run/pg_subj
until curl -sf $(U $P_SUBJ)/health >/dev/null; do
  sleep 10; kill -0 $(cat $ST/run/pg_subj) 2>/dev/null || { say "vLLM 4B morreu"; tail -30 $LOGD/vllm_subj.log >&2; exit 1; }
done
say "4B pronto"

behavior() {
  for s in pairs external cells; do
    if [ $s = external ]; then until [ -f $REPO/datasets/phase1/external/falsereject.jsonl ]; do sleep 60; done; fi
    if [ $s = cells ]; then
      until [ -f $D/CELLS_DONE ] || [ -f $D/GATE_DRY_FAIL ]; do sleep 120; done
      [ -f $D/CELLS_DONE ] || { say "sem células (gate falhou)"; return 0; }
    fi
    [ -f $D/behavior_raw_$s.jsonl ] || CRUN $DV/bin/python -m featgrpo.data.phase1.t4_behavior generate --set $s --url $(U $P_SUBJ)
  done
}
behavior > $LOGD/behavior.log 2>&1 &
BEH=$!

wait $CORP || { say "corpus falhou (ver $LOGD/corpus.log)"; tail -20 $LOGD/corpus.log >&2; exit 1; }
say "corpus: $(cat $CORPUS/report.json | tr -d '\n ')"

train() {  # gpu-idx camada
  local gi=$1 L=$2 out=$SAE/gemma-3-4b-it_L${2}_btk16k
  mkdir -p $out
  [ -f $out/DONE ] || CUDA_VISIBLE_DEVICES=${GPU[$gi]} CRUN $VPY -m featgrpo.interp.train_sae train --layer $L --out $out --corpus $CORPUS \
      > $LOGD/sae_L$L.log 2>&1 || { say "SAE L$L falhou"; tail -30 $LOGD/sae_L$L.log >&2; return 1; }
  CUDA_VISIBLE_DEVICES=${GPU[$gi]} CRUN $VPY -m featgrpo.interp.train_sae eval --layer $L --out $out --corpus $CORPUS \
      --ref-gemmascope resid_post/layer_${L}_width_16k_l0_medium,resid_post/layer_${L}_width_16k_l0_small \
      > $LOGD/sae_eval_L$L.log 2>&1 || say "eval L$L falhou"
  say "SAE L$L pronto"
}
train 1 22 & T22=$!
train 2 17 & T17=$!
train 0 29 & T29=$!
wait $BEH; say "comportamento rc=$?"
wait $T22 $T17 $T29
say "fim do job de modelo"
