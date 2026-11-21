#!/bin/bash
# Fase 1, frente A — modelo (3×H100). Na dgx-H100-03 usa o /raid; em outro nó, /tmp/$USER/nlp (staging da
# Fase 0) e empurra os SAEs para o /raid no fim.   sbatch -p h100n2 slurm/run_phase1_model.sbatch
#   1 GPU: vLLM do Gemma-3-4B-IT (comportamento) + SAEs das camadas 22 e 17 juntos; a 29 depois do comportamento
# Comportamento: pares e externos já; células quando o job de dados escrever CELLS_DONE.
# Retomável (checkpoints do SAE; arquivos de comportamento pulam se existem).
#SBATCH --job-name=p1-model
#SBATCH --partition=h100n2,b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
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
if [ "$(hostname -s)" = dgx-H100-03 ]; then
  ON_RAID=1; ST=$R/nlp_stage1; IMG=$R/images/pytorch-2.8.0-cu128-devel.sif; WBIND=$R/feat-grpo; XB="-B $R"
  SAE=$R/nlp_artifacts/sae_proprio; CORPUS=$R/nlp_artifacts/sae_corpus
else
  ON_RAID=0; ST=/tmp/$USER/nlp; IMG=$ST/pytorch-2.8.0-cu128-devel.sif; WBIND=$ST/w; XB=""
  SAE=$ST/artifacts/sae_proprio; CORPUS=$ST/artifacts/sae_corpus
fi
RUN=$ST/run/$SLURM_JOB_ID   # pids por job: o trap não mata servidores de outro job
mkdir -p $ST/hf $ST/hf_home $ST/cache $ST/tmp $RUN $ST/w/out $SAE $CORPUS
source $REPO/slurm/lib/raid.sh
if [ $ON_RAID = 0 ]; then   # imagem e venv do vLLM do /raid do h1003 (o /tmp dos nós é limpo)
  [ -f $IMG ] || raid_pull images/pytorch-2.8.0-cu128-devel.sif $ST/ --size-only || exit 1
  [ -x $ST/w/out/judge_serve_venv/bin/python ] || raid_pull feat-grpo/out/judge_serve_venv $ST/w/out/ || exit 1
fi
D1=data/phase1; D=$REPO/$D1; mkdir -p $D
LOGD=$REPO/slurm-logs/$SLURM_JOB_ID; mkdir -p $LOGD
read -r -a GPU <<< "${CUDA_VISIBLE_DEVICES//,/ }"

P_SUBJ=$((21000 + (SLURM_JOB_ID % 2000) * 10))
U() { echo "http://127.0.0.1:$1"; }
SUBJ=google/gemma-3-4b-it

trap 'say "saindo"; for f in $RUN/pg_*; do [ -f "$f" ] && kill -TERM -- -$(cat $f) 2>/dev/null; done; sleep 5; for f in $RUN/pg_*; do [ -f "$f" ] && kill -KILL -- -$(cat $f) 2>/dev/null; rm -f $f; done; jobs -p | xargs -r kill 2>/dev/null' EXIT

set -a; source $REPO/.env; set +a
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
export HF_HUB_DISABLE_SHARED_BLOBS=1 HF_HUB_CACHE=$ST/hf HF_HOME=$ST/hf_home HF_DATASETS_CACHE=$ST/hf_datasets XDG_CACHE_HOME=$ST/cache \
       TMPDIR=$ST/tmp VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_XET_HIGH_PERFORMANCE=1 \
       PYTHONUNBUFFERED=1 PYTHONPATH=$REPO/src OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       VLLM_CACHE_ROOT=$ST/cache/vllm TRITON_CACHE_DIR=$ST/cache/triton \
       PHASE0_CONFIG=configs/data/phase0_v5b.yaml PHASE0_CACHE=data/phase0_cache PHASE0_RAW=data/phase0_raw \
       PHASE1_DATA=$D1
CRUN() { apptainer exec --nv -B $WBIND:/w -B /home/$USER $XB "$IMG" "$@"; }
VPY=/w/out/judge_serve_venv/bin/python
DV=/tmp/$USER/nlp/driver_venv   # mesmo caminho em todos os nós (copiado da h1002 para a h1003)
REQ_MD5=$(md5sum $REPO/requirements-phase0.txt | cut -c1-12)
if [ ! -f $DV/.req_$REQ_MD5 ]; then
  say "criando venv do driver"; rm -rf $DV
  CRUN python3 -m venv $DV && CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -U pip && \
  CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -r $REPO/requirements-phase0.txt && touch $DV/.req_$REQ_MD5 || exit 1
fi

# pesos do 4B e corpus: já no /raid (job dl-raid); fora da h1003, puxa pela rede interna
if [ $ON_RAID = 1 ]; then export HF_HUB_CACHE=$R/nlp_artifacts/hf_cache
else
  m=models--google--gemma-3-4b-it
  { raid_pull nlp_artifacts/hf_cache/$m $ST/hf/ --copy-unsafe-links && raid_pull nlp_artifacts/hf_cache/.ok_google--gemma-3-4b-it $ST/hf/; } || exit 1   # sempre (incremental; conserta links quebrados)
  [ -f $CORPUS/train.jsonl ] || raid_pull nlp_artifacts/sae_corpus/ $CORPUS/ || exit 1
fi
[ -f $CORPUS/train.jsonl ] || { say "sem corpus"; exit 1; }
CORP=

# GPU0: vLLM do 4B (fração menor, para dividir com o SAE L29)
CUDA_VISIBLE_DEVICES=${GPU[0]} HF_HUB_OFFLINE=1 setsid apptainer exec --nv -B $WBIND:/w -B /home/$USER $XB "$IMG" \
  /w/out/judge_serve_venv/bin/vllm serve $SUBJ --host 127.0.0.1 --port $P_SUBJ --gpu-memory-utilization 0.40 \
  --max-model-len 4096 --max-num-seqs 128 --limit-mm-per-prompt '{"image": 0}' > $LOGD/vllm_subj.log 2>&1 &
echo $! > $RUN/pg_subj
until curl -sf $(U $P_SUBJ)/health >/dev/null; do
  sleep 10; kill -0 $(cat $RUN/pg_subj) 2>/dev/null || { say "vLLM 4B morreu"; tail -30 $LOGD/vllm_subj.log >&2; exit 1; }
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

say "corpus: $(cat $CORPUS/report.json | tr -d '\n ')"

train() {  # gpu-idx camada  (todos na mesma GPU: buffer menor)
  local gi=$1 L=$2 out=$SAE/gemma-3-4b-it_L${2}_btk16k
  mkdir -p $out
  [ -f $out/DONE ] || CUDA_VISIBLE_DEVICES=${GPU[$gi]} CRUN $VPY -m featgrpo.interp.train_sae train --layer $L --out $out --corpus $CORPUS --buffer 400000 --seqs 16 \
      > $LOGD/sae_L$L.log 2>&1 || { say "SAE L$L falhou"; tail -30 $LOGD/sae_L$L.log >&2; return 1; }
  CUDA_VISIBLE_DEVICES=${GPU[$gi]} CRUN $VPY -m featgrpo.interp.train_sae eval --layer $L --out $out --corpus $CORPUS \
      --ref-gemmascope resid_post/layer_${L}_width_16k_l0_medium,resid_post/layer_${L}_width_16k_l0_small \
      > $LOGD/sae_eval_L$L.log 2>&1 || say "eval L$L falhou"
  [ $ON_RAID = 0 ] && raid_push $out nlp_artifacts/sae_proprio/ && say "SAE L$L → raid"
  say "SAE L$L pronto"
}
train 0 22 & T22=$!
train 0 17 & T17=$!
wait $BEH; say "comportamento rc=$?"
kill -TERM -- -$(cat $RUN/pg_subj) 2>/dev/null; rm -f $RUN/pg_subj; say "vLLM do 4B parado"
train 0 29 & T29=$!
wait $T22 $T17 $T29
say "fim do job de modelo"
