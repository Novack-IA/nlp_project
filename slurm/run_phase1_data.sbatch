#!/bin/bash
# Fase 1, frente A — dados (3×H100). Na h100n3 (dgx-H100-03) usa imagem, venv do vLLM e pesos do /raid
# (nlp_artifacts/hf_cache); em outro nó, staging em /tmp/$USER/nlp como na Fase 0.
#   sbatch -p h100n3 slurm/run_phase1_data.sbatch
#   GPU0+1: juiz Llama-3.3-70B FP8 (TP=2) + Qwen3Guard (GPU0) + WildGuard (GPU1) + embedder (GPU1)
#   gerador dos moldes = o próprio juiz (Llama-3.3-70B); a internet lenta inviabilizou o gemma-4-31B
# Passos (retomáveis; cache de chamadas compartilhado com a Fase 0):
#   t1 tipologia ∥ externos (embedder)  →  t2 rodada seca (50)  →  gate rendimento ≥ 40%  →  t2 completo
#   → controles  →  rotulagem do comportamento do 4B (arquivos gerados pelo job de modelo, via NFS)
#SBATCH --job-name=p1-data
#SBATCH --partition=h100n2,b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=30:00:00
#SBATCH --chdir=/home/user_gustavonovack/nlp_project
#SBATCH -o /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.out
#SBATCH -e /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.err
set -uo pipefail
ts() { date +%H:%M:%S; }
say() { echo "[$(ts)] $*"; echo "[$(ts)] $*" >&2; }
say "job $SLURM_JOB_ID em $(hostname), GPUs=$CUDA_VISIBLE_DEVICES"

REPO=/home/user_gustavonovack/nlp_project
ST=/tmp/$USER/nlp
D1=data/phase1
LOGD=$REPO/slurm-logs/$SLURM_JOB_ID; RUN=$ST/run/$SLURM_JOB_ID   # pids por job: o trap não mata servidores de outro job
mkdir -p $LOGD $ST/w/out $ST/hf $ST/home $ST/cache $ST/tmp $RUN $REPO/$D1
IMG=$ST/pytorch-2.8.0-cu128-devel.sif; WB=$ST/w; XB=""
R=/raid/user_gustavonovack
if [ "$(hostname -s)" = dgx-H100-03 ]; then
  IMG=$R/images/pytorch-2.8.0-cu128-devel.sif; WB=$R/feat-grpo; XB="-B $R"; ON_RAID=1
else ON_RAID=0; fi
read -r -a GPU <<< "${CUDA_VISIBLE_DEVICES//,/ }"
[ ${#GPU[@]} -ge 2 ] || { say "precisa de 2 GPUs"; exit 1; }
BASE=$((20000 + (SLURM_JOB_ID % 2000) * 10))
P_JUDGE=$BASE; P_QG=$((BASE+1)); P_WG=$((BASE+2)); P_EMB=$((BASE+3)); P_GEN=$((BASE+4))
U() { echo "http://127.0.0.1:$1"; }
JUDGE=RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic; QG=Qwen/Qwen3Guard-Gen-8B; WG=allenai/wildguard
EMB=Qwen/Qwen3-Embedding-0.6B; GEN=$JUDGE   # gerador = juiz (internet lenta: sem gemma-4-31B)

trap 'say "saindo; derrubando servidores"; for f in $RUN/pg_*; do [ -f "$f" ] && kill -TERM -- -$(cat $f) 2>/dev/null; done; sleep 5; for f in $RUN/pg_*; do [ -f "$f" ] && kill -KILL -- -$(cat $f) 2>/dev/null; rm -f $f; done; jobs -p | xargs -r kill 2>/dev/null' EXIT

source $REPO/slurm/lib/raid.sh
if [ $ON_RAID = 0 ] && [ ! -f $IMG ]; then say "puxando imagem do raid"; raid_pull images/pytorch-2.8.0-cu128-devel.sif $ST/ --size-only || exit 1; fi
if [ $ON_RAID = 0 ] && [ ! -x $ST/w/out/judge_serve_venv/bin/python ]; then say "puxando venv do vLLM"; raid_pull feat-grpo/out/judge_serve_venv $ST/w/out/ || exit 1; fi

set -a; source $REPO/.env; set +a
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
HFC=$ST/hf; [ $ON_RAID = 1 ] && HFC=$R/nlp_artifacts/hf_cache
export HF_HUB_DISABLE_SHARED_BLOBS=1 HF_HUB_CACHE=$HFC HF_HOME=$ST/hf_home HF_DATASETS_CACHE=$ST/hf_datasets XDG_CACHE_HOME=$ST/cache \
       TMPDIR=$ST/tmp VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_XET_HIGH_PERFORMANCE=1 \
       PYTHONUNBUFFERED=1 PYTHONPATH=$REPO/src OMP_NUM_THREADS=4 \
       VLLM_CACHE_ROOT=$ST/cache/vllm TRITON_CACHE_DIR=$ST/cache/triton \
       PHASE0_CONFIG=configs/data/phase0_v5b.yaml PHASE0_CACHE=data/phase0_cache PHASE0_RAW=data/phase0_raw \
       PHASE0_DATA=data/phase0_v5b PHASE1_DATA=$D1 PHASE1_GEN_MODEL=$JUDGE
CRUN() { apptainer exec --nv -B $WB:/w $XB -B /home/$USER "$IMG" "$@"; }
DV=$ST/driver_venv
REQ_MD5=$(md5sum $REPO/requirements-phase0.txt | cut -c1-12)
if [ ! -f $DV/.req_$REQ_MD5 ]; then
  say "criando venv do driver"; rm -rf $DV
  CRUN python3 -m venv $DV && CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -U pip && \
  CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -r $REPO/requirements-phase0.txt && touch $DV/.req_$REQ_MD5 || exit 1
fi
PY() { CRUN $DV/bin/python -m "featgrpo.data.phase1.$1" "${@:2}"; }
CRUN $DV/bin/python -m pytest -q $REPO/tests/test_phase1.py || { say "testes da fase 1 falharam"; exit 1; }

if [ $ON_RAID = 0 ]; then   # pesos vêm do /raid do h1003 (rede interna), não da internet
  for m in $EMB $JUDGE $QG $WG; do d=models--${m//\//--}
    { raid_pull nlp_artifacts/hf_cache/$d $HFC/ --copy-unsafe-links && raid_pull nlp_artifacts/hf_cache/.ok_${m//\//--} $HFC/; } || exit 1
  done
fi
( CRUN $DV/bin/python -m featgrpo.data.phase0.download_models $EMB $JUDGE $QG $WG ) > $LOGD/download.log 2>&1 &
DL_PID=$!
wait_model() { local mk=$HF_HUB_CACHE/.ok_${1//\//--}
  until [ -f $mk ]; do kill -0 $DL_PID 2>/dev/null || [ -f $mk ] || { say "download de $1 falhou"; return 1; }; sleep 15; done; }
serve() {  # nome modelo porta idx-gpus util max_len [args...]
  local name=$1 model=$2 port=$3 gi=$4 util=$5 len=$6; shift 6
  local devs="" i; for i in ${gi//,/ }; do devs+="${devs:+,}${GPU[$i]}"; done
  wait_model "$model" || return 1
  say "serve $name ($model) gpus=$devs util=$util"
  CUDA_VISIBLE_DEVICES=$devs HF_HUB_OFFLINE=1 setsid apptainer exec --nv -B $WB:/w $XB -B /home/$USER "$IMG" \
    /w/out/judge_serve_venv/bin/vllm serve "$model" --host 127.0.0.1 --port $port \
    --gpu-memory-utilization $util --max-model-len $len "$@" > $LOGD/vllm_$name.log 2>&1 &
  local pid=$!; echo $pid > $RUN/pg_$name
  until curl -sf $(U $port)/health >/dev/null; do
    sleep 10; kill -0 $pid 2>/dev/null || { say "vLLM $name morreu"; tail -20 $LOGD/vllm_$name.log >&2; return 1; }
  done
  say "pronto $name; $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $devs | tr '\n' ' ')"
}
step() { say "== $*"; local t0=$SECONDS; "$@"; local rc=$?; say "== fim $2 rc=$rc ($((SECONDS - t0))s)"; return $rc; }

# servidores (juiz primeiro; gerador em paralelo na GPU2)
serve judge $JUDGE $P_JUDGE 0,1 0.62 16384 --tensor-parallel-size 2 --kv-cache-dtype fp8 --max-num-seqs 64 || exit 1
serve qg $QG $P_QG 0 0.28 8192 || exit 1
serve wg $WG $P_WG 1 0.25 8192 || exit 1
serve emb $EMB $P_EMB 1 0.05 8192 --runner pooling || serve emb $EMB $P_EMB 1 0.05 8192 --task embed || exit 1
LBL=(--qg-url $(U $P_QG) --wg-url $(U $P_WG) --judge-url $(U $P_JUDGE))
D=$REPO/$D1; O=$REPO/datasets/phase1

( [ -f $O/external/xstest.jsonl ] || step PY t3_controls_external external --embed-url $(U $P_EMB) ) > $LOGD/external.log 2>&1 &
EXT=$!
[ -f $O/trigger_types.jsonl ] || step PY t1_types --judge-url $(U $P_JUDGE) || exit 1
CELLS=(--gen-url $(U $P_JUDGE) "${LBL[@]}")
step PY t2_cells "${CELLS[@]}" --tag dry --n 50 || exit 1
Y=$(CRUN $DV/bin/python -c "import json;print(json.load(open('$D/cells_dry_report.json'))['yield'])")
say "rendimento da rodada seca: $Y"
if CRUN $DV/bin/python -c "import sys;sys.exit(0 if $Y >= 0.40 else 1)"; then
  touch $D/GATE_DRY_OK
  [ -f $O/cells_2x2.jsonl ] || step PY t2_cells "${CELLS[@]}" --tag full || exit 1
  step PY t3_controls_external controls || exit 1
  touch $D/CELLS_DONE
else
  touch $D/GATE_DRY_FAIL; say "GATE: rendimento < 40% — 2×2 completo NÃO disparado (plano B)"
fi
wait $EXT; say "externos rc=$?"

# rotulagem do comportamento do 4B (o job de modelo escreve behavior_raw_<set>.jsonl)
for s in pairs external cells; do
  [ "$s" = cells ] && [ ! -f $D/CELLS_DONE ] && continue
  until [ -f $D/behavior_raw_$s.jsonl ]; do sleep 60; done
  [ -f $D/behavior_$s.jsonl ] || step PY t4_behavior label --set $s "${LBL[@]}" || say "rótulo $s falhou"
done
say "fim do job de dados"
