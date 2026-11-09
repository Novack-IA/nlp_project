#!/bin/bash
# Fase 0 completa (phase0-v5-h100: P0 → P8 provisório) em UM job: 2×H100 (h100n2) ou 1×B200 (b200n1).
#
# Por que assim: só o dgx-H100-03 tem /raid/user_gustavonovack (e as GPUs dele estão ocupadas);
# na h100n2/b200n1 o /raid não é nosso. Então o job faz staging local em /tmp/$USER/nlp (pesos, venv
# do vLLM, imagem) e grava dados/cache no repo em /home (NFS, compartilhado).
# Servidores vLLM só em 127.0.0.1, dentro do job. Com 2 GPUs os modelos se revezam em 3 fases:
#   fase 1 (rótulo) : juiz Llama-3.3-70B FP8 TP=2 + Qwen3Guard (GPU0) + WildGuard (GPU1) + embedder
#                     → P0, P4.1 calibração, P1
#   fase 2 (geração): gemma-4-31B (GPU0) → P2; gemma-4-12B (GPU1) → P3, pré-geração do P5, ref gemma;
#                     3 referências (GPU0) → P6 generate
#   fase 3 (rótulo) : juiz + guards → P4, P5 (regeneração só do cache), P6 label, P7 export, P8
# Retomável: cada passo pula se a saída existe; o cache de chamadas fica em data/phase0/cache.
#
# Uso:  sbatch slurm/run_phase0.sbatch                                  (2×H100 na h100n2)
#       sbatch -p b200n1 --gres=gpu:1 slurm/run_phase0.sbatch            (1×B200: tudo na mesma GPU)
#       PHASE0_DATA=data/phase0_dry P1_ARGS='--per-domain 20' sbatch ...   (rodada seca)
#SBATCH --job-name=phase0
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --chdir=/home/user_gustavonovack/nlp_project
#SBATCH -o /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.out
#SBATCH -e /home/user_gustavonovack/nlp_project/slurm-logs/%x-%j.err
set -uo pipefail
ts() { date +%H:%M:%S; }
say() { echo "[$(ts)] $*"; echo "[$(ts)] $*" >&2; }
say "job $SLURM_JOB_ID em $(hostname), GPUs=$CUDA_VISIBLE_DEVICES"

REPO=/home/user_gustavonovack/nlp_project
STAGE_SRC=/home/user_gustavonovack/nlp_stage            # imagem + venv do vLLM copiados do /raid do dgx-H100-03
ST=/tmp/$USER/nlp                                       # staging local do nó
DATA=${PHASE0_DATA:-data/phase0}
LOGD=$REPO/slurm-logs/$SLURM_JOB_ID; mkdir -p $LOGD $ST/w/out $ST/hf $ST/home $ST/cache $ST/tmp $ST/run
IMG=$ST/pytorch-2.8.0-cu128-devel.sif
read -r -a GPU <<< "${CUDA_VISIBLE_DEVICES//,/ }"
BASE=$((20000 + (SLURM_JOB_ID % 2000) * 10))
P_JUDGE=$BASE; P_QG=$((BASE+1)); P_WG=$((BASE+2)); P_EMB=$((BASE+3)); P_GEN=$((BASE+4)); P_SUBJ=$((BASE+5))
P_REF=($((BASE+6)) $((BASE+7)) $((BASE+8)))
U() { echo "http://127.0.0.1:$1"; }

# layout: índices de GPU (dentro do job) e frações de memória por servidor
if [ ${#GPU[@]} -ge 2 ]; then   # 2×H100 80 GB
  J_GI=0,1; J_TP=2; J_U=0.62; QG_GI=0; QG_U=0.28; WG_GI=1; WG_U=0.25; EMB_GI=1; EMB_U=0.05
  GEN_GI=0; GEN_U=0.92; SUBJ_GI=1; SUBJ_U=0.85; REF_GI=0; REF_U=0.30
else                            # 1×B200 180 GB: todos na GPU 0
  J_GI=0; J_TP=1; J_U=0.50; QG_GI=0; QG_U=0.14; WG_GI=0; WG_U=0.13; EMB_GI=0; EMB_U=0.03
  GEN_GI=0; GEN_U=0.45; SUBJ_GI=0; SUBJ_U=0.30; REF_GI=0; REF_U=0.14
fi
say "layout: ${#GPU[@]} GPU(s); juiz TP=$J_TP util=$J_U"

JUDGE=RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic; QG=Qwen/Qwen3Guard-Gen-8B; WG=allenai/wildguard
EMB=Qwen/Qwen3-Embedding-0.6B; GEN=google/gemma-4-31B-it; SUBJ=google/gemma-4-12B-it
REFS=(meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3)

trap 'say "saindo; derrubando servidores"; for f in $ST/run/pg_*; do [ -f "$f" ] && kill -TERM -- -$(cat $f) 2>/dev/null; done; sleep 5; for f in $ST/run/pg_*; do [ -f "$f" ] && kill -KILL -- -$(cat $f) 2>/dev/null; rm -f $f; done; jobs -p | xargs -r kill 2>/dev/null' EXIT

# ---------------------------------------------------------------- staging (idempotente)
if [ ! -f $IMG ]; then say "copiando imagem"; cp $STAGE_SRC/pytorch-2.8.0-cu128-devel.sif $IMG.part && mv $IMG.part $IMG; fi
if [ ! -x $ST/w/out/judge_serve_venv/bin/python ]; then say "extraindo venv do vLLM"; tar xf $STAGE_SRC/judge_serve_venv.tar -C $ST/w/out; fi

set -a; source $REPO/.env; set +a                       # HF_TOKEN
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
export HF_HUB_CACHE=$ST/hf HF_HOME=$ST/hf_home HF_DATASETS_CACHE=$ST/hf_datasets XDG_CACHE_HOME=$ST/cache \
       TMPDIR=$ST/tmp VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_XET_HIGH_PERFORMANCE=1 \
       PYTHONUNBUFFERED=1 PYTHONPATH=$REPO/src PHASE0_DATA=$DATA OMP_NUM_THREADS=4 \
       VLLM_CACHE_ROOT=$ST/cache/vllm TRITON_CACHE_DIR=$ST/cache/triton
CRUN() { apptainer exec --nv -B $ST/w:/w -B /home/$USER "$IMG" "$@"; }
DV=$ST/driver_venv
# a imagem define PIP_CONSTRAINT=/etc/pip/constraint.txt (fsspec==2025.2.0 etc.), que quebra o `datasets`;
# o venv do driver é isolado e instalado sem essa restrição. Recriado se o requirements mudar.
REQ_MD5=$(md5sum $REPO/requirements-phase0.txt | cut -c1-12)
if [ ! -f $DV/.req_$REQ_MD5 ]; then
  say "criando venv do driver"
  rm -rf $DV
  CRUN python3 -m venv $DV && CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -U pip && \
  CRUN env -u PIP_CONSTRAINT $DV/bin/pip install -q -r $REPO/requirements-phase0.txt && touch $DV/.req_$REQ_MD5 \
    || { say "falha no venv do driver"; exit 1; }
  CRUN $DV/bin/python -c "import datasets, fsspec; print('datasets', datasets.__version__, 'fsspec', fsspec.__version__)"
fi
PY() { CRUN $DV/bin/python -m "featgrpo.data.phase0.$1" "${@:2}"; }
CRUN $DV/bin/python -m pytest -q $REPO/tests || { say "testes falharam"; exit 1; }

# downloads em segundo plano, na ordem de uso
( CRUN $DV/bin/python -m featgrpo.data.phase0.download_models $EMB $JUDGE $QG $WG $GEN $SUBJ "${REFS[@]}" ) > $LOGD/download.log 2>&1 &
DL_PID=$!
( PY p0_download --fetch ) > $LOGD/p0_fetch.log 2>&1 &
FETCH_PID=$!

# ---------------------------------------------------------------- servidores
wait_model() {  # espera o marcador do download
  local mk=$HF_HUB_CACHE/.ok_${1//\//--}
  until [ -f $mk ]; do kill -0 $DL_PID 2>/dev/null || [ -f $mk ] || { say "download de $1 falhou (ver $LOGD/download.log)"; return 1; }; sleep 15; done
}
serve() {  # nome modelo porta "idx-gpus" util max_len [args extras do vLLM...]
  local name=$1 model=$2 port=$3 gi=$4 util=$5 len=$6; shift 6
  local devs="" i; for i in ${gi//,/ }; do devs+="${devs:+,}${GPU[$i]}"; done
  wait_model "$model" || return 1
  say "serve $name ($model) gpus=$devs util=$util len=$len"
  CUDA_VISIBLE_DEVICES=$devs HF_HUB_OFFLINE=1 setsid apptainer exec --nv -B $ST/w:/w -B /home/$USER "$IMG" \
    /w/out/judge_serve_venv/bin/vllm serve "$model" --host 127.0.0.1 --port $port \
    --gpu-memory-utilization $util --max-model-len $len "$@" > $LOGD/vllm_$name.log 2>&1 &
  local pid=$!; echo $pid > $ST/run/pg_$name
  until curl -sf $(U $port)/health >/dev/null; do
    sleep 10; kill -0 $pid 2>/dev/null || { say "vLLM $name morreu (ver $LOGD/vllm_$name.log)"; tail -20 $LOGD/vllm_$name.log >&2; return 1; }
  done
  say "pronto $name; $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $devs | tr '\n' ' ')"
}
stop() {  # nome...
  local n; for n in "$@"; do
    [ -f $ST/run/pg_$n ] || continue
    kill -TERM -- -$(cat $ST/run/pg_$n) 2>/dev/null; sleep 8; kill -KILL -- -$(cat $ST/run/pg_$n) 2>/dev/null; rm -f $ST/run/pg_$n
  done
  sleep 5; say "parados: $*; $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $CUDA_VISIBLE_DEVICES | tr '\n' ' ')"
}
JUDGE_ARGS=(--tensor-parallel-size $J_TP --kv-cache-dtype fp8 --max-num-seqs 48)
label_stack() {  # juiz primeiro (a maior fatia), depois os guards
  serve judge $JUDGE $P_JUDGE $J_GI $J_U 16384 "${JUDGE_ARGS[@]}" && \
  serve qg $QG $P_QG $QG_GI $QG_U 8192 && serve wg $WG $P_WG $WG_GI $WG_U 8192
}
LBL=(--qg-url $(U $P_QG) --wg-url $(U $P_WG) --judge-url $(U $P_JUDGE))
D=$REPO/$DATA
step() { say "== $*"; local t0=$SECONDS; "$@"; local rc=$?; say "== fim $1 $2 rc=$rc ($((SECONDS - t0))s)"; return $rc; }
gate() {  # arquivo expressão-sobre-r descrição  (aviso; não para o job)
  CRUN $DV/bin/python -c "import json,sys; r=json.load(open('$D/$1')); sys.exit(0 if ($2) else 1)" \
    && say "gate ok: $3" || say "GATE FALHOU (aviso): $3 (ver $D/$1)"
}

# ---------------------------------------------------------------- fase 1: P0, calibração, P1
if [ ! -f $D/seeds.jsonl ] || [ ! -f $D/calibration.json ]; then
  say "### fase 1"
  serve emb $EMB $P_EMB $EMB_GI $EMB_U 8192 --runner pooling || serve emb $EMB $P_EMB $EMB_GI $EMB_U 8192 --task embed || exit 1
  wait $FETCH_PID || { say "P0 fetch falhou (ver $LOGD/p0_fetch.log)"; exit 1; }
  [ -f $D/p0_report.json ] || step PY p0_download --embed-url $(U $P_EMB) || exit 1
  label_stack || exit 1
  if [ ! -f $D/calibration.json ]; then ( step PY p4_label calibrate "${LBL[@]}" ) > $LOGD/calibrate.log 2>&1 & CAL_PID=$!; else CAL_PID=; fi
  [ -f $D/seeds.jsonl ] || step PY p1_seeds --embed-url $(U $P_EMB) --judge-url $(U $P_JUDGE) ${P1_ARGS:-} || exit 1
  [ -n "$CAL_PID" ] && { wait $CAL_PID; tail -40 $LOGD/calibrate.log >&2; }
  stop judge qg wg
fi
grep -q '"passed": true' $D/calibration.json || { say "GATE S3 FALHOU: calibração (ver $D/calibration.json)"; exit 1; }
say "gate S3 ok: calibração"

# ---------------------------------------------------------------- fase 2: P2, P3, pré-geração P5, P6 generate
p6_done() { local m; for m in $SUBJ "${REFS[@]}"; do [ -f $D/p6/${m##*/}.json ] || return 1; done; }
if [ ! -f $D/groups_p3.jsonl ] || [ ! -f $D/p5_pregen.ok ] || ! p6_done; then
  say "### fase 2"
  [ -f $ST/run/pg_emb ] || serve emb $EMB $P_EMB $EMB_GI $EMB_U 8192 --runner pooling || serve emb $EMB $P_EMB $EMB_GI $EMB_U 8192 --task embed || exit 1
  if [ ! -f $D/groups_p2.jsonl ]; then
    serve gen $GEN $P_GEN $GEN_GI $GEN_U 2048 --max-num-seqs 128 || exit 1
    serve subj $SUBJ $P_SUBJ $SUBJ_GI $SUBJ_U 4096 --max-num-seqs 128 &       # sobe a 12B na GPU1 enquanto o P2 roda
    SUBJ_UP=$!
    step PY p2_border --gen-url $(U $P_GEN) --embed-url $(U $P_EMB) || exit 1
    stop gen
    wait $SUBJ_UP || exit 1
  else
    serve subj $SUBJ $P_SUBJ $SUBJ_GI $SUBJ_U 4096 --max-num-seqs 128 || exit 1
  fi
  stop emb
  ( i=0; for m in "${REFS[@]}"; do serve ref$i $m ${P_REF[$i]} $REF_GI $REF_U 4096 --max-num-seqs 64 || exit 1; i=$((i+1)); done ) &
  REFS_UP=$!
  [ -f $D/groups_p3.jsonl ] || step PY p3_responses --url $(U $P_SUBJ) || exit 1
  PIDS=()
  [ -f $D/p5_pregen.ok ] || { ( step PY p5_validate --pregen --subject-url $(U $P_SUBJ) && touch $D/p5_pregen.ok ) & PIDS+=($!); }
  [ -f $D/p6/${SUBJ##*/}.json ] || { ( step PY p6_hardness generate --model $SUBJ --url $(U $P_SUBJ) ) & PIDS+=($!); }
  wait $REFS_UP || exit 1
  i=0; for m in "${REFS[@]}"; do
    [ -f $D/p6/${m##*/}.json ] || { ( step PY p6_hardness generate --model $m --url $(U ${P_REF[$i]}) ) & PIDS+=($!); }
    i=$((i+1))
  done
  for p in "${PIDS[@]}"; do wait $p || { say "fase 2: um passo falhou"; exit 1; }; done
  stop subj ref0 ref1 ref2
fi

# ---------------------------------------------------------------- fase 3: P4, P5, P6 label, P7, P8
say "### fase 3"
label_stack || exit 1
[ -f $D/groups_p4.jsonl ] || step PY p4_label label "${LBL[@]}" || exit 1
gate groups_p4_report.json "r['incoherent_subject_rate'] < 0.05" "S7 incoerência do sujeito < 5%"
[ -f $D/groups_p5.jsonl ] || step PY p5_validate "${LBL[@]}" || exit 1
gate groups_p5_report.json "r['valid'] >= 450" "S9 ≥ 450 grupos válidos"
[ -f $D/groups_p6.jsonl ] || step PY p6_hardness label "${LBL[@]}" || exit 1
stop judge qg wg
[ -f $D/audit/assignment.json ] || step PY p7_audit export || exit 1
step PY p8_finalize || exit 1
say "fim; relatório em $D/final/report.md"
