#!/bin/bash
# Fase 1, frente A — caminho (b): 2×2 do plano B com reescrita em ESTILO COMUM (t2d_style) + QC.
#   1×B200:  sbatch -p b200n1 --gres=gpu:1 slurm/run_phase1_style.sbatch   (tudo numa GPU)
#   2×H100:  sbatch -p h100n2 --gres=gpu:2 slurm/run_phase1_style.sbatch   (layout do job de dados)
# Se dois forem submetidos, o primeiro que começar pega a trava ($D1/STYLE_LOCK) e o outro sai.
# Passos (retomáveis): smoke (40 grupos, fora do test) + QC  →  espera $D1/STYLE_GO_FULL (ou STYLE_STOP)
#   →  completo (2.428) → datasets/phase1/cells_2x2_style.jsonl  →  QC
#SBATCH --job-name=p1-style
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
NG=${#GPU[@]}
BASE=$((20000 + (SLURM_JOB_ID % 2000) * 10))
freeport() { local p=$1; while (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null || ss -ltn 2>/dev/null | grep -q ":$p "; do p=$((p+97)); done; echo $p; }
P_JUDGE=$(freeport $BASE); P_QG=$(freeport $((BASE+1))); P_WG=$(freeport $((BASE+2))); P_EMB=$(freeport $((BASE+3))); P_GEN=$((BASE+4))
echo "[portas] juiz=$P_JUDGE qg=$P_QG wg=$P_WG emb=$P_EMB" >&2
U() { echo "http://127.0.0.1:$1"; }
JUDGE=RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic; QG=Qwen/Qwen3Guard-Gen-8B; WG=allenai/wildguard
EMB=Qwen/Qwen3-Embedding-0.6B; GEN=$JUDGE   # gerador = juiz (internet lenta: sem gemma-4-31B)

mkdir $REPO/$D1/STYLE_LOCK 2>/dev/null || { say "outro job do caminho (b) já está rodando; saindo"; exit 0; }
trap 'rmdir $REPO/$D1/STYLE_LOCK 2>/dev/null; say "saindo; derrubando servidores"; for f in $RUN/pg_*; do [ -f "$f" ] && kill -TERM -- -$(cat $f) 2>/dev/null; done; sleep 5; for f in $RUN/pg_*; do [ -f "$f" ] && kill -KILL -- -$(cat $f) 2>/dev/null; rm -f $f; done; jobs -p | xargs -r kill 2>/dev/null' EXIT

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

# pesos: fora da h1003, copia só o snapshot de cada modelo do /raid (arquivos reais, -L) para $ST/models/<id>
# e serve pelo caminho (--served-model-name mantém o id). Na h1003, usa o cache HF do /raid direto.
MP() { if [ $ON_RAID = 1 ]; then echo "$1"; else echo "$ST/models/$1"; fi; }
if [ $ON_RAID = 0 ]; then
  for m in $JUDGE $QG $WG; do
    say "puxando $m"
    raid_pull "nlp_artifacts/hf_cache/models--${m//\//--}/snapshots/*/" "$ST/models/$m/" -L || exit 1
  done
fi
wait_model() { return 0; }
serve() {  # nome modelo porta idx-gpus util max_len [args...]
  local name=$1 model=$2 port=$3 gi=$4 util=$5 len=$6; shift 6
  local devs="" i; for i in ${gi//,/ }; do devs+="${devs:+,}${GPU[$i]}"; done
  wait_model "$model" || return 1
  say "serve $name ($model) gpus=$devs util=$util"
  CUDA_VISIBLE_DEVICES=$devs HF_HUB_OFFLINE=1 setsid apptainer exec --nv -B $WB:/w $XB -B /home/$USER "$IMG" \
    /w/out/judge_serve_venv/bin/vllm serve "$(MP $model)" --served-model-name "$model" --host 127.0.0.1 --port $port \
    --gpu-memory-utilization $util --max-model-len $len "$@" > $LOGD/vllm_$name.log 2>&1 &
  local pid=$!; echo $pid > $RUN/pg_$name
  until curl -sf $(U $port)/health >/dev/null; do
    sleep 10; kill -0 $pid 2>/dev/null || { say "vLLM $name morreu"; tail -20 $LOGD/vllm_$name.log >&2; return 1; }
  done
  say "pronto $name; $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $devs | tr '\n' ' ')"
}
step() { say "== $*"; local t0=$SECONDS; "$@"; local rc=$?; say "== fim $2 rc=$rc ($((SECONDS - t0))s)"; return $rc; }

# servidores (juiz primeiro; gerador em paralelo na GPU2)
if [ $NG -ge 2 ]; then
  serve judge $JUDGE $P_JUDGE 0,1 0.62 16384 --tensor-parallel-size 2 --kv-cache-dtype fp8 --max-num-seqs 64 || exit 1
  serve qg $QG $P_QG 0 0.28 8192 || exit 1
  serve wg $WG $P_WG 1 0.25 8192 || exit 1
else   # uma B200 (~180 GB): tudo na GPU 0
  serve judge $JUDGE $P_JUDGE 0 0.55 16384 --kv-cache-dtype fp8 --max-num-seqs 64 || exit 1
  serve qg $QG $P_QG 0 0.12 8192 || exit 1
  serve wg $WG $P_WG 0 0.12 8192 || exit 1
fi
LBL=(--qg-url $(U $P_QG) --wg-url $(U $P_WG) --judge-url $(U $P_JUDGE))
GENA=(--gen-url $(U $P_JUDGE) "${LBL[@]}")
D=$REPO/$D1; O=$REPO/datasets/phase1

step PY t2d_style "${GENA[@]}" --tag style_smoke --n 40 || exit 1
[ -f $D/cells_style_smoke_qc_report.json ] || step PY t2c_qc --judge-url $(U $P_JUDGE) --path $D/cells_style_smoke.jsonl --report cells_style_smoke_qc_report.json || say "qc smoke falhou"
say "smoke: $(CRUN $DV/bin/python -c "import json;r=json.load(open('$D/cells_style_smoke_report.json'));print(r['ok'], r['groups'], round(r['yield'],2))")"
touch $D/STYLE_SMOKE_DONE
say "esperando $D/STYLE_GO_FULL (ou STYLE_STOP)"
until [ -f $D/STYLE_GO_FULL ] || [ -f $D/STYLE_STOP ]; do sleep 60; done
[ -f $D/STYLE_STOP ] && { say "STYLE_STOP: fim sem o completo"; exit 0; }

[ -f $O/cells_2x2_style.jsonl ] || step PY t2d_style "${GENA[@]}" --tag style_full --workers 96 || exit 1
say "completo: $(CRUN $DV/bin/python -c "import json;r=json.load(open('$D/cells_style_full_report.json'));print(r['ok'], r['groups'], round(r['yield'],2))")"
head -1 $O/cells_2x2_style.jsonl | grep -q '"qc_pass"' || step PY t2c_qc --judge-url $(U $P_JUDGE) --path $O/cells_2x2_style.jsonl --report cells_style_full_qc_report.json || say "qc completo falhou"
touch $D/STYLE_FULL_DONE
while [ -f $D/KEEP_SERVERS ]; do sleep 120; done
say "fim do job do caminho (b)"
