#!/bin/bash
# Rodada (S4 seca ou S5–S6 real) — job de GERAÇÃO: P1 → P2.
#   P1: embedder na GPU + juiz (sub-buckets de bias); sem vLLM.
#   P2: gemma-4-31B-it sobe UMA vez (0.78; o embedder roda no mesmo nó).
# Seca:  PHASE0_DATA=data/phase0_dry sbatch --export=ALL,P1_ARGS='--per-domain 20' slurm/run_gen.sbatch
# Real:  sbatch slurm/run_gen.sbatch
#SBATCH --job-name=gen
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=97G                        # máximo que o Slurm aceita (RealMemory=100000M)
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/beegfs/home/user_gustavonovack/nlp
#SBATCH -o /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.out
#SBATCH -e /beegfs/home/user_gustavonovack/nlp/slurm-logs/%x-%j.err
set -uo pipefail
echo "[job $SLURM_JOB_ID] $(hostname) $(uname -m) inicio=$(date -Is) PHASE0_DATA=${PHASE0_DATA:-data/phase0}" >&2

# Vigia de memória: o Slurm daqui NÃO aplica --mem. Cancela o job se o
# consumo do nó (queda do MemAvailable, inclui GPU) passar do pedido.
LIMITE_KB=$((112 * 1024 * 1024))         # vigia: 112G de ~117 GiB livres no nó (o Slurm não aplica --mem)
A0=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
( while sleep 5; do
    A=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if (( A0 - A > LIMITE_KB )); then
      echo "[vigia] $(( (A0 - A) / 1024 ))M > --mem; cancelando" >&2
      scancel "$SLURM_JOB_ID"; exit
    fi
  done ) &
trap 'kill $(jobs -p) 2>/dev/null; pkill -9 -u $USER -f -i vllm; sleep 5; rm -rf /tmp/$USER-enroot' EXIT

# ---- container: imagem vLLM NGC, repo e HF_HOME montados ----
export ENROOT_CACHE_PATH=/tmp/$USER-enroot/cache ENROOT_DATA_PATH=/tmp/$USER-enroot/data ENROOT_TEMP_PATH=/tmp/$USER-enroot/tmp
enroot create -n pipe /beegfs/home/$USER/images/vllm-26.09-py3.sqsh
set -a; source .env; set +a
RUN="enroot start -m /beegfs/home/$USER:/beegfs/home/$USER \
  -e HOME=/beegfs/home/$USER -e PYTHONUNBUFFERED=1 -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/beegfs/home/$USER/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/beegfs/home/$USER/nlp/src -e PHASE0_DATA=${PHASE0_DATA:-data/phase0} pipe"
PY="$RUN /beegfs/home/$USER/nlp/.venv/bin/python -m featgrpo.data.phase0"

# ---- servidores vLLM ----
# Na GB10 o CUDA só enxerga o MemFree: o page cache dos pesos já lidos não conta
# como livre e faz o próximo servidor falhar com CUDA OOM. drop_cache tira do
# cache os arquivos de pesos e os do container descompactado em /tmp (~40 GiB de
# cache já no início do job) — fadvise DONTNEED via dd nocache; não precisa de root.
drop_cache() {
  find /beegfs/home/$USER/.cache/huggingface/hub/blobs /tmp/$USER-enroot -type f -size +1M -print0 2>/dev/null |
    while IFS= read -r -d '' f; do dd if="$f" iflag=nocache count=0 status=none 2>/dev/null; done
}
mem() { awk '/MemFree/ {f=$2} /MemAvailable/ {a=$2} END {printf "MemFree %.1f GiB, MemAvailable %.1f GiB", f/1048576, a/1048576}' /proc/meminfo; }
# O drop_cache não alcança todo o cache (milhares de arquivos pequenos). reclaim
# força o kernel a devolver o resto: se o MemFree não cobre o que o vLLM vai pedir
# (fração × 121.7 GiB + 3 GiB), aloca e preenche esse total por um instante e solta.
reclaim() {  # $1 fração da memória da GPU
  local need free avail x
  need=$(awk -v u="$1" 'BEGIN {printf "%d", u * 121.7 + 3}')
  free=$(awk '/MemFree/ {printf "%d", $2/1048576}' /proc/meminfo)
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  (( free >= need )) && return 0                    # já há MemFree suficiente
  x=$need; (( x > avail - 2 )) && x=$(( avail - 2 ))   # aloca o TOTAL: só assim o kernel precisa devolver cache
  (( x > 0 )) && python3 -c "import sys; b = b'\x01' * (int(sys.argv[1]) << 30); del b" "$x"
}
serve() {  # $1 modelo  $2 porta  $3 fração da memória da GPU  $4 max-model-len
  local tag="[vllm ${1##*/}]"   # prefixo das linhas do servidor no .out/.err do job
  echo "== vllm serve $1 (porta $2, util $3, len $4) $(date -Is)"
  drop_cache; reclaim "$3"; echo "   memória antes: $(mem)"
  $RUN vllm serve "$1" --host 127.0.0.1 --port "$2" --gpu-memory-utilization "$3" \
    --max-model-len "$4" --dtype bfloat16 \
    > >(sed -u "s/^/$tag /") 2> >(sed -u "s/^/$tag /" >&2) &
  until curl -sf "http://127.0.0.1:$2/health" >/dev/null; do
    sleep 10
    pgrep -u $USER -f "vllm serve $1" >/dev/null || { echo "vllm $1 morreu; ver as linhas $tag no .err"; return 1; }
  done
  drop_cache; echo "   pronto $(date -Is); memória depois: $(mem)"
}
stop_all() {  # mata o servidor e o processo filho VLLM::EngineCore, e espera a memória voltar
  pkill -u $USER -f -i "vllm"; sleep 5; pkill -9 -u $USER -f -i "vllm"
  while pgrep -u $USER -f -i "vllm" >/dev/null; do sleep 2; done
  echo "   servidores parados; memória: $(mem)"
}

# ---- trabalho ----
echo "== P1 $(date -Is)"
$PY.p1_seeds ${P1_ARGS:-} || exit 1
echo "== P2 embeddings (GPU livre, antes do gerador) $(date -Is)"
$PY.p2_border --precompute || exit 1
echo "== P2 $(date -Is)"
serve google/gemma-4-31B-it 8000 0.78 2048 || exit 1
$PY.p2_border --url http://127.0.0.1:8000 || exit 1
echo "fim=$(date -Is)"
