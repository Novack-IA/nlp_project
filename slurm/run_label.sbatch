#!/bin/bash
# Rodada (S4 seca ou S7–S10 real) — job de SUJEITO + RÓTULOS: P3 → P4 → P5 → P6.
#   Fase A: gemma-4-12B-it (0.40) + Qwen3Guard (0.20) + WildGuard (0.20) no ar juntos;
#           P3, P4, P5 e a referência gemma-12B do P6 sem recarregar nada.
#   Fase B: guards continuam no ar; a 12B sai e as 3 outras referências sobem
#           uma por vez (0.38) para o P6; por fim, rótulos do P6.
#   Pausa sozinho por cota do Claude e retoma do cache; se cair, basta reenviar.
# Seca:  PHASE0_DATA=data/phase0_dry sbatch slurm/run_label.sbatch
# Real:  sbatch slurm/run_label.sbatch
#SBATCH --job-name=label
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=97G                        # máximo que o Slurm aceita (RealMemory=100000M)
#SBATCH --time=3-00:00:00
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
declare -A PG   # porta → grupo de processos do servidor
serve() {  # $1 modelo  $2 porta  $3 fração da memória da GPU  $4 max-model-len
  local tag="[vllm ${1##*/}]"   # prefixo das linhas do servidor no .out/.err do job
  echo "== vllm serve $1 (porta $2, util $3, len $4) $(date -Is)"
  drop_cache; reclaim "$3"; echo "   memória antes: $(mem)"
  setsid $RUN vllm serve "$1" --host 127.0.0.1 --port "$2" --gpu-memory-utilization "$3" \
    --max-model-len "$4" --dtype bfloat16 \
    > >(sed -u "s/^/$tag /") 2> >(sed -u "s/^/$tag /" >&2) &
  PG[$2]=$!   # grupo de processos do servidor (inclui o VLLM::EngineCore)
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
SUBJ=http://127.0.0.1:8000; QG=http://127.0.0.1:8001; WG=http://127.0.0.1:8002

$RUN bash -c 'cd /beegfs/home/$USER/nlp && .venv/bin/python -m pytest -q tests/' || exit 1   # testes antes de subir modelos
echo "== fase A: sujeito + guards $(date -Is)"
serve google/gemma-4-12B-it 8000 0.40 4096 || exit 1
serve Qwen/Qwen3Guard-Gen-8B 8001 0.20 4096 || exit 1
serve allenai/wildguard 8002 0.20 4096 || exit 1
echo "== P3 $(date -Is)"; $PY.p3_responses --url $SUBJ || exit 1
echo "== P4 $(date -Is)"; $PY.p4_label label --qg-url $QG --wg-url $WG || exit 1

# gates da §10 (só na rodada real; a seca tem 80 candidatos)
DATA=${PHASE0_DATA:-data/phase0}
gate() {  # $1 relatório  $2 expressão Python sobre r (o JSON)  $3 descrição
  [ "$DATA" = data/phase0 ] || return 0
  $RUN /beegfs/home/$USER/nlp/.venv/bin/python -c "import json,sys; r=json.load(open('/beegfs/home/$USER/nlp/$DATA/$1')); sys.exit(0 if ($2) else 1)" \
    && echo "   gate ok: $3" || { echo "   GATE FALHOU: $3 (ver $DATA/$1)"; exit 1; }
}
gate groups_p4_report.json "r['incoherent_subject_rate'] < 0.05" "S7: incoerência do sujeito < 5%"
echo "== P5 $(date -Is)"; $PY.p5_validate --subject-url $SUBJ --qg-url $QG --wg-url $WG || exit 1
gate groups_p5_report.json "r['valid'] >= 450" "S9: ≥ 450 grupos válidos"
echo "== P6 ref gemma-4-12B-it $(date -Is)"; $PY.p6_hardness generate --model google/gemma-4-12B-it --url $SUBJ || exit 1

echo "== fase B: referências (guards continuam no ar) $(date -Is)"
stop_one() {  # derruba só o servidor da porta $1 (grupo inteiro, com o EngineCore) e espera a memória voltar
  kill -TERM -- -${PG[$1]} 2>/dev/null; sleep 10; kill -KILL -- -${PG[$1]} 2>/dev/null
  while pgrep -g ${PG[$1]} >/dev/null; do sleep 2; done
  echo "   porta $1 parada; memória: $(mem)"
}
stop_one 8000
for M in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3; do
  serve "$M" 8000 0.38 4096 || exit 1
  echo "== P6 ref $M $(date -Is)"; $PY.p6_hardness generate --model "$M" --url $SUBJ || exit 1
  stop_one 8000
done
echo "== P6 rótulos $(date -Is)"; $PY.p6_hardness label --qg-url $QG --wg-url $WG || exit 1
echo "fim=$(date -Is)"
