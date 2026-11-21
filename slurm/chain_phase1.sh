#!/bin/bash
# Encadeia, a partir do login (h1003), os passos manuais da frente A dentro dos jobs já alocados.
# DATA=<job de dados>  MODEL=<job de modelo>
cd ~/nlp_project; D=data/phase1; U=http://127.0.0.1
DP=$((20000 + (DATA % 2000) * 10)); MP=$((21000 + (MODEL % 2000) * 10))
L=slurm-logs/$DATA/chain.log
say() { echo "[$(date +%H:%M:%S)] $*" >> $L; }
att() { srun --jobid=$1 --overlap -n1 bash slurm/attach.sh "${@:2}" >> $L 2>&1; }
# 1. respostas do 4B às variantes da mineração (job de modelo)
until [ -f $D/mining_variants.jsonl ]; do sleep 30; done
say "variantes prontas; gerando respostas do 4B"
[ -f $D/behavior_raw_mining.jsonl ] || att $MODEL featgrpo.data.phase1.t4_behavior generate --set mining --url $U:$MP
say "respostas da mineração: $(wc -l < $D/behavior_raw_mining.jsonl)"
# 2. 2×2 completo → controles → libera o comportamento das células
until [ -f datasets/phase1/cells_2x2.jsonl ]; do sleep 30; done
say "2×2 pronto: $(wc -l < datasets/phase1/cells_2x2.jsonl) grupos"
att $DATA featgrpo.data.phase1.t3_controls_external controls && touch $D/CELLS_DONE && say "controles ok; CELLS_DONE"
# 3. rótulo do comportamento nas células (o job de dados já passou desse ponto do roteiro)
until [ -f $D/behavior_raw_cells.jsonl ]; do sleep 30; done
[ -f $D/behavior_cells.jsonl ] || att $DATA featgrpo.data.phase1.t4_behavior label --set cells --qg-url $U:$((DP+1)) --wg-url $U:$((DP+2)) --judge-url $U:$DP
say "comportamento das células rotulado"
say "fim da cadeia"
