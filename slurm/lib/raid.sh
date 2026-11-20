# shellcheck shell=bash
# Acesso ao /raid do dgx-H100-03 (h1003) a partir de QUALQUER nó do cluster.
#
# O /raid é disco local de cada nó; só o dgx-H100-03 tem /raid/$USER. Em outros nós, o acesso é
# rsync sobre SSH (mesmo método dos scripts de admin em /cm/shared/scripts). A rede entre os nós
# é 1 GbE (~60–110 MB/s), então use para dados e resultados, não para centenas de GB de pesos.
#
# Uso (dentro de um job):
#   source slurm/lib/raid.sh
#   raid_pull nlp_project/data/phase0_v5b/final  /tmp/$USER/work/       # h1003:/raid → local
#   raid_push /tmp/$USER/work/saida/             nlp_project/data/run_X/ # local → h1003:/raid
#   raid_ls   nlp_project/data
#
# Caminhos remotos são relativos a $RAID_ROOT. No próprio dgx-H100-03 vira cópia local.

RAID_HOST=${RAID_HOST:-dgx-H100-03}
RAID_ROOT=${RAID_ROOT:-/raid/$USER}
_RAID_CTL="/tmp/$USER-raid-ssh-%C"
RAID_SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=10
          -o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPath="$_RAID_CTL" -o ControlPersist=15m)

raid_is_local() { [ "$(hostname -s)" = "$RAID_HOST" ]; }

_raid_rsync() {  # opções comuns: arquivamento, retomada, sem dono/grupo remoto
  rsync -a --no-owner --no-group --partial --partial-dir=.rsync-partial --info=stats1 "$@"
}

raid_pull() {  # <remoto relativo> <destino local> [opções extras do rsync]
  local src=$1 dst=$2; shift 2
  mkdir -p "$dst"
  if raid_is_local; then _raid_rsync "$@" "$RAID_ROOT/$src" "$dst"
  else _raid_rsync -e "${RAID_SSH[*]}" "$@" "$RAID_HOST:$RAID_ROOT/$src" "$dst"; fi
}

raid_push() {  # <origem local> <destino remoto relativo> [opções extras do rsync]
  local src=$1 dst=$2; shift 2
  if raid_is_local; then mkdir -p "$RAID_ROOT/$dst"; _raid_rsync "$@" "$src" "$RAID_ROOT/$dst"
  else "${RAID_SSH[@]}" "$RAID_HOST" mkdir -p "$RAID_ROOT/$dst" && _raid_rsync -e "${RAID_SSH[*]}" "$@" "$src" "$RAID_HOST:$RAID_ROOT/$dst"; fi
}

raid_ls() {  # <remoto relativo>
  if raid_is_local; then ls -la "$RAID_ROOT/${1:-}"; else "${RAID_SSH[@]}" "$RAID_HOST" ls -la "$RAID_ROOT/${1:-}"; fi
}

raid_quota() {  # cota do usuário no /raid do h1003
  if raid_is_local; then quota -s 2>/dev/null; else "${RAID_SSH[@]}" "$RAID_HOST" quota -s 2>/dev/null; fi
}
