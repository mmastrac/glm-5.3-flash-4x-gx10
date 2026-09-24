#!/bin/bash
# Take the model down on every node and confirm no rank is left, which must
# hold before up.sh starts a new stack. mentatd and mentatd-serve stay up.
#   usage: down.sh <node-ip>...
set -euo pipefail
SSH_USER=${SSH_USER:-$USER}
REPO_DIR=${REPO_DIR:-glm-5.3-flash-4x-gx10}   # relative to the remote home
(( $# > 0 )) || { echo "usage: down.sh <node-ip>..." >&2; exit 2; }

for n in "$@"; do
  ssh "$SSH_USER@$n" "cd $REPO_DIR && sudo docker compose -f compose/glm53.yaml down --timeout 60" &
done; wait

left=0
for n in "$@"; do
  if [[ -n "$(ssh "$SSH_USER@$n" 'sudo docker ps -aq --filter name=^glm53$')" ]]; then
    echo "glm53 is still present on $n" >&2
    left=1
  fi
done
exit "$left"
