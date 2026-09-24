#!/bin/bash
# Bring up the cluster from a workstation with ssh access to every node.
#   usage: up.sh <head-ip> <worker-ip> <worker-ip> <worker-ip>
#
# Each node needs a clone of this repo at REPO_DIR (with its submodule), the
# image built or pulled, and a filled-in compose/.env. The order is mentatd
# everywhere, mentatd-serve on the head, then the model on the head and, a few
# seconds later, on the workers.
set -euo pipefail
SSH_USER=${SSH_USER:-$USER}
REPO_DIR=${REPO_DIR:-glm-5.3-flash-4x-gx10}   # relative to the remote home
HEAD=${1:?usage: up.sh <head-ip> <worker-ip>...}
shift
WORKERS=("$@")
NODES=("$HEAD" "${WORKERS[@]}")

on() { local n="$1"; shift; ssh "$SSH_USER@$n" "cd $REPO_DIR && $*"; }

# Starting ranks while an old group is still tearing down hangs the new group
# just past NCCL setup with nothing in any log. Refuse rather than race it.
for n in "${NODES[@]}"; do
  if [[ -n "$(ssh "$SSH_USER@$n" 'sudo docker ps -aq --filter name=^glm53$')" ]]; then
    echo "a glm53 container still exists on $n; run scripts/down.sh first" >&2
    exit 1
  fi
done

echo "== mentatd on every node =="
for n in "${NODES[@]}"; do
  on "$n" sudo docker compose -f compose/mentatd.yaml up -d &
done; wait

echo "== mentatd-serve on the head (:6381) =="
on "$HEAD" sudo docker compose -f compose/mentatd-serve.yaml up -d

echo "== glm53 on the head, then the workers =="
on "$HEAD" sudo docker compose -f compose/glm53.yaml up -d
sleep 10
for n in "${WORKERS[@]}"; do
  on "$n" sudo docker compose -f compose/glm53.yaml up -d &
done; wait

# The weight load takes about ten minutes.
echo "== waiting for the head's API on :8002 =="
for _ in $(seq 1 90); do
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://$HEAD:8002/v1/models")" == 200 ]]; then
    echo "serving; check it with: smoketest/run.sh http://$HEAD:6381"
    exit 0
  fi
  sleep 15
done
echo "timed out; check: ssh $SSH_USER@$HEAD 'sudo docker logs glm53 | tail -40'" >&2
exit 1
