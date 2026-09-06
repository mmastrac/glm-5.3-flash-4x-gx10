#!/bin/bash
# Bring up the cluster. Run from a workstation with ssh access to every node.
# usage: up.sh <head-ip> <worker-ip>...
set -euo pipefail
SSH_USER=${SSH_USER:-admin}
HEAD=${1:?usage: up.sh <head-ip> <worker-ip>...}
shift
NODES=("$HEAD" "$@")
COMPOSE_DIR=${COMPOSE_DIR:-compose/glm53}
FILES="-f glm53.yaml -f pp-mtp-override.yaml -f dflash2-full-override.yaml -f spin-wait-override.yaml"

echo "== mentatd (Ray replacement) on every node =="
for n in "${NODES[@]}"; do
  ssh "$SSH_USER"@"$n" "cd ~/compose/mentatd && sudo docker compose -f mentatd.yaml up -d" &
done; wait

echo "== mentatd-serve (OpenAI front door, :6381) on the head =="
ssh "$SSH_USER"@"$HEAD" "cd ~/compose/mentatd-serve && sudo docker compose -f mentatd-serve.yaml up -d"

echo "== vLLM on every node =="
for n in "${NODES[@]}"; do
  ssh "$SSH_USER"@"$n" "cd ~/$COMPOSE_DIR && sudo docker compose $FILES up -d" &
done; wait

echo "== waiting for :8002 =="
for _ in $(seq 1 90); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://$HEAD:8002/v1/models")" == "200" ]] && { echo "serving"; exit 0; }
  sleep 15
done
echo "timed out; check: ssh ${SSH_USER}@${HEAD} 'sudo docker logs glm53 | tail -40'" >&2
exit 1
