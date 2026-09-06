#!/bin/bash
# usage: down.sh <node-ip>...
set -euo pipefail
SSH_USER=${SSH_USER:-admin}
for n in "$@"; do ssh "$SSH_USER"@"$n" 'sudo docker stop glm53' & done; wait
