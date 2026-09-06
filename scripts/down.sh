#!/bin/bash
# usage: down.sh <node-ip>...
set -euo pipefail
for n in "$@"; do ssh admin@"$n" 'sudo docker stop glm53' & done; wait
