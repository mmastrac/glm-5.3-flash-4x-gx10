#!/bin/bash
# vLLM's shm queue spins for busy_loop_s after the last message before it will
# block on zmq. The default is 1s; decode messages arrive every few ms, so the
# blocking path is never taken and the cores spin at full power. On GB10 the CPU
# and GPU share one package, so that heat comes out of the GPU's budget.
# Measured here: 1 -> 0.002 cut vLLM CPU 185%->109%, the SoC ~20C, and RAISED
# decode 66.9 -> 70.6 tok/s. 0 (always block) is cooler but 11% slower.
set -euo pipefail
F=/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py
OUT=${1:-/home/admin/spin-patch}
mkdir -p "$OUT"
sudo docker create --name _spin_tmp "${IMAGE:-glm53-spark:sm90-v21}" >/dev/null
sudo docker cp "_spin_tmp:$F" "$OUT/shm_broadcast.py"
sudo docker rm _spin_tmp >/dev/null
sudo chown "$(id -u):$(id -g)" "$OUT/shm_broadcast.py"
sed -i 's/busy_loop_s: float = [0-9.]*/busy_loop_s: float = 0.002/' "$OUT/shm_broadcast.py"
grep -oE 'busy_loop_s: float = [0-9.]+' "$OUT/shm_broadcast.py" | head -1
