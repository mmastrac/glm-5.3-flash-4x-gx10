#!/usr/bin/env bash
# Launch nccl_collective_test.py inside the spark-glm53 image with the SAME
# per-box network derivation entrypoint.sh performs (LAN address for the
# bootstrap, RoCE devices + GID index for the data plane). Run one container
# per box; see README.md for the docker command.
#
# Required env: RANK (0 on node0), WORLD_SIZE=4, MASTER_ADDR=192.0.2.1,
# MASTER_PORT. Optional: CLUSTER_SUBNET (198.51.100.), FABRIC_SUBNETS,
# VLLM_HOST_IP (the LAN address; derived from the default route if unset),
# any NCCL_* override (NCCL_PROTO, NCCL_ALGO, NCCL_IB_DISABLE, ...).
# Remaining arguments go to the python test.
set -euo pipefail

CLUSTER_SUBNET="${CLUSTER_SUBNET:-198.51.100.}"
FABRIC_SUBNETS="${FABRIC_SUBNETS:-$CLUSTER_SUBNET}"

if [[ -z "${VLLM_HOST_IP:-}" ]]; then
  VLLM_HOST_IP=$(ip -o -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
fi
export VLLM_HOST_IP
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$(ip -o -4 addr show | awk -v ip="$VLLM_HOST_IP" '$4 ~ "^"ip"/" {print $2; exit}')}"
export GLOO_SOCKET_IFNAME
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}"

# Same as entrypoint.sh fabric_port(): "<rdma-device> <gid-index>" for the
# port holding an address in $1.
fabric_port() {
  local prefix="$1" found addr ifname dev="" hex i t g d n
  found=$(ip -o -4 addr show 2>/dev/null \
      | awk -v p="$prefix" '$4 ~ "^"p {split($4,a,"/"); print $2, a[1]; exit}')
  [[ -n "$found" ]] || return 1
  ifname="${found%% *}"; addr="${found##* }"
  for d in /sys/class/infiniband/*; do
    for n in "$d"/ports/1/gid_attrs/ndevs/*; do
      [[ -f "$n" ]] || continue
      [[ "$(cat "$n" 2>/dev/null)" == "$ifname" ]] || continue
      dev="$(basename "$d")"; break 2
    done
  done
  [[ -n "$dev" ]] || return 1
  hex=$(printf '%s' "$addr" | awk -F. '{printf "%02x%02x:%02x%02x", $1,$2,$3,$4}')
  for i in $(seq 0 15); do
    t=$(cat "/sys/class/infiniband/$dev/ports/1/gid_attrs/types/$i" 2>/dev/null) || continue
    g=$(cat "/sys/class/infiniband/$dev/ports/1/gids/$i" 2>/dev/null) || continue
    [[ "$t" == "RoCE v2" && "$g" == *"ffff:$hex" ]] || continue
    printf '%s %s\n' "$dev" "$i"; return 0
  done
  return 1
}

if [[ "${NCCL_IB_DISABLE:-0}" != "1" && ( -z "${NCCL_IB_HCA:-}" || -z "${NCCL_IB_GID_INDEX:-}" ) ]]; then
  _hcas=""; _gid=""
  for _p in $FABRIC_SUBNETS; do
    _r=$(fabric_port "$_p") || continue
    _d="${_r%% *}"; _i="${_r##* }"
    if [[ -n "$_gid" && "$_i" != "$_gid" ]]; then
      echo "FATAL: fabric ports disagree on GID index ($_d at $_i, expected $_gid)" >&2; exit 1
    fi
    _gid="${_gid:-$_i}"; _hcas="${_hcas:+$_hcas,}$_d"
  done
  if [[ -z "$_hcas" ]]; then
    echo "FATAL: no RoCE v2 GID for any of: $FABRIC_SUBNETS" >&2; ip -br addr show >&2; exit 1
  fi
  export NCCL_IB_HCA="${NCCL_IB_HCA:-$_hcas}" NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-$_gid}"
fi
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "rank=${RANK} host_ip=${VLLM_HOST_IP} ifname=${GLOO_SOCKET_IFNAME} NCCL_IB_HCA=${NCCL_IB_HCA:-} gid=${NCCL_IB_GID_INDEX:-} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} PROTO=${NCCL_PROTO:-} ALGO=${NCCL_ALGO:-}"
cd /
exec python3 "$(dirname "$0")/nccl_collective_test.py" "$@"
