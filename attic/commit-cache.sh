#!/usr/bin/env bash
# Bless the FlashInfer autotune cache of a KNOWN-GOOD glm53 boot, and give the
# same bytes to all four nodes.
#
# Run from a machine that can ssh to every node, after a boot that actually
# serves. Pair with `spark-flashinfer-cache.sh restore`, which every node runs
# before `docker compose up`, so a boot can only ever start from what this
# script blessed.
#
# The point is not that the snapshot is immutable, it is that it is IDENTICAL.
# One node is the source and its bytes are copied everywhere, so the four cannot
# drift. Uneven caches are what deadlock a TP=4 boot: a rank holding an entry
# takes the cache-hit branch in flashinfer's cutlass_fused_moe while a rank
# without it profiles and calls all_reduce, and the collective never matches.
# See spark-flashinfer-cache.sh for the full chain.
#
#   ./commit-cache.sh [--dry-run]
set -euo pipefail

HEAD="${HEAD:-192.0.2.1}"
NODES="${NODES:-192.0.2.1 192.0.2.4 192.0.2.2 192.0.2.3}"
USER_AT="${USER_AT:-admin}"
PORT="${PORT:-8002}"
PATTERN="glm53-121-*"
SCRATCH=/home/admin/container-cache
COMMITTED=/home/admin/cache-committed
SSH=(ssh -o ConnectTimeout=10 -o BatchMode=yes)
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1

say() { printf '%s\n' "$*"; }

# 1. The snapshot is only worth blessing if this boot works. Health alone is not
#    enough -- the endpoint answers /health while the engine is still warming --
#    so ask for real tokens.
say "verifying ${HEAD} actually serves"
code=$("${SSH[@]}" "${USER_AT}@${HEAD}" "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/health" || true)
[ "$code" = "200" ] || { say "head health is ${code:-unreachable}; refusing to commit"; exit 1; }
gen=$("${SSH[@]}" "${USER_AT}@${HEAD}" "curl -s -m 180 http://127.0.0.1:${PORT}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"glm53\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":400,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d[\"choices\"][0].get(\"finish_reason\"),d[\"usage\"][\"completion_tokens\"])'" || true)
case "$gen" in
  stop\ *) say "  generation ok: $gen" ;;
  *) say "  generation failed or truncated ($gen); refusing to commit"; exit 1 ;;
esac

# 2. One source of truth.
say "taking the snapshot from ${HEAD}"
"${SSH[@]}" "${USER_AT}@${HEAD}" "sudo -n ls -d ${SCRATCH}/${PATTERN} >/dev/null 2>&1" \
  || { say "  no ${PATTERN} in ${SCRATCH} on the head"; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# `cd` rather than tar -C: the remote shell expands the glob before tar runs, so
# -C would have it matching in the login directory instead of the cache.
"${SSH[@]}" "${USER_AT}@${HEAD}" "cd ${SCRATCH} && sudo -n tar -cf - ${PATTERN}" > "$TMP/cache.tar"
say "  $(du -h "$TMP/cache.tar" | cut -f1)"

[ -n "$DRY" ] && { say "dry run; not distributing"; exit 0; }

# 3. Same bytes everywhere. A partial distribution recreates the very split this
#    exists to prevent, so a failure here is loud.
for n in $NODES; do
  say "committing to ${n}"
  "${SSH[@]}" "${USER_AT}@${n}" "sudo -n mkdir -p ${COMMITTED} && sudo -n rm -rf ${COMMITTED}/${PATTERN}"
  "${SSH[@]}" "${USER_AT}@${n}" "sudo -n tar -C ${COMMITTED} -xf -" < "$TMP/cache.tar"
done

# 4. Prove it rather than assume it.
say "verifying every node matches"
ref=""
for n in $NODES; do
  sum=$("${SSH[@]}" "${USER_AT}@${n}" "cd ${COMMITTED} && sudo -n find ${PATTERN} -type f -exec md5sum {} + 2>/dev/null | sort -k2 | md5sum | cut -d' ' -f1")
  say "  ${n}  ${sum}"
  [ -z "$ref" ] && ref="$sum"
  [ "$sum" = "$ref" ] || { say "MISMATCH on ${n}; the nodes are NOT uniform"; exit 1; }
done
say "committed and uniform across all four"
