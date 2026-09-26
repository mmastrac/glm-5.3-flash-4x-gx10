#!/usr/bin/env bash
# Build on the box that will run it -- there is no cross-build for aarch64 here.
# The context is the repo root, not image/, so the vllm symlink (into the
# spark-agent submodule) stays inside it. Copy the tree to a build box with
# `rsync -a`, which keeps the hidden .submodules/ and the symlink.
set -euo pipefail
TAG="${TAG:-spark-glm53:v8}"
cd "$(dirname "$0")/.."
args=(-t "$TAG" -f image/Dockerfile)
[ -n "${BASE:-}" ]           && args+=(--build-arg "BASE=$BASE")
[ -n "${MENTAT_VERSION:-}" ] && args+=(--build-arg "MENTAT_VERSION=$MENTAT_VERSION")
sudo docker build "${args[@]}" .
echo "built $TAG"
