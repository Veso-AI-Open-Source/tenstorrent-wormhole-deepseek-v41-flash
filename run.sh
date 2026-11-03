#!/usr/bin/env bash
# Run a Python script from this repo on the cards, inside the tt-metal container.
#   ./run.sh bench/bench_real.py
# Required: TT_METAL_TREE, a built tt-metal v0.81.0-dev20261020 tree.
# Optional: IMAGE, HF_HUB, DSV41_CACHE, DSV41_WEIGHTS, and DSV41_ENV (extra KEY=VALUE pairs, space separated).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TT_METAL_TREE:?set TT_METAL_TREE to a built tt-metal v0.81.0-dev20261020 tree}"
IMAGE="${IMAGE:-dsv41-tt:v081}"
HF_HUB="${HF_HUB:-$HOME/.cache/huggingface/hub}"
CACHE="${DSV41_CACHE:-$HOME/.cache/dsv41}"
WEIGHTS="${DSV41_WEIGHTS:-$REPO/dsv41/weights}"
mkdir -p "$CACHE" "$WEIGHTS" "$REPO/dsv41/weights"
EXTRA=()
for kv in ${DSV41_ENV:-}; do EXTRA+=(-e "$kv"); done
exec docker run --rm -i --privileged --name "dsv41-$$" --network=host --security-opt label=disable \
  --device=/dev/tenstorrent:/dev/tenstorrent --volume=/dev/hugepages-1G:/dev/hugepages-1G \
  --volume="$REPO":/work --volume="$WEIGHTS":/work/dsv41/weights --volume="$HF_HUB":/hf:ro \
  --volume="$TT_METAL_TREE":/tt-metal --volume="$CACHE":/cache \
  -e TT_METAL_HOME=/tt-metal -e TT_METAL_RUNTIME_ROOT=/tt-metal -e ARCH_NAME=wormhole_b0 \
  -e TT_METAL_CACHE=/cache/tt-metal -e TT_METAL_LOGS_PATH=/cache/logs -e HOME=/cache/home \
  -e PYTHONPATH=/tt-metal:/tt-metal/ttnn -e PYTHONUNBUFFERED=1 -e DSV41_ROOT=/work/dsv41 \
  -e 'DSV41_SNAP=/hf/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*' "${EXTRA[@]}" \
  --workdir /work --entrypoint /bin/bash "$IMAGE" -c 'source /tt-metal/python_env/bin/activate && exec python3 "$@"' _ "$@"
