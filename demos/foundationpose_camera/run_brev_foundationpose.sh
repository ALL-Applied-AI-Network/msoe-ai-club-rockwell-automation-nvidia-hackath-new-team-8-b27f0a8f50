#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CAPTURE_DIRECTORY OUTPUT_DIRECTORY" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
capture=$(realpath "$1")
output=$(realpath -m "$2")
fp_repo=${FOUNDATIONPOSE_REPO:-/home/ubuntu/foundation-pose-inference-library}

if [[ ! -f "$capture/manifest.json" ]]; then
  echo "capture directory has no manifest.json: $capture" >&2
  exit 2
fi
if [[ -e "$output/estimates.json" ]]; then
  echo "refusing to overwrite $output/estimates.json" >&2
  exit 2
fi
mkdir -p "$output/debug"

cd "$fp_repo"
./run_dev.sh run --rm \
  -e PYTHONPATH=/work/python/src \
  -v "$repo_root:/app:ro" \
  -v "$capture:/inputs:ro" \
  -v "$output:/outputs" \
  --entrypoint /opt/bench/bin/python \
  dev \
  /app/demos/foundationpose_camera/run_foundationpose.py \
  --capture /inputs \
  --output /outputs/estimates.json \
  --debug-dir /outputs/debug \
  --engine-cache-dir /work/engine_cache \
  --n-hypotheses "${FP_N_HYPOTHESES:-252}" \
  --batch-size "${FP_BATCH_SIZE:-0}"

echo "wrote $output/estimates.json"
