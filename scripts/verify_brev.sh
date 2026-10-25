#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/brev/versions.env"

FULL=0
if [[ "${1:-}" == "--full" ]]; then
    FULL=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/verify_brev.sh [--full]" >&2
    exit 2
fi

EXTERNAL_ROOT="${ROCKWELL_EXTERNAL_ROOT:-/workspace}"
MODEL_ROOT="${COSMOS3_NANO_PATH:-$EXTERNAL_ROOT/models/Cosmos3-Nano}"
COSMOS_ENV="${COSMOS_ENV:-$EXTERNAL_ROOT/envs/cosmos3}"
UV_ENV="${UV_ENV:-$EXTERNAL_ROOT/tools/uv}"
HF_ENV="${HF_ENV:-$EXTERNAL_ROOT/tools/hf-cli}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$EXTERNAL_ROOT/cache/uv}"
RUNTIME_ROOT="${ROCKWELL_RUNTIME_ROOT:-$EXTERNAL_ROOT/rockwell-runtime}"
RUNS_ROOT="${ROCKWELL_RUNS_ROOT:-$EXTERNAL_ROOT/runs}"
ISAAC="${ISAAC:-$ISAAC_DEFAULT}"

UV_LOCK="$REPO_ROOT/brev/uv-requirements.lock.txt"
HOST_LOCK="$REPO_ROOT/brev/host-requirements.lock.txt"
HF_LOCK="$REPO_ROOT/brev/hf-cli-requirements.lock.txt"
COSMOS_LOCK="$REPO_ROOT/brev/cosmos-requirements.lock.txt"
ENV_MARKER=.rockwell-brev-managed-env
RUNTIME_MARKER=.rockwell-brev-runtime
MARKER_PROJECT=rockwell-fanuc-digital-twin-v1

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

verify_twin_manifest() {
    python3 - "$1" <<'PY'
import hashlib
import json
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest_path = root / "MANIFEST.json"
if not manifest_path.is_file() or manifest_path.is_symlink():
    raise SystemExit(f"Missing or unsafe manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
records = manifest.get("files")
if not isinstance(records, list) or not records:
    raise SystemExit(f"Invalid manifest records: {manifest_path}")

seen = {"MANIFEST.json"}
for record in records:
    relative = Path(record["path"])
    relative_text = relative.as_posix()
    if (
        relative.is_absolute()
        or relative_text in {"", "."}
        or ".." in relative.parts
        or relative_text in seen
    ):
        raise SystemExit(f"Unsafe or duplicate manifest path: {relative_text}")
    path = root / relative
    if path.is_symlink() or not path.exists():
        raise SystemExit(f"Manifest file missing or is a symlink: {path}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise SystemExit(f"Manifest entry is not a regular file: {path}")
    resolved = path.resolve()
    if root not in resolved.parents:
        raise SystemExit(f"Manifest path escapes twin root: {path}")
    data = path.read_bytes()
    if len(data) != record["size_bytes"]:
        raise SystemExit(f"Manifest size mismatch: {path}")
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise SystemExit(f"Manifest SHA-256 mismatch: {path}")
    seen.add(relative_text)
print(f"[verify] manifest OK: {root} ({len(records)} payload files)")
PY
}

verify_reference_asset_manifest() {
    local root="$1"
    local require_tracked="${2:-0}"
    python3 - "$root" "$require_tracked" "$REPO_ROOT" <<'PY'
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
require_tracked = sys.argv[2] == "1"
repo = Path(sys.argv[3]).resolve()
manifest_path = root / "MANIFEST.json"
if not manifest_path.is_file() or manifest_path.is_symlink():
    raise SystemExit(f"Missing or unsafe reference-asset manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
records = manifest.get("files")
if not isinstance(records, list) or not records:
    raise SystemExit(f"Invalid reference-asset manifest records: {manifest_path}")

expected = {"MANIFEST.json"}
for record in records:
    if not isinstance(record, dict):
        raise SystemExit(f"Invalid reference-asset manifest record: {record!r}")
    relative = Path(record["path"])
    relative_text = relative.as_posix()
    if (
        relative.is_absolute()
        or relative_text in {"", "."}
        or ".." in relative.parts
        or relative_text in expected
    ):
        raise SystemExit(f"Unsafe or duplicate reference-asset path: {relative_text}")
    path = root / relative
    if path.is_symlink() or not path.exists():
        raise SystemExit(f"Reference asset missing or is a symlink: {path}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise SystemExit(f"Reference asset is not a regular file: {path}")
    resolved = path.resolve()
    if root not in resolved.parents:
        raise SystemExit(f"Reference-asset path escapes its root: {path}")
    data = path.read_bytes()
    if len(data) != record["size_bytes"]:
        raise SystemExit(f"Reference-asset size mismatch: {path}")
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise SystemExit(f"Reference-asset SHA-256 mismatch: {path}")
    expected.add(relative_text)

actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() or path.is_symlink()
}
if actual != expected:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise SystemExit(
        f"Reference-asset inventory differs from manifest; "
        f"missing={missing}, extra={extra}"
    )
if require_tracked:
    try:
        prefix = root.relative_to(repo).as_posix()
    except ValueError as error:
        raise SystemExit(f"Reference assets are outside repository {repo}: {root}") from error
    output = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--", prefix],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    tracked = {item.decode() for item in output.split(b"\0") if item}
    expected_tracked = {f"{prefix}/{item}" for item in expected}
    if tracked != expected_tracked:
        missing = sorted(expected_tracked - tracked)
        extra = sorted(tracked - expected_tracked)
        raise SystemExit(
            f"Reference-asset Git inventory differs from manifest; "
            f"missing={missing}, extra={extra}"
        )
print(f"[verify] reference-asset manifest OK: {root} ({len(records)} payload files)")
PY
}

verify_runtime_inventory() {
    python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

root_arg = Path(sys.argv[1])
if root_arg.is_symlink() or not root_arg.is_dir():
    raise SystemExit(f"Runtime twin is missing, not a directory, or a symlink: {root_arg}")
root = root_arg.resolve()
manifest = json.loads((root / "MANIFEST.json").read_text())
expected = {"MANIFEST.json", "assets/mat_print.png"}
expected.update(record["path"] for record in manifest["files"])
generated_roots = {".venv", "textures", "__pycache__"}
actual = set()
for path in root.rglob("*"):
    relative = path.relative_to(root)
    if relative.parts[0] in generated_roots:
        continue
    if path.is_symlink():
        raise SystemExit(f"Unexpected runtime symlink outside generated roots: {path}")
    if path.is_file():
        actual.add(relative.as_posix())
if actual != expected:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise SystemExit(f"Unexpected runtime inventory; missing={missing}, extra={extra}")
for name in ("textures", "__pycache__"):
    generated = root / name
    if generated.exists():
        if generated.is_symlink() or not generated.is_dir():
            raise SystemExit(f"Unsafe generated runtime path: {generated}")
        for path in generated.rglob("*"):
            if path.is_symlink():
                raise SystemExit(f"Unexpected symlink in generated runtime path: {path}")
print("[verify] runtime inventory contains only supplied files, the table overlay, and known generated roots")
PY
}

verify_managed_env() {
    local path="$1"
    local kind="$2"
    local marker="$path/$ENV_MARKER"
    [[ -d "$path" && ! -L "$path" ]] || fail "$kind environment is unsafe or missing: $path"
    [[ -f "$path/pyvenv.cfg" && -x "$path/bin/python" ]] || \
        fail "$kind environment is incomplete: $path"
    [[ -f "$marker" && ! -L "$marker" ]] || \
        fail "$kind environment is not repository-managed: $path"
    grep -Fxq "format=1" "$marker" || fail "Unsupported environment marker: $path"
    grep -Fxq "project=$MARKER_PROJECT" "$marker" || \
        fail "$kind environment ownership marker belongs to another project: $path"
    grep -Fxq "kind=$kind" "$marker" || fail "Environment kind mismatch: $path"
    grep -Fxq "python=$PYTHON_MAJOR_MINOR" "$marker" || \
        fail "Environment Python marker mismatch: $path"
    local actual
    actual="$("$path/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "$actual" == "$PYTHON_MAJOR_MINOR" ]] || \
        fail "$kind environment uses Python $actual, expected $PYTHON_MAJOR_MINOR"
}

verify_exact_lock() {
    local lock="$1"
    local python="$2"
    local label="$3"
    "$UV_ENV/bin/uv" pip check --python "$python"
    if ! diff -u \
        <(grep -Ev '^[[:space:]]*(#|$)' "$lock" | LC_ALL=C sort) \
        <("$UV_ENV/bin/uv" pip freeze --python "$python" | LC_ALL=C sort); then
        fail "$label environment differs from committed lock: $lock"
    fi
    echo "[verify] exact lock OK: $label"
}

for command in git python3 nvidia-smi tar timeout grep sort diff tee awk tr cmp; do
    command -v "$command" >/dev/null 2>&1 || fail "Required command is missing: $command"
done
for lock in "$UV_LOCK" "$HOST_LOCK" "$HF_LOCK" "$COSMOS_LOCK"; do
    [[ -f "$lock" ]] || fail "Required lock file is missing: $lock"
done
[[ -d "$EXTERNAL_ROOT" ]] || fail "External root is missing: $EXTERNAL_ROOT"
[[ -x "$ISAAC/python.sh" ]] || fail "Isaac Sim launcher is missing: $ISAAC/python.sh"
[[ -f "$ISAAC/VERSION" ]] || fail "Isaac Sim VERSION file is missing: $ISAAC/VERSION"
ACTUAL_ISAAC="$(tr -d '\r\n' < "$ISAAC/VERSION")"
[[ "$ACTUAL_ISAAC" == "$ISAAC_VERSION" ]] || \
    fail "Expected Isaac Sim $ISAAC_VERSION, found $ACTUAL_ISAAC"

cd "$REPO_ROOT"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
    fail "Repository is not clean; setup must not depend on uncommitted files."
REPO_COMMIT="$(git rev-parse HEAD)"
TWIN_SOURCE="$REPO_ROOT/digital-twin-setup"
REFERENCE_ASSETS_SOURCE="$REPO_ROOT/reference-assets"
RUNTIME_DIR="$RUNTIME_ROOT/$REPO_COMMIT"
RUNTIME_TWIN="$RUNTIME_DIR/twin"
RUNTIME_ASSETS="$RUNTIME_DIR/reference-assets"

python3 - "$REPO_ROOT" "$EXTERNAL_ROOT" \
    "$MODEL_ROOT" "$COSMOS_ENV" "$UV_ENV" "$HF_ENV" "$UV_CACHE_DIR" \
    "$RUNTIME_ROOT" "$RUNS_ROOT" "$RUNTIME_DIR" "$RUNTIME_TWIN" \
    "$RUNTIME_ASSETS" <<'PY'
import os
import sys

if len(sys.argv) != 13:
    raise SystemExit(f"Internal path-preflight argument mismatch: {len(sys.argv)}")
repo = os.path.realpath(sys.argv[1])
external_raw = sys.argv[2]
if not os.path.isabs(external_raw):
    raise SystemExit(f"External root must be absolute: {external_raw}")
external = os.path.realpath(external_raw)
if external == repo or external.startswith(repo + os.sep):
    raise SystemExit(f"External root resolves inside repository: {external}")

names = [
    "model", "cosmos env", "uv env", "HF env", "uv cache",
    "runtime root", "runs root",
]
managed = []
for name, raw in zip(names, sys.argv[3:10]):
    if not os.path.isabs(raw):
        raise SystemExit(f"{name} path must be absolute: {raw}")
    path = os.path.realpath(raw)
    if path == external or not path.startswith(external + os.sep):
        raise SystemExit(f"{name} must be below external root {external}: {path}")
    if path == repo or path.startswith(repo + os.sep):
        raise SystemExit(f"{name} resolves inside repository: {path}")
    managed.append((name, path))

for index, (left_name, left) in enumerate(managed):
    for right_name, right in managed[index + 1:]:
        if (
            left == right
            or left.startswith(right + os.sep)
            or right.startswith(left + os.sep)
        ):
            raise SystemExit(
                f"Managed paths overlap: {left_name}={left}, {right_name}={right}"
            )

runtime_root = managed[5][1]
for name, raw in zip(
    ("runtime directory", "runtime twin", "runtime reference assets"),
    sys.argv[10:13],
):
    if not os.path.isabs(raw):
        raise SystemExit(f"{name} path must be absolute: {raw}")
    path = os.path.realpath(raw)
    if not path.startswith(runtime_root + os.sep):
        raise SystemExit(f"{name} escapes runtime root {runtime_root}: {path}")
    if path == repo or path.startswith(repo + os.sep):
        raise SystemExit(f"{name} resolves inside repository: {path}")
print("[verify] runtime paths are isolated below", external)
PY

GPU_LIST="$(nvidia-smi -L)" || fail "nvidia-smi could not query a GPU"
[[ -n "$GPU_LIST" ]] || fail "No NVIDIA GPU was reported"
printf '%s\n' "$GPU_LIST"
GPU_MEMORY_MIB="$(
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
        awk 'BEGIN { max = 0 } { gsub(/[^0-9]/, "", $1); if ($1 > max) max = $1 } END { print max }'
)"
[[ "$GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] || fail "Could not determine GPU memory"
(( GPU_MEMORY_MIB >= MIN_GPU_MEMORY_MIB )) || \
    fail "Expected at least $MIN_GPU_MEMORY_MIB MiB GPU memory; found $GPU_MEMORY_MIB MiB"

git ls-files --error-unmatch digital-twin-setup/MANIFEST.json >/dev/null 2>&1 || \
    fail "The supplied twin is not committed in this checkout."
git ls-files --error-unmatch reference-assets/MANIFEST.json >/dev/null 2>&1 || \
    fail "The reference-asset manifest is not committed in this checkout."
for executable in \
    scripts/setup_brev.sh \
    scripts/verify_brev.sh \
    digital-twin-setup/bootstrap.sh \
    digital-twin-setup/run_sort_pipeline.sh \
    digital-twin-setup/share.sh; do
    GIT_MODE="$(git ls-files -s -- "$executable" | awk 'NR == 1 { print $1 }')"
    [[ "$GIT_MODE" == 100755 ]] || \
        fail "Git executable mode is missing for $executable (found ${GIT_MODE:-untracked})"
done
verify_twin_manifest "$TWIN_SOURCE"
verify_reference_asset_manifest "$REFERENCE_ASSETS_SOURCE" 1
[[ -d "$RUNTIME_DIR" && ! -L "$RUNTIME_DIR" ]] || fail "Runtime is missing: $RUNTIME_DIR"
[[ -f "$RUNTIME_DIR/$RUNTIME_MARKER" && ! -L "$RUNTIME_DIR/$RUNTIME_MARKER" ]] || \
    fail "Runtime ownership marker is missing: $RUNTIME_DIR"
grep -Fxq "format=1" "$RUNTIME_DIR/$RUNTIME_MARKER" || fail "Unsupported runtime marker"
grep -Fxq "project=$MARKER_PROJECT" "$RUNTIME_DIR/$RUNTIME_MARKER" || \
    fail "Runtime ownership marker belongs to another project"
grep -Fxq "commit=$REPO_COMMIT" "$RUNTIME_DIR/$RUNTIME_MARKER" || \
    fail "Runtime marker commit differs from checkout"
[[ -f "$RUNTIME_DIR/source-commit" ]] || fail "Runtime source marker is missing"
[[ "$(tr -d '\r\n' < "$RUNTIME_DIR/source-commit")" == "$REPO_COMMIT" ]] || \
    fail "Runtime copy was created from another repository commit"
[[ -d "$RUNTIME_TWIN" && ! -L "$RUNTIME_TWIN" ]] || \
    fail "Runtime twin is missing, not a directory, or a symlink: $RUNTIME_TWIN"
[[ -d "$RUNTIME_ASSETS" && ! -L "$RUNTIME_ASSETS" ]] || \
    fail "Runtime reference assets are missing, not a directory, or a symlink: $RUNTIME_ASSETS"
verify_twin_manifest "$RUNTIME_TWIN"
verify_reference_asset_manifest "$RUNTIME_ASSETS"
cmp -s -- \
    "$RUNTIME_ASSETS/table/manipbench-table-surface.png" \
    "$RUNTIME_TWIN/assets/mat_print.png" || \
    fail "Runtime twin texture differs from the committed reference asset"
verify_runtime_inventory "$RUNTIME_TWIN"
for executable in bootstrap.sh run_sort_pipeline.sh share.sh; do
    [[ -x "$RUNTIME_TWIN/$executable" ]] || \
        fail "Runtime lost executable mode: $RUNTIME_TWIN/$executable"
done

verify_managed_env "$UV_ENV" uv
verify_managed_env "$HF_ENV" hf-cli
verify_managed_env "$COSMOS_ENV" cosmos
verify_managed_env "$RUNTIME_TWIN/.venv" host
[[ "$("$UV_ENV/bin/uv" --version)" == "uv $UV_VERSION"* ]] || fail "Unexpected uv version"
[[ "$("$HF_ENV/bin/hf" version)" == *"$HUGGINGFACE_HUB_VERSION"* ]] || \
    fail "Unexpected Hugging Face CLI version"

verify_exact_lock "$UV_LOCK" "$UV_ENV/bin/python" uv
verify_exact_lock "$HF_LOCK" "$HF_ENV/bin/python" hf-cli
verify_exact_lock "$HOST_LOCK" "$RUNTIME_TWIN/.venv/bin/python" host
verify_exact_lock "$COSMOS_LOCK" "$COSMOS_ENV/bin/python" cosmos

(
    cd "$RUNTIME_TWIN"
    .venv/bin/python -m unittest \
        test_sort_planning test_cube_perception test_cube_pose test_runtime_config
)

"$COSMOS_ENV/bin/python" - "$MODEL_ROOT" <<'PY'
import json
import stat
import sys
from pathlib import Path

from safetensors import safe_open

root = Path(sys.argv[1]).resolve()
required = [root / "config.json", root / "model.safetensors.index.json"]
for path in required:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Required model file missing, empty, or unsafe: {path}")

index = json.loads(required[1].read_text())
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit("Model index contains no weight map")
shards = sorted(set(weight_map.values()))
for relative_text in shards:
    if not isinstance(relative_text, str):
        raise SystemExit(f"Non-string shard path in model index: {relative_text!r}")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"Unsafe shard path in model index: {relative_text}")
    path = root / relative
    if path.is_symlink() or not path.exists():
        raise SystemExit(f"Model shard missing or is a symlink: {path}")
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size == 0:
        raise SystemExit(f"Model shard is empty or not a regular file: {path}")
    resolved = path.resolve()
    if root not in resolved.parents:
        raise SystemExit(f"Model shard escapes model root: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if not list(handle.keys()):
            raise SystemExit(f"Safetensors shard contains no tensors: {path}")

zero = [p for p in root.rglob("*.safetensors") if p.stat().st_size == 0]
if zero:
    raise SystemExit(f"Zero-byte safetensor files: {zero}")
print(f"[verify] model index and safetensors headers OK: {len(shards)} indexed shards")
PY

DRY_RUN_OUTPUT="$("$HF_ENV/bin/hf" download "$COSMOS3_NANO_REPO" \
    --revision "$COSMOS3_NANO_REVISION" \
    --local-dir "$MODEL_ROOT" \
    --dry-run 2>&1)"
printf '%s\n' "$DRY_RUN_OUTPUT"
grep -q "Will download 0 files" <<<"$DRY_RUN_OUTPUT" || \
    fail "Pinned Hugging Face snapshot is incomplete or differs from the requested revision."

COSMOS3_NANO_PATH="$MODEL_ROOT" "$COSMOS_ENV/bin/python" - <<'PY'
import cv2
import scipy
import torch
import transformers
import transformers_cosmos3

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch")
print("[verify] CUDA:", torch.cuda.get_device_name(0))
print("[verify] torch:", torch.__version__)
print("[verify] transformers:", transformers.__version__)
print("[verify] scipy:", scipy.__version__)
print("[verify] cv2:", cv2.__version__)
print("[verify] shim:", transformers_cosmos3.__file__)
PY

if [[ "$FULL" -eq 1 ]]; then
    mkdir -p "$RUNS_ROOT"
    VALIDATION_RUN="$(mktemp -d "$RUNS_ROOT/setup-$REPO_COMMIT.XXXXXX")"
    printf 'commit=%s\n' "$REPO_COMMIT" > "$VALIDATION_RUN/.rockwell-validation-run"
    SCENE="$VALIDATION_RUN/scene.usd"
    ISAAC_LOG="$VALIDATION_RUN/build_scene.log"
    if ! (
        cd "$RUNTIME_TWIN"
        timeout --signal=TERM --kill-after=60s 20m \
            "$ISAAC/python.sh" build_scene.py --headless --save "$SCENE"
    ) 2>&1 | tee "$ISAAC_LOG"; then
        fail "Isaac scene build failed or timed out; inspect $ISAAC_LOG"
    fi
    [[ -s "$SCENE" ]] || fail "Isaac scene was not created: $SCENE"
    ! grep -q "Traceback (most recent call last)" "$ISAAC_LOG" || \
        fail "Isaac log contains a Python traceback: $ISAAC_LOG"
    ! grep -Fq "mat_print.png missing" "$ISAAC_LOG" || \
        fail "Isaac could not find the installed work-surface texture: $ISAAC_LOG"

    if ! COSMOS3_NANO_PATH="$MODEL_ROOT" \
        timeout --signal=TERM --kill-after=60s 20m \
        "$COSMOS_ENV/bin/python" - "$RUNTIME_TWIN" "$MODEL_ROOT" <<'PY'
import sys
from pathlib import Path

twin = Path(sys.argv[1]).resolve()
model_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(twin))
from perceive import ask, load_model

model, processor = load_model()
image = model_root / "assets" / "example_reasoning_input.png"
if not image.is_file():
    raise SystemExit(f"Validation image missing: {image}")
reply = ask(
    model,
    processor,
    image,
    "In one short sentence, describe this image.",
    max_new_tokens=48,
)
if not reply.strip():
    raise SystemExit("Cosmos inference returned an empty response")
print("[verify] inference reply:", reply)
PY
    then
        fail "Cosmos model load or inference failed or timed out"
    fi
    echo "[verify] full smoke outputs: $VALIDATION_RUN"
fi

verify_twin_manifest "$TWIN_SOURCE"
verify_twin_manifest "$RUNTIME_TWIN"
verify_reference_asset_manifest "$REFERENCE_ASSETS_SOURCE" 1
verify_reference_asset_manifest "$RUNTIME_ASSETS"
cmp -s -- \
    "$RUNTIME_ASSETS/table/manipbench-table-surface.png" \
    "$RUNTIME_TWIN/assets/mat_print.png" || \
    fail "Runtime twin texture changed during verification"
verify_runtime_inventory "$RUNTIME_TWIN"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
    fail "Verification changed the repository checkout."

echo "Brev verification passed for repository commit $REPO_COMMIT"
