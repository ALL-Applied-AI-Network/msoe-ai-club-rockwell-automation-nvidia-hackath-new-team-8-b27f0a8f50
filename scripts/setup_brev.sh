#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/brev/versions.env"

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/setup_brev.sh" >&2
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
    local root="$1"
    local strict_inventory="${2:-0}"
    python3 - "$root" "$strict_inventory" <<'PY'
import hashlib
import json
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
strict_inventory = sys.argv[2] == "1"
manifest_path = root / "MANIFEST.json"
if not manifest_path.is_file() or manifest_path.is_symlink():
    raise SystemExit(f"Missing or unsafe manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
records = manifest.get("files")
if not isinstance(records, list) or not records:
    raise SystemExit(f"Invalid manifest records: {manifest_path}")

expected = {"MANIFEST.json"}
for record in records:
    relative = Path(record["path"])
    relative_text = relative.as_posix()
    if (
        relative.is_absolute()
        or relative_text in {"", "."}
        or ".." in relative.parts
        or relative_text in expected
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
    expected.add(relative_text)

if strict_inventory:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SystemExit(
            f"Runtime inventory differs from manifest; missing={missing}, extra={extra}"
        )

print(f"[setup] manifest OK: {root} ({len(records)} payload files)")
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

print(f"[setup] reference-asset manifest OK: {root} ({len(records)} payload files)")
PY
}

show_python_prerequisite() {
    cat >&2 <<'EOF'
From the Brev host, install the documented Python prerequisites, then rerun setup:

  docker exec --user root vscode bash -lc \
    'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip'
EOF
}

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is missing in the Brev vscode container." >&2
    show_python_prerequisite
    exit 2
fi
if ! python3 -c 'import ensurepip, venv' >/dev/null 2>&1; then
    echo "ERROR: Python venv/ensurepip support is missing in the Brev vscode container." >&2
    show_python_prerequisite
    exit 2
fi

for command in git nvidia-smi sha256sum tar timeout df du awk grep sort diff find tr cmp; do
    command -v "$command" >/dev/null 2>&1 || fail "Required command is missing: $command"
done
for lock in "$UV_LOCK" "$HOST_LOCK" "$HF_LOCK" "$COSMOS_LOCK"; do
    [[ -f "$lock" ]] || fail "Required lock file is missing: $lock"
done
[[ -d "$EXTERNAL_ROOT" ]] || \
    fail "External root does not exist; create it first: $EXTERNAL_ROOT"
[[ -x "$ISAAC/python.sh" ]] || fail "Isaac Sim launcher is missing: $ISAAC/python.sh"
[[ -f "$ISAAC/VERSION" ]] || fail "Isaac Sim VERSION file is missing: $ISAAC/VERSION"

ACTUAL_PYTHON="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$ACTUAL_PYTHON" == "$PYTHON_MAJOR_MINOR" ]] || \
    fail "Expected Python $PYTHON_MAJOR_MINOR, found $ACTUAL_PYTHON"
ACTUAL_ISAAC="$(tr -d '\r\n' < "$ISAAC/VERSION")"
[[ "$ACTUAL_ISAAC" == "$ISAAC_VERSION" ]] || \
    fail "Expected Isaac Sim $ISAAC_VERSION, found $ACTUAL_ISAAC"

cd "$REPO_ROOT"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
    fail "Repository must be a clean clone before setup. Commit, stash, or remove local changes."
REPO_COMMIT="$(git rev-parse HEAD)"
TWIN_SOURCE="$REPO_ROOT/digital-twin-setup"
REFERENCE_ASSETS_SOURCE="$REPO_ROOT/reference-assets"
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
verify_twin_manifest "$TWIN_SOURCE" 0
verify_reference_asset_manifest "$REFERENCE_ASSETS_SOURCE" 1

RUNTIME_DIR="$RUNTIME_ROOT/$REPO_COMMIT"
RUNTIME_TWIN="$RUNTIME_DIR/twin"
RUNTIME_ASSETS="$RUNTIME_DIR/reference-assets"
RUNTIME_ENV="$RUNTIME_DIR/env.sh"
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
raw_managed = sys.argv[3:10]
managed = []
for name, raw in zip(names, raw_managed):
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

print("[setup] all writable paths are isolated below", external)
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
    fail "Cosmos validation requires at least $MIN_GPU_MEMORY_MIB MiB GPU memory; found $GPU_MEMORY_MIB MiB"

AVAILABLE_KB="$(df -Pk "$EXTERNAL_ROOT" | awk 'NR==2 {print $4}')"
MODEL_KB=0
if [[ -d "$MODEL_ROOT" ]]; then
    MODEL_KB="$(du -sk "$MODEL_ROOT" | awk '{print $1}')"
fi
if (( MODEL_KB < 30000000 )); then
    REQUIRED_FREE_KB=62914560
else
    REQUIRED_FREE_KB=15728640
fi
(( AVAILABLE_KB >= REQUIRED_FREE_KB )) || \
    fail "Insufficient free space under $EXTERNAL_ROOT; need at least $((REQUIRED_FREE_KB / 1048576)) GiB free"

TEMP_RUNTIME=
cleanup_runtime_temp() {
    if [[ -n "$TEMP_RUNTIME" && -d "$TEMP_RUNTIME" ]]; then
        rm -rf -- "$TEMP_RUNTIME"
    fi
}
trap cleanup_runtime_temp EXIT

if [[ -e "$RUNTIME_DIR" ]]; then
    [[ -d "$RUNTIME_DIR" && ! -L "$RUNTIME_DIR" ]] || \
        fail "Runtime path is not an owned directory: $RUNTIME_DIR"
    [[ -f "$RUNTIME_DIR/$RUNTIME_MARKER" && ! -L "$RUNTIME_DIR/$RUNTIME_MARKER" ]] || \
        fail "Existing runtime is not marked as repository-owned: $RUNTIME_DIR"
    grep -Fxq "format=1" "$RUNTIME_DIR/$RUNTIME_MARKER" || \
        fail "Unsupported runtime marker format: $RUNTIME_DIR"
    grep -Fxq "project=$MARKER_PROJECT" "$RUNTIME_DIR/$RUNTIME_MARKER" || \
        fail "Runtime ownership marker belongs to another project: $RUNTIME_DIR"
    grep -Fxq "commit=$REPO_COMMIT" "$RUNTIME_DIR/$RUNTIME_MARKER" || \
        fail "Existing runtime belongs to another commit: $RUNTIME_DIR"
    [[ -f "$RUNTIME_DIR/source-commit" ]] || \
        fail "Existing runtime has no source marker: $RUNTIME_DIR"
    [[ "$(tr -d '\r\n' < "$RUNTIME_DIR/source-commit")" == "$REPO_COMMIT" ]] || \
        fail "Existing runtime belongs to another repository commit: $RUNTIME_DIR"
else
    mkdir -p "$RUNTIME_ROOT"
    mkdir "$RUNTIME_DIR"
    printf '%s\n' "$REPO_COMMIT" > "$RUNTIME_DIR/source-commit"
    {
        printf 'format=1\n'
        printf 'project=%s\n' "$MARKER_PROJECT"
        printf 'commit=%s\n' "$REPO_COMMIT"
    } > "$RUNTIME_DIR/$RUNTIME_MARKER"
fi

# Never leave a previously published environment file pointing at a runtime
# that is about to be rebuilt. Reject symlinks before removing the old file;
# a new file is published atomically only after full verification succeeds.
if [[ -e "$RUNTIME_ENV" || -L "$RUNTIME_ENV" ]]; then
    [[ -f "$RUNTIME_ENV" && ! -L "$RUNTIME_ENV" ]] || \
        fail "Runtime environment file is unsafe: $RUNTIME_ENV"
    rm -f -- "$RUNTIME_ENV"
fi

# The runtime twin and reference assets are disposable. Rebuild both from
# committed Git objects on every setup run so stale files cannot survive a
# rerun. Only the verified table texture is overlaid into the disposable twin.
TEMP_RUNTIME="$(mktemp -d "$RUNTIME_ROOT/.setup-$REPO_COMMIT.XXXXXX")"
git archive --format=tar --output="$TEMP_RUNTIME/source.tar" \
    "$REPO_COMMIT" digital-twin-setup reference-assets
tar -xf "$TEMP_RUNTIME/source.tar" -C "$TEMP_RUNTIME"
rm -f -- "$TEMP_RUNTIME/source.tar"
mv "$TEMP_RUNTIME/digital-twin-setup" "$TEMP_RUNTIME/twin"
verify_twin_manifest "$TEMP_RUNTIME/twin" 1
verify_reference_asset_manifest "$TEMP_RUNTIME/reference-assets"
[[ -d "$TEMP_RUNTIME/twin/assets" && ! -L "$TEMP_RUNTIME/twin/assets" ]] || \
    fail "Disposable twin has no safe assets directory"
TABLE_TEXTURE="$TEMP_RUNTIME/reference-assets/table/manipbench-table-surface.png"
TWIN_TABLE_TEXTURE="$TEMP_RUNTIME/twin/assets/mat_print.png"
[[ ! -e "$TWIN_TABLE_TEXTURE" && ! -L "$TWIN_TABLE_TEXTURE" ]] || \
    fail "Supplied twin unexpectedly already contains assets/mat_print.png"
cp -- "$TABLE_TEXTURE" "$TWIN_TABLE_TEXTURE"
cmp -s -- "$TABLE_TEXTURE" "$TWIN_TABLE_TEXTURE" || \
    fail "Could not reproduce the table texture in the disposable twin"

OLD_TWIN="$RUNTIME_DIR/.previous-twin"
OLD_ASSETS="$RUNTIME_DIR/.previous-reference-assets"
for current in "$RUNTIME_TWIN" "$RUNTIME_ASSETS"; do
    if [[ -e "$current" || -L "$current" ]]; then
        [[ -d "$current" && ! -L "$current" ]] || \
            fail "Runtime component is not a regular directory: $current"
    fi
done
for previous in "$OLD_TWIN" "$OLD_ASSETS"; do
    if [[ -e "$previous" || -L "$previous" ]]; then
        [[ -d "$previous" && ! -L "$previous" ]] || \
            fail "Unsafe previous runtime path: $previous"
    fi
done
# Recover a component left only in its backup path by an interrupted rerun.
if [[ -d "$OLD_TWIN" ]]; then
    if [[ -d "$RUNTIME_TWIN" ]]; then
        rm -rf -- "$OLD_TWIN"
    else
        mv "$OLD_TWIN" "$RUNTIME_TWIN"
    fi
fi
if [[ -d "$OLD_ASSETS" ]]; then
    if [[ -d "$RUNTIME_ASSETS" ]]; then
        rm -rf -- "$OLD_ASSETS"
    else
        mv "$OLD_ASSETS" "$RUNTIME_ASSETS"
    fi
fi

TWIN_BACKED_UP=0
ASSETS_BACKED_UP=0
if [[ -d "$RUNTIME_TWIN" ]]; then
    mv "$RUNTIME_TWIN" "$OLD_TWIN"
    TWIN_BACKED_UP=1
fi
if [[ -d "$RUNTIME_ASSETS" ]]; then
    if ! mv "$RUNTIME_ASSETS" "$OLD_ASSETS"; then
        if [[ "$TWIN_BACKED_UP" -eq 1 ]]; then
            mv "$OLD_TWIN" "$RUNTIME_TWIN"
        fi
        fail "Could not preserve the previous runtime reference assets"
    fi
    ASSETS_BACKED_UP=1
fi

if ! mv "$TEMP_RUNTIME/reference-assets" "$RUNTIME_ASSETS"; then
    if [[ "$ASSETS_BACKED_UP" -eq 1 ]]; then
        mv "$OLD_ASSETS" "$RUNTIME_ASSETS"
    fi
    if [[ "$TWIN_BACKED_UP" -eq 1 ]]; then
        mv "$OLD_TWIN" "$RUNTIME_TWIN"
    fi
    fail "Could not install the fresh runtime reference assets"
fi
if ! mv "$TEMP_RUNTIME/twin" "$RUNTIME_TWIN"; then
    if ! mv "$RUNTIME_ASSETS" "$TEMP_RUNTIME/failed-reference-assets"; then
        fail "Could not roll back a partial runtime install; preserved backups remain in $RUNTIME_DIR"
    fi
    if [[ "$ASSETS_BACKED_UP" -eq 1 ]]; then
        mv "$OLD_ASSETS" "$RUNTIME_ASSETS"
    fi
    if [[ "$TWIN_BACKED_UP" -eq 1 ]]; then
        mv "$OLD_TWIN" "$RUNTIME_TWIN"
    fi
    fail "Could not install the fresh runtime twin"
fi

for previous in "$OLD_TWIN" "$OLD_ASSETS"; do
    if [[ -d "$previous" ]]; then
        rm -rf -- "$previous"
    fi
done
rm -rf -- "$TEMP_RUNTIME"
TEMP_RUNTIME=
trap - EXIT
verify_reference_asset_manifest "$RUNTIME_ASSETS"
cmp -s -- \
    "$RUNTIME_ASSETS/table/manipbench-table-surface.png" \
    "$RUNTIME_TWIN/assets/mat_print.png" || \
    fail "Installed twin texture differs from the committed reference asset"
for executable in bootstrap.sh run_sort_pipeline.sh share.sh; do
    [[ -x "$RUNTIME_TWIN/$executable" ]] || \
        fail "Runtime lost executable mode: $RUNTIME_TWIN/$executable"
done

ensure_managed_env() {
    local path="$1"
    local kind="$2"
    local marker="$path/$ENV_MARKER"
    local contents
    if [[ -e "$path" ]]; then
        [[ -d "$path" && ! -L "$path" ]] || \
            fail "$kind environment is not a regular directory: $path"
        if [[ ! -e "$marker" ]]; then
            contents="$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)"
            [[ -z "$contents" ]] || \
                fail "$kind environment exists but is not repository-managed: $path"
        fi
    else
        mkdir -p "$(dirname "$path")"
        mkdir "$path"
    fi
    if [[ ! -e "$marker" ]]; then
        {
            printf 'format=1\n'
            printf 'project=%s\n' "$MARKER_PROJECT"
            printf 'kind=%s\n' "$kind"
            printf 'python=%s\n' "$PYTHON_MAJOR_MINOR"
        } > "$marker"
    fi
    [[ -f "$marker" && ! -L "$marker" ]] || \
        fail "$kind environment has an unsafe ownership marker: $path"
    grep -Fxq "format=1" "$marker" || fail "Unsupported environment marker: $path"
    grep -Fxq "project=$MARKER_PROJECT" "$marker" || \
        fail "$kind environment ownership marker belongs to another project: $path"
    grep -Fxq "kind=$kind" "$marker" || fail "Environment kind mismatch: $path"
    grep -Fxq "python=$PYTHON_MAJOR_MINOR" "$marker" || \
        fail "Environment Python marker mismatch: $path"
    if [[ ! -f "$path/pyvenv.cfg" || ! -x "$path/bin/python" ]]; then
        echo "[setup] creating or repairing managed $kind environment: $path"
        python3 -m venv "$path"
    fi
    [[ -f "$path/pyvenv.cfg" && -x "$path/bin/python" ]] || \
        fail "$kind environment could not be created: $path"
    local actual
    actual="$("$path/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "$actual" == "$PYTHON_MAJOR_MINOR" ]] || \
        fail "$kind environment uses Python $actual, expected $PYTHON_MAJOR_MINOR"
}

ensure_managed_env "$UV_ENV" uv
"$UV_ENV/bin/python" -m pip install --disable-pip-version-check --no-input \
    "pip==$PIP_VERSION" "uv==$UV_VERSION"
"$UV_ENV/bin/uv" pip sync --python "$UV_ENV/bin/python" "$UV_LOCK"
[[ "$("$UV_ENV/bin/uv" --version)" == "uv $UV_VERSION"* ]] || \
    fail "Unexpected uv version"

ensure_managed_env "$RUNTIME_TWIN/.venv" host
mkdir -p "$UV_CACHE_DIR"
UV_CACHE_DIR="$UV_CACHE_DIR" "$UV_ENV/bin/uv" pip sync \
    --python "$RUNTIME_TWIN/.venv/bin/python" "$HOST_LOCK"

echo "[setup] running the supplied host bootstrap in the verified runtime copy"
(
    cd "$RUNTIME_TWIN"
    bash bootstrap.sh
)
UV_CACHE_DIR="$UV_CACHE_DIR" "$UV_ENV/bin/uv" pip sync \
    --python "$RUNTIME_TWIN/.venv/bin/python" "$HOST_LOCK"

ensure_managed_env "$HF_ENV" hf-cli
UV_CACHE_DIR="$UV_CACHE_DIR" "$UV_ENV/bin/uv" pip sync \
    --python "$HF_ENV/bin/python" "$HF_LOCK"
[[ "$("$HF_ENV/bin/hf" version)" == *"$HUGGINGFACE_HUB_VERSION"* ]] || \
    fail "Unexpected Hugging Face CLI version"

ensure_managed_env "$COSMOS_ENV" cosmos
UV_CACHE_DIR="$UV_CACHE_DIR" "$UV_ENV/bin/uv" pip sync \
    --python "$COSMOS_ENV/bin/python" \
    --torch-backend "$TORCH_BACKEND" \
    "$COSMOS_LOCK"

echo "[setup] checking and downloading pinned model revision"
if ! "$HF_ENV/bin/hf" download "$COSMOS3_NANO_REPO" \
    --revision "$COSMOS3_NANO_REVISION" \
    --dry-run; then
    echo "Hugging Face access failed. If authentication is required, run:" >&2
    printf '  %q auth login\n' "$HF_ENV/bin/hf" >&2
    fail "Could not inspect the pinned model revision"
fi
mkdir -p "$MODEL_ROOT"
if ! "$HF_ENV/bin/hf" download "$COSMOS3_NANO_REPO" \
    --revision "$COSMOS3_NANO_REVISION" \
    --local-dir "$MODEL_ROOT"; then
    echo "If authentication is required, run:" >&2
    printf '  %q auth login\n' "$HF_ENV/bin/hf" >&2
    fail "Pinned model download failed; rerun setup to resume after correcting access"
fi

mkdir -p "$RUNS_ROOT"

ROCKWELL_EXTERNAL_ROOT="$EXTERNAL_ROOT" \
COSMOS3_NANO_PATH="$MODEL_ROOT" \
COSMOS_ENV="$COSMOS_ENV" \
UV_ENV="$UV_ENV" \
HF_ENV="$HF_ENV" \
UV_CACHE_DIR="$UV_CACHE_DIR" \
ROCKWELL_RUNTIME_ROOT="$RUNTIME_ROOT" \
ROCKWELL_RUNS_ROOT="$RUNS_ROOT" \
ISAAC="$ISAAC" \
    bash "$SCRIPT_DIR/verify_brev.sh" --full

ENV_TEMP=
cleanup_env_temp() {
    if [[ -n "$ENV_TEMP" && -f "$ENV_TEMP" && ! -L "$ENV_TEMP" ]]; then
        rm -f -- "$ENV_TEMP"
    fi
}
trap cleanup_env_temp EXIT
ENV_TEMP="$(mktemp "$RUNTIME_DIR/.env.sh.XXXXXX")"
{
    printf 'export ISAAC=%q\n' "$ISAAC"
    printf 'export COSMOS=%q\n' "$COSMOS_ENV/bin/python"
    printf 'export HOST_PYTHON=%q\n' "$RUNTIME_TWIN/.venv/bin/python"
    printf 'export COSMOS3_NANO_PATH=%q\n' "$MODEL_ROOT"
    printf 'export ROCKWELL_TWIN_RUNTIME=%q\n' "$RUNTIME_TWIN"
    printf 'export ROCKWELL_REFERENCE_ASSETS=%q\n' "$RUNTIME_ASSETS"
    printf 'export ROCKWELL_RUNS_ROOT=%q\n' "$RUNS_ROOT"
} > "$ENV_TEMP"
mv -f -- "$ENV_TEMP" "$RUNTIME_ENV"
ENV_TEMP=
trap - EXIT

echo
echo "Setup complete. For future sessions run:"
printf '  source %q\n' "$RUNTIME_ENV"
echo "Runtime twin: $RUNTIME_TWIN"
echo "Reference assets: $RUNTIME_ASSETS"
echo "External runs: $RUNS_ROOT"
echo "Optional Hugging Face login (never store a token in Git):"
printf '  %q auth login\n' "$HF_ENV/bin/hf"
