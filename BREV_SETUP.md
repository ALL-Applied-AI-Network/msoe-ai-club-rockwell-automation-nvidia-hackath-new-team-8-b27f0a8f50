# Reproducible Brev setup

This guide prepares the supplied FANUC digital twin on the event's
[Brev Isaac Sim launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-35JP2ywERLgqtD0b0MIeK1HnF46).
The validated launchable ID is `env-35JP2ywERLgqtD0b0MIeK1HnF46`, with Isaac
Sim `6.0.1-rc.7+release.42383.32955d8d.gl`. The Git checkout remains unchanged:
setup uses a
commit-specific disposable copy and stores environments, weights, caches,
captures, runs, and videos outside the repository.

This setup is terminal-only and headless. It does not open the Isaac Sim GUI or
browser viewer and does not require the user to operate an interactive window.
The supplied twin documents an optional interactive scene command, but that is
not part of this setup workflow.

Coding agents must also read `AGENTS.md`, `digital-twin-setup/AGENTS.md`,
`digital-twin-setup/SETUP.md`, and
`digital-twin-setup/docs/MODEL_NOTES.md` before acting. Read
`reference-assets/README.md` before using the separately supplied CAD or
tabletop image.

## 1. Clone into the `vscode` container

Use a clean clone under `/workspace` and confirm that it has no local changes:

```bash
cd /workspace
git clone https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-8-b27f0a8f50 rockwell-hackathon
cd /workspace/rockwell-hackathon
git status --porcelain
```

The final command must print nothing. The setup deliberately refuses to use an
unclean checkout so it cannot depend on files teammates do not have.

On a fresh Brev instance, first verify that GitHub CLI is available inside the
`vscode` container:

```bash
command -v gh
```

If that prints no path, run this once from the Brev host, then repeat the check
inside the `vscode` terminal:

```bash
docker exec --user root vscode bash -lc \
  'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y gh'
```

For every fresh clone, and whenever a resumed Brev session will fetch from or
push to GitHub, initialize the repository's GitHub credential helper from an
interactive terminal in the clone:

```bash
cd /workspace/rockwell-hackathon
bash scripts/ensure_github_auth.sh
```

The command is idempotent. If no valid login is available, the agent must tell
the human that authorization is required and start GitHub CLI's browser/device
flow in that interactive terminal. The human approves the displayed ephemeral
URL and one-time code through GitHub; the agent must not ask the human to paste
a reusable token. The script then reports the active account, verifies through
GitHub's API that it has push permission to this clone's `origin`, and only
then runs `gh auth setup-git`. A valid login on a resumed instance is reused,
but account and repository permission are checked again. A successful login
without push permission is intentionally not reported as ready.

The script does not copy a token into the checkout. Never copy authentication
files from another computer or put a personal token, GitHub CLI configuration
file, device-flow output, or credential output in this repository, logs, or
chat. If a noninteractive invocation reports that human authorization is
required, reopen the clone in an interactive Brev terminal and run the same
script there.

The script also reports credential persistence without printing the secret.
`GH_TOKEN` and `GITHUB_TOKEN` are transient environment credentials. A login
created by `gh auth login` persists beyond the shell and, on a headless Brev
container without a secure credential store, may be stored in plaintext in
`~/.config/gh/hosts.yml`. Anyone able to read that file can reuse the token.
Do not automatically log out while authorized work is running. Before handing
a shared instance to an untrusted user, use the cleanup command printed by the
gate:

```bash
# Stored login (replace ACCOUNT with the account printed by the gate):
gh auth logout --hostname github.com --user ACCOUNT

# Environment-provided login:
unset GH_TOKEN GITHUB_TOKEN
```

After cleanup, `gh auth status --hostname github.com >/dev/null 2>&1` must no
longer succeed for the removed login. The `gh auth setup-git` helper contains
no token itself and may remain configured; without a valid GitHub CLI login it
cannot supply credentials.

## 2. Install the small system prerequisite when necessary

The standard launchable may not include host Python in the `vscode` container.
Check from its terminal:

```bash
python3 --version
```

If that command is missing, run this once from the Brev host—not from inside
the `vscode` terminal:

```bash
docker exec --user root vscode bash -lc \
  'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip'
```

An agent may run that host command. It must not install packages into the Git
checkout or edit the supplied twin.

## 3. Optional Hugging Face authentication

`nvidia/Cosmos3-Nano` is currently public, so anonymous setup works. A personal
read-only Hugging Face token can provide better rate limits. Never paste a token
into chat or save one in Git. If desired, the user authenticates directly after
the setup tool environment exists:

The setup is designed to work anonymously. If the download service later
requires authentication, let setup create the managed CLI environment first,
then run the exact `hf auth login` command printed by setup. Download failures
also print that command. Do not manually create or reuse a shared environment
at that path.

If setup has not created that environment yet, run setup anonymously first; a
failed or interrupted model download is resumable.

## 4. Run the repository setup

From the clean checkout:

```bash
bash scripts/setup_brev.sh
```

The script is fail-closed, resumable, and safe to rerun. Each rerun rebuilds the
small disposable twin runtime from the current Git commit and creates a fresh
validation-output directory. It:

1. Checks Python 3.12, at least 40,000 MiB of GPU memory, free space, and the
   exact validated Isaac Sim version.
2. Verifies every supplied twin file against `digital-twin-setup/MANIFEST.json`
   and every separately supplied asset against `reference-assets/MANIFEST.json`.
3. Exports both committed bundles from Git—not the working directory—to a
   runtime named for the current commit, then verifies their exact inventories
   and required Linux executable modes. It overlays only the verified tabletop
   PNG at the path expected by the disposable runtime twin; source files remain
   unchanged.
4. Creates marked, project-owned environments, applies the committed exact-
   version locks, and runs the twin's original `bootstrap.sh` only in the
   external runtime copy.
5. Installs the validated CUDA 13 Cosmos environment from
   `brev/cosmos-requirements.lock.txt`, including the immutable NVIDIA shim
   revision recorded in that lock.
6. Downloads the pinned `nvidia/Cosmos3-Nano` revision outside Git, confirms no
   revision files are missing, validates index paths, and opens every indexed
   safetensors header.
7. Builds a fresh headless Isaac scene outside the twin and rejects tracebacks
   or a missing USD.
8. Loads Nano on CUDA and performs one real image inference through the twin's
   unchanged `load_model()` and `ask()` functions.
9. Rechecks the source manifest and requires the Git checkout to remain clean.

The launchable, Isaac, model, and tool versions are centralized in
`brev/versions.env`; exact Python resolutions and the immutable NVIDIA shim
commit are in the four lock files under `brev/`. Python 3.12 is enforced, while
3.12.3 is the tested patch release. These locks pin package versions and the
NVIDIA Git commit, but are not bit-for-bit wheel hash locks. Do not replace
them with unpinned `latest` packages.

## External layout

With the defaults, setup creates:

```text
/workspace/rockwell-hackathon/                 # unchanged Git checkout
/workspace/rockwell-hackathon/reference-assets/ # committed source CAD/texture
/workspace/rockwell-runtime/<git-sha>/twin/    # disposable runtime twin
/workspace/rockwell-runtime/<git-sha>/reference-assets/ # verified asset bundle
/workspace/rockwell-runtime/<git-sha>/env.sh   # non-secret runtime paths
/workspace/models/Cosmos3-Nano/                # pinned model snapshot
/workspace/envs/cosmos3/                       # pinned CUDA environment
/workspace/tools/                              # pinned uv and HF CLI tools
/workspace/cache/uv/                           # package cache
/workspace/runs/setup-<git-sha>.<random>/       # fresh validation USD and logs
/workspace/runs/                               # later experiment outputs
```

Set `ROCKWELL_EXTERNAL_ROOT` before setup to replace `/workspace` with another
existing external storage root. Every managed path must be a distinct
descendant of that root and outside the repository. Existing environments are
accepted only when they carry the matching ownership marker; this prevents an
exact package sync from deleting packages in a shared or unrelated environment.

## 5. Verify an existing setup

Run the inexpensive checks, host tests, model-shard checks, and CUDA imports:

```bash
bash scripts/verify_brev.sh
```

Also rebuild the Isaac scene, reload Nano, and run image inference:

```bash
bash scripts/verify_brev.sh --full
```

Both commands must finish with `Brev verification passed`. Verification compares
all managed environments with their committed locks. A zero process exit from
Isaac alone is insufficient; the full verifier creates a new output directory,
requires a nonempty USD, and rejects Python tracebacks. This is a setup smoke
test (scene export plus model inference), not a rendered robot-motion or full
sorting-pipeline test.

The pristine starter omits `assets/mat_print.png`. Setup fills that one path in
the disposable runtime with the manifest-verified
`reference-assets/table/manipbench-table-surface.png`, so the old missing-file
warning must not appear. `MAT_MARKINGS = False` remains unchanged, which means
the image is available but is not rendered as tabletop artwork by default.
The STEP, STL, and USDA reference models are copied to Brev but are not silently
substituted for the starter's geometry.

## 6. Use the configured runtime

Load the generated, non-secret environment file for the checked-out commit:

```bash
EXTERNAL_ROOT="${ROCKWELL_EXTERNAL_ROOT:-/workspace}"
RUNTIME_ROOT="${ROCKWELL_RUNTIME_ROOT:-$EXTERNAL_ROOT/rockwell-runtime}"
source "$RUNTIME_ROOT/$(git rev-parse HEAD)/env.sh"
cd "$ROCKWELL_TWIN_RUNTIME"
```

The sourced file also exports `ROCKWELL_REFERENCE_ASSETS`, the verified runtime
copy of the committed CAD and tabletop bundle.

For a complete pipeline run, always choose a new empty directory outside the
twin:

```bash
bash run_sort_pipeline.sh "$ROCKWELL_RUNS_ROOT/stack_01"
```

Do not claim the full pipeline succeeded unless `execution_report.json` exists,
contains `task_complete: true`, and reports six verified pickups and six
verified placements. Inspect the perception evaluation, plan, logs, overlays,
and videos as described by the twin's own instructions.

## Security and trust boundary

Use only the pinned official `nvidia/Cosmos3-Nano` revision. The compatible
2026 stack includes `transformers==5.8.0` and `accelerate==1.13.0`, which are
affected by published advisories CVE-2026-80047 and CVE-2026-69112. The former
code path is not used here; the latter makes untrusted checkpoint indexes
unsafe. The verifier rejects index path traversal before loading, but this does
not make arbitrary third-party weights trusted. Revisit these pins when NVIDIA
publishes a shim-compatible fixed stack; do not blindly upgrade them.

If setup stops after creating a partial marked environment, rerun it; package
sync is resumable. If it reports an unmarked pre-existing environment, move
that directory aside or choose new external paths instead of asking setup to
take ownership of unknown contents.

Repository maintainers committing from Windows must preserve executable mode
`100755` for `scripts/setup_brev.sh`, `scripts/verify_brev.sh`, and the supplied
`bootstrap.sh`, `run_sort_pipeline.sh`, and `share.sh`. Setup refuses a commit
that loses those modes; changing the Git mode does not change twin file bytes.

## What is never committed

Do not commit model weights, authentication tokens, environments, package
caches, generated textures, simulator captures, run directories, videos, or
machine-specific paths. The small source/reference files under
`reference-assets/` are an intentional exception and must continue to match
their manifest. The project owner confirmed redistribution in this repository
on 2026-10-24; retain the provenance and attribution recorded in
`reference-assets/README.md`.

