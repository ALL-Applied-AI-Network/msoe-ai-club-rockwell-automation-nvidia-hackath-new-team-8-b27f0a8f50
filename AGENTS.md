# Repository instructions for coding agents

## Default response to "Set this up"

When the user says `Set this up`, `Get this running`, or an equivalent setup
request without naming a different target, treat it as authorization to perform
the complete clean-clone Brev setup. Do not ask the user to restate these
instructions or confirm routine setup steps.

1. Read `BREV_SETUP.md`, then the nested agent/setup/model notes named below.
2. Require a clean Git checkout. Never discard, stash, overwrite, or silently
   include local changes. If the checkout is not clean, report the exact paths
   and stop before setup because the runtime must come from committed Git.
3. Run `bash scripts/setup_brev.sh` from the repository root. Let that script
   create the external environments, download the pinned Nano revision, build
   the disposable runtime, and perform full verification. Do not edit tracked
   files, improvise package versions, or manually copy weights into the repo.
4. If host Python is missing, follow only the prerequisite procedure in
   `BREV_SETUP.md`. If Hugging Face authentication becomes necessary, give the
   user the printed local `hf auth login` command; never request a token in chat.
   Safe interrupted downloads and marked project environments may be rerun.
5. Do not report success unless setup ends with `Brev verification passed`, the
   generated `env.sh` exists, a nonempty headless Isaac USD was exported, real
   Nano CUDA inference completed, and `git status --porcelain` is empty.
6. Report the commit SHA, GPU, Isaac version, verification result, runtime/model/
   run paths, and any warnings that affect use. A full sorting or motion run is
   a separate task and is not implied by a setup-only request.

If a required launchable, GPU capacity, disk space, network service, or user
authentication is unavailable, complete every safe independent check first,
then report the precise blocker without weakening validation.

## Setup objective

Make the project reproducible from a clean clone on the event's Brev Isaac Sim
launchable. Read `BREV_SETUP.md` first, then read
`digital-twin-setup/AGENTS.md`, `digital-twin-setup/SETUP.md`, and
`digital-twin-setup/docs/MODEL_NOTES.md` before running or changing the twin.
Read `reference-assets/README.md` before using the separately supplied CAD or
tabletop image.

## Repository boundary

- Treat `digital-twin-setup/` as pristine supplied source unless the user
  explicitly requests a source change.
- Treat `reference-assets/` as an exact, manifest-verified source bundle. Do not
  edit its files in place or silently substitute its reference CAD into the
  scene. The setup overlay activates only the tabletop file's expected runtime
  path; geometry changes require separate Isaac validation.
- Run `scripts/setup_brev.sh` from the checkout. It creates a commit-specific
  disposable copy from committed Git objects and puts all environments,
  models, caches, and outputs outside the repository.
- Do not point setup at a shared Python environment or add ownership-marker
  files by hand. Let setup create its marked environments.
- Never place weights, credentials, environments, generated textures, captures,
  runs, or videos in Git.
- Never ask the user to paste a Hugging Face token into chat. If authentication
  is necessary, have the user run `hf auth login` directly in their terminal.
- Do not weaken tests or runtime gates to make setup appear successful. Report
  the exact completed checks and any remaining failure.

## Reproducibility

Versions and immutable upstream revisions live in `brev/versions.env`; exact
Python resolutions live in the four `brev/*requirements.lock.txt` files.
The pristine twin and reference-asset byte inventories are pinned separately in
their respective `MANIFEST.json` files.
Update those files only after testing the replacement on a clean runtime.
For a new setup, the full verifier invoked by `scripts/setup_brev.sh` satisfies
the validation requirement. When checking an existing setup without rebuilding
it, run `bash scripts/verify_brev.sh --full`. In either case, confirm that
`git status --porcelain` is empty before reporting success.

