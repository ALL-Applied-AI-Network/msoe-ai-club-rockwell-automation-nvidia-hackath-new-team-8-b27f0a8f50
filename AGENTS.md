# Repository instructions for coding agents

## One-time role selection gate

At the start of setup for a fresh clone, or whenever the user's role has not
already been established, ask exactly one short question before running setup
or editing files:

> Are you working as **computer vision**, **grasping**, or **lead developer**?

Do not ask again in every turn. If the user already named a role, use it without
reconfirmation. The role controls the onboarding guide, working branch, and
normal edit boundary:

| Role | Read before work | Working branch | Normal edit boundary |
|---|---|---|---|
| Computer vision | `team_plugins/computer_vision/SETUP.md`, then its `AGENTS.md` | `computer-vision` | `team_plugins/computer_vision/**` |
| Grasping | `team_plugins/grasping/SETUP.md`, then its `AGENTS.md` | `grasping` | `team_plugins/grasping/**` |
| Lead developer | `team_plugins/lead/SETUP.md`, then `team_plugins/lead/AGENTS.md` | `main` | Reviewed integration and shared files |

The canonical integration branch is `main`, not `master`. A team agent must
check its branch before editing, regularly merge `origin/main` into its own
clean team branch as described in the role setup guide, and never push team
work directly to `main`. If a needed change falls outside the team's normal
boundary and could affect another person's work, stop, describe the interface
need or failing fixture, and ask the lead to integrate it. Do not evade this by
copying shared logic into the team folder.

Immediately after the role is selected, the agent must clearly tell the
teammate: (1) their assignment, (2) the concrete first deliverable, (3) which
folder and branch they own, (4) what they must not change independently, and
(5) the first small test or command to run. Do not merely read the role files
silently or assume that the teammate already understands the division of work.

Role selection is the one permitted setup question when the user merely says
"Set this up." It selects a workflow; it does not authorize physical robot
motion, controller changes, destructive Git operations, or secret handling.

## Beginner coaching expected from agents

Assume teammates are still learning AI and robotics. Before an unfamiliar
command or design step, briefly state its prerequisite, input, output, and how
success will be checked. Prefer one small experiment over an unexplained large
rewrite. When useful, suggest one to three low-risk ideas that fit the remaining
hackathon scope, and distinguish proven behavior from a hypothesis.

Invite the teammate to ask the AI when a term, error, tradeoff, or next action
is unclear. A useful reminder belongs at onboarding, blockers, major decisions,
and handoff—not mechanically in every response. Never hide uncertainty to make
the project sound more complete.

## Brev GitHub authentication startup gate

On a fresh Brev instance or fresh clone, establish GitHub CLI access before the
first authenticated GitHub operation. First locate the CLI inside the `vscode`
container:

```bash
command -v gh
```

If that prints no path, install the distro's GitHub CLI package from the Brev
host, then verify that `command -v gh` succeeds in the `vscode` terminal:

```bash
docker exec --user root vscode bash -lc \
  'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y gh'
```

Then run this from the repository root in an interactive Brev terminal:

```bash
bash scripts/ensure_github_auth.sh
```

The agent must explicitly tell the human when GitHub authorization is needed,
start the interactive `gh auth login --web` browser/device flow, and wait for
the human to approve it through GitHub. Only the ephemeral GitHub URL and
one-time device code may be shown, and only in the intended human's visible
interactive terminal. Do not report GitHub access as ready merely because login
succeeded. The gate must identify the active account, verify through GitHub's
API that it has push permission to this clone's `origin`, and then run
`gh auth setup-git` so HTTPS Git operations use GitHub CLI's credential helper.
On a resumed Brev instance, the same gate may reuse a valid existing login, but
it must repeat the account and push-permission checks.

If the check is being run without an interactive terminal and authorization is
needed, stop and ask the human to open the interactive Brev terminal; do not
replace the device flow with a token prompt. After the human authorizes the
displayed one-time code, rerun the script if the login command was interrupted.

Never put a GitHub token, the contents of `hosts.yml`, credential-helper output,
or another reusable credential in the repository, command history, logs, chat,
or agent messages. Suppress `gh auth status` output in automation because some
versions disclose credential-source details; the gate prints only the account
name and storage category.
Never attempt an unattended login by copying credentials from the local
computer. The one-time device code is entered only into GitHub's authorization
page as directed by the CLI; it is not a reusable credential and must not be
committed or preserved. GitHub authorization is user-specific: the gate may
reuse a login already stored by GitHub CLI on that Brev instance, but it cannot
safely create one without the human's interactive approval.

An environment-provided `GH_TOKEN` or `GITHUB_TOKEN` is transient, while
`gh auth login` normally persists beyond the current shell and may fall back to
plaintext `~/.config/gh/hosts.yml` when no secure credential store is available.
The gate reports which case applies without printing the credential. On a
shared Brev instance, tell the human about that risk and give the cleanup
command printed by the gate. For a stored login this is
`gh auth logout --hostname github.com --user <reported-account>`; for an
environment credential it is `unset GH_TOKEN GITHUB_TOKEN`. Never log the user
out automatically, because the same instance may still be doing authorized
work.

## Default response to "Set this up"

When the user says `Set this up`, `Get this running`, or an equivalent setup
request without naming a different target, apply the one-time role gate above,
read that role's setup guide, and then treat the request as authorization to
perform the complete clean-clone Brev setup. Do not ask the user to restate
these instructions or confirm routine setup steps.

Default setup is terminal-only and headless. Do not open the Isaac Sim GUI,
browser viewer, or another interactive simulator window, and do not ask the
user to operate one. The optional interactive command in the supplied twin's
`SETUP.md` is for a later explicitly requested inspection, not setup.

1. Read the selected role's `SETUP.md` and `AGENTS.md`, then `BREV_SETUP.md`
   and the nested agent/setup/model notes named below.
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

