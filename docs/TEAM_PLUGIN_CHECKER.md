# One-command team plugin check

Run this check before handing vision or grasp work to the integration lead. It
validates one selected integration profile, runs the shared golden handoff and
component contracts with that profile selected, then unsets the profile and
runs the complete CPU baseline suite.

The checker does **not** start Isaac, use a GPU, contact the physical robot,
submit a hardware request, download a model, or fetch Git refs. It is a source
and CPU-contract gate, not evidence that perception or a grasp works in the
real world.

## Before running it

From the repository root, update the local remote references while network
access is available:

```bash
git status --short
git fetch origin
```

`git status --short` should be empty before merging `origin/main`; do not reset
or discard another person's work. On `computer-vision` or `grasping`, the
checker fails when the locally fetched `origin/main` is not an ancestor of the
current branch. It also fails a handoff when tracked or untracked working-tree
changes remain. It never performs the fetch itself.

During development only, `--allow-dirty` changes the dirty-tree failure into a
visible warning so you can run an in-progress check. Remove that flag, commit
the intended files, and rerun before handoff. Commits that are ahead of the
configured upstream are shown as a warning rather than a failure because this
check normally runs before `git push`; push them after the check passes.

## Run your actual profile

Computer-vision example:

```bash
uv run --project hardware/fanuc_cell python scripts/check_team_plugin.py \
  --profile team_plugins/computer_vision/my_profile.v1.json
```

Grasping example:

```bash
uv run --project hardware/fanuc_cell python scripts/check_team_plugin.py \
  --profile team_plugins/grasping/my_profile.v1.json
```

If `--profile` is omitted, the checker uses the current branch's checked-in
example profile. That is useful for setup, but it does not test a teammate's
new provider or scorer. Always pass the new versioned profile before handoff.

The command checks, in order:

1. clean working-tree state and visible upstream status;
2. team-branch ancestry against the already-fetched `origin/main`;
3. strict profile JSON and every selected `module:factory` import;
4. integration-profile contracts;
5. golden vision-to-grasp serialized fixtures;
6. shared perception and/or grasp contracts selected by the profile;
7. optional tests below the owning team's `tests/` folder; and
8. the complete baseline suite with every inherited `ROCKWELL_*` variable
   removed. Active phases receive only the selected
   `ROCKWELL_INTEGRATION_PROFILE`.

Any failure stops the handoff. Do not bypass a failed phase by choosing a
different example profile. Fix the stated profile, fixture, contract, or
baseline regression and rerun the same command.

The shared golden-fixture phase exercises serialization and the baseline
vision-to-grasp boundary. Profile validation imports a selected factory but
does not run heavyweight model inference. A teammate's own tests must
instantiate that factory and consume representative saved inputs; GPU/model
quality and physical behavior remain separate evidence gates.

For a fast profile/factory import diagnosis without tests:

```bash
uv run --project hardware/fanuc_cell python scripts/check_team_plugin.py \
  --profile path/to/profile.json --validate-only
```

`--skip-branch-sync` exists for detached CI checkouts. Teammates should not use
it to hide a branch that has not merged current `origin/main`.

## GitHub Actions

`.github/workflows/cpu-contracts.yml` runs the same checker for pushes and pull
requests involving `main`, `computer-vision`, and `grasping`. It uses only the
pinned CPU environment. A push to either team branch is also rejected when it
does not contain the fetched `origin/main`. For pull requests, CI chooses a
recognized team profile from the head branch, then the base branch, otherwise
the combined baseline profile. This is deliberately a smoke/profile-import and
CPU-contract gate. It does **not** know which new versioned profile a feature
branch intends to deliver. The teammate must run that exact profile locally
with `--profile` before handoff. The baseline suite also imports and validates
every checked-in profile-shaped JSON below `team_plugins`. Isaac/Brev rendering,
model-quality evaluation, and physical-hardware gates remain explicit separate
steps.

If a failure message or term is unclear, ask the coding AI to explain the
prerequisite, expected output, and smallest test that isolates the problem.
