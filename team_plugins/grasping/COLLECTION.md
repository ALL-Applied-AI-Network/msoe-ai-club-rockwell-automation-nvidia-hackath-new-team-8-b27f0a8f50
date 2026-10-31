# Matched Isaac collection procedure

## Exact-commit preconditions

Work from a clean, committed `grasping` branch. Run the full Brev setup for
that exact commit before collection; the preparer, launcher, isolated runner,
and collector all fail closed on relevant dirty source paths or a commit
mismatch. Generated cases, runs, logs, collections, videos, datasets, and model
artifacts must stay outside Git.

Load the setup-created environment for the checked-out commit. Do not assume a
plain `python` command exists on Brev:

```bash
cd /workspace/rockwell-hackathon
RUNTIME_ROOT="${ROCKWELL_RUNTIME_ROOT:-/workspace/rockwell-runtime}"
source "$RUNTIME_ROOT/$(git rev-parse HEAD)/env.sh"
test -x "$HOST_PYTHON"
```

The `HOST_PYTHON` path is commit-addressed by `setup_brev.sh`. Isaac
candidate processes still run through `/isaac-sim/python.sh`.

## Five-object policy smoke

The older five-object smoke is a policy-level regression, not independent
training data:

1. Prepare fresh external cases with
   `"$HOST_PYTHON" -m team_plugins.grasping.prepare_screw_trials OUTPUT`.
2. Use a new run directory and its case root as `ROCKWELL_REPO_ROOT`, then run
   the mixed demo with `--physics-only`.
3. Preserve the complete report, case manifest, pre-action features, commit,
   simulator/contact settings, and source hashes.
4. Do not convert the two sequential screw events into independent rows. One
   global offset affects both screws, and earlier actions can change the later
   scene.

A positive complete-grasp label means the intended object was acquired,
retained through lift and transport, released at the intended destination, and
still passed final scene verification. Cancelled, crashed, or unobserved
attempts are unresolved rather than fabricated negatives.

## Target-isolated screw pilot

The grasping-owned isolated runner keeps all five configured objects in the
physics world but acts on exactly one target. Every candidate launches in a
fresh Isaac process and a new case directory. The runner transforms a catalog
candidate generated from the configured target onto the settled target pose;
it rejects more than 2 mm position drift or 5 degrees axial-yaw drift. This is
simulation-only runtime control from simulator truth, not camera perception or
hardware execution.

After exact-commit setup, create and run the external 4-layout by 3-candidate
matrix:

```bash
PILOT_ROOT="/workspace/grasp-experiments/screw-pilot-$(git rev-parse --short=12 HEAD)"

"$HOST_PYTHON" -m team_plugins.grasping.prepare_screw_policy_matrix \
  "$PILOT_ROOT"

"$HOST_PYTHON" -m team_plugins.grasping.run_screw_policy_matrix \
  "$PILOT_ROOT/experiment_manifest.json" \
  "$PILOT_ROOT/runs" \
  --twin-runtime "$ROCKWELL_TWIN_RUNTIME" \
  --reference-assets "$ROCKWELL_REFERENCE_ASSETS"

"$HOST_PYTHON" -m team_plugins.grasping.collect_screw_policy_matrix \
  "$PILOT_ROOT/experiment_manifest.json" \
  "$PILOT_ROOT/runs" \
  "$PILOT_ROOT/collection"
```

If a reboot interrupts execution, rerun the launcher with `--resume`. A
resolved case is reused only after its identity, commit, scenario hash, split,
metadata, report status, and target-only label agree. An incomplete or
inconsistent case and log are moved without deletion under
`runs/_archived_incomplete/` before a fresh attempt.

The pilot locks these layouts:

- clear at 20 degrees, train;
- 9 mm tip gap at 20 degrees, train;
- 8 mm tip gap at 20 degrees, train;
- 8 mm tip gap at 160 degrees, held-out test.

Each layout contains matched 12/16/20 mm candidates with independent resets.
The 8-9 mm gaps reflect the corrected jaw-axis footprint and leave margin for
approximately 1 mm of pre-action settling observed in the original smoke.

## Label and feature limits

A screw target-only label is resolved only when:

- there is exactly one target event and one-item completion denominator;
- pre-descent target revalidation passes;
- lift, carry, lower, release, settle, and retreat truth records exist;
- the measured screw grasp point stays within 8 mm of the commanded lift,
  carry, and lower endpoints, using the source grasp-point-to-tip offset;
- pickup, placement, trajectory, task, and final-scene checks resolve.

Missing or malformed command/phase evidence remains unresolved. A measured
retention failure is a resolved negative even if the screw later falls into the
correct bin.

This label is explicitly target-only. The other four configured objects remain
physics obstacles, but their final poses and collision/contact safety are not
evaluated. Therefore these rows are not collision-safe labels. Feature
recomputation uses settled simulator truth for the target and configured poses
for neighboring objects; neighboring-object settling is not measured. Both
limitations are recorded in the report and provenance sidecars.

The collector verifies the manifest, matrix report, attempt metadata, report,
scenario hashes, exact current commit, clean source state, group completeness,
and train/test family isolation. It excludes unresolved attempts,
post-settle-rejected candidates, and entire incomplete groups.

## Pilot readiness rule

The 12-attempt collection is diagnostic and always emits
`"trainer_ready": false`, even when all rows are structurally valid. It reports
`training_input_structurally_valid` separately and records each group's
12/16/20 mm outcome pattern. A fail/success/fail pattern is non-monotone in the
v1 raw `axial_offset_m` feature; introduce and review a suitable nonlinear
feature or schema/model revision, then run a larger locked collection before
training. Do not fabricate failures, train a one-class model, or promote this
pilot to a hardware profile.

## Cube/block smoke scope

The same isolated runner can execute cube yaw candidates and verifies the
commanded parallel-jaw axis. Those results remain simulation smoke labels
because synchronized cube retention truth is not recorded throughout carry.
The supported project "blocks" are the configured 10 mm square cube class;
arbitrary rectangular or tilted blocks fail closed. Cube/block smoke results
are never emitted as complete-grasp trainer labels.
