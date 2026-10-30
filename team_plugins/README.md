# Team plugin workspace

This directory is the working boundary for the computer-vision and grasping
teams. Teammates normally add files below their own subdirectory and select
them through an integration profile. They do not need to modify the working
cube demo, physical-robot policy, Isaac adapters, or shared contracts.

Start with the setup guide for your assigned role:

- `computer_vision/SETUP.md` on branch `computer-vision`;
- `grasping/SETUP.md` on branch `grasping`; or
- `lead/SETUP.md` on the integration branch `main`.

The active project scope is known cubes plus the supplied ISO 10642 M6×30
screw. Vision hands off a typed `SceneObservation` (or its validated JSON
serialization), never a prose string. Grasping selects the feasible candidate
with the highest predicted probability of retaining the object through the
complete move and commanded release. Predicting or balancing execution time is
not part of the current project objective.

The checked-in example factories deliberately reproduce the current baseline:

- `computer_vision/baseline_provider.py` wraps the deterministic table-camera
  colour-cube provider;
- `grasping/baseline_pipeline.py` builds the existing single top-down cube
  grasp pipeline.

Importing an example does not activate it. With
`ROCKWELL_INTEGRATION_PROFILE` unset, the application uses exactly the
pre-plugin baseline. To exercise both example factories from the repository
root:

```bash
export ROCKWELL_INTEGRATION_PROFILE=team_plugins/profiles/baseline_plugins.json
```

The same relative profile path works in WSL and Brev because it resolves from
the repository root. Restart the Python process after changing it: the
official client reads requested cameras while importing `Policy`.

Each team also has an independently selectable `profile.example.json`. Copy
that file under a new descriptive name and change its factory/options; do not
have both teams repeatedly edit one shared working profile during development.

## Parallel-work and merge protocol

The repository uses `main`, not `master`. Each team changes only its owned
directory and its own named profile while developing. Small serialized fixtures
may be exchanged through reviewed Git files; datasets and model weights stay in
documented external storage.

At the beginning of a work session, before a costly Brev run, and before a
handoff, update a **clean** team branch. Substitute `grasping` for
`computer-vision` when appropriate:

```bash
git switch computer-vision
git status --short
# Stop and ask the AI/lead if the status is not empty.
git fetch origin
git merge --ff-only origin/computer-vision
git merge --no-edit origin/main
```

If Git reports a conflict, do not reset, discard, or guess at another person's
change. Show the conflicting paths to the AI and integration lead. After the
merge succeeds, run the role guide's focused tests. Then publish only the team
branch:

```bash
git push origin computer-vision
```

Team members never merge their own branch into `main`. The integration lead:

1. reviews each team handoff and its tests;
2. creates and version-bumps a **new** combined profile under `profiles/`;
3. integrates through a reviewed merge or pull request;
4. runs the combined profile first on offline replays, then Isaac camera input;
5. treats a physical run as a separate reviewed step—successful imports or
   simulator motion are not evidence of physical readiness.

If either component needs a shared-contract change, stop and describe the
missing field or behavior with a failing fixture. The integration lead owns the
reviewed contract/version change so one team cannot silently break the other
implementation or the baseline.

This keeps normal merges additive: provider/pipeline modules, their tests, and
their individual profiles should not overlap. Only the integration lead edits
the combined profile or shared contracts.

## Package and dependency model

The supported extension mechanism is an explicit `module.path:callable`
factory named in a reviewed profile. It intentionally does not scan arbitrary
plugin folders or discover Python entry points: explicit selection makes the
loaded code and its provenance visible in logs and code review.

The lowest-risk path for this hackathon is to keep the lightweight provider or
pipeline adapter in this checked-in `team_plugins` package. A separate Python
package also works without core edits when it is installed reproducibly in the
process environment and its module is importable, but it must be pinned by a
lockfile and tested in both the intended WSL and Brev launch commands. Do not
depend on a machine-wide `PYTHONPATH` or an unrecorded editable install.

Heavy vision runtimes with conflicting dependencies should normally run in a
separately pinned local service. Keep only a small, timeout-bounded adapter in
this package. The profile is authorization to import and execute Python code,
so select only reviewed factories and never point it at an untrusted checkout
or model artifact.

## Profile contract

A strict schema-version-1 profile has these fields:

```json
{
  "schema_version": 1,
  "profile_id": "unique-versioned-name",
  "vision": {
    "factory": "package.module:callable",
    "camera_ids": ["table"],
    "depth_camera_ids": [],
    "calibration_ids": {
      "table": "sidclab-cameras.json:table"
    },
    "depth_calibration_ids": {},
    "options": {}
  },
  "grasp": {
    "factory": "package.module:callable",
    "options": {}
  }
}
```

A profile may select only vision or only grasping; the omitted component keeps
the baseline. Factory modules are imported lazily only when their component is
constructed. Invalid profiles, imports, factories, and returned objects fail
closed. Profiles do not change which object classes the current task adapter
can execute.

The recorded SHA-256 covers the profile JSON, not imported Python, model
weights, or external services. Any behavior-affecting change requires a new
`profile_id`; run/evaluation provenance must also record the Git commit,
provider/scorer version, model artifact hash, and applicable calibration or
feature-schema version.

## Ownership

| Directory | Owner | Responsibility |
|---|---|---|
| `computer_vision/` | Vision team | Camera inference, pose estimation, tracking, and `SceneObservation` output |
| `grasping/` | Grasp team | Candidate generation, deterministic feasibility, features, scoring, and selection |
| `lead/` and `profiles/` | Integration lead | Role onboarding, combined profiles, reviewed cross-team integration |
| `sorting_core/` | Integration lead | Stable shared contracts and runtime validation |
| `isaac_bridge/` | Integration lead | Isaac frame and execution adapter |
| `hardware/fanuc_cell/` | Integration lead/operator workflow | Official physical-cell client and bounded action execution |

Read the `SETUP.md` and then `AGENTS.md` inside your role directory before
changing anything. `AGENTS.md` is both a human technical guide and the
instruction boundary for coding agents working in that subtree.

## Rules shared by both teams

1. Keep model code, adapters, fixtures, and plugin-specific tests in your team
   directory whenever possible.
2. Do not commit model weights, datasets, generated frames, videos, virtual
   environments, caches, tokens, or machine-specific absolute paths.
3. Heavy model dependencies belong in a plugin-owned, pinned environment or
   service. Do not install packages ad hoc into the repository's managed Brev,
   Cosmos, Isaac, or hardware-client environments.
4. An explicitly selected plugin fails closed. An import, checkpoint,
   inference, timeout, schema, or scoring error must be reported; it must not
   silently switch to the baseline and continue moving a robot.
5. Simulator truth is evaluator data, never runtime perception input.
6. Use the exact shared typed contracts. Do not hand off an informal string or
   an unversioned dictionary between teams.
7. Preserve the current baseline as an evaluation comparison after a learned
   component works.

## Current scope boundary

The generic vision contract can describe arbitrary known objects and full 6-DoF
poses. The current task and executor cannot yet act on all of them: the task
adapter projects supported detections into a colour-cube scene, and the
executable pose is a vertical top-down fingertip pose with yaw. Screws, nails,
vials, tilted parts, and piles require completed object-catalog entries plus
appropriate task, grasp, feasibility, and motion support. Do not bypass that
boundary by labeling a new object as a cube.

The current simulation baseline uses **10 mm cubes**. A separately selected
physical vision profile supports 16 mm detection, but this does not make the
whole planner dimension-generic. Record the actual object size in every test
and never silently mix the 10 mm and 16 mm assumptions.

## Shared references

- [`docs/TEAM_INTEGRATION.md`](../docs/TEAM_INTEGRATION.md): exact runtime
  contracts and examples.
- [`docs/OBJECT_CATALOG.md`](../docs/OBJECT_CATALOG.md): object-local axes,
  symmetry, dimensions, and new-class checklist.
- [`docs/schemas/scene-observation.v1.schema.json`](../docs/schemas/scene-observation.v1.schema.json):
  serialized vision-result shape.
- [`docs/SYNTHETIC_DATA.md`](../docs/SYNTHETIC_DATA.md): simulator data,
  camera-domain limitations, and label provenance.
