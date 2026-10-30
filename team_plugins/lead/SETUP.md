# Lead developer setup

This guide is for the person integrating computer vision, grasp selection,
Isaac simulation and the supervised FANUC hardware path. If a command or
interface is unclear, ask the AI to explain the prerequisite, expected output
and rollback before running it.

## Assignment

The lead owns:

- shared typed contracts, task/destination adapters and versioning;
- combined integration profiles and cross-team fixtures;
- reproducible WSL/Brev environments and complete regression testing;
- Isaac camera/training integration, including the missing realistic screw
  perception-data path;
- hardware-client integration, run metadata and supervised test procedure;
- branch review/merge, end-to-end evaluation and honest project claims.

The lead does not replace the computer-vision or grasping teams' model work.
Require each team to satisfy its `SETUP.md` and `AGENTS.md` handoff checklist.

## Branch and integration workflow

The canonical branch is `main`, not `master`. Begin only from a clean checkout:

```bash
git switch main
git status --short
# Continue only when the status is empty.
git fetch origin
git merge --ff-only origin/main
```

Vision develops on `computer-vision`; grasping develops on `grasping`. They
merge `origin/main` into their own branches, never directly into `main`.

When a team requests integration:

1. fetch and inspect the remote branch and its diff;
2. read its handoff evidence and run its focused tests;
3. require a valid independent profile and small serialized fixtures;
4. resolve interface changes on a reviewed branch/PR—never by weakening tests;
5. create a new versioned combined profile under `team_plugins/profiles/`;
6. run offline replay, combined host tests, then Isaac camera input;
7. merge only reviewed, reproducible work and push `main`;
8. tell both teams to merge the new `origin/main` back into their branches.

If Git conflicts, preserve both sides, identify the semantic owner, and ask the
affected teammate. Do not use destructive reset/checkout or silently choose one
implementation.

For each independent or combined handoff, use the one-command gate in
[`docs/TEAM_PLUGIN_CHECKER.md`](../../docs/TEAM_PLUGIN_CHECKER.md) and pass the
exact versioned profile under review:

```bash
uv run --project hardware/fanuc_cell python scripts/check_team_plugin.py \
  --profile team_plugins/profiles/combined.v1.json
```

CI is a smoke/profile-import gate; this explicit local command verifies that
the intended versioned profile was selected for the CPU contract gate. A
handoff run must be clean. `--allow-dirty` is for development only.

## Active architecture and objective

The cross-team handoff is `SceneObservation`, not an informal string. JSON is
only its validated serialized representation. Preserve object identity,
classes, poses/units/frames, dimensions, uncertainty/quality, timestamps,
camera/calibration IDs and provider provenance.

The grasp pipeline first rejects deterministically invalid candidates, then
uses `HighestSuccessSelector`. “Success” covers acquisition, retained lift and
transport, correct placement and commanded release. Do not activate or evaluate
success/time selection. Continue logging measured timing as operational data.

The required object scope is cubes plus the supplied ISO 10642 M6×30 screw.
Never route screws through the cube compatibility mapping. A real screw path
requires its catalog entry, vision evidence, destination semantics, candidate
grasps, feasibility checks and supported executable motion.

## First integration tasks

1. Confirm each role has checked out its proper branch and can run focused
   tests without shared edits.
2. Freeze the cube and provisional screw catalog entries, including CAD hashes,
   local axes, symmetry, dimensions, grasp regions and unknown physical facts.
3. Keep cube camera/training mode working while adding screw instances and
   camera-derived labels to a realistic synthetic capture path. The existing
   screw physics demo has configured poses and no training-data mode.
4. Validate the vision provider independently on saved real and held-out
   synthetic frames.
5. Validate grasp generation/scoring independently from serialized scenes.
6. Combine through a new profile, replay first, then run Isaac camera-derived
   decisions. Evaluator truth may score outcomes but may not alter decisions.
7. Use the physical runbook only after the official client/mock and supervised
   hardware prerequisites are current.

## Realistic-mode responsibility

The existing cube `--training-data` mode is the reference implementation for
native multicamera frames, calibrated appearance, domain randomization, sensor
effects, synchronized labels and evidence manifests. It is not proof of
photorealism or real performance.

The lead owns the shared change needed to represent screws in an equivalent
camera/training path. Reuse `docs/SYNTHETIC_DATA.md` and its label/provenance
contract rather than inventing a second dataset format. Preserve the ordinary
cube demo and default physics/rendering regression.

## Practical innovation filter

Encourage small experiments such as:

- pretrained segmentation plus calibrated geometry versus training from
  scratch;
- screw-axis keypoints or mask principal axis versus a full pose network;
- a few explicit cube/screw grasp families versus unconstrained generation;
- logistic regression or a small random forest versus a deep grasp scorer;
- targeted second-camera use only where measured occlusion requires it.

Defer RL, learned world models, arbitrary clutter/piles, moving interception and
general 6-DoF manipulation until the complete cube-and-screw baseline works.

## Required lead deliverables

- reviewed team profiles and one versioned combined profile;
- complete cube/screw catalog and interface decisions;
- repeatable realistic cube-and-screw data-generation instructions;
- combined offline replay and Isaac camera-derived evaluation;
- unchanged plugin-unset baseline and full passing host suite;
- current physical runbook, submission/layout metadata and experiment logs;
- explicit unsupported conditions, failed trials and remaining risks;
- Git commit, model/profile/calibration/feature versions for every result.

## Done criteria

Integration is ready for a demonstration only when:

1. a clean clone can reproduce setup and tests;
2. both independent profiles and the combined profile load fail-closed;
3. the vision-to-grasp fixture round-trips without lost units/provenance;
4. cube and any claimed screw decisions come only from camera observations;
5. deterministic feasibility precedes highest-success scoring;
6. simulation and hardware adapters preserve the same semantics;
7. evaluation includes failures and does not overclaim untested hardware;
8. team branches have merged the latest `origin/main` and all work is pushed.

At every blocked dependency, ask the AI for the smallest read-only check or
experiment that can resolve it before introducing another framework.
