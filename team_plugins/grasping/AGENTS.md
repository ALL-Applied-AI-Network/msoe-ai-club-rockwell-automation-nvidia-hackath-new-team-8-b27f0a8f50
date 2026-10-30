# Grasping plugin instructions

These instructions apply to every file below `team_plugins/grasping/`. They are
onboarding for human teammates and binding guidance for coding agents.
The repository-root `AGENTS.md` also applies.

Read `SETUP.md` before the first change. The normal working branch is
`grasping`; verify it before editing. Keep changes in this directory unless the
integration lead explicitly accepts a shared-interface change. When the
teammate is uncertain about terminology, model choice, an error, or a next
experiment, explain the options and remind them they can ask the AI.

At the first grasping onboarding response, clearly summarize the assignment,
first candidate/decision-fixture deliverable, owned branch/folder, prohibited
shared edits, and first replay/contract test. Do not only point at this file.

## Mission and boundary

The grasping team turns a camera-derived, scene-bound target into auditable
grasp candidates, rejects geometrically or mechanically invalid candidates,
extracts only pre-action features, estimates outcomes for the remaining
candidates, and selects one according to an explicit objective.

The team owns:

- candidate generation for supported object/gripper families;
- deterministic jaw-fit, approach, lift, transport, placement, and collision
  preflight appropriate to the available geometry;
- the shared feature extractor used during data collection and inference;
- complete-grasp success-probability prediction;
- candidate selection and complete decision provenance;
- grasp-only offline evaluation.

The team does **not** own camera calibration, object detection/tracking,
destinations, low-level controller commands, physical interlocks, or simulator
ground truth. A learned score never grants permission to bypass deterministic
feasibility checks.

For this project, a grasp succeeds only when the gripper acquires the intended
object and retains it through the complete relocation until the commanded
release. Merely closing the jaws, lifting briefly, or touching the object is
not success.

The active objective is simple: after deterministic feasibility rejection,
select the candidate with the highest predicted complete-grasp success
probability. Do not train a time predictor or balance success against execution
time. Measured execution/recovery time may still be logged as diagnostic data,
but it does not affect candidate ranking.

The required object scope is cubes plus the supplied ISO 10642 M6×30 screw.
The screw simulation demonstrates a configured shaft grasp, but camera-derived
screw targeting and physical retention are not yet proven. Treat those as work
to validate, not assumptions. Consume the shared typed `SceneObservation` (or
its validated JSON serialization), never an informal object string.

## Activation profile

Plugins are selected through `ROCKWELL_INTEGRATION_PROFILE`. A relative JSON
path resolves from the repository root. With the variable unset, the original
single top-down cube pipeline remains active.

A grasp entry has this exact shape:

```json
{
  "factory": "team_plugins.grasping.my_pipeline:create",
  "options": {
    "selector": "highest_success"
  }
}
```

The profile may select grasping without selecting vision; the omitted vision
component remains the baseline. See
`team_plugins/profiles/baseline_plugins.json` for a complete example.

## Start here: first implementation loop

Work from the repository root and preserve a geometric baseline throughout:

1. Read this file, `docs/TEAM_INTEGRATION.md`, and every relevant object entry
   in `docs/OBJECT_CATALOG.md`.
2. Copy `baseline_pipeline.py` to a descriptively named module. First replace
   one component at a time (generator, checker, feature extractor, scorer, or
   selector) so failures have an identifiable source.
3. Copy `profile.example.json` to a new, versioned profile name and point it at
   your `create(context)` factory. Never overwrite the vision team's profile.
4. Build tests from serialized camera-derived scenes. The grasp model must be
   testable without importing the vision model, Isaac, or the physical client.
5. Select the profile before Python starts, run focused contract tests, compare
   against the unchanged geometric baseline, then unset the profile and run
   the full regression suite.
6. Hand off the factory, profile, feature schema, model/checkpoint hash,
   candidate/rejection examples, probability-calibration/ranking results,
   measured latency, known failures, and supported object/gripper combinations.

In WSL Ubuntu or Brev, the shared validation loop is:

```bash
export ROCKWELL_INTEGRATION_PROFILE=team_plugins/grasping/profile.example.json
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_integration_plugins.py' -v
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_grasping_integration.py' -v
unset ROCKWELL_INTEGRATION_PROFILE
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -v
```

The checked-in example proves only that the plugin wiring reproduces the
legacy pipeline. Add team-owned tests below this directory for every new
component and run those in the component's locked environment as well.

Expected team-owned additions normally look like:

```text
team_plugins/grasping/
  my_pipeline.py
  my_profile.v1.json
  tests/
    test_my_pipeline.py
  FEATURE_SCHEMA.md
  MODEL_NOTES.md             # label, artifact hash, calibration and limits
```

Do not rename or edit `AGENTS.md`; update it only through a reviewed change to
the team-wide working agreement.

## Exact factory contract

The `factory` is `module.path:callable`. It receives exactly one positional
argument and returns `GraspPlanningPipeline`:

```python
from sorting_core.grasping import GraspPlanningPipeline
from sorting_core.integration_plugins import GraspFactoryContext

def create(context: GraspFactoryContext) -> GraspPlanningPipeline:
    ...
```

`GraspFactoryContext` contains these exact fields:

```text
config
checkpoint
options
profile_id
profile_path
profile_sha256
```

`options` is recursively frozen. Treat the context as immutable. Load models
inside `create`, not at module import. Use `context.checkpoint` or a documented
profile option for artifacts and record an immutable model version in every
`GraspScore`.

The current pipeline is composed from five structural interfaces:

```python
GraspPlanningPipeline(
    generator=GraspCandidateGenerator,
    checker=GraspFeasibilityChecker,
    feature_extractor=GraspFeatureExtractor,
    scorer=GraspScorer,
    selector=GraspSelector,
)
```

The pipeline validates candidate identities, scene/target binding, component
return types, and failure provenance. Reuse it instead of writing an opaque
function that generates, scores, and selects without intermediate evidence.

## Current input and output contract

The compatibility pipeline currently receives:

```python
decide_for_target(
    scene: SceneDetections,
    target: CubeDetection,
    context: GraspPlanningContext,
) -> GraspPlanningDecision
```

`GraspPlanningContext` provides destination and release pose, configured pick
height/approach clearance, optional complete generic `SceneObservation`, the
objects the cube adapter could not support, and optional measured robot state.
Do not ignore `unsupported_objects`: an unmodeled object cannot be assumed to
be clear space.

The decision contains:

- the selected executable `PickPlacePlan`;
- the selected evaluated candidate;
- every accepted candidate and its features/score;
- every deterministic rejection and its reasons/metrics.

Every candidate must carry a unique `candidate_id`, the current `scene_id`, the
selected `target_track_id`, pick and approach fingertip poses, and the required
jaw opening when known. Candidate provenance must identify the generator.

## Units, frames, timestamps, and geometry

- Positions, openings, dimensions, and clearances are metres.
- Angles are radians.
- Runtime object poses are in `robot_base` and follow the axes/symmetry in
  `docs/OBJECT_CATALOG.md`.
- Candidate scene and target IDs must match the current observation. Never
  reuse a candidate after reobservation without regenerating/revalidating it.
- `TipPoseBase` currently represents fingertip XYZ plus yaw only. It is not a
  general 6-DoF gripper pose.
- FANUC UTOOL1/TCP is treated as the fingertip action point in the current
  integration. Do not add another unexplained TCP offset.
- Current task heights and release poses are configuration assumptions, not
  perception-derived general geometry.

For new classes, first complete all object-catalog fields: CAD/version, local
origin and axes, dimensions, symmetries, permitted grasp regions/families,
destination semantics, and required evidence. The actual custom aluminum
finger geometry, jaw travel, hooks, hoses, and held-object sweep matter; a
model score cannot make an impossible grasp feasible.

## Deterministic feasibility comes first

Reject a candidate before learned scoring when known constraints fail. As
applicable, check:

- object/gripper/finger fit and required opening;
- reachable pose and joint limits;
- finger contact locations and exposed grasp region;
- approach corridor and descent clearance;
- full fingers, gripper body, robot links, hoses, table, neighboring objects,
  fixtures, and bin rim;
- lift while holding the object;
- swept transport path and placement/release clearance;
- uncertainty inflation from perception/calibration/timing;
- stale scene/target identity or unsupported/unknown obstacles.

The checked-in legacy checker only verifies radial reach and center spacing. It
is a baseline, not collision-free motion validation. Physics contact response
is not a substitute for collision avoidance.

## Features, labels, and learned scores

Use the same versioned feature extractor for dataset collection and runtime.
Features must be available before the action. Examples include target pose and
quality, observation age, object dimensions/class, grasp orientation, required
opening, finger/object clearances, approach/transport properties, and nominal
duration. Never train on exact simulator truth and deploy using noisy camera
estimates.

Each `GraspFeatures` has an immutable `schema_id`. A feature definition or unit
change requires a new schema ID and retraining/evaluation.

Each `GraspScore` may contain:

```text
success_probability
expected_total_episode_time_s or None
expected_execution_time_s or None
expected_recovery_time_s or None
scorer_id
model_version
```

The time fields remain in the shared schema for backward compatibility and
logging. Current project scorers should normally leave predicted time fields
`None`; the active selector uses only `success_probability`.

Probability must be calibrated against complete-grasp outcomes, not merely
classification accuracy. Start with empirical rates, logistic regression, or
a small random forest before using a more complex model.

Use `HighestSuccessSelector`. A zero or undefined probability is ineligible;
ties must be deterministic and auditable. The dormant `SuccessTimeSelector`
is not part of this project's evaluation and must not be selected by a team
profile.

## Dataset and outcome rules

Define one episode before collection:

1. primary attempt;
2. reobservation after failure;
3. at most the agreed fallback retry;
4. end in correct placement or unresolved failure.

Log scene and candidate snapshots, pre-action features, selected and rejected
candidates, model/profile/checkpoint versions, perception/planning/execution/
recovery timing, correct placement, failed lift, slip/drop, wrong destination,
contact, cancellation, intervention, retry, and unresolved outcome. Keep
evaluator-only simulator truth separate from runtime features.

Replay matched layouts with alternative feasible candidates where practical.
Split by layout/arrival sequence, not random adjacent frames, to avoid temporal
leakage. Evaluate calibration and action ranking in addition to classification
accuracy.

## Dependencies and model artifacts

- Do not commit weights, datasets, generated runs, or environment directories.
- Pin plugin-owned dependencies and store only reproducible configuration,
  code, and small fixtures in Git.
- Brev and the hardware client use managed environments. Do not `pip install`
  into them ad hoc.
- Small tabular models should run in the existing CPU process when their
  dependencies are available. Heavy/conflicting models should use a separately
  pinned local service and a lightweight plugin client with bounded timeouts.
- Load `context.checkpoint` deliberately and validate its expected format/hash.
  The hardware-specific override is `ROCKWELL_GRASP_CHECKPOINT`; Isaac passes
  `grasp_checkpoint=` to its adapter. Do not make the grasp plugin interpret
  `ROCKWELL_VISION_CHECKPOINT` or the vision artifact. Never load untrusted
  pickle/joblib/PyTorch artifacts merely because a path exists.
- Do not put secrets, credentials, personal paths, or writable global state in
  a profile.

## Required tests before handoff

1. **Import/factory:** select the profile and construct the pipeline through a
   real `GraspFactoryContext` with no shared-file edits.
2. **Candidate integrity:** no candidates, duplicates, stale scene IDs, wrong
   target IDs, and invalid poses/openings fail predictably.
3. **Feasibility:** each known rejection condition has a focused test and
   auditable finite metrics.
4. **Feature parity:** collection and inference produce identical feature keys,
   definitions, ordering-independent values, schema ID, and units.
5. **Scoring:** reject NaN/out-of-range probabilities and test deterministic
   tie-breaking.
6. **Selection policy:** compare the geometric baseline with highest predicted
   success using identical feasible sets and recovery rules.
7. **Simulation:** use camera-derived scenes for decisions; vary layout,
   occlusion, friction/compliance, pose error, and relevant grasp geometry.
8. **Retention:** verify the object remains held through lift and transport and
   is released correctly; pickup contact alone is not success.
9. **Real replay/physical ladder:** replay recorded camera-derived scenes first,
   then supervised hover before any permitted physical descent.
10. **Regression:** the integration profile unset still reproduces the working
    baseline; the baseline example profile produces the same decisions.

Report rejected candidates and failures, not only successful selected grasps.

## Failure rules

- A selected plugin import/checkpoint/model failure is explicit and fail-closed;
  never silently switch to constant-success or the legacy grasp.
- No feasible candidates means stop/reobserve/replan or report unresolved; it
  does not justify scoring a rejected candidate.
- A zero/undefined probability is ineligible.
- Stale observations, moved targets, new/unknown obstacles, and changed grasp
  clearance require candidate regeneration and revalidation.
- Never use simulator truth to repair camera-derived target or obstacle poses.
- Never claim collision-free or hardware-safe based only on IK success or
  successful simulator contact physics.

## Files this team must not edit independently

- `hardware/fanuc_cell/**`
- `isaac_bridge/**`
- `sorting_core/perception.py`, `sorting_core/grasping.py`,
  `sorting_core/integration_plugins.py`, or `sorting_core/team_integrations.py`
- `digital-twin-setup/**`
- `reference-assets/**`
- shared files under `config/`
- the computer-vision team's directory

Ask the integration lead for a reviewed shared-interface change. Do not weaken
validation gates or tests to make a model output pass.

## Sim-to-real cautions

- Isaac friction, compliance, contact, materials, finger geometry, and hose
  behavior are not physical proof. Run sensitivity studies and preserve the
  parameter values with every episode.
- Camera-derived estimates, not configured simulator poses, must feed the
  selection pipeline in evaluation intended to represent deployment.
- The physical cross-camera and full-arm swept-clearance problems remain
  incomplete. Inflate obstacles for measured uncertainty.
- Validate actual gripper/finger dimensions and object fit before using a new
  class. Vials, nails, and screws may require different contact regions even
  if the same gripper can physically hold them.
- The first new-object scope should be known, separated objects on a flat
  table. Defer piles, tangles, moving interception, and arbitrary 6-DoF until
  the simpler complete pipeline is measured.

## Handoff checklist

- [ ] Factory is importable and returns `GraspPlanningPipeline`.
- [ ] Profile options and checkpoint/model hashes are documented.
- [ ] Supported class, CAD revision, axes, symmetry, gripper/finger geometry,
      and grasp families are complete in the object catalog.
- [ ] Candidate, feasibility, feature, scorer, and selector IDs are versioned.
- [ ] Feature schema and complete-grasp label definition are frozen.
- [ ] Deterministic checks run before learned scoring.
- [ ] Baseline/highest-success comparisons use matched conditions and identical
      recovery rules.
- [ ] Probability calibration, ranking, complete-placement success, total time,
      contacts, drops, interventions, and unresolved outcomes are reported.
- [ ] Simulation decisions use camera-derived observations.
- [ ] No weights, datasets, secrets, generated artifacts, or absolute paths are
      committed.
- [ ] Profile-unset and baseline-example regressions pass.
- [ ] Unsupported conditions and known mechanical assumptions are explicit.
- [ ] The tested cube size is explicit: 10 mm and 16 mm are never mixed.
- [ ] A saved scene/candidate fixture reproduces a decision without importing
      the vision model, Isaac, or the physical client.
- [ ] Model licensing, source, artifact retrieval, expected CPU/GPU/memory,
      and bounded inference latency are documented in team-owned model notes.

## Current limitation that must remain visible

The shared grasp pipeline is modular internally, but its current public target
is still `CubeDetection`, its task scene is `SceneDetections`, and its
executable fingertip pose is XYZ plus yaw. A plugin can improve cube candidate
generation/scoring without shared changes. True screw, nail, vial, tilted-part,
or side-grasp execution needs a reviewed generic task/grasp and 6-DoF motion
extension. Do not pretend the legacy top-down executor already provides it.
