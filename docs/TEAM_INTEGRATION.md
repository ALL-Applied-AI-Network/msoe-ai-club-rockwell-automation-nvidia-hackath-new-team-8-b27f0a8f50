# Team integration guide

This guide defines the handoff between the computer-vision team, the grasping
team, and the existing Isaac/FANUC adapters. The goal is to let either team
replace one component without rewriting the working cube demo or low-level
robot action code.

The checked-in defaults preserve the current externally visible behavior:

- the legacy provider detects red, yellow, and blue cubes;
- the legacy grasp pipeline returns the same single vertical top-down grasp;
- the existing `PickPlacePlan`, motion phases, and backend action limits still
  control Isaac and the FANUC;
- model weights, datasets, generated captures, and simulator truth remain
  outside Git and outside the runtime decision contract.

These extension points do **not** claim that arbitrary-object perception,
collision-free planning, or learned grasp selection already works.

Object-local axes and symmetry are part of the interface, not a model detail.
The current cube convention and the required template for new classes are in
[`OBJECT_CATALOG.md`](OBJECT_CATALOG.md).

## Pipeline and ownership

```text
calibrated camera frames
        |
        v
VisionProvider.observe(PerceptionRequest)          computer-vision team
        |
        v
SceneObservation v1                               shared contract
        |
        v
cube/color task adapter chooses a target           existing baseline
        |
        v
GraspPlanningPipeline.plan_for_target(...)         grasping team
        |
        +--> candidate generation
        +--> deterministic feasibility checks
        +--> feature extraction
        +--> learned scoring of feasible candidates only
        +--> selection
        |
        v
PickPlacePlan and semantic phases                   existing integration
        |
        +--> Isaac adapter
        +--> supervised FANUC policy
```

Vision owns observations, not robot commands. Grasping owns candidates and
scores, not camera calibration or low-level control. Isaac and FANUC remain
free to reject a selected plan that violates backend limits.

## Where each team works

| Purpose | Exact file or method | Input | Output |
|---|---|---|---|
| Vision contracts | `sorting_core/perception.py` | No backend dependencies | `CameraFrame`, `DepthEncoding`, `RobotState`, `PerceptionRequest`, `DetectedObject`, `SceneObservation`, `VisionProvider` |
| Grasp contracts and default pipeline | `sorting_core/grasping.py` | Camera-derived legacy task scene, target, context | `PickPlacePlan` |
| Select active team components | `ROCKWELL_INTEGRATION_PROFILE` plus a strict JSON profile | Lightweight camera metadata and `module:factory` references | Provider and grasp-pipeline factories |
| Existing cube detector | `sorting_core/vision.py::detect_coloured_cubes` | RGB, calibrated projector, config | Legacy `SceneDetections` |
| Existing cube task adapter | `sorting_core/task_perception.py::cube_scene_from_observation` and `sorting_core/planner.py` | Generic scene plus configured cube/color task | Legacy cube scene and destination intent |
| Isaac integration | `isaac_bridge/camera_sort.py::IsaacCameraSortAdapter` | Isaac frames plus selected integrations | Scene and selected plans |
| Hardware integration | `hardware/fanuc_cell/policy.py::Policy` | Official observations plus selected integrations | Bounded official TCP action chunks |

Implement model code below `team_plugins/computer_vision/` or
`team_plugins/grasping/`, or in another importable package owned by that team.
Add a new profile instead of editing `sorting_core/team_integrations.py`, the
Isaac adapter, or the hardware policy. The nested `AGENTS.md` in each team
directory is the canonical human-and-agent onboarding guide. Do not scatter
provider-selection conditionals through the backends.

## Drop-in activation

An integration profile selects either team independently or both together:

```json
{
  "schema_version": 1,
  "profile_id": "my-team-integration-v1",
  "vision": {
    "factory": "team_plugins.computer_vision.my_provider:create",
    "camera_ids": ["table"],
    "depth_camera_ids": [],
    "calibration_ids": {
      "table": "sidclab-cameras.json:table"
    },
    "depth_calibration_ids": {},
    "options": {}
  },
  "grasp": {
    "factory": "team_plugins.grasping.my_pipeline:create",
    "options": {}
  }
}
```

Set the profile before starting Python, because the official hardware client
declares its requested cameras while importing `Policy`:

```bash
export ROCKWELL_INTEGRATION_PROFILE=team_plugins/profiles/my_integration.json
```

Relative paths resolve from the repository root. Restart the process after a
profile change. With the variable unset, the original HSV cube provider and
single top-down grasp remain active. Profile metadata is validated at import,
but the selected model modules are imported lazily only when their factory is
constructed. An explicitly selected profile fails closed on malformed JSON,
missing dependencies, factory errors, or wrong return types; it never silently
falls back and continues execution.

Factories use one stable context argument:

```python
def create(context: VisionFactoryContext) -> VisionProvider: ...
def create(context: GraspFactoryContext) -> GraspPlanningPipeline: ...
```

The context supplies existing runtime inputs, recursively frozen profile
options, and the profile ID/path/SHA-256 for provenance. See
`team_plugins/computer_vision/AGENTS.md` and
`team_plugins/grasping/AGENTS.md` for every field and the required handoff
tests. Plugin modules are trusted executable code; selection is explicit and
there is no automatic directory scan.

### Exact integration signatures

These are the intended seams. Treat a signature change as a shared-interface
change requiring both teams' review:

```python
class VisionProvider(Protocol):
    def observe(self, request: PerceptionRequest) -> SceneObservation: ...

def validate_scene_for_request(
    request: PerceptionRequest,
    scene: SceneObservation,
) -> SceneObservation: ...

def create_vision_provider(
    *,
    config: SortConfig,
    camera_projectors: Mapping[str, Any],
    calibration_ids: Mapping[str, str] | None = None,
    expected_cube_edge_m: float | None = None,
    checkpoint: str | None = None,
    detector: Any | None = None,
) -> VisionProvider: ...

def cube_scene_from_observation(
    observation: SceneObservation,
    *,
    class_ids: Iterable[str] = ("cube",),
    maximum_tilt_rad: float = math.radians(10.0),
) -> CubeSceneProjection: ...

def build_colour_sort_plans(
    scene: SceneDetections,
    config: SortConfig,
    *,
    grasp_pipeline: GraspPlanningPipeline | None = None,
    generic_scene: SceneObservation | None = None,
    unsupported_objects: tuple[DetectedObject, ...] | None = None,
    robot_state: RobotState | None = None,
) -> tuple[PickPlacePlan, ...]: ...

class GraspPlanningPipeline:
    def decide_for_target(
        self,
        scene: SceneDetections,
        target: CubeDetection,
        context: GraspPlanningContext,
    ) -> GraspPlanningDecision: ...

    def plan_for_target(
        self,
        scene: SceneDetections,
        target: CubeDetection,
        context: GraspPlanningContext,
    ) -> PickPlacePlan: ...

class GraspCandidateGenerator(Protocol):
    def generate(
        self, scene: SceneDetections, target: CubeDetection,
        context: GraspPlanningContext,
    ) -> tuple[GraspCandidate, ...]: ...

class GraspFeasibilityChecker(Protocol):
    def check(
        self, scene: SceneDetections, candidate: GraspCandidate,
        context: GraspPlanningContext,
    ) -> FeasibilityResult: ...

class GraspFeatureExtractor(Protocol):
    def extract(
        self, scene: SceneDetections, candidate: GraspCandidate,
        context: GraspPlanningContext,
    ) -> GraspFeatures: ...

class GraspScorer(Protocol):
    def score(
        self, candidate: GraspCandidate, features: GraspFeatures,
    ) -> GraspScore: ...

class GraspSelector(Protocol):
    def select(
        self, evaluations: tuple[EvaluatedGrasp, ...],
    ) -> EvaluatedGrasp: ...

def create_grasp_pipeline(
    *, config: SortConfig, checkpoint: str | None = None,
) -> GraspPlanningPipeline: ...

class IsaacCameraSortAdapter:
    def __init__(
        self,
        camera: CalibratedPinholeCamera,
        config: SortConfig,
        *,
        perception_provider: VisionProvider | None = None,
        grasp_pipeline: GraspPlanningPipeline | None = None,
        vision_checkpoint: str | None = None,
        grasp_checkpoint: str | None = None,
    ): ...

    def observe_request(
        self, request: PerceptionRequest,
    ) -> SceneObservation: ...

    def plan_observation(
        self,
        observation: SceneObservation,
        *,
        robot_state: RobotState | None = None,
    ) -> CameraSortDecision: ...

    def observe_request_and_plan(
        self, request: PerceptionRequest,
    ) -> CameraSortDecision: ...

class Policy:
    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        perception_provider: VisionProvider | None = None,
        grasp_pipeline: GraspPlanningPipeline | None = None,
    ): ...
```

Direct constructor injection is intended for unit tests, offline replay, and
development. The official hardware runner constructs `Policy(checkpoint)`
itself, so deployed components must be selected with an integration profile.
`sorting_core/team_integrations.py` is the stable host and should not be edited
by either team. Camera metadata is read from the selected profile when that
module is imported; set the profile first and restart the process after
changing it.

## Shared units, frames, and timestamps

These rules are part of the interface:

- Positions, dimensions, openings, and clearances use **metres**.
- Angles use **radians**.
- Times use **seconds** on the backend-supplied clock.
- Object poses use the `robot_base` coordinate frame.
- Quaternions use `(x, y, z, w)` order and must already have unit length.
- `Pose3D.position_base_m` is the object origin agreed in the object catalog,
  normally its geometric center. It is not automatically the visible top face.
- `Pose3D.orientation_xyzw` rotates the catalog-defined object-local frame into
  `robot_base`; it is not the inverse transform. Dimensions follow the
  catalog's local X/Y/Z axes.
- `PerceptionRequest.capture_time_s` is the scene fusion/reference time.
- Every `CameraFrame` retains a `capture_time_s`. The official hardware adapter
  currently has only the observation-level `obs["t"]`, so it copies that one
  reference time into every frame; this does **not** prove simultaneous camera
  exposure. Other adapters may supply genuinely different camera timestamps.
- Every returned object's `capture_time_s` must equal the scene reference time.
  A tracker may propagate an estimate to that time, but must not hide the
  original frame times while doing so.
- RGB `calibration_id` and `depth_calibration_id` are distinct because the
  streams can have different intrinsics, extrinsics, sizes, and units. Every
  returned scene must cite at least one calibration from its request, and each
  object must cite at least one requested source camera. Model confidence is
  not a measurement of calibration error.
- Strict JSON has no `NaN` or infinity. Use `json.dumps(..., allow_nan=False)`;
  the Python contracts independently reject nonfinite numeric values.
- Missing detections and unknown objects are **not proven clear space**. An
  occluded or unknown object that might intersect a grasp or path must be
  represented conservatively or cause that candidate to be rejected.
- Simulator truth is evaluator data only. It must never repair runtime camera
  observations.
- `RobotState.arm_joints_rad` contains exactly J1--J6 in radians;
  `tcp_pose_base_m_wpr_rad` contains base-frame XYZ in metres followed by
  FANUC W/P/R parameters in radians; `gripper_state` is in `[0, 1]`. These are
  typed SI values, not the official client's raw degree/mm arrays.

## SceneObservation v1

Runtime components exchange typed Python objects, not a hand-built string.
JSON is for logs, fixtures, and optional cross-process handoff. Its strict
shape is defined by
[`schemas/scene-observation.v1.schema.json`](schemas/scene-observation.v1.schema.json)
and by `scene_observation_to_dict()` / `scene_observation_from_dict()`.

The JSON schema validates structure, but it cannot express every runtime
invariant: unique track IDs, object/scene timestamp equality, unit-quaternion
norm, or whether cited calibrations and cameras came from the request. Always
parse with `scene_observation_from_dict()` and call
`validate_scene_for_request()` before planning; schema validation alone is not
enough.

```json
{
  "schema_version": 1,
  "scene_id": "trial-04:scene-17",
  "capture_time_s": 1793300000.125,
  "coordinate_frame": "robot_base",
  "objects": [
    {
      "track_id": "trial-04:red:0",
      "class_id": "cube",
      "sort_label": "red",
      "capture_time_s": 1793300000.125,
      "pose": {
        "position_base_m": [0.442, -0.188, 0.0029],
        "orientation_xyzw": [0.0, 0.0, 0.1951, 0.9808]
      },
      "dimensions_m": [0.016, 0.016, 0.016],
      "confidence": 0.94,
      "pose_quality": 0.82,
      "visibility": 0.87,
      "source_camera_ids": ["table"],
      "attributes": {}
    }
  ],
  "provider_name": "legacy-colour-cube",
  "provider_version": "1",
  "calibration_ids": ["sidclab-cameras.json:table"]
}
```

`class_id` describes geometry/category. `sort_label` carries the configured
destination label. The current compatibility adapter accepts `class_id="cube"`
by default. If the object catalog later adopts a more specific ID such as
`cube_16mm`, add that ID explicitly to `CUBE_TASK_CLASS_IDS` only after checking
that the current geometry, destination, and grasp assumptions still apply.
Geometrically identical cubes may have different sort labels. Unknown objects
should be returned with an explicit catalog class such as `unknown` when
possible rather than silently discarded.

The baseline detector's IDs are scene-local. A tracking provider may return
persistent IDs, but it must document identity creation, loss, and reacquisition.
No caller may assume that current baseline IDs survive between frames.

The current task adapter still understands only the configured cube/color
task. Returning a new class does not automatically create a destination,
grasp, or executable motion for it.

## Computer-vision team

### Provider input

`VisionProvider.observe(request)` receives a `PerceptionRequest` containing:

- a scene ID and fusion/reference timestamp;
- a tuple of `CameraFrame` values, each with camera ID, adapter-supplied frame
  timestamp, separate RGB/depth calibration IDs, RGB and/or depth. On the
  official hardware those
  frame timestamps are currently copies of the one observation reference time,
  not measured per-camera exposure times;
- `request.robot_state` when available, which matters for a moving wrist
  camera and contains the typed SI fields defined above;
- compact JSON-compatible metadata.

Use `request.frame("table")` to retrieve a named frame. Calibrated pixel-to-base
projectors are injected when the provider is created; they are not serialized
inside the request.

The baseline camera requirements remain defined by the stable host. A selected
vision plugin declares its own requirements in profile metadata, which the
hardware client can read without importing a heavyweight model:

```json
{
  "camera_ids": ["table"],
  "depth_camera_ids": [],
  "calibration_ids": {
    "table": "sidclab-cameras.json:table"
  },
  "depth_calibration_ids": {}
}
```

Add a camera only when the provider consumes it and its calibration/timestamp
path has a test. Requesting several images is not itself multi-camera fusion.
Keep depth disabled until units, invalid pixels, RGB alignment, and calibration
are validated. A depth payload is rejected unless it has both its own
calibration ID and a `DepthEncoding`. Official hardware depth is native
`uint16`, `0` means invalid, and `meters_per_unit=0.001`; it is not declared
pixel-aligned to the differently sized RGB image. Isaac depth is commonly
floating-point metres and must declare its own encoding instead of copying the
hardware convention.

`CUBE_TASK_CLASS_IDS` remains integration-owned and is deliberately not a
profile option. It is the explicit compatibility mapping into today's cube
planner. Ask for a reviewed shared change only after confirming that a class
has the same cube geometry, destination, and grasp assumptions. Other detected
classes remain in `SceneObservation` and are returned as unsupported. The
checked-in legacy checker/planner blocks execution when unsupported objects
are present; do not resume motion until an object-aware feasibility checker or
task adapter accounts for every such object.

The checked-in factory injects a calibrated projector only for the table
camera. A profile can request fixed-side, moving-wrist, or depth streams, but
that declaration does not create their projectors. Before one of those streams
drives motion coordinates, the integration lead must expose a reviewed adapter
that reuses the pinned calibration and existing geometry helper. A team plugin
may consume that injected adapter; it must not create a second divergent
calibration implementation merely to make a demo pass. Wrist geometry also
requires joints from the same observation. The wrist RGB calibration is native
1280×720 while the current policy requests RGB at 1920×1080; scale intrinsics
consistently in the reviewed adapter or add an explicitly validated
native-resolution path. Depth streams have their own native sizes and
invalid-pixel conventions. The existing
`hardware/fanuc_cell/calibration/geometry.py::flange_pose()` helper expects raw
joint **degrees**; do not pass `RobotState.arm_joints_rad` into it. Use an
SI-native wrapper, or explicitly convert with `np.degrees()` at that legacy
helper boundary and test the transform.

`PERCEPTION_CAMERA_IDS` and `PERCEPTION_DEPTH_CAMERA_IDS` are combined when
building requests, so a depth-only camera may produce a `CameraFrame` with no
RGB. `IsaacCameraSortAdapter.observe_request()` is the provider-neutral
perception seam for multi-camera Isaac inputs. Use
`observe_request_and_plan()` to exercise that request through task projection
and grasp selection end to end; the older `observe_generic(rgb, ...)` helper
is the single-table-camera compatibility path. Strict depth-only profile
parsing and camera declarations are tested, but depth-only operation through
the official physical client is still unverified and must not be treated as a
hardware-ready capability until an official-observation replay proves it.

Official RGB payloads are `H x W x 3`, `uint8`, **RGB** channel order. OpenCV
usually decodes BGR; the official client already converts once. A provider
must not swap the channels again. Validate the same convention for every Isaac
capture path.

### Provider output

Return one `SceneObservation`. The provider must:

1. preserve `request.scene_id` and `request.capture_time_s`;
2. transform every pose into `robot_base`;
3. identify the provider and at least one calibration revision from the request;
4. attach class, confidence, and at least one requested source camera;
5. attach pose quality and visibility when the model can support those values;
6. keep simulator truth and manual corrections out of runtime results;
7. reject an uncalibrated conversion instead of fabricating a plausible pose.

### Minimal provider skeleton

```python
from sorting_core.perception import (
    DetectedObject,
    PerceptionRequest,
    Pose3D,
    SceneObservation,
)


class TeamVisionProvider:
    def __init__(self, *, model, camera_projectors, calibration_ids):
        self.model = model              # Load once, not once per frame.
        self.projectors = dict(camera_projectors)
        self.calibration_ids = dict(calibration_ids)

    def observe(self, request: PerceptionRequest) -> SceneObservation:
        table = request.frame("table")
        # Run inference on table.rgb, then use the injected calibrated
        # projector to produce robot-base metres. Fuse/track here if needed.
        objects = (
            DetectedObject(
                track_id=f"{request.scene_id}:example",
                # The current adapter accepts "cube" by default. If the team
                # adopts IDs such as "cube_16mm", request an integration-lead
                # review of the cube-task compatibility mapping.
                class_id="cube",
                sort_label="red",
                capture_time_s=request.capture_time_s,
                pose=Pose3D.from_yaw((0.44, -0.19, 0.0029), 0.0),
                dimensions_m=(0.016, 0.016, 0.016),
                confidence=0.90,
                pose_quality=0.80,
                visibility=0.95,
                source_camera_ids=("table",),
                attributes={},
            ),
        )
        return SceneObservation(
            scene_id=request.scene_id,
            capture_time_s=request.capture_time_s,
            objects=objects,
            provider_name="team-vision",
            provider_version="0.1",
            # Cite only calibrations actually consumed from this request, not
            # every calibration configured on disk.
            calibration_ids=(table.calibration_id,),
        )
```

Place the provider and its factory in the vision team's plugin package. The
stable host constructs it from `VisionFactoryContext`:

```python
from sorting_core.integration_plugins import VisionFactoryContext


def create(context: VisionFactoryContext):
    return TeamVisionProvider(
        model=load_vision_model(context.checkpoint, context.options),
        camera_projectors=context.camera_projectors,
        calibration_ids=context.calibration_ids,
    )
```

Reference this callable as `package.module:create` in the vision profile. The
host distinguishes `None` (use declared defaults) from an explicitly empty
runtime calibration map; plugins must not replace that distinction with a
truthiness fallback.

Unit-test this provider with saved frames and injected fake projectors. A basic
provider test should not require Isaac or access to the physical cell.

The official client has one legacy `checkpoint` argument, but vision and grasp
models are separate artifacts. On hardware, use
`ROCKWELL_VISION_CHECKPOINT` and `ROCKWELL_GRASP_CHECKPOINT` to override them
independently. In Isaac, pass `vision_checkpoint=` and `grasp_checkpoint=`.
Each loader must verify its model version plus the compatible object-catalog,
calibration (vision), and feature-schema (grasp) version before returning;
never let both teams interpret the same untyped checkpoint file as theirs.

## Grasping team

> **Current scope boundary:** the executable task adapter is still cube/color
> specific. Its protocol target is `CubeDetection`, its destination is a
> color-matched `BinTarget`, and it applies upright square-yaw symmetry. A
> screw, vial, or nail must **not** be added to `CUBE_TASK_CLASS_IDS` or trained
> as though this were a generic-object executor. Once the team freezes the new
> object/destination list, add a generic `DetectedObject` task-target and
> placement-intent layer, then adapt the cube demo into that layer. The current
> contracts intentionally preserve the working cube hardware path until those
> semantics are known.

The current grasp boundary is intentionally narrower than the generic vision
contract. `GraspPlanningPipeline.plan_for_target(scene, target, context)` still
receives legacy `SceneDetections` and `CubeDetection` from the cube/color task
adapter. `context.generic_scene` carries the complete `SceneObservation` so a
new feature extractor can use class, dimensions, pose quality, visibility, and
unknown obstacles without reading simulator truth.

`decide_for_target(scene, target, context)` returns the fuller
`GraspPlanningDecision`: the selected evaluation, every scored feasible
candidate, every deterministically rejected candidate, and the executable
plan. `plan_for_target(...)` remains the compatibility shortcut that returns
only `.plan`. The hardware adapter logs this candidate-level evidence so the
team can connect runtime choices to later success/failure labels.

`GraspPlanningContext` also supplies the destination, release pose, configured
pick height, approach clearance, and the same typed `RobotState` supplied to
perception. `unsupported_objects` contains only the objects explicitly rejected
by the current cube task adapter; supported cubes filtered out as destinations
or non-target evidence are not mislabeled as unknown. Whenever
`generic_scene` is supplied, callers must also pass the projection's explicit
tuple—even when it is empty—so omission cannot silently mean clear space. The method returns an
existing `PickPlacePlan`, which preserves the executor boundary.

### Mandatory processing order

1. `GraspCandidateGenerator.generate(scene, target, context)` proposes grasps.
2. `GraspFeasibilityChecker.check(...)` deterministically rejects candidates
   that violate known fit, reach, approach, collision, or path constraints.
3. `GraspFeatureExtractor.extract(...)` creates the same versioned numeric
   features used during dataset collection and runtime.
4. `GraspScorer.score(candidate, features)` predicts **complete grasp
   success**: acquire the object, retain it throughout transport, and release
   it only at the destination.
5. `GraspSelector.select(evaluations)` applies the experiment policy.
6. `decide_for_target()` converts the selected candidate into
   `PickPlacePlan` and retains all accepted/rejected evidence;
   `plan_for_target()` is the shortcut returning only that plan.

The default highest-success selector treats a predicted probability of zero as
ineligible; if every candidate scores zero, planning fails instead of sending
an action that the model says cannot succeed.

For the success/time experiment, `GraspScore.expected_total_episode_time_s`
means the **unconditional expected duration of the bounded episode**, including
the same fixed retry/recovery policy used by every compared action. It is not
`E[recovery | failure]`. `SuccessTimeSelector` minimizes the explicit local
heuristic
`expected_total_episode_time_s / success_probability`; it fails if any
eligible candidate lacks the total-time prediction. This is a throughput
heuristic, not a whole-scene scheduling optimum. The selector ID and objective
are stored in every selected plan so highest-success and success/time runs
remain distinguishable even when their predictions are identical.

A learned score must never override deterministic rejection. If all candidates
are rejected, no plan is returned. If model inference raises or returns an
invalid probability/time, the pipeline raises `GraspPlanningError`; an
explicit, tested fallback may be added later, but accidental exception-based
fallback is not allowed.

Perception and scoring run synchronously in the current policy. The hardware
adapter rechecks observation age after those components finish; if their
combined processing exceeds the configured observation-age limit, it cancels
or reobserves and emits only a one-row hold, never forward motion from the
expired scene. Benchmark end-to-end latency, not model inference alone.

The existing checker is only radial-reach and center-spacing preflight. It is
not full-arm inverse kinematics, collision checking, swept-path validation, or
physical grasp validation.

### Minimal learned-scorer skeleton

Most initial grasp-team work can replace only `GraspScorer`; the generator,
checker, feature extractor, and selector can remain simple while their data is
being validated.

```python
from sorting_core.grasping import GraspScore


class TeamSuccessScorer:
    def __init__(self, model):
        self.model = model

    def score(self, candidate, features):
        ordered_names = sorted(features.values)
        row = [features.values[name] for name in ordered_names]
        probability = float(self.model.predict_proba([row])[0, 1])
        return GraspScore(
            success_probability=probability,
            expected_total_episode_time_s=None,
            expected_execution_time_s=None,
            expected_recovery_time_s=None,
            scorer_id="team-success-scorer",
            model_version="0.1",
        )
```

Construct a complete `GraspPlanningPipeline` in the grasp team's plugin
factory:

```python
from sorting_core.grasping import (
    GraspPlanningPipeline,
    HighestSuccessSelector,
    LegacyFeatureExtractor,
    LegacyGeometricChecker,
    LegacyTopDownCubeGenerator,
)
from sorting_core.integration_plugins import GraspFactoryContext


def create(context: GraspFactoryContext) -> GraspPlanningPipeline:
    return GraspPlanningPipeline(
        generator=LegacyTopDownCubeGenerator(),
        checker=LegacyGeometricChecker(
            maximum_radial_reach_m=(
                context.config.hardware.maximum_radial_reach_m
            ),
            minimum_centre_clearance_m=(
                context.config.task.minimum_distinct_centres_m
            ),
        ),
        feature_extractor=LegacyFeatureExtractor(),
        scorer=TeamSuccessScorer(
            load_grasp_model(context.checkpoint, context.options)
        ),
        selector=HighestSuccessSelector(),
    )
```

Keep model weights outside Git and do not download or initialize a model during
an active physical rollout.

Hardware logs join candidate evidence to outcomes using the attempt ID and
`grasp_decision_key`. Accepted candidate logs include poses, deterministic
feasibility metrics, features, model/version, selector, probability, and total
time prediction. `plan_complete` or `policy_blocked` records the measured
backend-reference episode elapsed time; all events also carry a wall-clock
timestamp. Preserve these fields when building the training table, and do not
label camera-verified placement as independent force/grasp confirmation.

## Default behavior and compatibility

Until the factories are changed:

- `create_vision_provider()` returns the legacy HSV cube provider;
- `create_grasp_pipeline()` returns
  `GraspPlanningPipeline.legacy_default(config)`;
- that pipeline generates one legacy top-down grasp, checks it geometrically,
  gives it a constant score, and selects it;
- square-cube yaw remains normalized modulo 90 degrees;
- the executor supports vertical top-down pick pose plus yaw, not an arbitrary
  six-degree-of-freedom wrist orientation;
- color order, destinations, reobservation gates, phase names, and action
  bounds remain unchanged.

This default is the required fallback. Enabling and then disabling a teammate
implementation must restore the baseline outputs.

## Required tests

Use small fake providers and scorers before testing a large model:

1. Round-trip `SceneObservation` through `scene_observation_to_dict()` and
   `scene_observation_from_dict()` without losing decision-relevant values.
2. Reject missing calibrations, wrong frames, stale/mismatched timestamps,
   nonfinite poses, non-unit quaternions, and unknown schema versions.
3. Confirm the provider cannot receive simulator truth through its request.
4. Confirm the scene cites at least one calibration from the request, every
   object cites at least one requested source camera, and unknown evidence IDs
   are rejected. Confirm `pose_quality` and `visibility` may remain `None` when
   a provider cannot support those estimates.
5. Confirm a fake provider drives the task adapter without OpenCV.
6. Confirm several candidate grasps can be generated.
7. Confirm rejected candidates never reach `GraspScorer.score()`.
8. Confirm scoring changes candidate ranking but never makes a rejected
   candidate executable.
9. Confirm model exceptions or invalid probabilities produce no plan.
10. Run a golden compatibility test: defaults produce the same plan poses,
    phase order, and bounded action rows as the pre-refactor baseline.
11. Confirm official mm/degrees feedback becomes the expected SI `RobotState`,
    and that the same state reaches Isaac and hardware grasp features.
12. Confirm success/time selection uses `expected_total_episode_time_s / p`,
    logs the selector ID, and joins the executed candidate to its final timing
    and outcome.
13. Delay a fake provider and scorer beyond the observation-age limit; confirm
    the plan is cancelled/reobserved and no forward-motion row is returned.

The following `uv` commands were validated from the repository root in WSL
Ubuntu. If `uv` is not on the Windows PowerShell `PATH`, enter WSL first rather
than changing the command or creating a second environment:

```bash
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -v
```

For a quick team-contract loop before the full suite:

```bash
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_integration_plugins.py' -v
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_perception_integration.py' -v
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_grasping_integration.py' -v
```

On the pinned Brev runtime, also run:

```bash
bash scripts/verify_brev.sh --full
```

The official mock checks client protocol shape and action limits only. It does
not validate perception, grasp retention, collision avoidance, or physical
success.

## Files teammates should not edit

- `digital-twin-setup/`: pristine supplied source. Use the adapter/overlay path.
- `reference-assets/`: exact manifest-verified CAD and reference files.
- `hardware/fanuc_cell/cell_client.py`, `run.py`, and `mock_cell.py`: pinned
  official-client behavior.
- `hardware/fanuc_cell/calibration/`: change only through a measured,
  documented calibration procedure and assign a new calibration ID.
- Low-level TCP conversion, interpolation, action validation, or physical
  bounds merely to make a model test pass.
- Simulator evaluator/label code to feed runtime observations.
- Baseline tests to hide a behavior change. Add a new mode and tests instead.

Before editing `hardware/fanuc_cell/policy.py` or an Isaac demo, first ask
whether the change belongs behind `VisionProvider`, `GraspPlanningPipeline`, or
the cube/color task adapter. Most teammate work should not require direct edits
to either backend.

## Limitations that must remain explicit

- The active task adapter is still cube/color specific.
- The executor accepts top-down pick pose plus yaw, not arbitrary 6-DoF grasps.
- Baseline IDs are scene-local, not persistent tracks.
- Current geometric checks are not full collision or swept-path validation.
- Deterministic feasibility must run before any learned score.
- Unknown or unobserved objects are not evidence of a clear path.
- Multiple camera timestamps can differ; requesting more cameras does not
  automatically solve occlusion or synchronization.
- Simulator truth may evaluate perception but may not repair it.
- Supporting a new object requires a catalog entry, perception evidence,
  destination semantics, candidate grasps, feasibility checks, representative
  training/evaluation examples, and executor compatibility—not only a class
  name.
