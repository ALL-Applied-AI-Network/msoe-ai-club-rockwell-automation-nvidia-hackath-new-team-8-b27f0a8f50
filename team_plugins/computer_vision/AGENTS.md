# Computer-vision plugin instructions

These instructions apply to every file below `team_plugins/computer_vision/`.
They are onboarding for human teammates and binding guidance for coding agents.
The repository-root `AGENTS.md` also applies.

Read `SETUP.md` before the first change. The normal working branch is
`computer-vision`; verify it before editing. Keep changes in this directory
unless the integration lead explicitly accepts a shared-interface change.
When the teammate is uncertain about terminology, model choice, an error, or a
next experiment, explain the options and remind them they can ask the AI.

At the first computer-vision onboarding response, clearly summarize the
assignment, first scene/overlay deliverable, owned branch/folder, prohibited
shared edits, and first replay/contract test. Do not only point at this file.

## Mission and boundary

The vision team converts runtime camera observations into a versioned,
camera-derived `SceneObservation`. The team owns:

- model or deterministic image inference;
- instance classification, segmentation, keypoints, and pose estimation;
- conversion from image/depth coordinates through the supplied calibration;
- tracking and propagation to the request's reference timestamp;
- detection confidence, pose quality, visibility, and provenance;
- vision-only offline evaluation and overlays.

The vision team does **not** own grasp selection, destinations, robot commands,
motion timing, controller limits, physical interlocks, or simulator ground
truth. A vision plugin must never command the robot.

The current required object scope is cubes plus the supplied ISO 10642 M6×30
screw. Cube camera/training capture already exists. The separate screw physics
demo currently uses configured simulator poses and has no camera/training-data
mode, so realistic screw perception data is a known lead-owned integration gap,
not something the vision team may claim is already solved.

Implement new work in this directory. Do not edit shared contracts merely to
fit one model; adapt the model output to the existing contract. Request an
integration-lead review only when evidence shows a shared contract cannot
represent required information.

The handoff is structured data, not a human-readable string. In-process code
returns `SceneObservation`; file or service boundaries use its schema-validated
JSON serialization. Do not invent a second object-list format.

## Activation profile

Plugins are selected by the environment variable
`ROCKWELL_INTEGRATION_PROFILE`. The value is an integration-profile JSON path;
a relative path resolves from the repository root. With the variable unset,
the original baseline remains active.

A vision entry has this exact shape:

```json
{
  "factory": "team_plugins.computer_vision.my_provider:create",
  "camera_ids": ["table"],
  "depth_camera_ids": [],
  "calibration_ids": {
    "table": "sidclab-cameras.json:table"
  },
  "depth_calibration_ids": {},
  "options": {
    "example_threshold": 0.5
  }
}
```

Every requested RGB camera must have exactly one RGB calibration mapping.
Every requested depth camera must have exactly one depth calibration mapping.
Do not list unused cameras: the official client requests and transmits the
declared streams. A profile change requires a new Python process because the
hardware client reads its camera list while importing `Policy`.

The complete example is
`team_plugins/profiles/baseline_plugins.json`.

## Start here: first implementation loop

Work from the repository root and keep the first change deliberately small:

1. Read this file, `docs/TEAM_INTEGRATION.md`, and the entries for your object
   classes in `docs/OBJECT_CATALOG.md`.
2. Copy `baseline_provider.py` to a descriptively named module. Keep a single
   public `create(context)` factory and adapt your model's result behind it.
3. Copy `profile.example.json` to a new, versioned profile name. Change only
   the factory, cameras/calibrations actually consumed, and documented
   options. Never overwrite another team's working profile.
4. Add small replay fixtures and focused tests below this directory. Keep
   large images, datasets, checkpoints, and generated overlays outside Git.
5. Select the profile **before** starting Python, run the focused contract
   tests, then run your replay/evaluation. Unset it to prove the baseline is
   unchanged.
6. Hand off the factory, profile, immutable model/checkpoint identity, metrics,
   known failures, and one valid serialized `SceneObservation` fixture to the
   integration lead and grasp team.

In WSL Ubuntu or Brev, the shared validation loop is:

```bash
export ROCKWELL_INTEGRATION_PROFILE=team_plugins/computer_vision/profile.example.json
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_integration_plugins.py' -v
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_perception_integration.py' -v
unset ROCKWELL_INTEGRATION_PROFILE
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -v
```

The checked-in example profile is only a wiring smoke test; replace it with
your own profile when evaluating your provider. If a model needs a different
environment, run its plugin-owned tests there too and document the exact
locked environment. Passing only that model environment is not sufficient:
the lightweight factory/client must still pass the repository contract tests.

Expected team-owned additions normally look like:

```text
team_plugins/computer_vision/
  my_provider.py
  my_profile.v1.json
  tests/
    test_my_provider.py
  MODEL_NOTES.md             # architecture, preprocessing, artifact hash
```

Do not rename or edit `AGENTS.md`; update it only through a reviewed change to
the team-wide working agreement.

## Exact factory contract

The `factory` string is `module.path:callable`. The callable receives exactly
one positional argument and must return an object implementing
`VisionProvider`:

```python
from sorting_core.integration_plugins import VisionFactoryContext
from sorting_core.perception import VisionProvider

def create(context: VisionFactoryContext) -> VisionProvider:
    ...
```

`VisionFactoryContext` provides these exact fields:

```text
config
camera_projectors
calibration_ids
expected_cube_edge_m
checkpoint
detector
options
camera_ids
depth_camera_ids
declared_calibration_ids
declared_depth_calibration_ids
profile_id
profile_path
profile_sha256
```

The mappings and `options` are frozen. Treat them as immutable. The declared
calibration fields are profile provenance; `calibration_ids` contains the
effective runtime calibration map passed to the provider. Model imports and
checkpoint loading should happen inside `create`, not at module import.

The returned object implements:

```python
class VisionProvider:
    def observe(self, request: PerceptionRequest) -> SceneObservation:
        ...
```

The runtime validates the returned scene with
`validate_scene_for_request()`. Do not rely only on JSON Schema validation.

## Input contract

`PerceptionRequest` contains:

- `scene_id`: identity that the result must copy exactly;
- `capture_time_s`: fusion/reference time that the scene and every object must
  use exactly;
- `frames`: one `CameraFrame` per requested camera;
- `robot_state`: optional measured J1--J6, TCP, and gripper state;
- `metadata`: read-only backend context.

Each `CameraFrame` retains its own `camera_id`, capture time, RGB/depth payload,
calibration IDs, and depth encoding. Different camera frame times are not proof
of synchronization. A moving wrist-camera transform must use the corresponding
measured robot state and must account for time skew conservatively.

The official hardware RGB path supplies decoded image arrays. Inspect and test
the actual shape, channel order, and dtype at the adapter boundary; do not infer
them from downloaded visualization videos. Do not rescale calibration pixels
without rescaling the intrinsics in the same documented operation.

The current hardware factory injects a ready-to-use pixel-to-base projector
for the fixed table camera only. A profile can request side/wrist RGB or depth
streams, but that does not create a backend-neutral projector for them. Before
using those streams for motion coordinates, ask the integration lead to expose
and test a reviewed adapter that reuses the pinned calibration/geometry helper,
including the fixed-camera transform, moving wrist transform, depth encoding,
robot-state timing, and uncertainty. A plugin must not create a second
divergent calibration implementation merely to make a demo pass. The wrist RGB
calibration is native 1280×720 while the current policy requests 1920×1080 RGB;
that mismatch needs consistent intrinsic scaling or a separately validated
native-resolution path. Depth-only profile declarations are contract-tested,
but depth-only operation through the official physical client is still
unverified and must be proven with an official-observation replay before use.

## Output contract

Return `SceneObservation` from `sorting_core.perception`, not an informal
string. Every object is a `DetectedObject` with:

```text
track_id
class_id
sort_label or None
capture_time_s
Pose3D(position_base_m, orientation_xyzw)
dimensions_m or None
confidence
pose_quality or None
visibility or None
source_camera_ids
attributes
```

The scene also identifies `provider_name`, `provider_version`, and every
calibration ID actually used. Track IDs must be unique in one scene and stable
across frames when tracking is claimed.

Detection confidence is not pose accuracy. `confidence` says how strongly the
system supports the object/class detection. `pose_quality` separately reports
pose reliability. If useful uncertainty cannot fit the scalar field, store a
versioned, finite uncertainty record in `attributes` and document it before
the grasp team consumes it.

Masks, boxes, keypoints, original frame times, and model-specific evidence may
be retained in `attributes` or external logs. Large masks should normally be
stored as artifacts referenced by stable IDs rather than expanded into every
runtime log.

## Units, coordinate frames, and object axes

- Positions and dimensions are metres.
- Angles are radians.
- `Pose3D.orientation_xyzw` is a unit quaternion in `(x, y, z, w)` order.
- The pose is `T_robot_base_object`: it maps object-local vectors/points into
  robot base, not the inverse.
- The object origin, local axes, dimensions, symmetry, and canonical pose rule
  come from `docs/OBJECT_CATALOG.md`.
- Times are backend clock seconds; do not substitute wall-clock time.
- Every returned object's timestamp equals the scene reference time. A tracker
  may propagate to that time, but must retain original frame times in evidence.
- Camera pixels are not robot coordinates. Use the declared calibration and
  tested projection/backprojection, never a learned simulator-specific
  pixel-to-millimetre shortcut.

For isolated flat screws or nails, record the shaft axis and head/tip polarity
when they affect grasping. For an upright vial, record base center, cylinder
axis, diameter/height, and upright/lying state; yaw about a symmetric cylinder
may be unobservable and should be treated according to its catalog symmetry.

## Dependencies and model artifacts

- Do not commit weights or downloaded model repositories.
- Receive a checkpoint through `context.checkpoint` or a profile option. The
  hardware-specific override is `ROCKWELL_VISION_CHECKPOINT`; Isaac passes
  `vision_checkpoint=` to its adapter. Record the artifact's immutable
  hash/version in `provider_version` and run logs. Do not make the vision
  plugin interpret `ROCKWELL_GRASP_CHECKPOINT` or the grasp artifact.
- Never put credentials in profiles, code, logs, or chat.
- Keep dependency versions in a plugin-owned `pyproject.toml` and `uv.lock` if
  the plugin needs packages absent from the base runtime.
- Brev setup uses exact package synchronization, and Isaac has a separate
  Python runtime. Ad-hoc `pip install` will be removed or will work in only one
  backend.
- For heavyweight or conflicting dependencies, run the model in a separately
  pinned local service and keep this provider as a small bounded-time client.
  Define startup, health check, request timeout, output validation, and clean
  shutdown. A service failure must fail the provider call closed.

## Required tests before handoff

1. **Import/factory:** import the selected module and construct it using a
   real `VisionFactoryContext` without changing shared files.
2. **Contract:** return a valid empty scene and a valid known-object scene;
   verify stale timestamps, unknown cameras/calibrations, duplicate track IDs,
   nonfinite values, and invalid quaternions are rejected.
3. **Determinism/replay:** freeze the model/checkpoint and replay identical
   frames. Explain any nondeterminism.
4. **Synthetic evaluation:** use camera-derived frames, never simulator truth
   to repair decisions. Measure detection, pose, occlusion, and class errors on
   held-out seeds/layout sequences.
5. **Real evaluation:** use native real-camera frames kept separate from tuning.
   Report position error in robot-base millimetres, orientation error with
   symmetry handled correctly, recall, false positives, dropouts, and latency.
6. **Integration replay:** serialize with `scene_observation_to_dict`, parse
   with `scene_observation_from_dict`, validate against the original request,
   and pass the result to the existing task adapter without robot motion.
7. **Failure cases:** close same-colour objects, partial occlusion, blue hose,
   hands/background clutter, touching objects, image blur, exposure shifts,
   stale frames, missing cameras, and objects outside the workspace.

Set thresholds from measured grasp clearance and stopping behavior, not from a
generic confidence number. Preserve full per-case results, including failures.

## Failure rules

- An inference exception, timeout, corrupt checkpoint, invalid calibration,
  malformed output, or nonfinite pose is an error, not an empty clear scene.
- A valid empty scene means “no supported object detected,” not “workspace is
  proven empty.” Unknown or occluded regions remain unknown.
- Do not invent pose coordinates for low-confidence detections. Retain the
  detection as unsupported/uncertain or omit it according to a documented
  policy, and ensure motion cannot interpret the missing area as clearance.
- Never silently use simulator truth, configured object spawn positions, old
  detections, or another camera's calibration to fill a failure.
- Never silently fall back to the baseline after an explicitly selected plugin
  fails. Operators and logs must know which provider produced each scene.

## Files this team must not edit independently

- `hardware/fanuc_cell/**`
- `isaac_bridge/**`
- `sorting_core/perception.py`, `sorting_core/grasping.py`,
  `sorting_core/integration_plugins.py`, or `sorting_core/team_integrations.py`
- `digital-twin-setup/**`
- `reference-assets/**`
- shared task/configuration files under `config/`
- another team's plugin directory

Ask the integration lead for a reviewed interface change instead. Model code,
plugin-specific configuration examples, fixtures, and documentation belong
here. Do not weaken or delete shared tests to accommodate a provider.

## Sim-to-real cautions

- The appearance fit is a first pass derived from compressed real video; it is
  not a calibrated native camera-response model.
- The table framing audit still found an approximately 52-pixel maximum
  landmark residual. Diagnose calibration versus geometry before relying on
  pixel-perfect transfer.
- Synthetic wrist-hose occlusion appears only in explicitly requested derived
  wrist previews, not native raw/sensor frames. Real rollout 4 was heavily
  obstructed throughout and rollout 5 for about 60% of sampled time.
- Table is the only camera currently trusted for motion coordinates. Side and
  wrist are auxiliary until their physical cross-camera biases, timing, and
  calibration are validated.
- Start with known, separated objects on a flat table. Arbitrary piles and
  unconstrained 6-DoF pose are not the first milestone.

## Handoff checklist

- [ ] Factory module and callable are importable from the repository root.
- [ ] Integration profile lists only consumed cameras and exact calibrations.
- [ ] Profile options and checkpoint/model hashes are documented.
- [ ] Provider name/version uniquely identify the model and preprocessing.
- [ ] Every supported class has a complete object-catalog entry.
- [ ] Units, axes, symmetry, timestamp handling, and image preprocessing are
      documented with one worked example.
- [ ] Contract, replay, synthetic, real, latency, and failure-case results are
      attached.
- [ ] Output validates through `validate_scene_for_request()`.
- [ ] No weights, datasets, secrets, generated artifacts, or absolute paths are
      committed.
- [ ] Baseline behavior still works with the profile variable unset.
- [ ] Remaining unsupported classes/conditions are explicit.
- [ ] The tested cube size is explicit: 10 mm and 16 mm are never mixed.
- [ ] A small serialized scene fixture lets the grasp team integrate without
      importing the vision model or receiving the training dataset.
- [ ] Model licensing, source, preprocessing, artifact retrieval, and expected
      CPU/GPU/memory/latency are recorded in the team-owned model notes.

## Current limitation that must remain visible

The current task adapter executes only the configured cube class and the
current executor produces a vertical top-down pose plus yaw. A provider may
correctly report screws, nails, vials, or tilted objects, but they remain
unsupported until task/destination mapping, grasp feasibility, and motion
execution support them. Do not mislabel them or drop roll/pitch merely to make
the old cube path accept them.
