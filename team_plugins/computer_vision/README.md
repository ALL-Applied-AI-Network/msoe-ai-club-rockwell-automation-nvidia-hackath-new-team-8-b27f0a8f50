# Cube and screw perception experiments

This directory contains computer-vision-team code only. The production default
is still the profile-unset HSV cube baseline. Nothing here changes hardware or
Isaac behavior unless an integration profile is explicitly selected.

## Leading bounded implementation

`hybrid_geometry_provider.py` combines:

1. the unchanged HSV/calibrated-geometry cube detector;
2. a conservative silhouette/width-profile detector for the supplied
   `screw_iso10642_m6x30` while it lies on the white table; and
3. a small nearest-neighbour tracker for separated, slowly changing scenes.
   Screw promotion requires three observations with strictly newer
   camera-frame timestamps; repeated calls on one frozen frame do not count.
   A match must remain within 25 mm and within configured cross-frame length,
   width, directed-axis, and head-width-ratio bounds. This is still not a
   motion model: close/crossing parts can swap IDs, and a persistent lookalike
   can pass unless the negative-case gate rejects it. A disappearance is not
   labeled as occlusion because it could be a successful pickup.

The official adapter currently copies the backend observation timestamp into
`CameraFrame`; it does not expose a camera exposure ID. The timestamp rule
therefore blocks repeated calls carrying the same timestamp, but cannot prove
that two differently stamped images came from different physical exposures.

The exact opt-in profile is `profile.hybrid-geometry.v1.json`.
Raw cube observations outside the pickup region are preserved because the
existing hardware policy needs them for virtual-table/bin destination
verification; this provider only annotates whether each cube is in the pickup
region and leaves eligibility to downstream logic.

The screw pose uses an **experimental canonical representative** of the
catalog's restricted lying-pose family:

- position: estimated center of the head face in robot-base metres;
- local `-Z`: head-to-tip shaft direction;
- local `+X`: canonical robot-base `+Z` for a lying screw;
- local-axis roll: explicitly unobservable at this camera scale;
- sort label: `red`.

The current profile uses provisional CAD-centered projected-size gates of
26--34 mm length and 9--14 mm width. These are detector acceptance gates, not
physical-pose error bounds. They were tightened after archived robot/tool
details passed the broader 21--41 mm by 3--16 mm development gate. Native
images of the actual event screw are still required.

The provider projects the whole nonplanar silhouette onto one configured plane
at `surface_z + 6.5 mm` and uses the longitudinal extreme as the head-face
origin. Those are unverified approximations, not measured 3D pose. The object
catalog still marks screw axial symmetry as unresolved; this quaternion must
not be interpreted as a catalog-approved full 6D orientation.

A geometry-in-range dark candidate becomes `unknown_obstacle` when polarity,
boundary, workspace, table-context, confidence, or temporal-confirmation checks
fail. Components outside the area/geometry gates can still be omitted. Hands,
large robot/tool silhouettes, saturated cables, arbitrary unsupported objects,
and all occluded regions are not comprehensively segmented; an omitted region
is not proven clear.

This profile intentionally does not detect buckets. Destination handling stays
outside this provider. It also does not segment arbitrary hands, cables, or all
occluded image regions: only screw-like dark ambiguities become
`unknown_obstacle`. Unknown clear space is therefore not proven.

The shared task adapter still executes only class `cube`. A correctly reported
screw remains visible but unsupported until the grasp/task/executor work is
merged. Never rename a screw to `cube` to bypass that boundary.

## Runtime contract

Every target is returned in a `sorting_core.perception.SceneObservation`, not a
human-readable string. The important handoff fields are:

```text
track_id
class_id
sort_label
capture_time_s
pose.position_base_m
pose.orientation_xyzw
dimensions_m
confidence
pose_quality
visibility
source_camera_ids
attributes.catalog_id / symmetry_id / uncertainty_scope
```

`confidence` is a detector/shape score, not a calibrated probability of grasp
success. Cube `pose_quality` and visibility are left unset because neither is
independently measured. Screw `pose_quality` is only a mask/shape-fit score;
visibility remains unset and `occlusion_status` states that occlusion was not
estimated. Both cubes and screws carry
`absolute_base_accuracy_verified=false` until a measured physical-pose test is
completed.

## Run the team tests

From the repository root in WSL:

```bash
uv run --project hardware/fanuc_cell python -m unittest \
  team_plugins.computer_vision.tests.test_hybrid_geometry_provider -v
```

The tests cover:

- both head/tip polarities;
- several rotations;
- two separated screws;
- repeated-frame tracking and frozen-frame non-confirmation;
- cross-frame shape-change reset;
- severe mid-shaft occlusion;
- merged/crossing screws;
- empty and dark-square negatives;
- a uniform dark-bar ambiguity;
- a screw-shaped dark detail on a colored container;
- wrong input resolution; and
- calibration mismatch and stale/future frame timestamps;
- factory/profile provenance;
- boundary-straddling objects and complete-footprint checks;
- workspace-margin validation;
- non-monotonic time and non-fabrication of occlusion on disappearance; and
- preservation of the cube path.

The provider has 26 focused tests, the annotation evaluator has 12, replay and
overlay/provenance handling has 9, and the bounded live diagnostic has 12, for
59 computer-vision-team tests total.
Run the annotation evaluator tests directly with:

```bash
uv run --project hardware/fanuc_cell python -m unittest \
  team_plugins.computer_vision.tests.test_annotation_evaluation -v
```

See `ANNOTATION_EVALUATION.md` for the strict, hash-pinned manual annotation
format and symmetry-aware evaluation workflow needed to turn native hardware
captures into honest accuracy measurements.

## Run a no-motion real-video replay

Downloaded recordings are only 640x360 compressed video. Upscaling them is
useful for regression review but cannot restore native detail or establish
millimetre accuracy:

```bash
uv run --project hardware/fanuc_cell python -m \
  team_plugins.computer_vision.replay_camera \
  /path/to/table.mp4 \
  --output-dir /path/to/replay-output \
  --cube-edge-mm 16 \
  --every 10 \
  --allow-upscale-compressed
```

The command requires a new or empty output directory, loads the exact checked-in
profile, validates every returned scene against its request, and writes an
overlay video, sampled overlays, JSONL scene records, and a hash-pinned report.
It emits a schema-valid `evaluation-result.v1.json` only when the worktree is
clean and capture/calibration/profile/provider/tool/catalog/commit/branch
provenance stays unchanged throughout the run. A provenance change makes the
command exit nonzero. Timestamps are decoded `frame_index / FPS`, not measured
exposure times. Do not commit recordings, weights, or generated outputs.

The overlay uses numbered markers and a corner legend: yellow rows are cubes,
green rows are screw **geometry candidates**, red rows are non-executable
unknowns, and gray/crossed rows are outside the permitted region. `cube-region`
means only that a cube center is inside the configured pickup region; confidence,
ambiguity, complete-scene, clearance, and planner checks still apply.
`screw-cand` does not mean the cube-only executor can grasp a screw.

Use `--every 1` when testing temporal confirmation. Sparse sampling can hide
short-lived false tracks and does not represent the live 10 Hz decision stream.

The downloaded LeRobot archives are AV1. Use the optional PyAV environment
without changing the project lock; the tool selects PyAV for AV1 instead of
trusting partial OpenCV decoding:

```bash
uv run --project hardware/fanuc_cell --with av python -m \
  team_plugins.computer_vision.replay_camera /path/to/file-000.mp4 \
  --output-dir /path/to/replay-output --allow-upscale-compressed
```

## Select the plugin explicitly

For import/checker testing:

```bash
export ROCKWELL_INTEGRATION_PROFILE=team_plugins/computer_vision/profile.hybrid-geometry.v1.json
```

Remove it to prove the original baseline still loads:

```bash
unset ROCKWELL_INTEGRATION_PROFILE
```

Run the one-command checker before handoff:

```bash
uv run --project hardware/fanuc_cell python scripts/check_team_plugin.py \
  --profile team_plugins/computer_vision/profile.hybrid-geometry.v1.json
```

The profile does not encode cube size; freeze the commit, profile hash,
calibration hash, config, and explicit `--cube-mm 10|16` runtime choice
together. The checker proves software compatibility, not physical pose
accuracy.

Treat this hybrid profile as offline/no-motion until the native calibration,
timestamp, negative-case, and measured-pose gates in `HARDWARE_VALIDATION.md`
pass. The shared executor cannot execute screws at all. The exact no-motion
fallback from `hardware/fanuc_cell` is:

```bash
unset ROCKWELL_INTEGRATION_PROFILE
uv run supervised_run.py hold
```

That fallback verifies the cell connection and measured-pose hold but does not
invoke perception. For a bounded live perception capture, run this from the
repository root in WSL during the approved operator-supervised slot:

```bash
uv run --project hardware/fanuc_cell python -m \
  team_plugins.computer_vision.live_no_motion_diagnostic \
  --output-dir /mnt/c/path/outside-the-repository/live-vision-01 \
  --cube-mm 16 \
  --max-snapshots 10
```

The output directory must be new or empty. The tool selects the exact hybrid
profile before importing the official policy, clears inherited motion,
descent, and grasp-fit flags, refuses a policy that reports either interlock as
enabled after construction or rollout reset, and rejects a cached/different
integration profile instead of silently using it. It runs only the existing
perception boundary, and saves decoded requested-resolution RGB as PNG plus
canonical `SceneObservation` JSON, latency, hashes, calibration/provider
identity, the exact `cameras.json` SHA-256,
commit, branch, and dirty-state provenance. Each record also retains the current
six-joint-plus-gripper feedback and UF0 TCP pose with the backend observation
timestamp. That timestamp is not a camera exposure timestamp. Provider or
downstream task-partition failures produce an explicit error record and a
nonzero final exit status; they never become an empty clear scene.
`_detect_scene_views` includes the existing cube task's diagnostic partitioning,
so an error there can follow valid provider output. The returned action is still
the original strict measured TCP/gripper hold row.

The saved PNG is a lossless re-encoding of the RGB array delivered by the
backend. Its record states RGB channel order, requested wire format, decoded
shape, and both pixel/PNG hashes. It is not claimed to be an untouched sensor
frame; verify sensor-native resolution and upstream processing separately.

The provider's configured 0.15 s frame-age check does not bound inference or
end-to-end age in the current hardware adapter: both the request and
`CameraFrame` are stamped from the same `obs['t']`, so their difference is
normally zero. The policy separately enforces its gained-observation-age
deadline (3 s by default) after synchronous perception/planning. Neither value
measures camera exposure skew.

The official TCP protocol has no true no-op. Repeating the measured pose may
still cause a small controller correction, so this diagnostic is not an
emergency stop and the operator/controller remain responsible for stopping.
Run it only with the supervised cell stationary; do not jog the robot while it
is active. Perception and evidence writes happen before `act()` returns, so they
delay delivery of the already-constructed hold row. Each record separates
hold-construction time, perception/task-boundary latency, and a lower bound on
total action-return delay (the final record write/print is excluded). This is
intentionally synchronous; do not add asynchronous capture during the final
hardware session.
It uses the private lead-owned `_detect_scene_views` boundary because no public
live perception-only seam exists; recheck this adapter whenever the hardware
policy changes.

## Files and ownership

- `hybrid_geometry_provider.py`: provider, screw geometry, ambiguity handling,
  and tracking.
- `profile.hybrid-geometry.v1.json`: exact opt-in integration profile.
- `replay_camera.py`: offline no-motion replay/overlay tool.
- `live_no_motion_diagnostic.py`: bounded live perception capture while the
  official policy returns its measured-pose hold.
- `tests/`: team-owned contract and edge-case tests.
- `MODEL_EVALUATION.md`: model comparison and measured evidence.
- `DECISION_LOG.md`: hypotheses, results, and next experiments.
- `HARDWARE_VALIDATION.md`: the physical evidence still required.
- `ANNOTATION_EVALUATION.md`, `annotation-manifest.v1.schema.json`, and
  `evaluate_annotations.py`: reviewed ground-truth/evaluation workflow.

Lead/main follow-up: `docs/OBJECT_CATALOG.md` still describes the old
profile-unset HSV path as unable to detect screws. The lead should clarify on
`main` that the opt-in hybrid profile can emit a restricted, experimental lying
screw observation but is not execution-ready. This vision branch intentionally
does not edit that shared file.

Ask the team AI when a field, coordinate frame, test, or failure message is
unclear. Do not edit hardware control, Isaac, shared contracts, or the grasp
team's files from this branch.
