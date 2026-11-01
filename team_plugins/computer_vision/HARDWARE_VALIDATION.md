# Final hardware-session vision checklist

The current software is not yet proven to return true robot-base poses on
hardware. The downloaded videos cannot supply that proof. Use this checklist to
turn the final session into measured evidence while keeping the existing
cube-only fallback intact.

Record references and predictions with the strict workflow in
`ANNOTATION_EVALUATION.md`; it emits the repository's shared evaluation-result
format without filling unavailable pose metrics with zeros.

## Freeze before arrival

Record:

- Git commit and branch;
- integration profile path and SHA-256;
- cube size profile (10 mm or 16 mm);
- camera calibration file/hash;
- provider/model version;
- object catalog/CAD hashes;
- expected native image sizes and RGB order;
- who records measured placements; and
- the exact fallback command with the profile unset.

Do not mix results from different commits, calibrations, cube sizes, or profiles
in one accuracy claim.

## Capture first; do not move the robot

The ordinary `supervised_run.py hold` path does not invoke perception. Use the
bounded diagnostic from the repository root during the approved slot:

```bash
uv run --project hardware/fanuc_cell python -m \
  team_plugins.computer_vision.live_no_motion_diagnostic \
  --output-dir /mnt/c/path/outside-the-repository/live-vision-01 \
  --cube-mm 16 --max-snapshots 10
```

### Opt-in YOLO11n hardware evidence

The default command above remains the HSV/geometry hybrid. To exercise the
fine-tuned YOLO11n proposal model, use its isolated runtime and external
checkpoint explicitly:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/rockwell-yolo11n-runtime \
  uv sync --project team_plugins/computer_vision/yolo11n_runtime --frozen

UV_PROJECT_ENVIRONMENT=/tmp/rockwell-yolo11n-runtime \
  uv run --project team_plugins/computer_vision/yolo11n_runtime python -m \
  team_plugins.computer_vision.live_no_motion_diagnostic \
  --vision yolo11n \
  --vision-checkpoint <absolute-path-to-seed55-best.pt> \
  --output-dir /mnt/c/path/outside-the-repository/live-yolo11n-01 \
  --cube-mm 16 --max-snapshots 10
```

This is the first real-hardware test, not permission to descend. YOLO11n emits
2D class/box proposals only. The provider still requires the checked-in table
camera calibration and deterministic cube geometry to produce robot-base
position and square-symmetry yaw. The checkpoint is accepted only when its
SHA-256 is
`010bb71b1b07fc77490a2bc2bd774102205640bdf9ba07d6fa8b3bd758788fbb`.
Keep the `.pt` file and captured evidence outside Git.

Before the cell slot, replay the exact profile/checkpoint on saved table-camera
video with the same isolated runtime. This exercises the same profile factory
without requesting hardware motion and records the selected provider-module and
checkpoint hashes:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/rockwell-yolo11n-runtime \
  uv run --project team_plugins/computer_vision/yolo11n_runtime python -m \
  team_plugins.computer_vision.replay_camera \
  /path/to/table.mp4 \
  --profile team_plugins/computer_vision/profile.yolo11n-cube.v1.json \
  --vision-checkpoint /absolute/path/outside/repository/seed55-best.pt \
  --output-dir /path/outside/repository/yolo-replay-output \
  --cube-edge-mm 16 --every 1 --allow-upscale-compressed \
  --record-vision-errors
```

Without `--record-vision-errors`, replay remains fail-fast. With it, each strict
YOLO whole-scene rejection is an explicit error/abstention record and `NO SCENE`
overlay—not an empty scene or successful observation. `report.json` includes
the count and maximum length of consecutive error streaks in **processed replay
frames**, plus 1-based processed ordinals, original source-frame ranges, and
reason counts. This measures acceptance/dropout behavior; compressed replay
still cannot validate robot-base millimetre pose accuracy. Any recorded
abstention suppresses `evaluation-result.v1.json`, even when other frames
succeeded, because that schema cannot represent the missing per-frame scene.
Use `report.json` and `scenes.jsonl` for mixed runs; the report states the
standardized-evaluation skip reason explicitly. A zero-error standardized run
binds its experiment identity to provider name/version and the calibration
SHA-256, and records that calibration hash without exposing an absolute path.

The YOLO11n profile intentionally abstains instead of guessing if it sees an
above-threshold screw prediction, if a YOLO cube box and calibrated cube
geometry do not agree one-to-one inside the crop, or if calibrated geometry
finds a pick-eligible cube outside neural crop coverage. Expect those cases to
produce a visible provider error and a hold, not a partial scene for motion.

The provider performs three blank-image inferences while `Policy` is constructed,
before the first live observation, so cold Torch/Ultralytics initialization is
not charged against a camera frame's freshness budget. On the project laptop's
WSL CPU on October 31, 2026, the exact seed-55 checkpoint loaded in about
1.1 s, the cold initialization/inference took about 11.8 s, and the next two
1000x650 passes took 150 ms and 128 ms. CUDA was unavailable in that test.
Re-measure on the machine used for the cell session; those three measurements
are a runtime smoke test, not a latency guarantee or pose-accuracy result.
In a later three-pass-warm-up replay, construction took 7.25 s, the warm-up
passes took 2.66 s, 116 ms, and 155 ms, and the first reconstructed real-frame
observation took 591 ms. The timings vary, which is why live records must retain
per-observation latency instead of assuming the best warm measurement.

After the no-motion records show the correct objects and stable calibrated
poses, dry-run the normal hardware wrapper:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/rockwell-yolo11n-runtime \
  uv run --project team_plugins/computer_vision/yolo11n_runtime python \
  hardware/fanuc_cell/supervised_run.py hover \
  --vision yolo11n \
  --vision-checkpoint <absolute-path-to-seed55-best.pt> \
  --cube-mm 16 --dry-run
```

The explicit `/tmp` environment keeps the roughly 5 GB Torch/CUDA dependency
tree on WSL's Linux filesystem instead of slowly copying thousands of files
into the Windows-mounted repository. It is intentionally disposable and must
be recreated after `/tmp` is cleared or WSL is reset. The lockfile, not that
temporary environment, is the reproducible artifact.

Remove `--dry-run` only for the already established operator-supervised hover
stage. Do not start with `sort`: the held-out real-video result for this
checkpoint was 85.3% precision and 82.9% recall on 35 provisionally labeled
cube instances, and screw performance was not evaluated because the held-out
set contained no screws. These are development measurements, not guarantees of
calibrated pose accuracy.

Choose `10` or `16` only after measuring the actual cubes. Start with
`--dry-run` to inspect the non-secret effective settings. The directory must be
new or empty and must remain outside Git. The tool clears motion/descent flags,
refuses an armed policy after construction or reset, never calls planning, and
saves decoded requested-resolution RGB plus the validated structured scene. It
still returns the official measured-pose hold row, which is not an emergency
stop and can produce a small controller correction. The assigned operator and
controller remain responsible for stopping.

`--max-snapshots` limits provider calls and saved evidence only. It does not
terminate the approved rollout; after reaching the limit, the policy continues
returning fresh measured-pose holds until the cell run ends or the operator
stops it. A downstream cube-task partition error cannot erase an already
validated provider scene, and an evidence-write failure is reported without
replacing the already-constructed hold row.

Keep the robot/controller stationary and do not jog while this tool is active.
The hold row is constructed first, but synchronous perception and file writes
delay its return to the client. Records separate hold-construction time,
perception/task-boundary latency, and a lower bound on total action-return delay;
the lower bound excludes the final record write and event print.

Every record includes current J1--J6 degrees, gripper feedback, and the UF0 TCP
pose in millimetres/WPR degrees. Their source timestamp is the backend
observation time copied into the adapter, **not** a measured camera exposure
time. Use it for replay/provenance, not as proof of camera/robot synchronization.

### Paired table and side evidence

When the blue cable or robot blocks the table view, collect the fixed side view
in the same no-motion observation without giving it pose or motion authority:

```bash
uv run --project hardware/fanuc_cell python -m \
  team_plugins.computer_vision.live_no_motion_diagnostic \
  --output-dir /mnt/c/path/outside-the-repository/live-table-side-01 \
  --cube-mm 16 --capture-camera table --aux-camera side \
  --max-snapshots 10
```

Both fixed cameras are requested at their pinned 1920x1080 size. The selected
provider receives **table only**; side is saved as capture-only evidence with
its own calibration ID and is not used to create, average, or repair a grasp
pose. A missing table or side frame makes that snapshot an explicit recording
failure while preserving the already-constructed measured hold row.

Do not pair wrist with table in this command. Wrist is pinned at 1280x720, the
client exposes one RGB request size, and the diagnostic rejects that mixed-size
pair instead of rescaling calibration. Capture wrist separately with
`--capture-camera wrist` while the arm is settled. The backend timestamp is not
an exposure timestamp, so even paired table/side files are evidence from one
backend observation—not proof of sensor synchronization. Do not enable side or
wrist coordinates for motion until their bias, timing, and pose error are
measured independently.

For table, side, and wrist cameras, save:

- decoded full-frame RGB at the requested resolution, and separately record
  whether the backend output is actually sensor-native;
- untouched sensor output too, if the backend exposes it;
- image shape, dtype, channel order, camera ID, and exposure timestamp;
- the exact robot observation timestamp and joints;
- calibration ID;
- depth plus its units/invalid value/alignment metadata when available; and
- a photo/note of every object actually present.

The table camera is the only current source permitted for robot-base target
coordinates. Side/wrist frames are diagnostic until cross-camera calibration
and timing are measured.

The diagnostic can request one RGB stream at a time at the resolution pinned by
`hardware/fanuc_cell/calibration/cameras.json`. Separate processes are required
because the official client exposes one global RGB request size while table and
side are 1920x1080 and wrist is 1280x720:

```bash
# Evidence only: no side/wrist provider inference and no coordinate fusion.
uv run --project hardware/fanuc_cell python -m \
  team_plugins.computer_vision.live_no_motion_diagnostic \
  --capture-camera side \
  --output-dir /mnt/c/path/outside-the-repository/live-side-01 \
  --cube-mm 16 --max-snapshots 10

uv run --project hardware/fanuc_cell python -m \
  team_plugins.computer_vision.live_no_motion_diagnostic \
  --capture-camera wrist \
  --output-dir /mnt/c/path/outside-the-repository/live-wrist-01 \
  --cube-mm 16 --max-snapshots 10
```

Those modes save the decoded frame, calibration ID/solve/RMS/transform
provenance, observation timestamp, joints and TCP, and mark the record
`capture_only`. They deliberately do not run the table provider or publish a
pose. The backend may also require the selected camera to be listed in the
approved submission; this has not been tested against the live service. Verify
that requirement with the organizer or a hold-only request before relying on
the commands. Requested native-calibration resolution is still not proof that
the returned pixels are untouched sensor-native data. Depth capture remains a
separate reviewed backend change. Merely saving another view does not calibrate
or fuse it.

## Minimal measured pose set

Use an organizer-approved fixture/grid whose transform to robot base is
independently surveyed or taught by the operator. A loose ruler or printed grid
is not base-frame truth by itself. Do not infer truth from the same camera being
evaluated, and do not add a visual marker that changes the runtime target
appearance unless that change is separately evaluated.

### Cubes

Place one measured cube at:

- four pickup-region corners;
- center;
- at least two intermediate positions; and
- several yaw values.

For every placement, record physical edge size, measured base XY, cube top
height, yaw modulo 90 degrees, image, detection, and error. Repeat frames without
moving the cube to measure jitter.

### Screws

First verify that the physical part is the supplied ISO 10642 M6x30 and record
finish/color. Place one isolated lying screw at several measured head-face
centers and shaft angles. Record:

- head-face center in base coordinates;
- head-to-tip shaft direction;
- whether the entire silhouette is visible;
- detector output or explicit abstention; and
- position/axis error using screw symmetry.

Record whether the physical event screw falls inside the current 26--34 mm
projected-length and 9--14 mm projected-width gates across the workspace. These
are archived-video-derived starting gates, not physical tolerances. Do not widen
them until the resulting negative-case false-target rate is remeasured.

Do not demand or claim axial roll from the current camera view.
The current canonical roll is an experimental representative; obtain lead/grasp
team approval before freezing it as a training-label convention. Independently
check the configured `surface_z + 6.5 mm` projection and head-extreme origin
approximation rather than treating them as measured geometry.

### Fast layout ladder

If session time is short, capture this bounded set before attempting motion:

1. five registered cube layouts spanning center, inset corners, intermediate
   positions, and yaw modulo 90 degrees; permute colors across locations;
2. six isolated screw placements spanning the pickup region and shaft angles;
3. two separated screws, then one touching/crossing layout that should abstain;
4. cable occlusion and a boundary-straddling screw that should abstain; and
5. empty table, bins, robot/tool, hand, and unsupported-dark-object negatives.

Capture 5--10 consecutive frames without moving each layout. Those repeats
measure jitter, not independent accuracy; keep the whole layout together when
splitting data. Prioritize independently measured truth and negative cases over
collecting many nearly identical frames.

## Native image/depth gate for FoundationPose

For one cube and one screw:

1. capture raw native RGB, depth, intrinsics, and timestamps;
2. verify depth is registered to RGB;
3. count valid target depth pixels and plot depth edges over RGB;
4. measure depth spread/bias on the part and table; and
5. stop the FoundationPose experiment if the 6 mm shaft is sparse,
   misregistered, or unstable.

Compressed 640x360 recordings are not usable for this gate.

## Required negative cases

Capture and label:

- empty workspace;
- bins/containers present;
- blue cable crossing or near a target;
- robot/tool in the camera view;
- a hand entering the image;
- one unsupported dark object;
- two separated screws;
- touching/crossing screws;
- partial screw occlusion;
- cube/screw near workspace boundary; and
- exposure/lighting variation.

The expected behavior for ambiguity is an explicit unknown/low-quality result or
no executable target—not a precise guess.

The current provider only covers screw-like dark ambiguities. A disappeared
track is not automatically an occlusion because it could be a successful
pickup. Hands, cables, arbitrary unsupported objects, and all occluded regions
are not comprehensively segmented. Keep them out of the active path or add a
separately validated obstacle/unknown-region provider plus execution feedback
before relying on this profile for motion. Touching/stacked screws are
intentionally unsupported.

## Acceptance gate before any new provider controls motion

Set numeric thresholds from measured finger/object clearance and the grasp
team's approach geometry. Compute the lateral perception budget rather than
choosing a round error target:

```text
usable_xy_budget = raw_half_capture_clearance
                 - calibration_uncertainty
                 - robot/TCP_uncertainty
                 - truth/placement_uncertainty
                 - object_tolerance
                 - motion/latency_allowance
                 - chosen_margin
```

Proceed only when that budget is positive. A nominal 10 mm cube in the stated
12 mm straight opening has only 1 mm raw clearance per side before any of these
terms. A nominal 16 mm cube is slightly larger than the modeled 15.99 mm hook
opening, so software cannot establish fit; measure the installed fingers and
part. Then require:

- target class/count correct on the frozen validation set;
- accepted base-position and symmetry-aware axis/yaw errors within that
  clearance-derived threshold;
- no executable false target on any negative case;
- repeated-frame jitter inside the same threshold;
- no stale/mismatched timestamp or calibration accepted (both opt-in providers
  enforce exact calibration ID, a 0.15 s maximum declared frame age, and
  request-boundary validation. The hybrid screw tracker additionally enforces
  monotonic camera-frame time for confirmation, so three calls on one frozen
  frame do not count as three screw observations. YOLO11n has no cross-call
  tracker). The current official adapter copies
  one observation timestamp into each `CameraFrame`, so this validates metadata
  consistency but does **not** measure real camera-exposure-to-robot skew;
- inference plus capture/service latency below the policy's separate gained-age
  deadline (3 s by default); and
- the same preprocessing/profile used in the test and runtime.

The provider's 0.15 s check is normally zero in the current hardware adapter,
because request time and `CameraFrame` time both come from the same `obs['t']`.
It therefore does **not** constrain inference or end-to-end latency. The policy's
separate gained-age deadline is the active synchronous processing limit, while
actual exposure-to-observation skew remains unmeasured.

If any gate fails, use the existing cube-only baseline for the physical demo.
Do not lower a gate just to make a model pass.

## Integration facts

- Selecting the hybrid profile can expose screws in `SceneObservation`.
- The shared executor still cannot pick screws. That requires the grasp team's
  screw candidates, deterministic finger/shaft fit, complete-path clearance,
  retention/release checks, destination mapping, and staged motion tests.
- Cubes preserve the legacy provider's raw inside/outside-pickup observations
  so existing destination verification can still partition them downstream.
  The hybrid provider adds provenance but leaves cube pose quality unset.
- Profile unset means the original baseline remains active.

## After the session

Preserve successes and failures. Save:

- raw inputs and measured truth;
- serialized `SceneObservation`;
- overlays with axes/confidence/quality;
- latency and abstention reason;
- calibration/model/profile hashes;
- grasp/outcome only if an action was actually executed; and
- every operator intervention.

Split future training/validation by complete trial/layout sequence, never by
adjacent frames from the same recording.
