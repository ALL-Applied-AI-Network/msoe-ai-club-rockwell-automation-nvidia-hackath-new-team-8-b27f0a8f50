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

The active hybrid profile requests table RGB only, so the diagnostic captures
only that stream. Capturing side/wrist/depth requires a separately reviewed
profile/backend change; merely listing another camera does not calibrate or
fuse it.

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
- no stale/mismatched timestamp or calibration accepted (the provider now
  enforces exact calibration ID, a 0.15 s maximum frame age, monotonic
  camera-frame time for tracking/confirmation, and request-boundary validation;
  three calls on one frozen frame do not count as three observations). The
  current official adapter copies
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
