# Synthetic training-data contract

The Brev/Isaac bucket video is a presentation smoke test. Its fixed studio
appearance, 1280×720 resolution, burned-in labels, and configured simulator
poses are not a training dataset. Do not train from its existing `frames/`
directory.

This repository now has a separate, host-testable data contract:

- `config/synthetic_data_generation.json` defines native table-camera capture
  and bounded, seeded randomization requests.
- `sorting_core.synthetic_data` validates that contract against the checked-in
  physical camera calibration, samples replayable per-episode manifests,
  applies deterministic derived sensor effects, and builds synchronized frame
  records.
- The demo's default presentation behavior remains unchanged. Its opt-in
  `--training-data` mode consumes this contract and its `--camera-perception`
  mode makes runtime decisions from rendered pixels rather than object poses.

## Non-negotiable capture rules

1. Capture the calibrated table camera at **1920×1080 RGB**.
2. Save a lossless, raw PNG before presentation text or lossy compression.
3. Keep any augmented image separate from the raw image.
4. Save the exact camera/calibration version, image timestamp, robot-state
   timestamp, scene-state timestamp, six measured arm joints, measured gripper
   opening/closure, FK-derived TCP, the separate last commanded state, scene
   version, motion phase, and per-episode manifest hash.
5. Apply the same geometric camera transform to RGB, masks, boxes, and points.
   Compute distorted amodal cuboid boxes from distorted projected 3D vertices,
   not by distorting the four corners of an already axis-aligned box.
6. Put simulator poses, masks, and other truth under `evaluator_only`; they are
   labels/evaluation data, never runtime policy inputs, pose repairs, or
   control-flow gates in camera/training modes.
7. Split datasets by complete episode/layout/arrival sequence. Adjacent video
   frames must not be scattered between training and evaluation.

The capture config references
`hardware/fanuc_cell/calibration/cameras.json` instead of copying intrinsics.
The loader rejects a resolution different from the table calibration. The
optional postprocess adds the table camera's measured one-term radial
distortion (`k1`) to a pinhole render and applies replayable exposure, white
balance, gamma, noise, blur, vignette, and JPEG effects. These display-RGB
effects are robustness approximations, not a physical camera model.

Training mode also enables the checked-in scan of the real work surface,
`reference-assets/table/manipbench-table-surface.png`, and removes the pristine
paper layer that otherwise covers it. This is an opt-in runtime override; the
default simulation and every hardware path remain unchanged. The first broad
appearance fit used `Rollout5TableCam.mp4`, while rollout 4 is reserved as a
held-out check. Both recordings are compressed 640x360 video, so they support
coarse exposure, colour-cast, vignetting, and tabletop matching only.

Reproduce that comparison without starting Isaac:

```powershell
uv run --project hardware/fanuc_cell scripts/compare_camera_appearance.py `
  --real-video HardwareResults/HardwareResultVideos/Rollout5TableCam.mp4 `
  --sim-dir path/to/run/raw_frames --sim-domain raw `
  --output-json path/to/raw-vs-real.json `
  --contact-sheet path/to/raw-vs-real.png
```

The tool downsamples synthetic frames to the real recording instead of
inventing detail by upsampling the video. Its exposure and white-balance hints
deliberately exclude the moving robot corridor and saturated objects. Collect
a native, lossless 1920x1080 real table-camera still before fitting sensor
noise, precise blur, camera response, or small-object edge appearance.

## Sample a replayable episode

This part runs in ordinary host Python and does not start Isaac:

```python
from sorting_core.synthetic_data import (
    load_synthetic_data_spec,
    sample_randomization,
)

spec = load_synthetic_data_spec()
manifest = sample_randomization(
    spec,
    episode_seed=20261025,
    episode_id="pilot-0001",
)
print(manifest.sha256)
```

Each parameter is seeded by its fully qualified name. Adding a new parameter
in a later config revision does not change values for existing names at the
same seed. The complete manifest, not just the seed, is the authoritative
replay record.

Sampling does **not** claim that Isaac applied anything. After the Isaac-side
adapter changes the stage, call `finalize_application_status(...)` with every
parameter classified as either applied or skipped with a reason. It rejects
partial accounting. Training mode marks stage parameters applied only after
reading their effective USD/PhysX consumers back. It writes the hashed evidence
to `isaac_randomization_readback.json`.

## Derived sensor image

```python
from sorting_core.synthetic_data import (
    augment_sensor_rgb,
    load_capture_calibration,
)

calibration = load_capture_calibration(spec)
result = augment_sensor_rgb(
    raw_rgb,
    manifest,
    frame_index=frame_number,
    camera_calibration=calibration,
)
derived_rgb = result.image_rgb
```

`raw_rgb` must be the native 1920×1080 `uint8` RGB render. Save it first. The
result remains 1920×1080 RGB and reports every applied sensor parameter.
Gaussian noise uses a recorded deterministic per-frame seed derived from the
episode seed and `frame_index`; adjacent frames do not repeat one noise field.
Instance masks should use `apply_radial_distortion(..., nearest=True)`; point
labels should use `distort_points_uv(...)`.

## Synchronized frame labels

`build_synchronized_frame_record(...)` rejects frames whose robot or scene
state differs from capture time by more than the configured tolerance. Its
record separates deployable inputs from evaluator-only truth:

```text
policy_input
  measured_robot_state
    arm_joint_positions_deg (six arm joints only)
    gripper
      jaw_half_opening_m
      closure_fraction (0=open, 1=closed over full mechanical stroke)
    tcp_pose_uf0_mm_deg (derived from measured joints through FK)
  scene_version
  phase

commanded_robot_state
  arm_joint_targets_deg
  gripper opening/closure target
  tcp_target_uf0_mm_deg

evaluator_only
  object IDs/classes/colours
  base-frame transforms and dimensions
  visibility/occlusion
  explicitly amodal, image-clipped projected-3D-cuboid boxes
  instance-mask paths
```

The projected cuboid boxes are not visible-pixel boxes and must not be used as
evidence of visibility or occlusion. Invalid/behind-camera/out-of-frame boxes
are recorded with an explicit status and a null value.

An episode-level report must additionally record candidate grasps, rejection
reasons, action rows, perception/planning/execution/recovery times, correct
placement, failed lift, drop, wrong bin, unintended contact, cancellation,
retry, and intervention. The default scripted demo's execution report is not a
substitute because its decisions come from simulator truth; the opt-in
camera-perception mode removes that particular decision leak but does not add
the missing episode-level annotations or camera-based post-place verification.
The Isaac demo therefore executes the remaining planned actions regardless of
evaluator-truth success/failure; truth scores `task_complete` but cannot control
the sequence or process exit status. `runtime_sequence_complete` is reported
separately and is not an independent placement-success measurement.

## What the provisional ranges do and do not mean

The checked-in ranges cover:

- coloured-plastic albedo and roughness, and table roughness;
- dome/key/fill/rim intensity and light-source size (shadow softness);
- small camera pose and focal perturbations;
- exposure, white balance, gamma, noise, blur, vignette, and compression;
- independent cube position/yaw perturbations; and
- cube mass, physically ordered static/dynamic friction, and restitution.

Every range remains marked `provisional_unmeasured`. Broad light intensity,
white-balance, exposure, gamma, and vignette bounds now use compressed hardware
video as a starting constraint, but that is not a native photometric camera
calibration. Other values remain engineering starting points. Broad
randomization can reduce accuracy or erase the red/yellow/blue class meaning.
Replace/bound the ranges using native real samples and physical measurements
before claiming sim-to-real accuracy.

## Isaac integration and remaining annotation work

The opt-in collector deliberately mutates only the disposable runtime and
stage; it does not modify the pristine `digital-twin-setup/` source. It now
applies the sampled material, light, camera, scene and physics values, captures
raw and sensor-derived RGB, snapshots robot/object state, writes synchronized
records, and refuses an incompletely classified application manifest.

The task-cube visuals are referenced USD assets with their own variant
materials. Training mode therefore authors a separate local material for each
`/World/SortCube*/Visual/Geom/Die`, binds it on that exact mesh, verifies the
effective binding and shader-input readback, and only then classifies the
albedo/roughness requests as applied. The table material, lights, cube
transforms/masses and PhysX coefficients receive the same readback treatment.
When the real printed tabletop artwork overlay is present, its texture remains
enabled and the collector changes/verifies that overlay shader's roughness too;
otherwise varying only the hidden material below it would not change pixels.

`effective_camera.json` contains the focal/aperture and USD transform read from
the live camera prim, not just the requested config. It records both
`camera_to_world` and `camera_to_robot_base` explicitly (the axes align in this
twin, while the origins do not) and carries a SHA-256 digest. Every frame label
embeds that effective metadata and references both episode readback hashes. The
real-camera lens distortion remains a distinct derived-image transform; it is
not falsely described as an Isaac camera intrinsic.

The following work remains before calling the images a complete perception
training dataset:

1. vary object count and create deliberate clutter, touching/overlap,
   occlusion, blocked approaches, and arrival/motion sequences; the current
   episode always contains exactly three cubes and only perturbs each cube's
   XY/yaw within a small range;
2. independently materialize bin materials too if individual-bin appearance
   variation becomes useful; task cubes are already materialized per instance;
3. add semantic/instance render passes, measured visibility/occlusion,
   instance masks and—only where the selected model needs them—depth/normals;
4. validate randomized layouts with complete swept-volume collision checks,
   not only separation, IK and physical execution gates;
5. model motion blur, rolling shutter, auto-exposure and the measured camera
   response instead of generic display-RGB approximations;
6. add a deployable camera-based post-place outcome check rather than using
   simulator-truth placement scores;
7. replace every provisional range using physical samples/measurements and
   validate held-out real-camera performance;
8. decouple capture/label writing from policy success (or label failed decision
   frames), because the current training run aborts before dataset frames when
   randomized appearance defeats the HSV policy and therefore undersamples the
   hardest conditions; and
9. record the clean project commit, Isaac/renderer build and settings, GPU, and
   driver in each episode manifest; current config/calibration/readback hashes
   do not fully identify the generator runtime; and
10. measure yaw accuracy at the real camera's native part scale. In the current
    wide table view a 10 mm cube spans only about 14--16 pixels. A fixed 3x3
    morphology opening was therefore disabled after it changed a stationary
    seed-7 cube's estimated yaw by 18.7 degrees. The shared physical-size,
    aspect, and fill gates rejected the observed robot-colour distractors in
    replay, but held-out real images are still required before deployment.

Path-traced imagery may improve shadows and reflections, but it does not repair
wrong geometry, calibration, time synchronization, physics, or labels. Render
quality and data correctness must be tested separately.

## Missing measurements

Before these data can be described as accurate for the physical FANUC cell, we
still need:

- current table-camera validation images and exposure/white-balance behavior;
- real cube/bin RGB examples across the work area and under expected lighting;
- cube/bin dimensions, bevels, mass, material, wear, and surface reflectance;
- table/cube/finger and cube/bin friction plus finger compliance;
- pneumatic close/open delay and effective grasp force;
- controller/network latency, interpolation, and tracking error; and
- confirmation of the active **10 mm cube** and fingertip TCP convention.

The pinned client/mock labels its action point as the fingertip. Separately, an
organizer clarified on October 26, 2026 that the standard fingers touch the
table near UF0 Z = -335 mm; the former -345 mm FAQ value came from another
group's shorter fingers and was reverted. These facts are consistent, but a
supervised high check must still verify the installed fingers and active
UTOOL1/UF0 before physical descent. Synthetic randomization cannot make an
incorrect or stale tool frame safe.
