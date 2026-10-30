# Synthetic training-data contract

The Brev/Isaac bucket video is a presentation smoke test. Its fixed studio
appearance, 1280×720 resolution, burned-in labels, and configured simulator
poses are not a training dataset. Do not train from its existing `frames/`
directory.

This repository now has a separate, host-testable data contract:

- `config/synthetic_data_generation.json` defines the primary native
  table-camera contract and bounded, seeded randomization requests; collection
  validates table, side, and wrist against the same calibration file.
- `sorting_core.synthetic_data` validates that contract against the checked-in
  physical camera calibration, samples replayable per-episode manifests,
  applies deterministic derived sensor effects, and builds synchronized frame
  records.
- The demo's default presentation behavior remains unchanged. Its opt-in
  `--training-data` mode consumes this contract and its `--camera-perception`
  mode makes runtime decisions from rendered pixels rather than object poses.
- `config/hardware_frame_curation.json` default-denies downloaded real frames.
  Ranges with observed hands, people, laptops, phones, or setup activity are
  excluded. The retained ranges are provisional because the current
  contact-sheet review sampled at 20-second intervals; fine-scrub every frame
  used for training or publication.
- `config/real_camera_profiles.json` provides separate coarse table, side, and
  wrist appearance targets. Those targets come from compressed 640x360 video,
  not a photometric calibration.

## Non-negotiable capture rules

1. Capture table and side at **1920×1080 RGB** and wrist at **1280×720 RGB**,
   matching each checked-in calibration. Only table currently drives motion.
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
The loader rejects a resolution different from the selected calibration. The
postprocess adds each selected camera's measured one-term radial distortion
(`k1`) to its pinhole render and applies replayable, camera-separated exposure,
white balance, gamma, noise, blur, vignette, and JPEG effects. These display-RGB
effects are robustness approximations, not a physical camera model. A separate
archive preview applies fixed per-camera gains and an I420 round trip. A global
neutral-luma curve was tested and rejected: on the matched seed-7 render it
moved every audited cube/container patch farther from the two-frame physical
targets. Revisit contrast fitting only with better camera-response evidence;
the current preview does not claim to reproduce a native camera response or
the downloaded AV1 bitstream.

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
  --real-video /path/to/frame-by-frame-reviewed-table-video.mp4 `
  --sim-dir path/to/run/raw_frames --sim-domain raw `
  --camera table --assume-precurated-input `
  --output-json path/to/raw-vs-real.json `
  --contact-sheet path/to/raw-vs-real.png
```

`--assume-precurated-input` is an explicit acknowledgement for a video that was
already reviewed frame by frame outside the checked-in archive manifest; the
flag does not perform a privacy scrub. When comparing a downloaded archive episode, use
`--curation-manifest config/hardware_frame_curation.json`, `--dataset-id`,
`--episode`, and the intended `--curation-split` instead. A missing or
nonmatching reviewed range is an error rather than permission to read the whole
video. Repeat the comparison with `--camera side` and `--camera wrist`; each
view uses a different neutral region and must be assessed independently.

The tool downsamples synthetic frames to the real recording instead of
inventing detail by upsampling the video. Its exposure and white-balance hints
deliberately exclude the moving robot corridor and saturated objects. Collect
a native, lossless 1920x1080 real table-camera still before fitting sensor
noise, precise blur, camera response, or small-object edge appearance.

An Isaac training run now defaults to `--training-cameras table,side,wrist` and
saves native raw and sensor-derived RGB for all three views plus camera-specific
640x360 archive-domain previews. Raw renders are never overwritten. The
optional `--wrist-hose-stress` flag adds only a seeded derived wrist occluder,
rendered as layered tapered strands with an alpha-blended shadow, dark rim,
body highlight, and slight defocus. It also stores the unblurred cable/shadow
geometry mask and an ignore mask dilated by at least the finite three-sigma
shadow-blur support plus one deployment pixel for I420 chroma spill; table and
side views are never hose-augmented.
Exclude the ignore mask from supervised pixel loss or treat hose views as
unlabeled robustness data. See the
[scripted demo guide](../demos/scripted_colour_bucket_sort/README.md) for the
complete command.

For camera/framing checks at measured robot poses, use the
[hardware-pose shadow replay](../demos/hardware_pose_replay/README.md). It sets
the simulated articulation from measured joint state, renders all three views,
and never sends recorded actions to hardware. It is a visual diagnostic, not a
controller or proof of timing equivalence.

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
    load_camera_calibration,
)

camera_id = "wrist"
calibration = load_camera_calibration(spec, calibration_entry=camera_id)
result = augment_sensor_rgb(
    raw_rgb,
    manifest,
    frame_index=frame_number,
    camera_calibration=calibration,
    camera_id=camera_id,
    expected_resolution=calibration["size"],
    apply_calibrated_radial_distortion=True,
)
derived_rgb = result.image_rgb
```

`raw_rgb` must be that camera's native `uint8` RGB render. Save it first. The
result retains native resolution and reports every applied sensor parameter.
Gaussian noise uses a recorded deterministic seed derived from episode,
camera, and frame; adjacent frames and different cameras do not share a field.
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

- independent cube and orange/yellow/light-blue destination hue, saturation,
  brightness, roughness, and table roughness;
- dome/key/fill/rim intensity, RGB tint, and light-source size (shadow softness);
- small camera pose and focal perturbations;
- per-camera exposure, white balance, gamma, noise, blur, vignette, distortion,
  and decoded-image compression approximations;
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
appearance requests as applied. Logical destination labels stay red/yellow/blue
while independently bound materials render orange/yellow/light-blue to match
the observed containers. The table material, lights, cube transforms/masses
and PhysX coefficients receive the same readback treatment.
Cube and destination nominals now use the first-pass paired patch audit in
`config/real_camera_colour_patches.json`. The manifest records the source
image hashes, manually reviewed ROIs, retained core-pixel counts, physical and
seed-7 simulated display-RGB medians, and the fitted authored diffuse values.
It also retains the previous authored diffuse inputs so both the two-frame
physical aggregation and channel-ratio fit can be reproduced and validated.
This is deliberately a reproducible rendering iteration rather than a claim of
physical albedo: the evidence is compressed 640x360 display RGB, the cubes
span only a few pixels, and the fit still confounds renderer lighting and tone
processing. Accept or replace the values only after a new Brev render repeats
the same patch measurement and a separate held-out rollout improves.
When the real printed tabletop artwork overlay is present, its texture remains
enabled and the collector changes/verifies that overlay shader's roughness too;
otherwise varying only the hidden material below it would not change pixels.

`effective_camera.json` contains the primary table focal/aperture and USD transform read from
the live camera prim, not just the requested config. It records both
`camera_to_world` and `camera_to_robot_base` explicitly (the axes align in this
twin, while the origins do not) and carries a SHA-256 digest. Every frame label
embeds that effective metadata and references both episode readback hashes. The
real-camera lens distortion remains a distinct derived-image transform; it is
not falsely described as an Isaac camera intrinsic. `camera_validation.json`
checks table, side, and wrist stage readback against named calibration entries,
and `geometry_validation.json` records asset hashes, measured dimensions, and
the known sharp-collider/rounded-visual mismatch.

The following work remains before calling the images a complete perception
training dataset:

1. extend the new seeded visual-only background distractors into deliberate
   touching/overlap, target occlusion, blocked approaches, varied target count,
   and arrival/motion sequences; current clutter is deliberately outside the
   pickup region, has no collider, and does not prove blocked-grasp handling;
2. add semantic/instance render passes, measured visibility/occlusion,
   instance masks and—only where the selected model needs them—depth/normals;
3. validate randomized layouts with complete swept-volume collision checks,
   not only separation, IK and physical execution gates;
4. model physical motion blur, rolling shutter, auto-exposure and the measured camera
   response instead of generic display-RGB approximations;
5. add a deployable camera-based post-place outcome check rather than using
   simulator-truth placement scores;
6. replace every provisional range using physical samples/measurements and
   validate held-out real-camera performance;
7. decouple capture/label writing from policy success (or label failed decision
   frames), because the current training run aborts before dataset frames when
   randomized appearance defeats the HSV policy and therefore undersamples the
   hardest conditions; and
8. record the clean project commit, Isaac/renderer build and settings, GPU, and
   driver in each episode manifest; current config/calibration/readback hashes
   do not fully identify the generator runtime; and
9. measure yaw accuracy at the real camera's native part scale. In the current
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
