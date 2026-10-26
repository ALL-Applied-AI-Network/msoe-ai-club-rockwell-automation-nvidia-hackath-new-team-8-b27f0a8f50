# Scripted colour bucket-sort demo

This demo drives the supplied FANUC/SMC Isaac Sim twin through three physical
pick-and-place sequences: red, yellow, and blue cubes are placed into matching
bins made from the supplied `manipbench-bin-50x50x55.stl` model.

The **default mode** is a scripted simulation smoke test, not the final sorting
system. It uses configured simulator poses and simulator ground truth; it does
not use camera perception, Cosmos, a learned selector, or a real-robot control
interface. Its purpose is to preserve the known-good scene, inverse-kinematics,
contact-grasping, placement, verification, and rendering regression path. The
explicit `--camera-perception` and `--training-data` modes described below add
camera-derived decisions without changing that default; neither mode uses
Cosmos.

The simulator and FANUC hardware adapter share
`config/colour_bucket_sort.json` for the colour order, cube size and starting
poses, bin centres and dimensions, clearances, and frame assumptions. The
active choice is a **10 mm cube** and **UTOOL1/TCP at the fingertips**. The live
event site has separately described the TCP as 10 mm beyond the fingertips;
that conflict is unresolved and must be checked with a supervised physical
hover/contact test before any real descent. Every run prints the assumption and
records it, the config hash, and the shared semantic phase sequence in
`execution_report.json`.

The semantic pre-descent revalidation gate is also present here. Default mode
checks simulator ground truth and says so in the report; the opt-in camera
modes satisfy the same gate with a fresh raw-camera observation. The hardware
adapter must likewise use a fresh camera observation. Simulation truth must
never be used to make a real-robot decision.

## Prerequisites

Run the repository's verified Brev setup first:

```bash
cd /workspace/rockwell-hackathon
bash scripts/setup_brev.sh
```

Setup creates a commit-specific runtime outside Git. The generated `env.sh`
exports the Isaac path, runtime twin, reference assets, and external runs root.

## Run

From the clean repository checkout inside the Brev `vscode` container:

```bash
cd /workspace/rockwell-hackathon
EXTERNAL_ROOT="${ROCKWELL_EXTERNAL_ROOT:-/workspace}"
RUNTIME_ROOT="${ROCKWELL_RUNTIME_ROOT:-$EXTERNAL_ROOT/rockwell-runtime}"
source "$RUNTIME_ROOT/$(git rev-parse HEAD)/env.sh"
cd "$ROCKWELL_TWIN_RUNTIME"
"$ISAAC/python.sh" \
  /workspace/rockwell-hackathon/demos/scripted_colour_bucket_sort/colour_bucket_demo.py \
  --run "$ROCKWELL_RUNS_ROOT/scripted_colour_bucket_sort_01"
```

The script locates the repository from its own absolute path, so changing into
the runtime twin does not hide `sorting_core`. If the script itself is copied
outside the checkout, set `ROCKWELL_REPO_ROOT` to the clean repository root.

The `--run` directory must be new. Use `--physics-only` to skip PNG rendering
while retaining physics and placement verification. A rendered run writes PNG
frames plus `execution_events.json` and `execution_report.json`; encode frames
to video outside the repository. Never commit generated runs, frames, reports,
videos, environments, weights, caches, or credentials.

Success requires `execution_report.json` to contain `task_complete: true`, three
verified pickups, and three verified placements. A prior development run of
this exact script completed 3/3 pickups and 3/3 placements on the verified
runtime for commit `5bca8bd`; each new committed runtime still needs its own
validation.

## Camera parity and training-data modes

Do not use the default presentation run's `frames/` directory as training
data: it is 1280×720, contains burned-in text, uses fixed studio appearance,
and is driven by configured simulator poses.  That known-good default remains
the physics/rendering regression path.

Use raw native-resolution RGB for runtime selection and fresh pre-descent
revalidation without Cosmos or simulator-truth decisions:

```bash
"$ISAAC/python.sh" \
  /workspace/rockwell-hackathon/demos/scripted_colour_bucket_sort/colour_bucket_demo.py \
  --run "$ROCKWELL_RUNS_ROOT/camera_bucket_sort_01" \
  --camera-perception
```

Both camera modes retain the exact unannotated RGB observations used for the
initial plan and each pre-descent revalidation under `decision_frames/`. The
execution report links those files, so a failed or questionable decision can
be reproduced from pixels instead of inferred from presentation video.
The pre-descent gate checks count, confidence, XY displacement, and
square-equivalent yaw for every remaining target, rejects a completed colour
that reappears, and rebinds post-gate pick motion to the refreshed current
target rather than descending on the original pose.
Camera-based post-place verification is not implemented in this Isaac demo.
Simulator truth scores pickup/placement outcomes for evaluation, but in camera
and training modes it cannot stop or alter later actions or determine process
success. `runtime_sequence_complete` reports whether the planned sequence ran;
truth-derived `task_complete` remains an evaluator score. This keeps truth out
of runtime control while making the parity gap with the hardware policy
explicit; do not interpret a truth score as deployable outcome sensing.
The raw detector retains cube-like colour components across the calibrated
image. The camera adapter promotes only components inside the configured
base-frame pickup region to pick targets. The destination bins are outside that
provisional region, but moving robot, hose, and gripper colours can project
inside it because the image is back-projected onto the cube-top plane. The
physical cell must validate the bounds and boundary uncertainty before
operation. This is not an obstacle map or a guarantee that the arm's swept
volume is clear.
Known-cube silhouette gates also reject sparse or elongated colour patches
from the robot, hoses, lettering, and gripper before target counting. Their
fill/aspect thresholds are shared with the hardware adapter, preserve
fail-closed ambiguity handling for shapes that pass the gates, and are only
provisional until checked against native 1920x1080 images from the real cell.
They can also reject a partially occluded or damaged real cube, so a missing
detection is not by itself proof that a placed or dropped part is absent.

Add seeded, provisional domain randomization and synchronized training-data
artifacts with:

```bash
"$ISAAC/python.sh" \
  /workspace/rockwell-hackathon/demos/scripted_colour_bucket_sort/colour_bucket_demo.py \
  --run "$ROCKWELL_RUNS_ROOT/training_bucket_sort_seed_7" \
  --training-data --seed 7 --dataset-stride 15
```

Training mode implies camera perception and native 1920×1080 rendering. It
keeps presentation overlays under `frames/`, lossless unannotated renderer
images under `raw_frames/`, derived distorted/sensor-augmented images under
`sensor_frames/`, and synchronized JSON under `frame_labels/`. It also writes
requested and finalized randomization manifests, hashed USD/PhysX evidence in
`isaac_randomization_readback.json`, and hashed live-camera intrinsics plus
world/base-frame extrinsics in `effective_camera.json`. Each frame label embeds
the effective camera readback and both evidence hashes. A run fails rather than
claiming a sampled parameter that was not actually applied and read back. Each
label keeps six measured arm-joint angles separate from measured gripper
opening/closure, derives the measured TCP from articulation readback through
FK, and records the last commanded state separately.

The [synthetic-data contract](../../docs/SYNTHETIC_DATA.md) explains the
provisional ranges and remaining gaps. In particular, the available 2D boxes
are explicitly **amodal, image-clipped projections of the simulator-truth 3D
cuboids**. They are not visible-pixel boxes and do not measure occlusion. The
pipeline distorts the eight projected cuboid vertices before computing the
derived-image box. Timestamped 3D poses are present, but visibility fractions,
instance masks, depth, normals, motion blur, rolling shutter, and a measured
real-camera response are not yet available.
