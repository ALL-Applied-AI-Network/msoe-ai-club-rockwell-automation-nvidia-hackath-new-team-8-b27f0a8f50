# Scripted cube-and-screw bucket-sort demo

This is a **separate** Isaac Sim demonstration. It leaves the existing
`scripted_colour_bucket_sort` demo, its shared cube configuration, and the
FANUC hardware adapter unchanged.

The robot first performs the familiar three cube placements—red, yellow, and
blue cubes into matching supplied bins—then physically grasps two supplied ISO
10642 M6 x 30 screw CAD models by their 6 mm shafts and places both screws in
the red bin. The screws and cubes remain dynamic rigid bodies; the demo never
attaches an object to the gripper.

The scripted robot motion remains a configured-pose simulation regression; it
does not use an AI estimate to command the robot. An optional, evaluation-only
camera path records RGB-D, instance masks, calibration, and meshes for NVIDIA
FoundationPose. Simulator truth is stored outside the inference capture so the
pose estimator cannot read it. This makes it possible to measure camera-model
error and confidence before using estimates for control.

## Prerequisite

Run the clean-clone Brev setup from the repository root:

```bash
bash scripts/setup_brev.sh
```

The generated `env.sh` exports the disposable Isaac twin, reference CAD bundle,
and external runs directory.

## Run

From the clean repository checkout in the Brev `vscode` container:

```bash
EXTERNAL_ROOT="${ROCKWELL_EXTERNAL_ROOT:-/workspace}"
RUNTIME_ROOT="${ROCKWELL_RUNTIME_ROOT:-$EXTERNAL_ROOT/rockwell-runtime}"
source "$RUNTIME_ROOT/$(git rev-parse HEAD)/env.sh"
cd "$ROCKWELL_TWIN_RUNTIME"
"$ISAAC/python.sh" \
  /workspace/rockwell-hackathon/demos/scripted_cube_screw_bucket_sort/cube_screw_bucket_demo.py \
  --run "$ROCKWELL_RUNS_ROOT/cube_screw_bucket_sort_01"
```

Use a new output directory for every run. Add `--physics-only` for a faster
contact/IK check without PNG rendering.

Add `--foundationpose-capture` to record both the initial pickup scene and the
final sorted scene:

```bash
"$ISAAC/python.sh" \
  /workspace/rockwell-hackathon/demos/scripted_cube_screw_bucket_sort/cube_screw_bucket_demo.py \
  --run "$ROCKWELL_RUNS_ROOT/cube_screw_foundationpose_01" \
  --random-spawn --seed 20261030 --foundationpose-capture
```

Each phase contains `capture/manifest.json`, raw RGB, depth in metres, one
instance mask per object, and metre-scale OBJ meshes. Its sibling
`ground_truth.json` contains the actual Isaac poses. Never mount that sibling
file into the inference container. See `demos/foundationpose_camera/README.md`
for inference and evaluation commands. The masks are Isaac-generated in this
first version; FoundationPose derives pose from camera RGB-D and CAD, while a
future hardware version should replace mask generation with a detector or SAM.

Add `--random-spawn` to place all five objects at a new, separated layout
inside the configured pickup region. Each object's orientation and destination
stay fixed, and the pickup plan uses the sampled pose. The generated seed,
spawn coordinates, initial simulator poses, final poses, and per-object pickup
and placement results are recorded in `execution_report.json`. Pass `--seed`
to reproduce a layout; if omitted, the run generates and records a seed:

```bash
"$ISAAC/python.sh" \
  /workspace/rockwell-hackathon/demos/scripted_cube_screw_bucket_sort/cube_screw_bucket_demo.py \
  --run "$ROCKWELL_RUNS_ROOT/cube_screw_bucket_sort_random_01" \
  --random-spawn --seed 20261029 --physics-only
```

To encode a rendered run:

```bash
"$HOST_PYTHON" encode_video.py \
  --frames "$ROCKWELL_RUNS_ROOT/cube_screw_bucket_sort_01/frames" \
  --out "$ROCKWELL_RUNS_ROOT/cube_screw_bucket_sort_01/demo.mp4" \
  --fps 30
```

`HOST_PYTHON` is exported by the generated `env.sh` and points at the
commit-specific disposable twin's host environment.

## Success contract

`execution_report.json` must report:

- `task_complete: true`;
- five verified pickups and five verified placements;
- three cube placements and two screw placements;
- both screws assigned to the red destination; and
- the exact supplied screw SHA-256, 1,910 triangles, and CAD bounds.

Screw placement is not accepted from its centre alone. The verifier transforms
every CAD vertex using the final rigid-body pose and requires the full screw to
be inside the 35 mm red-bin pocket, above its floor, and below its rim. This is
important because a 30 mm screw has only 2.5 mm nominal axial clearance at each
end. The configured lanes put the screws 8 mm apart, point their heads toward
opposite ends of the bin, and release them 16.5 mm above the floor so the held
CAD clears the already-sorted red cube. The jaws open only enough to release the
6 mm shank while inside the pocket and open fully after retreating above it.
After all five actions, the demo samples every object's pose again and repeats
the geometry and stability checks, so a later insertion cannot silently leave
an earlier success counted after it has been disturbed. Rendered runs save this
verified final-state tail so the video's ending agrees with the JSON report.

The model has no geometric threads, and the 7.5 g steel mass is CAD-volume based,
not a measurement of the event's physical screws. Passing this simulation does
not prove that the real gripper can pick screws or that a hardware trajectory is
collision-free.

Generated runs, frames, reports, and videos belong outside Git.
