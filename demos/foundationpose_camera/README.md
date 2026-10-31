# FoundationPose RGB-D evaluation path

This directory contains the camera-only pose-estimation and isolated evaluation
path for the cube-and-screw task. It uses NVIDIA's GPU-resident FoundationPose
inference library in model-based RGB-D mode. Inference may read only:

- an RGB image;
- a depth image in metres;
- camera intrinsics and a calibrated camera-to-robot-base transform;
- one mask and CAD mesh per object.

The estimator does **not** accept a spawn seed, configured object coordinates,
or an Isaac rigid-body pose.  Isaac truth is written to a separate file and is
opened only by `evaluate_pose_estimates.py` after inference completes.

## Capture contract

Each capture directory contains:

```text
capture/
  manifest.json
  rgb.png
  depth_m.npy
  masks/<object-id>.png
  meshes/...
ground_truth.json          # outside capture/; never mounted for inference
```

`manifest.json` has schema version 1, a 3x3 intrinsic matrix, an explicit
column-vector `camera_to_base_column_matrix`, and an object list:

```json
{
  "schema_version": 1,
  "capture_id": "trial-001-initial",
  "rgb_file": "rgb.png",
  "depth_file": "depth_m.npy",
  "camera": {
    "intrinsics": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
    "camera_to_base_column_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  },
  "objects": [
    {
      "object_id": "screw-red-01",
      "class_id": "screw",
      "sort_label": "red",
      "mask_file": "masks/screw-red-01.png",
      "mask_source": "isaac_instance_segmentation",
      "mesh_file": "meshes/screw.obj",
      "mesh_units": "m"
    }
  ]
}
```

The first deadline-oriented version may use Isaac instance segmentation for the
mask.  That means the *pose* is camera/AI-derived, but the mask is synthetic.
For hardware, replace the mask source with SAM 2, a trained detector, or a
manual region without changing the FoundationPose or evaluator contracts.

## Run inference

Run this inside the official NVIDIA FoundationPose inference container:

```bash
python demos/foundationpose_camera/run_foundationpose.py \
  --capture /workspace/runs/trial-001/capture \
  --output /workspace/runs/trial-001/estimates.json \
  --debug-dir /workspace/runs/trial-001/foundationpose-debug \
  --engine-cache-dir /workspace/foundationpose-engines
```

FoundationPose ScoreNet emits an unbounded ranking score. The output preserves
it as `foundationpose_raw_score` and reports a monotonic arctangent-normalized
`confidence` in `[0, 1]` for dashboards. That normalized number is **not** a
probability. Calibrate the raw score against accumulated simulator or hardware
pose errors before using confidence as an automatic rejection threshold.

## Evaluate without truth leakage

Run the evaluator only after inference has written `estimates.json`:

```bash
python demos/foundationpose_camera/evaluate_pose_estimates.py \
  --estimates /workspace/runs/trial-001/estimates.json \
  --truth /workspace/runs/trial-001/ground_truth.json \
  --output-json /workspace/runs/trial-001/pose_errors.json \
  --output-csv /workspace/runs/trial-001/pose_errors.csv
```

The result includes estimated and true poses, position error in millimetres,
full 3D rotation error in degrees, optional task-axis error, confidence, missing
objects, and aggregate mean/RMSE metrics.

## Hardware path

For an Intel RealSense camera, capture an RGB image, depth aligned to RGB in
metres, and the color-camera intrinsic matrix on Windows or Linux:

```bash
python demos/foundationpose_camera/capture_realsense_rgbd.py \
  --output hardware_capture
```

Install `pyrealsense2`, NumPy, and OpenCV in that camera environment. Create a
hand-eye-calibrated 4x4 `camera_to_base.json` matrix and an `objects.json` list
whose entries provide `object_id`, `class_id`, `mask_file`, `mesh_file`, and
`mesh_units`. Masks can come from a detector/SAM, or be drawn manually during
bring-up. Package and validate those files without exposing any truth pose:

```bash
python demos/foundationpose_camera/prepare_hardware_capture.py \
  --rgbd hardware_rgbd \
  --camera-to-base camera_to_base.json \
  --objects objects.json \
  --capture-id hardware-001 \
  --output hardware_capture
```

On the Brev host (or another Linux NVIDIA GPU with the official library built),
run the reusable inference command:

```bash
bash demos/foundationpose_camera/run_brev_foundationpose.sh \
  /absolute/path/to/hardware_capture \
  /absolute/path/to/hardware_output
```

For an 8 GB laptop GPU, begin conservatively with fewer registration
hypotheses and a matching micro-batch, then increase only after measuring VRAM:

```bash
FP_N_HYPOTHESES=42 FP_BATCH_SIZE=14 \
  bash demos/foundationpose_camera/run_brev_foundationpose.sh \
  /absolute/path/to/hardware_capture \
  /absolute/path/to/hardware_output
```

The result is `hardware_output/estimates.json` in robot-base coordinates. The
camera-to-base calibration is what makes the same output directly consumable by
a robot planner; validate it with stationary fixtures and conservative motion
limits before commanding physical hardware.
