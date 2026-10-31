# YOLOE visual-prompt segmentation experiment

Status: bounded offline experiment, **not** a runtime provider and **not**
hardware-ready perception.

## Question being tested

Can the pretrained `yoloe-26s-seg` visual-prompt model produce repeatable cube
or ISO 10642 M6x30 screw instance masks at the physical table camera's native
pixel scale, sufficiently better than the deterministic geometry control to
justify its dependency, latency, and license burden?

The experiment stops at image-space boxes/masks. It deliberately does not
invent robot-base position, orientation, pose quality, visibility, or tracking.
If masks become reliable, the next reviewed step is calibrated geometry:

```text
native RGB tile -> YOLOE instance mask -> full-frame mask
                -> calibrated table projection / catalog geometry
                -> symmetry-aware pose + measured uncertainty
                -> SceneObservation
```

## Why tiling is mandatory

The archived 640x360 videos make the small parts only a few pixels wide. They
are useful for qualitative false-positive/dropout review, not millimetre pose
validation. The intended experiment uses lossless native 1920x1080 table RGB
and crops native 640x640 overlapping tiles before inference, so the entire
frame is not downscaled to 640 pixels. YOLOE may still internally letterbox a
tile to `imgsz`; the report records that value and every tile transform.

Do not upscale compressed footage and call it native evidence.

## Reproducible environment

The isolated environment is under `yoloe_env/`. From the repository root:

```bash
uv sync --project team_plugins/computer_vision/yoloe_env --frozen
```

The tested direct dependency is `ultralytics==8.4.194`. The tested checkpoint
was `yoloe-26s-seg.pt`, 31,072,171 bytes, SHA-256:

```text
48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654
```

The model capability/API reference is the
[official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/).

Weights, images, annotations, and outputs stay outside Git. Copy
`yoloe_experiment_manifest.example.json` beside an external `artifacts/` and
`evidence/` directory, replace every zero hash and placeholder coordinate, and
keep the manifest with the result. The command rejects mismatched hashes,
wrong native image size, duplicate classes, ambiguous prompt images, malformed
annotations, or an unpinned Ultralytics version before loading the model.

Run:

```bash
uv run --project team_plugins/computer_vision/yoloe_env \
  python -m team_plugins.computer_vision.yoloe_visual_prompt_experiment \
  --manifest /external/yoloe-pilot/experiment.json \
  --output-dir /external/yoloe-pilot/output
```

The output directory contains:

- `report.json`: hashes, commit/branch, environment, prompt classes, every
  tile transform, per-tile latency, predictions, and optional reviewed-box
  metrics;
- `predictions.jsonl`: one independent record per frame; and
- lossless PNG overlays.

The runner never contacts the robot and never reads simulator truth.

Run the dependency-free harness tests either in the shared client environment
or the exact model environment:

```bash
uv run --project team_plugins/computer_vision/yoloe_env --frozen \
  python -m unittest team_plugins.computer_vision.tests.test_yoloe_experiment -v
```

The current branch passes 15 harness tests and the complete 258-test
profile-unset repository baseline.

## Annotation and metrics

The optional annotation is intentionally small and reviewable:

```json
{
  "schema_version": 1,
  "image_sha256": "...",
  "instances": [
    {
      "class_id": "screw_iso10642_m6x30",
      "bbox_xyxy": [100, 100, 130, 120]
    }
  ]
}
```

The current harness reports class-aware 2D box true positives, false positives,
dropouts, precision, recall, and matched IoU. This is only a mask-provider gate.
It does not satisfy the final pose evaluation, which still needs independently
measured base-frame part positions and symmetry-aware orientations.

Split by complete trial/layout sequence. Never place adjacent frames from one
hardware rollout into both prompt/tuning and held-out evaluation.

## Measured smoke result

The initial technical smoke used:

- Ultralytics `8.4.194`;
- `yoloe-26s-seg.pt` with the hash above;
- CPU inference at `imgsz=640`;
- one approximate visual-prompt box around a black screw-like part in archived
  compressed table frame 620; and
- archived compressed target frame 990.

Import, checkpoint load, and inference completed, but the target returned zero
boxes and no masks. This proves only API compatibility. It is negative first
evidence for one-shot detection at compressed resolution, not a model-wide
accuracy result.

The committed harness then repeated the same unannotated compressed case in its
lockfile-pinned CPU environment:

| Confidence threshold | Output | Tile inference latency | Interpretation |
| --- | --- | ---: | --- |
| 0.25 | 0 instances | 17.14 s | Reproduces the original dropout at a normal threshold. |
| 0.05 | 1 instance at confidence 0.0606 | 24.64--45.54 s across two runs | Threshold-sensitive candidate; without annotation it is not a verified true positive. |

The low-threshold box was approximately
`[188.32, 296.35, 213.10, 307.73]` in the 640x360 target. Report SHA-256 values
are respectively
`e8ceb48f27d26d4989fe9445670419f707c6083419f6b19f37e62d1cc0b218ab`
and, for the final-code low-threshold rerun,
`b7fcd0011d1ddd764dc99d3247b72f618cebf0290f23f4c2a46776466e53f256`;
the reports and images remain external under
`HardwareResults/VisionExperiments/yoloe-harness/` and are not committed.
Those archived reports predate the cross-OS Git provenance fix and contain null
commit/branch fields. The measured inference outputs remain bounded exploratory
evidence, but the reports are not clean-commit-pinned artifacts. The current
runner resolves Windows-created worktree pointers under WSL, fails closed when
Git provenance cannot be read, and records start/end branch, commit, and dirty
state. Rerun the exact thresholds before using a YOLOE report as reproducible
handoff evidence.

Creating the isolated environment installed 60 packages and occupied about
4.9 GB in this WSL-on-Windows worktree. The first sync took 5 minutes 35
seconds after package preparation. CPU latency is clearly unsuitable for the
current synchronous robot loop; any usefulness test must use a GPU and include
tiling, service transfer, calibrated pose conversion, and serialization in the
end-to-end timing.

## Stop/continue gates

Continue toward a mask-provider adapter only if a frozen, reviewed native set
shows all of the following:

1. repeatable isolated-cube and isolated-screw detection across stationary
   frames;
2. useful abstention/dropout behavior under cable, hand, robot, container,
   lighting, boundary, and partial-occlusion negatives;
3. materially better held-out accepted-mask accuracy than the geometry control;
4. end-to-end latency compatible with the scene-age limit after tiling,
   calibrated pose conversion, serialization, and service overhead; and
5. an acceptable deployment license.

Stop if native tiling still drops isolated objects, prompt choice dominates the
result, lookalikes become executable instances, or latency/license cost is not
justified by measured improvement.

## Licensing gate

Ultralytics publishes AGPL-3.0 and Enterprise options in its
[official license guidance](https://www.ultralytics.com/license). This
repository does not make a legal determination. Every manifest must explicitly
use either:

- `evaluation-only-agpl-review`; or
- `enterprise-authorized`.

Do not present this experiment as a proprietary factory deployment until the
team/organizer resolves which terms apply. The model and dependencies are kept
out of the lightweight hardware client regardless; any future runtime should
be a separately pinned bounded-time service with health, timeout, output
validation, and clean-shutdown tests.

## Current blockers

- No reviewed native 1920x1080 cube/screw annotation set was available for this
  branch.
- Archived footage has no object masks, poses, or independently measured
  base-frame truth.
- The screw finish/count/identity in archived video is not annotated.
- A good mask still does not resolve screw axial roll or yield a validated 3D
  head-face origin.
- The shared executor remains cube-only; a detected screw is not executable.

Ask the team AI to explain visual prompting, tiling, IoU, the license gate, or
the next smallest experiment when any of these terms are unclear.
