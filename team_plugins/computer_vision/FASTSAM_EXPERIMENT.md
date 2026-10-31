# FastSAM mask-proposal experiment

## Decision

FastSAM is **plausible enough for a bounded offline comparison**, but it is not
an integrated perception provider and it is not currently a replacement for
the color/geometry control.  On one unannotated archived camera frame it
produced a compact proposal over the region previously identified as a
screw-like part.  It also produced many nested and scene-sized proposals.  That
is evidence that it can propose tiny regions, not evidence that it detected a
screw correctly.

This branch therefore provides an offline, hash-pinned evaluator only.  It does
not change default simulation or hardware behavior.

## What FastSAM can and cannot supply

Ultralytics documents FastSAM as a class-agnostic, single-class instance
segmenter: its first stage proposes all instances, then prompts select among
those masks.  It does not assign the project catalog classes `cube_10mm` or
`screw_iso10642_m6x30`.  See the [official FastSAM
documentation](https://docs.ultralytics.com/models/fast-sam/).

Useful output from this experiment is limited to image-space polygons, boxes,
centroids, mask areas, and model confidences.  A deployable vision plugin would
still need all of the following:

1. supported-object classification, while retaining unsupported masks as
   possible obstacles;
2. calibrated camera-to-robot-base conversion;
3. depth or validated table-plane geometry;
4. symmetry-aware orientation estimation;
5. multi-camera association and capture-time validation;
6. measured uncertainty and an abstention rule.

The confidence returned by the model is not treated as millimetre pose accuracy
or grasp-success probability.

## Initial smoke evidence

The prerequisite gate ran before this branch was created.  The test used
`FastSAM-s.pt`, `ultralytics==8.4.194`, CPU inference, and the archived
640x360 JPEG `source-000620.jpg`.  The image has no trustworthy ground-truth
mask, so the following are observations only:

| Setting | Observed output | Measured model-call wall time |
| --- | --- | ---: |
| confidence 0.25 | 66 boxes and 66 masks | about 6.96 s |
| confidence 0.05 | 214 boxes and 214 masks | about 3.50 s |

The two timings came from separate invocations with different cache state; they
are not a benchmark.  The subsequently locked CPU environment
(`torch==2.15.0`, `torchvision==0.29.2`) measured first calls between 17.34 and
82.49 seconds across fresh processes whose OS/cache state was not controlled,
and a repeated call in the same process at 3.36 seconds.  This wide startup
variation must be measured on the
intended GPU before considering runtime use.  At confidence 0.05, one proposal
at the inspected image point near `(206, 300)` had confidence `0.512`, box
`[200, 297, 212, 303]`, and area 69 pixels.  Nine proposals covered that point,
including several much larger masks.  These results pass only the narrow
"can it produce a tiny candidate mask?" gate.

### Practical comparison

| Approach | Archived-frame observation | Practical conclusion |
| --- | --- | --- |
| Existing color/geometry control | Already produces task-specific cube observations using known color and geometry; no same-harness latency measurement was made here. | Keep as the control and fallback. It is simpler and already connected to calibrated project geometry, but does not generalize automatically to screws, arbitrary materials, or severe color shifts. |
| Concurrent YOLOE smoke | A sibling experiment reported zero detections at confidence 0.25 and one unverified `0.0606` detection at 0.05, with roughly 17.1-45.5 s CPU calls. The frame was unannotated. | No evidence yet that this checkpoint finds the tiny screw region; do not reject it until its own reproducible branch evaluation is reviewed. |
| FastSAM-s | It proposed a compact region at the inspected point, but produced 66-214 total proposals and has no semantic class. Locked-environment CPU first calls measured 17.34-82.49 s across uncontrolled cache states and 3.36 s repeated. | Potential mask/unknown-obstacle proposal source. It needs aggressive measured filtering and a separate classifier/pose stage; it is not ready for the control loop. |

The comparison is deliberately asymmetric: it reports only what each experiment
actually measured.  It does not claim model accuracy from unannotated frames.

## Licensing gate

This harness imports `ultralytics.FastSAM`.  Ultralytics states that its code
and models are available under AGPL-3.0, with an Enterprise license for closed
commercial integration, including embedded robotics.  Review the [official
Ultralytics licensing page](https://www.ultralytics.com/license) before any
deployment.

The upstream FastSAM repository is itself inconsistent: its README describes
the model as Apache-2.0 while its checked-in [LICENSE
file](https://github.com/CASIA-LMC-Lab/FastSAM/blob/main/LICENSE) is AGPL-3.0.
This project therefore treats FastSAM as AGPL-constrained unless the organizers
or counsel establish another applicable license.  The example manifest accepts
only `evaluation-only-agpl-review` or `enterprise-authorized`; this field is a
recorded assertion, not a legal enforcement mechanism.

## Reproducible setup

Use Python 3.10-3.12 and keep both weights and evidence outside the repository:

```bash
uv sync --project team_plugins/computer_vision/fastsam_env --locked
mkdir -p /external/model-cache
cd /external/model-cache
uv run --project /path/to/repo/team_plugins/computer_vision/fastsam_env \
  python -c "from ultralytics import FastSAM; FastSAM('FastSAM-s.pt')"
sha256sum FastSAM-s.pt
```

The evaluated checkpoint had SHA-256
`c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844`.
Do not silently reuse that hash for a newly downloaded file; verify the bytes.

The checked-in lock currently resolves CUDA 13.2 runtime packages as PyTorch
dependencies even when the manifest selects CPU.  GPU compatibility with the
current Brev driver has **not** been tested.  Keep the example on CPU until
`torch.cuda.is_available()` and one pinned GPU inference smoke both pass; then
record the driver, CUDA, GPU, and measured latency in the external report.

Copy `fastsam_manifest.example.json` outside the repository and replace its
placeholder paths, image hash, image size, ROI, device, and annotation path.
For an unannotated smoke set `annotation_path` to `null`.  For evaluation, copy
`fastsam_annotation.example.json` outside the repository and annotate each
supported instance with a full-image `bbox_xyxy`.  Keep the annotations private
if the source recording is private.

Run from the repository root:

```bash
uv run --project team_plugins/computer_vision/fastsam_env --locked \
  python -m team_plugins.computer_vision.run_fastsam_experiment \
  /external/manifests/fastsam-pilot.json \
  --output-dir /external/results/fastsam-pilot
```

The runner requires a clean committed worktree by default.  `--allow-dirty` is
for development only and is recorded in the report.  Inputs and weights are
SHA-256 pinned.  Results include a lossless overlay, proposal JSON, per-tile
call latency, hashes, git/code provenance, and optional annotation coverage.

Native-resolution tiles prevent the 6 mm screw shaft from being unnecessarily
downscaled with the entire table frame.  Overlapping tiles can create duplicate
masks, so the harness performs cross-tile box-IoU suppression.  Same-tile masks
remain because nested proposals are meaningful evidence of over-segmentation.

## Interpreting evaluation output

`minimum_reference_bbox_iou` controls the exploratory coverage threshold.  A
nonzero overlap is not enough: a scene-sized mask would otherwise appear to
cover every tiny part.  The example value `0.1` is a review starting point, not
a physical acceptance criterion.  Replace box annotations with pixel masks if
FastSAM remains competitive, then score mask IoU and boundary quality.

Unmatched class-agnostic proposals are reported as `unmatched_proposals`, not
"false positives."  They may be hands, cables, bins, unsupported objects,
background fragments, or duplicate/nested masks.  Before integration, annotate
enough of the scene to separate useful unknown-obstacle masks from harmful
over-segmentation.

Advance this candidate only if native 1920x1080 archived evidence shows all of
the following relative to the geometry control and the best semantic model:

- supported objects are not dropped under the tested lighting and cable
  occlusion cases;
- proposal count can be filtered without removing cubes or screws;
- latency fits the measured observe/replan interval on the intended GPU;
- downstream classification and calibrated pose estimation improve end-to-end
  results enough to justify the added dependency and license burden.

Stop if tiny screws consistently disappear, giant/nested masks cannot be
filtered without losing targets, or the combined mask/classification/geometry
pipeline is slower or less reliable than the simpler control.

## Tests

The lightweight tests do not import or download Ultralytics:

```bash
uv run --project hardware/fanuc_cell \
  python -m unittest team_plugins.computer_vision.tests.test_fastsam_experiment
```

They cover native tiling, full-frame coordinate restoration, disconnected-mask
handling, cross-tile de-duplication, maximum-cardinality reference matching,
strict IoU coverage, hash pinning, external-weight enforcement, and manifest
validation.
