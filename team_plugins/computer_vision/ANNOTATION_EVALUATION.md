# Annotation and evaluation workflow

`annotation-manifest.v1.schema.json` is the reviewable source record for cube
and screw evaluation. `evaluate_annotations.py` validates the stricter runtime
semantics and emits:

- a `docs/schemas/evaluation-result.v1.schema.json`-compatible report; and
- a diagnostics sidecar containing identity switches and repeated-pose
  residual jitter.

The tool uses only the Python standard library. It never commands the robot,
loads model weights, or infers missing truth.

## What must be annotated

Each manifest is for one camera/capture revision and pins the capture hash,
Git commit, profile, provider, calibration, catalog, split, and reviewer. Each
frame records its layout, image hash, timestamp source, optional latency,
reference instances, provider predictions, and ignore/unknown regions.

`validation` and `test` manifests must be `reviewed` or `adjudicated`; the
evaluator rejects unreviewed evidence. Pilot/train manifests may remain
unreviewed while labels are being debugged. Any reference truth or prediction
claiming a robot-base pose requires at least one calibration ID. An ID records
which calibration was used; preserve the corresponding calibration artifact
and hash outside this manifest.

For every cube or screw reference, annotate:

- stable instance ID within one layout;
- class and cube color/sort label;
- visible polygon, bounding box, and center;
- screw head/tip endpoints plus whether their direction is observable;
- truncation, occlusion, touching group, and scoring status; and
- optional independently measured robot-base truth.

Do not obtain robot-base “truth” from the camera being evaluated. A registered
fixture/grid, operator-taught base points, measured part pose, and a stated
truth-uncertainty bound are prerequisites for `pose_truth`. Leave it `null` for
compressed recordings or ordinary manual 2D labels.

## Matching and metrics

A prediction matches a reference only when:

1. class and sort label agree;
2. its image center lies in the reference polygon; and
3. a one-to-one assignment has not already used either item.

Assignment first maximizes the number of valid matches over the whole frame,
then minimizes their summed center distance. Stable instance/prediction IDs
make exact-cost ties deterministic. This prevents a locally nearest greedy
pair from consuming the only prediction available to another reference.

This avoids inventing a hard IoU pass threshold for targets that may occupy
only a few compressed pixels. Mask/box quality should still be reviewed
separately. A wrong cube color becomes both a dropout for the reference and a
false positive for the prediction. Predictions centered in explicitly marked
`ignore`/`unknown` regions are excluded and counted in diagnostics, except
when their center is also inside a scored reference mask; a broad ignore region
cannot erase a scored object.

Predictions whose class is not in the supported catalog (including
`unknown_obstacle`) are **abstentions**, not executable-target false positives.
They are excluded from the shared per-class score and reported by class in the
diagnostics sidecar. A supported reference covered only by an abstention still
counts as a dropout.

Cube rotation error is the full 3D rotation error minimized over four
90-degree-equivalent upright orientations, so tilt is not hidden. Screw error
uses the catalog local `-Z` head-to-tip axis and respects `directed` versus
`undirected` annotations. Axial roll is never fabricated. Position error is
emitted only when both independently measured truth and a predicted base pose
exist.

Screw predictions also carry 2D `head_uv`, `tip_uv`, and
`axis_observability`. The diagnostics sidecar reports directed image-axis error
only when both label and prediction identify head-to-tip polarity. It reports
undirected shaft-axis error when the reference says polarity is not observable;
otherwise it records why the image-axis sample was unavailable. Symmetry IDs
are class-specific and validated against the catalog convention rather than
accepted as free text; image-axis observability remains a separate field.

Diagnostics stratify matched/false-positive/ignored/abstaining predictions by
fixed confidence ranges `[0,.5)`, `[.5,.8)`, and `[.8,1]`. These are reporting
bins, **not** runtime confidence thresholds. Reference true positives and
dropouts are also separated by occlusion category, truncation, and whether the
object touches another annotated instance.

Identity switches are computed per `(layout_id, instance_id)`. Position jitter
uses prediction-minus-truth residuals and therefore requires at least two
matched pose-truth samples. Adjacent frames are repeated measurements, not
independent trials; keep train/validation/test separation at layout/run level.
Timestamp monotonicity is checked across consecutive known timestamps even if
unknown timestamps occur between them.

## Run

From the repository root in WSL:

```bash
uv run --project hardware/fanuc_cell python -m \
  team_plugins.computer_vision.evaluate_annotations \
  /external/path/annotations.json \
  --evaluation-out /external/path/evaluation-result.v1.json \
  --diagnostics-out /external/path/diagnostics.v1.json

uv run --project hardware/fanuc_cell python scripts/summarize_evaluation.py \
  /external/path/evaluation-result.v1.json
```

Keep private recordings, annotations containing personal data, generated
reports, and overlays outside Git. Commit only the schema, evaluator, tests,
and small synthetic fixtures.

## Archived pilot frame ledger

Use the following deterministic 47-timestamp set only as a `pilot` split. It is
not final held-out evidence because development has already inspected these
recordings. Score table-camera targets; use synchronized side/wrist images only
for diagnostic visibility until their extrinsics and timing are revalidated.

- `r20261029-88a87b`, episode 0: `43, 348, 653, 958, 1263, 1568,
  1874, 2179, 2484, 2789, 3094, 3399`;
- `r20261027-4a4b7d`, episode 10: `0, 178, 357, 535, 713, 892, 1070,
  1248, 1427, 1605, 1783, 1961, 2140, 2318, 2496, 2675, 2853, 3031,
  3210, 3388`;
- `r20261027-4a4b7d`, episode 9 hose stress: `0, 71, 142, 213, 284,
  356, 427, 498, 569, 640`; and
- `r20261029-88a87b`, episode 0 hand/unsupported negatives: `3400, 3500,
  3600, 3700, 3800`.

Previously tuned frames such as 620 and 990 from the early `table.mp4` replay
may remain documented failure examples but must not enter a held-out claim. No
uncontaminated labeled screw set exists yet; collect it at native resolution.

## Evidence limits

Compressed RGB annotations can establish detection/count errors, image-axis
error, abstention behavior, identity switches, and latency. They cannot
establish robot-base millimetre accuracy, camera-to-robot timing, grasp
clearance, or physical success. A small hardware set with independent
base-frame truth is still required before a new provider controls motion.

The shared `evaluation-result.v1` schema has no fields for review provenance,
evaluator provenance, abstentions, 2D image-axis errors, confidence strata, or
occlusion/truncation/touching strata. To stay schema-compatible, those fields
are kept in the hash-linked diagnostics sidecar. The shared output contains
only supported-class counts, base-pose samples when justified, latency, and
placeholder `not_executed` episodes. Their timing contains `perception_s` only;
it deliberately does not mislabel summed inference time as total episode time.
