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
# Prepare exact clean frames and prefilled provider predictions from a clean
# replay. frames.txt contains one strictly increasing source-frame index per
# line; every index must exist in the replay's scenes.jsonl.
uv run --project hardware/fanuc_cell --with av python -m \
  team_plugins.computer_vision.prepare_annotation_bundle \
  /external/path/capture.mp4 \
  --replay-dir /external/path/clean-replay \
  --frame-indices-file /external/path/frames.txt \
  --output-dir /external/path/annotation-bundle \
  --annotation-id pilot-layout-01 \
  --dataset-id hardware-pilot \
  --source-run-id rollout-05 \
  --layout-id static-layout-01 \
  --annotator-id teammate-name

uv run --project hardware/fanuc_cell python -m \
  team_plugins.computer_vision.evaluate_annotations \
  /external/path/annotation-bundle/annotation-manifest.reviewed.json \
  --evaluation-out /external/path/evaluation-result.v1.json \
  --diagnostics-out /external/path/diagnostics.v1.json

uv run --project hardware/fanuc_cell python scripts/summarize_evaluation.py \
  /external/path/evaluation-result.v1.json
```

The bundle contains lossless decoded-source PNGs, lossless native-calibration
annotation PNGs, and separate prediction previews. Annotate only the clean
`annotation/` images; `preview/` overlays are non-authoritative review aids and
are explicitly marked as not ground truth. The starter manifest already
contains the frozen provider predictions but deliberately has empty
`references` and `regions` and is not score-ready. Preserve it unchanged, copy
it to `annotation-manifest.reviewed.json`, add human references/regions there,
and set review provenance only after a second person checks the labels. Run one
bundle per physical layout so repeated frames stay grouped by `layout_id`.

The tool rejects a dirty checkout, stale/mismatched replay artifacts, a capture
or calibration hash mismatch, missing frame indices, nonempty output, and any
output path inside the repository. The replay report pins the complete
`scenes.jsonl` byte hash, and the bundle verifies that hash before trusting any
prediction. Its bundle report pins the bundle-tool bytes, current Git
commit/branch/dirty state, and all replay inputs. It also requires
each re-decoded/upscaled RGB array to match the per-frame hash recorded by the
replay, so a decoder/backend difference cannot silently change annotation
pixels. It never imports the cell client or sends a robot command. An old replay may need to be
regenerated after a provider, profile, replay tool, calibration, or
object-catalog change; do not bypass that provenance gate for reported results.

Keep private recordings, annotations containing personal data, generated
reports, and overlays outside Git. Commit only the schema, evaluator, tests,
and small synthetic fixtures.

## Archived pilot frame ledger

Use the following deterministic 47-timestamp set only as a `pilot` split. It is
not final held-out evidence because development has already inspected these
recordings. Score table-camera targets; use the row/frame-aligned side/wrist
images only for diagnostic visibility until their extrinsics and exposure
timing are revalidated. Row alignment at 10 Hz is not exposure synchronization.

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
