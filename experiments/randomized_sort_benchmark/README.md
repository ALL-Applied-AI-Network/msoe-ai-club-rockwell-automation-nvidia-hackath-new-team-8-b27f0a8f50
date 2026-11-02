# Randomized realistic four-combination benchmark

This experiment evaluates the same frozen Isaac scenes with a 2×2 design:

| Cell | Vision | Grasp selection |
|---|---|---|
| DD | deterministic colour/silhouette geometry | deterministic geometric ranking |
| YD | YOLO11n box corroboration + the same calibrated geometry | deterministic geometric ranking |
| DA | deterministic colour/silhouette geometry | guarded pretrained-AI ranking |
| YA | YOLO11n box corroboration + calibrated geometry | guarded pretrained-AI ranking |

DD is the baseline. YOLO does **not** manufacture a 3D pose: for cubes it must
agree one-to-one with the camera-derived geometry result. The reviewed
checkpoint has no held-out screw evidence, so screws use the identical
deterministic silhouette/geometry path in D and Y cells. The AI ranker can choose
only an existing candidate that already passed deterministic reach, opening,
evidence, and local finger-envelope checks. Cosmos confidence is a ranking
signal, not a calibrated probability of retaining the object through release.

The benchmark is completely separate from the physical policy and the known-
good cube demo. It does not alter either path.

## Experimental unit

One condition contains 1–10 nonpenetrating objects sampled in the pickup
region:

- 10 mm red, yellow, or blue cubes;
- the exact supplied ISO 10642 M6×30 screw CAD.

Cubes map to the same logical colour destination. Screws and red cubes map to
red. Object count, type, position, and yaw are seeded. Appearance, physics,
and sensor seeds are independent and fixed before a policy cell is chosen.
The same manifest hash must appear in every matched cell.

Realistic mode varies the measured broad-fit tabletop, cube/bin materials,
roughness, camera pose/focal length, four light intensities, light tint/source
size, object mass/friction/restitution, exposure, white balance, gamma, noise,
blur, vignetting, and compression. The decision image—not only a saved dataset
image—receives the sensor effects. Radial distortion is intentionally omitted
from the decision image until perception has an explicit undistortion stage;
otherwise the pinhole backprojection would be knowingly inconsistent.

Simulator transforms are evaluator-only. Runtime selection, revalidation,
candidate generation, and motion use camera-derived `SceneObservation` data.

## Pre-registered sample size

The default suite has four conditions for every object count 1–10:

- 40 matched conditions;
- 160 end-to-end runs;
- 220 reference objects per cell;
- 880 object opportunities across four cells.

Start with counts 1, 3, 6, and 10 as a 16-run smoke. Do not start the full
batch until all four smoke cells emit complete reports with the same condition
hashes.

## Prepare a suite

Run from the repository root with the host Python environment:

```bash
python -m experiments.randomized_sort_benchmark.prepare_suite \
  --out /workspace/randomized-sort-suite \
  --base-seed 20261101 \
  --trials-per-object-count 4
```

For a first smoke:

```bash
python -m experiments.randomized_sort_benchmark.prepare_suite \
  --out /workspace/randomized-sort-smoke \
  --base-seed 20261101 \
  --trials-per-object-count 1 \
  --counts 1,3,6,10
```

## Run one Brev shard

First run `scripts/setup_brev.sh` for the exact clean commit, then source the
generated `env.sh`. Keep weights and all run data outside Git.

```bash
cd /workspace/rockwell-hackathon
source "$ROCKWELL_RUNTIME_ROOT/$(git rev-parse HEAD)/env.sh"

"$HOST_PYTHON" -m experiments.randomized_sort_benchmark.run_shard \
  --suite /workspace/randomized-sort-suite/suite_index.json \
  --runs /workspace/randomized-sort-runs \
  --cell DD
```

YOLO cells also need the external, SHA-pinned seed-55 `best.pt` via
`--yolo-endpoint` (recommended) or `--yolo-checkpoint`. Isaac Sim 6.0.1's
pinned Python does not contain PyTorch/Ultralytics, so do not inject another
environment's packages into it. Start the persistent localhost sidecar with a
Python environment that has the pinned YOLO runtime:

```bash
export YOLO_PYTHON=/workspace/rockwell-hackathon/team_plugins/computer_vision/fastsam_env/.venv/bin/python
export YOLO_CHECKPOINT=/workspace/yolo-detector-finetune-pilot/runs/train-seed-55/weights/best.pt

PYTHONPATH="$PWD" "$YOLO_PYTHON" \
  -m experiments.randomized_sort_benchmark.yolo_sidecar \
  --checkpoint "$YOLO_CHECKPOINT" \
  --host 127.0.0.1 \
  --port 18080
```

Check `http://127.0.0.1:18080/health`, then pass
`--yolo-endpoint http://127.0.0.1:18080`. The client revalidates the model hash,
class map, and lossless input-image hash for every result.

AI cells need the separately running Cosmos3-Nano
OpenAI-compatible endpoint via `--ai-endpoint`; each decision has a durable
receipt. A provider failure is recorded as deterministic fallback and must not
be counted as an AI-bound result.

Use one L40S instance per cell when possible. If instance speeds differ,
rotate cell-to-instance assignment or compare simulated task time and report
inference/wall latency separately.

## Summarize

```bash
python -m experiments.randomized_sort_benchmark.summarize \
  --suite /workspace/randomized-sort-suite/suite_index.json \
  --runs /workspace/randomized-sort-runs \
  --out /workspace/randomized-sort-comparison
```

The suite index is required so a crashed or missing run cannot silently vanish
from the denominator. By default the command writes a diagnostic summary but
returns exit code 2 until every condition has a valid report in every cell and
the matched randomization/runtime provenance agrees. Use `--allow-partial`
only for progress monitoring; partial results are explicitly ineligible for a
policy conclusion.

The completed summary reports object and full-scene completion, correctly
sorted objects per simulated minute, vision abstention/latency, outcome causes,
AI binding/fallback/changed-selection counts, class/destination opportunities,
capacity-stress strata, and all six pairwise policy comparisons. Object and
throughput intervals resample whole scene conditions so objects sharing one
scene are not incorrectly treated as independent trials. The 2×2 interaction
is also reported. Runtime reports must include a stable `runtime_provenance`
comparison key; host/instance metadata may be recorded separately and may
differ when equivalent L40S instances execute different cells.

## Interpretation limits

- The 35 mm pocket can be physically capacity-limited in high red-destination
  loads. Those cases are retained and should be reported as capacity stress,
  not relabeled as perception failure. The summarizer conservatively tags a
  condition as a capacity-stress candidate when more than two screws must reuse
  the two commanded screw slots, or a destination load exceeds the nine cube
  grid slots. This tag is not proof that a physical bin is full.
- Seed-55 YOLO has measured held-out cube performance but no held-out physical
  screw evidence. It therefore gates cubes only; screw perception remains the
  deterministic geometry baseline in every cell.
- The screw detector needs three distinct frames before publishing a stable
  executable track; early abstentions are logged.
- Only the table camera drives this revision. Side/wrist evidence is a future
  improvement, not silently fused uncalibrated data.
- Local finger-envelope checks are not complete swept-volume collision
  planning.
- Isaac success is training/evaluation evidence, not proof of physical screw
  pickup or FANUC safety.
