# Randomized realistic four-combination benchmark

This experiment evaluates the same frozen Isaac scenes with a 2×2 design:

| Cell | Vision | Grasp selection |
|---|---|---|
| DD | deterministic colour/silhouette geometry | deterministic geometric ranking |
| YD | YOLO11n box corroboration + the same calibrated geometry | deterministic geometric ranking |
| DA | deterministic colour/silhouette geometry | guarded pretrained-AI ranking |
| YA | YOLO11n box corroboration + calibrated geometry | guarded pretrained-AI ranking |

DD is the baseline. YOLO does **not** manufacture a 3D pose: it must agree
one-to-one with the camera-derived geometry result. The AI ranker can choose
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
`--yolo-checkpoint`. AI cells need the separately running Cosmos3-Nano
OpenAI-compatible endpoint via `--ai-endpoint`; each decision has a durable
receipt. A provider failure is recorded as deterministic fallback and must not
be counted as an AI-bound result.

Use one L40S instance per cell when possible. If instance speeds differ,
rotate cell-to-instance assignment or compare simulated task time and report
inference/wall latency separately.

## Summarize

```bash
python -m experiments.randomized_sort_benchmark.summarize \
  --runs /workspace/randomized-sort-runs \
  --out /workspace/randomized-sort-comparison
```

The summary reports object completion, full-scene completion, correctly sorted
objects per simulated minute, vision abstention/latency, outcome causes, AI
binding/fallback/changed-selection counts, Wilson intervals, and paired
bootstrap intervals against DD.

## Interpretation limits

- The 35 mm pocket can be physically capacity-limited in high red-destination
  loads. Those cases are retained and should be reported as capacity stress,
  not relabeled as perception failure.
- Seed-55 YOLO has measured held-out cube performance but no held-out physical
  screw evidence. Screw results are experimental.
- The screw detector needs three distinct frames before publishing a stable
  executable track; early abstentions are logged.
- Only the table camera drives this revision. Side/wrist evidence is a future
  improvement, not silently fused uncalibrated data.
- Local finger-envelope checks are not complete swept-volume collision
  planning.
- Isaac success is training/evaluation evidence, not proof of physical screw
  pickup or FANUC safety.
