# Scripted colour bucket-sort demo

This demo drives the supplied FANUC/SMC Isaac Sim twin through three physical
pick-and-place sequences: red, yellow, and blue cubes are placed into matching
bins made from the supplied `manipbench-bin-50x50x55.stl` model.

This is a scripted simulation smoke test, not the final sorting system. It uses
configured simulator poses and simulator ground truth; it does not use camera
perception, Cosmos, a learned selector, or a real-robot control interface. Its
purpose is to validate scene assets, inverse kinematics, contact-based grasping,
placement, verification, and rendering before those missing layers are added.

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
