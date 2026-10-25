# Vendored SIDC Lab FANUC cell client

This directory contains a pinned snapshot of the official physical-cell client
from:

- Source: <https://github.com/SIDCLab/PhysicalAI-Hackathon-2026>
- Commit: `ff7534e730f2b3db261c7f22d3d9b56a40c00eed`
- License: Apache License 2.0; see [LICENSE](LICENSE)

The snapshot was copied from a clean checkout. It contains no upstream Git
metadata, virtual environment, request credentials, recorded datasets, or
other generated state.

## Boundary

`cell_client.py`, `run.py`, `mock_cell.py`, `pyproject.toml`, `uv.lock`, the
upstream `README.md` and `LICENSE`, everything under `calibration/`, and
`submission/table_plan.png` are byte-pinned upstream interfaces or assets and
must remain unmodified. Keep upstream fixes as a clearly reviewed snapshot
update rather than editing those files in place.

`policy.py` is deliberately project-owned: it replaces the upstream example
with the thin hardware adapter around this repository's shared sorting core.
The two YAML files under `submission/` are also project-owned task metadata.
Keeping those three files next to the official runner is required by the
official client layout; it does not make them part of the pinned upstream
snapshot.

The upstream `.gitignore` is intentionally retained here. In particular,
`.request.json` contains the physical-run request secret and must never be
committed or shared.

## Updating the snapshot

1. Clone or fetch the official repository outside this project checkout.
2. Review the proposed upstream commit and check it out with a clean working
   tree. Record the full commit SHA.
3. Replace only the byte-pinned upstream files listed under **Boundary** from
   committed Git objects. Preserve the project-owned `policy.py` and submission
   YAML. Never copy `.git/`, `.request.json`, environments, caches,
   credentials, downloaded datasets, or generated outputs.
4. Preserve the relative paths and exact bytes, including
   `submission/table_plan.png`. Update the commit recorded above.
5. Compare every byte-pinned file with the selected upstream Git object,
   review the diff, run `uv sync`, and run `uv run run.py --mock`.

Passing the mock verifies the client protocol, shapes, limits, timing, and
rollout flow. It does not validate robot kinematics, collision avoidance,
perception accuracy, grasping, or physical task success.
