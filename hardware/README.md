# FANUC hardware onboarding

This directory contains the project's hardware adapter for the official SIDC
Lab supervised FANUC workflow. It does not bypass the event service, connect
directly to a controller or PLC, or authorize an unsupervised robot run.
The authoritative external references are the
[live event hardware instructions](https://hackathon.sidclab.org/) and the
[official SIDC Lab client](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026);
this repository pins the client revision recorded in
[`fanuc_cell/UPSTREAM.md`](fanuc_cell/UPSTREAM.md).

## Status

- The adapter uses the official `Policy.act(obs)` HTTPS client and absolute TCP
  action format.
- The official local mock is the first executable gate. Passing it verifies
  protocol shapes, configured limits, timing, and basic rollout behavior only.
- It does **not** verify camera accuracy, frames, reachability, collision-free
  motion, grasp success, bin placement, or emergency stopping.
- This project has run on the physical FANUC cell. Recorded rollout 5
  camera-verified all three cube placements in `virtual-table` mode. That is
  valuable integration evidence, not proof of force/contact sensing, complete
  swept-path clearance, arbitrary layouts, or screw support.

The preserved Isaac cube task assumes **10 mm cubes**. Hardware perception now
requires an explicit **10 mm or 16 mm** cube profile and the physical cube must
be measured or operator-confirmed before descent. The integration follows the
pinned executable client's definition of UTOOL1/TCP at the **fingertips (0 mm
extra offset)**. Separately, an organizer clarified on October 26, 2026 that
the standard fingers touch the table near UF0 Z = -335 mm; the former -345 mm
FAQ value was for another group's shorter fingers and was reverted. Those facts
are consistent but do not prove the live controller setup. Verify the installed
fingers and active UTOOL1/UF0 with the cell operator before physical descent.

The raw detector also uses a provisional calibrated pickup region of
`x=0.30..0.55 m` and `y=-0.30..0.30 m` in robot-base coordinates only when
deciding which detections may become pick targets. The policy classifies
matching-bin and wrong-bin observations first, and retains non-bin observations
outside the region—including weak ambiguous evidence—for fail-closed
placement-outcome checks. Confirm the whole
region against the real camera overlay, calibration uncertainty, cell fixtures,
and approved working envelope; do not widen it merely to force a missed
detection. It is not an obstacle map or a swept-volume collision guarantee.

## Run the official mock

Install [`uv`](https://docs.astral.sh/uv/), then run from the repository root:

```bash
cd hardware/fanuc_cell
uv sync
uv run supervised_run.py hover --cube-mm 16 --mock --trials 1 --trial-s 2
```

Replace `16` with `10` only for a measured/operator-confirmed 10 mm cube. The
mock starts without a measured TCP, so the wrapper uses its mock-only hover
bootstrap. This does **not** authorize a physical run. With physical motion
unset, the policy performs only a stationary hold-command diagnostic: it
requires a fresh measured TCP and gripper value, skips perception and planning,
and returns that one measured row. The official interface has no true no-op, so
even this repeated row can make a small correction toward an earlier measured
pose. Do not jog the robot while it is running, and do not treat it as a
read-only mode, emergency stop, or safety system.

`ROCKWELL_ENABLE_PHYSICAL_MOTION=1` enables the complete planned pre-descent
route, not merely one bounded Cartesian target. The project does not yet prove
full-arm/finger/hose/held-object swept clearance for arbitrary changed routes.
Use the supervised wrapper and repeat the hold/hover preflight for every new
profile, size, layout, or grasp; do not rely on an old successful rollout.

Never add `.request.json` to Git, print it, paste it into chat, or upload it.
The official `--submit` command creates it with the secret required for that
one request, and the repository ignores it intentionally.

## Prepare the supervised physical run

Read the maintained beginner procedure in
[`docs/PHYSICAL_HARDWARE_TEST_SETUP.md`](../docs/PHYSICAL_HARDWARE_TEST_SETUP.md)
before using a request or motion mode.

Before `--submit`:

1. Replace every `TODO` in
   [`fanuc_cell/submission/submission.yaml`](fanuc_cell/submission/submission.yaml).
2. Make the task and placement files match the actual measured cube size,
   `virtual-table` or physical-bin destination mode, objects, and layout. Ask
   the organizers to confirm the accepted labels. The checked-in draft still
   describes 10 mm cubes plus physical bins and must not be reused unchanged
   for a 16 mm/no-bucket run.
3. Run the mock and resolve every rejection without weakening a limit or safety
   gate.
4. Review the full phased checklist in
   [`docs/HARDWARE_TRANSFER.md`](../docs/HARDWARE_TRANSFER.md), including the
   missing-information checklist.

When the task metadata and physical scene are approved, use the official flow:

```bash
cd hardware/fanuc_cell
uv run run.py --submit
uv run run.py --status
```

Book the approved 30-minute slot using the request id printed by `--submit`.
Coordinate the first live connection and every motion check with the assigned
cell operator. First request an organizer-supported observation-only/no-op
procedure; the checked-in default can only repeat a measured command row. Keep
both motion variables unset until the operator has confirmed the observation
contract, table/camera calibration, object layout, gripper feedback, installed
fingers, and live UTOOL1/UF0.

Only after full clearance validation and the supervised frame-and-motion gates
may the same approved client eventually be started with physical descent
enabled. Use the wrapper so stale environment settings are cleared and the
selected object/destination assumptions are printed:

```bash
uv run supervised_run.py sort \
  --destination virtual-table \
  --cube-mm 16 \
  --confirm-grasp-fit
```

Use `--cube-mm 10` for a measured 10 mm cube; its sort does not require the
16 mm fit flag. Select `--destination bins` only for an approved scene that
actually contains the bins. These behavior-bearing arguments are deliberate
authorization gates, not a safety system. An operator remains responsible for
the controller's stop and recovery procedure. Do not run `sort` on the basis of
a passing mock or Isaac run alone.

The official runner creates and consumes the private request credential in
this directory. Use `uv run run.py --status` to inspect the request. Plain
`uv run run.py` runs the stationary hold-command diagnostic; an eventual
approved sorting run uses the wrapper shown above and may run only during the
supervised slot after every prerequisite is satisfied.

## What stays shared with Isaac

The task configuration, target/destination layout, selection order, semantic
phases, and fail-closed checks live outside the official client in shared
project modules. Isaac's preserved cube task remains 10 mm; the hardware vision
profile is explicitly 10 or 16 mm and must not silently change Isaac geometry.
Isaac and hardware have different low-level adapters: the simulator uses its
own IK and physics, while the real cell uses the official TCP service. Never
copy simulated joint values directly to the FANUC controller.

The separate scripted screw demo is not an executable physical screw path. A
screw remains unsupported until camera-derived pose, task/destination mapping,
grasp candidates, deterministic feasibility, execution orientation, and
outcome verification are integrated and tested.

For the exact frame conversion, validation gates, known gaps, and references,
read [`docs/HARDWARE_TRANSFER.md`](../docs/HARDWARE_TRANSFER.md). The provenance
and update boundary for the vendored official client are in
[`fanuc_cell/UPSTREAM.md`](fanuc_cell/UPSTREAM.md).
