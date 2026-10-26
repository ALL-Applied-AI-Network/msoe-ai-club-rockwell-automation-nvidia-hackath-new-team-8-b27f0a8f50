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
- This project has not yet been run on the physical FANUC cell.

The shared configuration currently assumes **10 mm cubes** and follows the
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
ROCKWELL_ENABLE_PHYSICAL_MOTION=1 uv run run.py --mock
```

The mock starts without a measured TCP, so its protocol test explicitly opens
the motion gate. This does **not** authorize a physical run. With the motion
gate unset, the policy performs only a stationary hold-command diagnostic: it
requires a fresh measured TCP and gripper value, skips perception and planning,
and returns that one measured row. The official interface has no true no-op, so
even this repeated row can make a small correction toward an earlier measured
pose. Do not jog the robot while it is running, and do not treat it as a
read-only mode, emergency stop, or safety system.

`ROCKWELL_ENABLE_PHYSICAL_MOTION=1` enables the complete planned pre-descent
route, not merely one bounded hover. The project does not yet implement the
required full-arm/finger/held-object swept-path collision validator. Therefore
keep this variable unset on physical hardware until that validator, the cell
geometry, and the supervised preflight gates are complete.

Never add `.request.json` to Git, print it, paste it into chat, or upload it.
The official `--submit` command creates it with the secret required for that
one request, and the repository ignores it intentionally.

## Prepare the supervised physical run

Before `--submit`:

1. Replace every `TODO` in
   [`fanuc_cell/submission/submission.yaml`](fanuc_cell/submission/submission.yaml).
2. Ask the organizers to confirm that the physical cell will have one red,
   yellow, and blue **10 mm** cube, three matching bins, the checked-in fixed
   layout, and the generic `cube`/`bin` labels used in
   [`placement.yaml`](fanuc_cell/submission/placement.yaml). These are not the
   official default eight-screw/yellow-bin scene.
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
enabled. Both explicit authorizations are required:

```bash
ROCKWELL_ENABLE_PHYSICAL_MOTION=1 \
ROCKWELL_ENABLE_PHYSICAL_DESCENT=1 \
uv run run.py
```

Setting these environment variables is a deliberate authorization gate, not a
safety system. An operator must remain responsible for the controller's stop
and recovery procedure. Do not set either variable on the basis of a passing
mock or Isaac run.

The official runner creates and consumes the private request credential in
this directory. Use `uv run run.py --status` to inspect the request. Plain
`uv run run.py` runs the stationary hold-command diagnostic; an eventual
approved sorting run requires both motion variables shown above and may run
only during the supervised slot after every prerequisite is satisfied.

## What stays shared with Isaac

The task configuration, 10 mm object assumption, target/bin layout, selection
order, semantic phases, and fail-closed checks live outside the official client
in the shared project modules. Isaac and hardware have different low-level
adapters: the simulator uses its own IK and physics, while the real cell uses
the official TCP service. Never copy simulated joint values directly to the
FANUC controller.

For the exact frame conversion, validation gates, known gaps, and references,
read [`docs/HARDWARE_TRANSFER.md`](../docs/HARDWARE_TRANSFER.md). The provenance
and update boundary for the vendored official client are in
[`fanuc_cell/UPSTREAM.md`](fanuc_cell/UPSTREAM.md).
