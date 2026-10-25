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

The shared configuration currently assumes **10 mm cubes** and defines the
UTOOL1/TCP at the **fingertips (0 mm extra offset)**. The live event site has
also said that the TCP is 10 mm beyond the fingertips. That unresolved 10 mm
frame difference can turn a safe height into a table contact. Keep the current
choice consistent in simulation and hardware, and verify it with the cell
operator before enabling any physical descent.

## Run the official mock

Install [`uv`](https://docs.astral.sh/uv/), then run from the repository root:

```bash
cd hardware/fanuc_cell
uv sync
uv run run.py --mock
```

The default policy is fail-closed. `ROCKWELL_ENABLE_PHYSICAL_DESCENT` is unset,
so it will not descend toward a detected cube. Depending on the observation it
may still command a hover pose; "descent disabled" does not mean "no motion."
The policy's hold command is also **not an emergency stop**. Only the cell
operator/controller provides the supported stop behavior.

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
cell operator. During the approved validation session, leave the descent
interlock unset and first confirm the table/camera calibration, object layout,
safe hover, working envelope, gripper, and the live UTOOL1/TCP convention. Even
this hover run can move the arm.

Only after that supervised frame-and-hover verification may the same approved
client be started with physical descent enabled:

```bash
ROCKWELL_ENABLE_PHYSICAL_DESCENT=1 uv run run.py
```

Setting this environment variable is a deliberate authorization gate, not a
safety system. An operator must remain responsible for the controller's stop
and recovery procedure. Do not set the variable on the basis of a passing mock
or Isaac run.

The official runner creates and consumes the private request credential in
this directory. Use `uv run run.py --status` to inspect the request and plain
`uv run run.py` (with or without the descent gate as appropriate) only during
the approved supervised slot.

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
