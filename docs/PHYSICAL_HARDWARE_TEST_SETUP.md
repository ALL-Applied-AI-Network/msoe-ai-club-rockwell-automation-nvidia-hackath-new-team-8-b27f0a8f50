# Physical FANUC test setup and runbook

This is the beginner-facing procedure for running this repository through the
official supervised SIDC Lab service. The live
[hackathon site](https://hackathon.sidclab.org/) is authoritative for the
current schedule, limits, approvals, and operator procedure. This file records
the repository state audited on **October 30, 2026**. Ask the project lead or
your coding AI when a step or term is unclear; do not guess at the robot.

## What works now, and what does not

- The official HTTPS client, shared camera pipeline, and cube policy have run
  on the physical FANUC ER-4iA. Recorded rollout 5 camera-verified all three
  red/yellow/blue cube placements in the fixed `virtual-table` destination
  zones.
- That result is not proof of force/contact sensing, general collision-free
  motion, arbitrary layouts, or every future vision/grasp plugin.
- The current executable hardware task expects exactly one red, yellow, and
  blue cube in the configured pickup region. It supports an explicit **10 mm or
  16 mm hardware vision profile**. Isaac's preserved cube task remains 10 mm.
- Physical bins are optional only when `--destination virtual-table` is used.
  The virtual destinations are fixed table regions; they are not perceived
  buckets.
- The project also has a scripted cube-and-screw Isaac demo. Its screw poses are
  configured, not camera-derived. The current physical task adapter/executor
  does **not** yet support screws. Do not put screws into a cube-policy hardware
  rollout and expect them to be sorted.
- Table RGB is currently the only camera trusted for motion coordinates. Side
  and wrist images are valuable diagnostic evidence, but their detector bias,
  wrist transform/timing, and hose occlusion are not yet validated for motion.
- Use the no-motion vision diagnostic's `--capture-camera table|side|wrist`
  option to collect each stream separately at its configured resolution. Side
  and wrist runs are capture-only and do not publish motion coordinates.

For screws to become hardware-ready, the teams must first complete and test:

1. camera-derived screw detection and head/shaft pose;
2. screw task and destination mapping;
3. several mechanically plausible grasp candidates;
4. deterministic finger-fit, approach, held-part, and swept-clearance checks;
5. success scoring among candidates that passed those checks;
6. execution support for the required orientation; and
7. lift/transport/release verification on replay, then Isaac, then supervised
   hover before contact.

## People and systems involved

The event operator and cell controller own physical execution, stopping, fault
recovery, and emergency-stop actions. Our computer runs `Policy.act(obs)` and
returns requested targets through the official HTTPS service. Do not connect
directly to the FANUC controller, PLC, safety network, or industrial network,
and do not change UTOOL, UF0, payload, camera placement, or controller programs.

The current policy uses absolute TCP targets: position in millimetres,
orientation in degrees, and a gripper command. Its assumed action point is the
fingertip in UF0. An organizer clarified that the standard fingers reach the
table near UF0 Z = -335 mm; the former -345 mm FAQ value belonged to shorter
fingers. Verify the installed fingers and active UTOOL1/UF0 at a high,
non-contact pose before every new contact configuration.

## Before requesting a physical slot

### 1. Freeze and record the exact test

Write down:

- project Git commit;
- integration-profile ID and profile SHA-256, if used;
- vision/grasp artifact hashes and versions;
- camera calibration version;
- physical object type and measured dimensions;
- `10` or `16` mm cube profile;
- `virtual-table` or `bins` destination mode;
- requested objects, positions, and number of rollouts;
- supported and explicitly unsupported conditions; and
- the expected observation, action, and success evidence.

Do not silently change any of these after the submission is approved. Create a
new submission or get explicit organizer agreement if the scene or behavior
changes.

### 2. Prepare the local environment

The lightweight cube client can run from WSL Ubuntu on the local computer;
Brev is not inherently required. If a selected model needs Brev/GPU inference,
the instance must remain active for the entire slot, its network latency must
be measured before motion, and no shutdown timer may overlap the slot.

From the repository root:

```bash
cd hardware/fanuc_cell
uv sync
```

Return to the repository root and run the tests:

```bash
cd ../..
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -v
```

If testing a team plugin, export its profile **before** Python starts and record
the value:

```bash
export ROCKWELL_INTEGRATION_PROFILE=team_plugins/profiles/your-profile.json
```

Run the tests once with that profile and once after
`unset ROCKWELL_INTEGRATION_PROFILE`. The second run proves the preserved
baseline was not accidentally changed.

### 3. Inspect the launcher without contacting the cell

From `hardware/fanuc_cell`:

```bash
uv run supervised_run.py hover --cube-mm 16 --dry-run
```

Use `10` only for a measured/operator-confirmed 10 mm cube. Compressed rollout
video suggested approximately 16 mm silhouettes, but video is not a dimensional
measurement.

This baseline command leaves the merged multigrasp system disabled. A reviewed
10 mm multigrasp model can be selected independently with `--grasp multigrasp`
and its checkpoint/hash; choose deterministic or YOLO-assisted perception with
`--vision baseline|yolo11n`. See `docs/TEAM_INTEGRATION.md`. Multigrasp v1 does
not support the 16 mm hardware profile, and component selection does not change
the default `smooth` motion settings.

### 4. Pass the localhost-only official mock

```bash
uv run supervised_run.py hover \
  --cube-mm 16 \
  --mock \
  --trials 1 \
  --trial-s 2
```

The mock starts without a measured TCP, so `hold --mock` is intentionally not
a valid bootstrap. A passing mock proves request/action format and configured
limits only. It does not prove camera accuracy, perception, reachability,
collision clearance, grasp retention, or physical success.

### 5. Make submission metadata match the real scene

`hardware/fanuc_cell/submission/submission.yaml` remains a draft until its team,
contact, and assigned task suffix TODOs are filled with real values. Do not
invent them. `placement.yaml` must describe what the operator will actually
place.

The checked-in draft currently describes three 10 mm cubes and three physical
bins. That does not match a 16 mm/no-bucket virtual-table test. Before
submitting, choose one coherent setup and get the object labels approved:

- **Cube-only no-bucket test:** list only the cubes, describe the measured cube
  size, and state that the policy uses fixed virtual-table destination zones.
- **Cube-and-bin test:** list cubes and bins and later launch with
  `--destination bins`.
- **Future screw test:** make a separate submission only after the screw
  perception/task/grasp/execution path passes its staged gates. The official
  default eight-screw/yellow-bin scene does not make our cube policy a screw
  policy.

Then submit and check status:

```bash
uv run run.py --submit
uv run run.py --status
```

Use the printed request ID to book the event's 30-minute slot. The generated
`.request.json` contains a secret. Never commit it, print its contents, paste it
into chat, upload it, or share the private results URL.

## Day-of supervised test ladder

Use the following order. A later command is not permission to skip an earlier
observation and review.

### Gate A: confirm the scene and stop procedure

Before starting the client, confirm with the operator:

- approved request/run and current rollout layout;
- object dimensions and colors;
- destination mode and clear destination footprints;
- installed custom fingers and mechanical object fit;
- active UTOOL1/UF0 and fingertip action convention;
- camera visibility and table clearance;
- who will stop the rollout and how; and
- that no person will jog or enter the working area while the client is active.

### Gate B: measured-pose hold diagnostic

```bash
uv run supervised_run.py hold
```

This skips perception/planning and returns one freshly measured TCP/gripper row.
It is **not** read-only, a true no-op, a safety system, or an emergency stop;
controller tracking can make a small correction. Do not jog the robot while it
is running.

Check camera names, RGB order, timestamps, current joints/TCP/gripper feedback,
and the expected scene. Stop on missing/malformed feedback.

### Gate C: hover only

```bash
uv run supervised_run.py hover --cube-mm 16
```

This permits the planned high approach but keeps the descent gate closed. Stop
if any target, overlay, frame, path, finger, hose, fixture, neighbor, or
destination clearance is not as expected. Verify the high TCP response and
fingertip/table relationship with the operator.

### Gate D: complete cube sort

Only after the hover is correct, the object size/fit is confirmed, and the
operator approves the path:

```bash
uv run supervised_run.py sort \
  --destination virtual-table \
  --cube-mm 16 \
  --confirm-grasp-fit
```

For a measured 10 mm cube, use `--cube-mm 10`; the special fit flag is required
only for 16 mm contact. For physical bins, select `--destination bins` only
when the approved submission and actual scene contain those bins.

The default action chunk is 32 rows. The supervised launcher now selects its
hardware-only `smooth` profile: 10 mm per 0.1-second row and 8 degrees per row,
or a nominal 100 mm/s Cartesian rate. That covers every recorded rollout-5
phase in one chunk while remaining below the current service limits. Shared
Isaac trajectories remain unchanged. Add `--motion-profile conservative` to
restore the former 5 mm / 5-degree hardware profile.

## Current service limits versus archived text

The live hackathon page is authoritative. On October 30, 2026 it listed
**150 mm/s**, **15 mm per row**, **60 mm queued ahead**, and **10 degrees per
row**. The historical site snapshot preserved in `scilabDocumentation.txt`
contains older values of 100 mm/s, 12 mm, 40 mm, and 8 degrees. The selected
hardware `smooth` profile intentionally uses the common lower envelope:
100 mm/s, 10 mm rows and 8-degree rows. It does not use the live service's
maximum 150 mm/s setting.

Other protocol bounds are enforced by the current official mock and server.
Recheck the live page and mock before each new request rather than copying old
numbers into planning code.

## Stop, fault, and recovery behavior

- Use **Ctrl+C**, not Ctrl+Z, to stop the client. Ctrl+Z only suspends it and can
  leave an old process waiting on the run.
- A policy log saying `holding` does not mean the robot is emergency-stopped.
  The operator/controller remains responsible for stopping.
- A stale/moved/ambiguous target or unexpected obstacle requires
  stop/reobserve/replan. Do not blind-retry or score a rejected grasp.
- Record controller faults and operator interventions. Do not clear faults or
  resume motion without the operator.
- Do not interpret gripper state `1` as proof that an object is held. The
  interface exposes commanded/closed state, not force or object contact.
- Verify the object after lift and after release with fresh camera evidence.

## What to record during every rollout

Store private session evidence outside tracked source or under the ignored
`hardware/fanuc_cell/hardware_runs/` directory:

- project/client commit, profile, model, feature-schema, and calibration IDs;
- request/run/trial/layout identifiers, but never the request secret;
- the actual object dimensions, placement, and destination mode;
- native camera observations and operator-visible overlays;
- observation timestamps, age, joints, TCP, and gripper feedback;
- every detected object, pose, confidence, pose quality, and unknown region;
- all generated candidates and deterministic rejection reasons;
- selected candidate and predicted **complete relocation success**;
- action rows and semantic phases;
- perception/planning/network/execution latency;
- post-lift and post-release verification;
- drop, contact, wrong destination, cancellation, fault, retry, and operator
  intervention; and
- final score plus an honest explanation of any disagreement between camera
  verification and the operator's result.

After the run:

```bash
uv run run.py --status
uv run run.py --download
```

The results URL is private. The downloaded LeRobotDataset includes recorded
camera streams and robot state. Official depth streams are not recorded; if a
future policy depends on depth, separately preserve the exact depth frame,
joints, calibration ID, and timestamp under the event's data rules.

## Minimum next experiments

For the current cube path:

1. Re-run offline table-video detection with the exact profile.
2. Pass the mock.
3. Hold diagnostic.
4. Hover over one target and compare commanded/observed pose.
5. One complete cube relocation with lift and release evidence.
6. Three-cube static sort.

For screws, stop after each failed prerequisite rather than forcing the cube
path to accept a screw:

1. Detect one isolated screw in realistic Isaac RGB and recorded real RGB.
2. Verify head/shaft position and symmetry-aware orientation error.
3. Replay several grasp candidates without motion.
4. Validate finger/shaft fit and held-screw sweep using the supplied CAD.
5. Run camera-derived Isaac pickup/transport/release trials.
6. Create an approved one-screw physical submission.
7. Hold, hover, then only an operator-approved incremental contact attempt.

If you are unsure what evidence or code belongs at a gate, ask the project AI
before proceeding. It is far cheaper to clarify a frame, label, or command in
software than during a physical slot.
