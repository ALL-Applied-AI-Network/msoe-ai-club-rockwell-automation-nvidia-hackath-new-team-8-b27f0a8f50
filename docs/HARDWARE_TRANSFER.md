# Hardware transfer guide

This document describes how to move the cube-sorting behavior from the supplied
Isaac Sim twin to the event's FANUC ER-4iA cell. It is a plan and an interface
contract, **not evidence that this repository has controlled the physical
robot**. The existing scripted colour-bucket demo uses configured simulator
poses and simulator truth; do not send its trajectory directly to the robot.

The physical interface described here is pinned to the official SIDC Lab client
at commit
[`ff7534e730f2b3db261c7f22d3d9b56a40c00eed`](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/tree/ff7534e730f2b3db261c7f22d3d9b56a40c00eed).
Re-audit the upstream package before updating that revision.

## Non-negotiable assumptions

> **10 mm reminder — print this at the start of every simulation and hardware
> run:** this project currently assumes a **10 mm cube**, a work-surface height
> of **−335.1 mm in UF0**, and a provisional pick target of **−332.6 mm** under
> the official repository/mock's **TCP means fingertip** convention. The live
> event site describes a different tool point, approximately 10 mm lower. The
> final low pick target must not be used on hardware until a supervisor confirms
> the tool convention with a non-contact hover test.

Keep these values in one configuration object rather than copying numbers into
perception, planning, simulation, and hardware code:

```text
CUBE_SIZE_MM = 10.0
SURFACE_Z_BASE_M = -0.0051
UF0_Z_OFFSET_MM = -330.0
SURFACE_Z_UF0_MM = -335.1
PICK_TIP_CLEARANCE_MM = 2.5
PICK_TIP_Z_UF0_MM = -332.6
```

The calculation is `−335.1 + 2.5 = −332.6 mm`. Under the chosen convention,
that target puts the fingertip 2.5 mm above the work surface and 7.5 mm below
the top of a 10 mm cube. Whether the custom hook actually grasps correctly at
that height is unverified.

Why 10 mm rather than 16 mm: the cell description says the custom SMC fingers'
straight faces are 12 mm apart when open. A 10 mm cube nominally leaves only
1 mm clearance on each side; a 16 mm cube does not fit between those straight
faces. Even the 10 mm choice is tight enough that camera, calibration, and
placement error must be measured before contact.

### The unresolved 10 mm tool-point conflict

The pinned client repository calls `obs["tcp_pose"]` the **fingertip**, defines
UF0 as the base frame shifted by −330 mm, and places the work surface at about
−335 mm. Its mock accepts TCP z only in `[−334, 300]` mm. Those facts support
the provisional −332.6 mm value above.

The current [event FAQ](https://hackathon.sidclab.org/) instead says the
controller's UTOOL1 is 10 mm past the fingertips and that the TCP is about
−345 mm when the fingertips touch the surface. If that description is applied
literally, the same physical fingertip height would use a TCP z value 10 mm
lower. The two descriptions cannot both be used as the command convention.

We deliberately choose the repository/mock fingertip convention because it is
the executable contract and because the higher target is the conservative
failure direction: a mismatch should miss the cube high instead of driving the
fingers into the table. This is still only an assumption. Before the first
contact attempt, an assigned cell supervisor must confirm:

1. what point `obs["tcp_pose"]` reports;
2. what point a `tcp` action commands;
3. UTOOL1 and UF0 currently loaded on the controller; and
4. the observed fingertip-to-table clearance at a high, non-contact target.

Start well above the table, approach in supervised increments, and stop before
the low pick height. Do not “test” the convention by touching the table. If the
live-site convention is confirmed, change the single shared transform and its
tests; do not add an unexplained `−10` in the hardware adapter.

## Use the event client, not a direct robot driver

For this event, the supported hardware boundary is the official HTTPS client:

```text
cell observations -> Policy.act(obs) on our computer -> action rows -> cell server -> FANUC
```

The operator and cell server own physical execution and safety enforcement.
Our software implements `Policy.act(obs)`. It must not configure the robot
controller, PLC, industrial network, safety system, camera placement, UTOOL, or
UF0.

FANUC's public ROS 2 driver does list `er4ia` as a supported description, but
direct hardware control also depends on controller versions, licensed motion
options, network configuration, and cell authority that have not been
established for this testbed. The official driver requirements are useful
background, not permission to bypass the event client. Likewise, an Isaac Sim
ROS joint-state tutorial demonstrates simulator transport; it does not prove a
safe connection to this FANUC cell.

## Exact policy contract

The pinned upstream [`policy.py`](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/blob/ff7534e730f2b3db261c7f22d3d9b56a40c00eed/policy.py)
defines the participant-facing API. `Policy.reset()` is called at the start of
every rollout. `Policy.act(obs)` returns one target row or a list of target rows.
Rows are **absolute targets**, not deltas.

### Observations

| Field | Meaning |
|---|---|
| `obs["images"]` | Dictionary of requested `table`, `side`, and/or `wrist` images. Arrays are `(H, W, 3)` `uint8` **RGB**. The default request is the entire frame resized, not cropped, to 640×360. |
| `obs["joints"]` | `[J1, J2, J3, J4, J5, J6, gripper]`; degrees in the v0.5 twin convention, with gripper 0 open and 1 closed. |
| `obs["tcp_pose"]` | `[x, y, z, w, p, r]`; position in mm and FANUC orientation in degrees. This project provisionally interprets it as the fingertip in UF0. |
| `obs["trial"]` | Current rollout information, including `instruction`. The policy may use or ignore the wording. |
| `obs["depth"]` | Present only when requested. Table depth is 640×400; wrist depth is 848×480. Values are `uint16` millimetres along the optical axis, with 0 meaning no return. |

The client observation also carries a timestamp, robot status, and queued-action
count internally. Preserve those when adapting it to a project observation;
they are needed to reject stale decisions and understand latency.

OpenCV normally treats decoded images as BGR, but the official client converts
them to RGB. Convert explicitly at one boundary. Otherwise red and blue sorting
can silently reverse.

### Actions

Choose exactly one mode:

- `ACTION = "tcp"`: `[x, y, z, w, p, r, gripper]` in mm/degrees. This is the
  recommended project mode because the same Cartesian plan can feed both
  adapters and it avoids exposing the FANUC/J3 convention to task logic.
- `ACTION = "joints"`: `[J1, J2, J3, J4, J5, J6, gripper]` in degrees. Joint
  values use the v0.5 twin convention, where `sim J3 = controller J3 + J2`.

For a down-facing gripper, the upstream convention is `w, p, r = −180, 0, R`.
The gripper command is binary in practice: 0 opens it; 0.5 or greater closes
it. The event site says the gripper changes state after the arm reaches that
row's target. Include a short dwell at the same pose after closing and opening;
do not assume a pneumatic jaw changes instantaneously.

The policy cannot declare the rollout complete. After verifying the task, it
must hold still by returning the current safe pose until the operator scores or
stops the rollout.

### Enforced mock limits

The pinned upstream
[`mock_cell.py`](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/blob/ff7534e730f2b3db261c7f22d3d9b56a40c00eed/mock_cell.py)
checks the following protocol limits:

| Limit | Accepted range |
|---|---|
| TCP x | 150 to 650 mm in UF0 |
| TCP y | −450 to 450 mm in UF0 |
| TCP z | −334 to 300 mm in UF0 |
| TCP w, p, r | each within ±360° |
| J1 | −170 to 170° |
| J2 | −110 to 120° |
| J3 | −69 to 205° |
| J4 | −190 to 190° |
| J5 | −120 to 120° |
| J6 | −360 to 360° |
| Row duration `DT` | 0.1 to 1.0 seconds |
| Rows per chunk | 1 to 100 |
| Total action queue | at most 100 rows |

All values must be finite. The gripper value is clipped to `[0, 1]`. The
provisional −332.6 mm pick target is only 1.4 mm above the mock's lower z
limit—another reason to keep it provisional and centrally visible.

Passing the mock is necessary but not sufficient. The mock does not implement
kinematics, collision checking, object motion, contact, or grasp success. The
live event page also says targets should remain within about 0.55 m of the base
axis; a rectangular x/y bounds check does not guarantee that reach constraint.
Our guard must reject `sqrt(x² + y²) > 550 mm` and must still perform IK,
joint-limit, and swept-path checks in the twin.

## What simulation/hardware parity means

“Behave the same” means both backends use the same decision-making contract,
not that physics and timing are identical.

Shared between backends:

- the observation schema, units, frame names, timestamps, and camera names;
- colour/pose perception and uncertainty outputs;
- tracked object IDs and scene-version invalidation;
- candidate generation and selection policy;
- the phase state machine and recovery rules;
- action rows, preflight limits, and success/failure logging; and
- the same 10 mm cube and provisional tool-point configuration.

Backend-specific:

- `IsaacBackend` obtains camera images and robot state from Isaac, converts them
  to the official observation schema, converts TCP rows through the existing IK
  and gripper controller, and uses physics only for execution/evaluation.
- `HardwarePolicy` receives the official `obs` dictionary and returns the same
  TCP rows through `Policy.act`. The event client, operator, and controller own
  actual motion.

Simulator ground truth may label evaluation data, but it must never repair a
runtime perception result or select a hardware action. A policy that reads
configured cube poses in Isaac does not have hardware parity.

Keep the existing Isaac demo entry point working. Add the common policy and
adapters alongside it; do not rewrite the proven demo around an unfinished
hardware client. Regression-test the existing 3/3 scripted run after every
shared motion or frame change.

## Minimal phase state machine

Use an explicit state machine rather than returning the entire sort as one
open-loop chunk:

```text
OBSERVE -> SELECT -> APPROACH -> REVALIDATE -> DESCEND
        -> CLOSE/DWELL -> LIFT/VERIFY -> TRANSPORT
        -> ABOVE_BIN -> DESCEND -> OPEN/DWELL -> RETREAT/VERIFY
```

Every action references a target track ID and the scene version from which it
was planned. If the target moved, a new obstacle appeared, the frame is stale,
or the target became ambiguous, hold and reobserve. Recovery depends on phase:
blindly retreating while between obstacles can create a second collision.

Use longer chunks only for well-cleared transit. Near cubes, bins, or the table,
use short chunks so a new observation arrives before descent. Do not enable
prefetch near contact until replacement timing and cancellation have been
measured.

## Phased validation ladder

Advance only when the current gate passes. Record failures; do not weaken a
gate to make a demonstration appear successful.

### Gate 0 — freeze the contract

- Pin the official client at upstream SHA `ff7534e` and record the project SHA.
- Centralize the five numeric assumptions shown at the top of this document.
- Print those assumptions, especially the 10 mm tool-point conflict, at every
  run start.
- Confirm the Git working tree is clean and no credentials or generated runs
  are tracked.

### Gate 1 — CPU contract tests

- Test RGB/BGR conversion, metres↔millimetres, degrees↔radians, UF0/base z, and
  the J3 conversion if joint mode is ever used.
- Test every numeric boundary, NaN/inf rejection, 0.55 m radial reach guard,
  chunk length, and hold behavior.
- Test `reset()` removes all prior-rollout tracks and phase state.
- Test no-detection and ambiguous-detection observations return a valid hold,
  not a guessed motion.

### Gate 2 — official mock

From a pinned official-client checkout or an attributed pinned copy inside this
repository:

```bash
uv sync
uv run run.py --mock
```

Success is the literal `✓ Ready for the real robot` verdict with zero rejected
actions. This proves only formats and connection behavior. Mock images are
synthetic, and the mock's initial TCP pose may be absent, so the policy needs a
tested safe initialization/hold path.

### Gate 3 — Isaac regression and parity tests

- Keep the existing scripted bucket-sort demo passing unchanged.
- Run the new shared policy using only simulated camera observations, never
  configured object poses.
- Compare the adapter's observation shapes, units, frame transforms, and action
  limits with the official contract.
- Verify approach, finger, held-object, lift, transport, bin-rim, and retreat
  clearances with inflated obstacles.
- Perturb camera pose, cube pose, friction, lighting, latency, and object
  arrival time. Report sensitivity instead of treating one physics setting as
  proof.

### Gate 4 — supervised read-only hardware check

During an approved slot, with the operator in control:

- start the client and inspect camera names, image colour order, timestamps,
  joint convention, TCP pose, and table/bin/cube visibility;
- save calibration overlays without commanding a contact motion;
- confirm available cube/bin colours and exact placement names; and
- measure observation-to-action round trip and actual pose lag.

Stop if calibration error consumes the available grasp clearance.

### Gate 5 — supervised hover confirmation

- With the operator's approved empty/clear workspace, command only a high,
  non-contact pose.
- Verify the observed TCP response and visible fingertip height.
- Approach in operator-approved increments, remaining well above the cube and
  table.
- Resolve the −335/−345 convention and custom hook geometry before enabling
  the provisional −332.6 mm target.

Any disagreement between expected and observed frames returns the project to
Gate 1. Do not compensate by trial-and-error at table height.

### Gate 6 — one cube, one bin

- Use one 10 mm cube at a conservative, known position and one known bin.
- Detect it from a real image, execute one pick/place, and independently verify
  lift and final placement.
- Save a trustworthy log containing all observations used for decisions,
  actions, timings, result, and any operator intervention.

This is the first complete physical milestone. Only after it succeeds should
the team add several cubes, several bins, changing layouts, or learning.

### Gate 7 — multi-object static sort

- Detect all cubes and bins needed as obstacles, not only the chosen target.
- Reject blocked, unreachable, stale, or ambiguous candidates.
- Compare the geometric baseline, highest predicted success, and success/time
  selector on matched layouts with identical recovery rules.

### Gate 8 — dynamic arrivals

- First demonstrate that a moved target or new obstacle cancels descent and
  causes hold/reobserve/replan.
- Confirm with organizers whether objects may be added during a rollout and
  whether feeding can be constrained.
- Do not attempt continuous interception until latency, stopping behavior, and
  swept clearance are measured. Cancel-and-replan is the initial policy.

## Perception and depth consequences

The official
[`calibration/`](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/tree/ff7534e730f2b3db261c7f22d3d9b56a40c00eed/calibration)
contains camera transforms and NumPy projection helpers. The starter twin's
cameras are intended to match the real views, but calibration residuals are not
the same as end-to-end grasp accuracy.

- A visible cube centroid lies on the cube's top face. For a 10 mm cube, map a
  top-face pixel at `surface_z + 0.010 m`, not at the table plane.
- The event FAQ warns that overhead table depth does not reliably see screws or
  small cubes. Start with table RGB plus known geometry. If needed, use wrist
  depth close to the table and pass the **same observation's joints** to the
  wrist transform.
- Depth frames are not included in the downloadable event recording. If a
  decision depends on depth, separately log the exact depth frame, joints,
  calibration version, and timestamp used, subject to event rules and without
  recording credentials.
- Camera entries are each the latest frame; do not assume perfect multi-camera
  synchronization. Treat old or inconsistent frames as uncertainty, not clear
  space.
- Colour thresholds that work in rendered images are not validated on the
  physical lighting, camera white balance, cube material, or bin material.
  Obtain real examples, reject ambiguous classifications, and keep an
  operator-visible overlay.

## Latency, verification, and logging

The live event page reports that the real arm follows targets about 0.8 seconds
behind the twin. Each returned row normally represents 0.1 seconds. A long
100-row chunk can therefore create roughly ten seconds without a new planning
observation. Network latency from Brev adds to perception and replanning time.

For the lightweight geometric baseline, prefer running the event client on a
reliable local Linux/macOS machine during the physical slot. If Brev is used,
verify latency before motion, prevent its shutdown timer from overlapping the
slot, and consider JPEG only after proving compression does not corrupt colour
classification.

The observation exposes no force, finger contact, or “object held” signal.
Gripper state 1 means commanded/closed, not a verified grasp. Use camera-based
verification after lift and after release. The official operator score is for
the whole rollout, so our log must also retain per-object evidence.

At minimum, log:

```text
project commit and upstream client commit
policy/config/calibration versions and 10 mm assumption banner
rollout, phase, target track ID, scene version, observation timestamp/age
camera-derived objects, pose quality, uncertainty/unknown regions
candidate action, preflight rejection reason, commanded rows
perception, planning, network, execution, recovery, and total episode time
lift/place verification, contact/drop/wrong-bin/cancellation/intervention
```

Simulator truth and physics settings belong in evaluator-only fields. Never
train on exact simulator coordinates and then claim deployment from camera
estimates. The same feature extractor must run during collection and runtime.

## Submission and supervised run workflow

The official
[`README.md`](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/blob/ff7534e730f2b3db261c7f22d3d9b56a40c00eed/README.md)
defines this sequence:

1. Install `git` and `uv`, then synchronize the pinned client with `uv sync`.
2. Implement the wrapper in `policy.py` and pass Gate 2 with
   `uv run run.py --mock`.
3. Fill `submission/submission.yaml`. Give the policy/task clear names, select
   the closest model type, and describe the sorting task honestly.
4. Fill `submission/placement.yaml` for any non-default scene. If objects and
   layout are both `same`, include one entry reused for every rollout. If either
   is `different`, include exactly one entry per requested rollout. Positions
   are x/y millimetres in the robot frame with optional yaw degrees.
5. Run `uv run run.py --submit`. It creates `.request.json` containing a secret.
   **Never commit, paste, or publish that file or a private results link.**
6. Use the printed request ID to book a 30-minute Central Time slot on the
   [event calendar](https://hackathon.sidclab.org/#calendar). The request must
   be approved.
7. During the approved slot, run `uv run run.py` and leave it running. The
   operator homes the robot, prepares each rollout, starts/stops it, and records
   faults/interventions.
8. Use `uv run run.py --status` for status/results and
   `uv run run.py --download` for the recorded LeRobotDataset.

The standard documentation describes operator-prepared starting layouts, not
objects entering while the arm moves. Dynamic feeding requires explicit
organizer approval and a safe test protocol.

## Missing information checklist

Do not call the hardware integration complete until these are answered:

- [ ] Is the action/report point the repository's virtual fingertip or raw
      controller UTOOL1? Resolve the −335.1 versus approximately −345 mm table
      convention.
- [ ] What UTOOL1, UF0, payload, and controller program are active for the
      approved event interface?
- [ ] Where is the custom aluminium hook-finger CAD, including notch dimensions
      and its relation to the reported TCP? The resource page currently exposes
      the standard SMC body model, which is insufficient for final clearance.
- [ ] Are red, yellow, and blue 10 mm cubes and three corresponding physical
      bins available? In what quantities?
- [ ] What exact object names does the submission service accept for coloured
      cubes/bins? Official examples show only generic `cube` and `bin`.
- [ ] Can the operator set up three bins and multiple coloured cubes, and can
      any object move or arrive during a rollout?
- [ ] How does the cell interpolate between TCP rows, and what speed,
      acceleration, stopping, and action-replacement behavior is guaranteed?
- [ ] What real camera samples and calibration-check targets can the team use
      before the first pick?
- [ ] Which supervised slot is approved, who will attend, and what is the
      operator-agreed stop/hold procedure?
- [ ] May separately captured depth and rollout data be used for training, or
      only for evaluation/debugging under the event's data rules?

## Primary references

- [Event participant dashboard, cell FAQ, CAD resources, and booking](https://hackathon.sidclab.org/)
- [Official client at the audited `ff7534e` revision](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/tree/ff7534e730f2b3db261c7f22d3d9b56a40c00eed)
- [Official policy contract](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/blob/ff7534e730f2b3db261c7f22d3d9b56a40c00eed/policy.py)
- [Official mock and protocol limits](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/blob/ff7534e730f2b3db261c7f22d3d9b56a40c00eed/mock_cell.py)
- [Official frame definitions](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/blob/ff7534e730f2b3db261c7f22d3d9b56a40c00eed/calibration/FRAMES.md)
- [Official camera geometry helpers](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/blob/ff7534e730f2b3db261c7f22d3d9b56a40c00eed/calibration/geometry.py)
- [FANUC ER-4iA specifications](https://www.fanucamerica.com/products/robot/er-4ia)
- [FANUC ROS 2 driver quick start and controller prerequisites](https://fanuc-corporation.github.io/fanuc_driver_doc/main/docs/quick_start/quick_start.html)
- [FANUC supported description models](https://fanuc-corporation.github.io/fanuc_driver_doc/main/docs/environment/supported_models.html)
- [Isaac Sim ROS 2 joint-control tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/robot_control/tutorial_ros2_manipulation.html)
