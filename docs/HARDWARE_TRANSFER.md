# Hardware transfer guide

This document describes how to move the cube-sorting behavior from the supplied
Isaac Sim twin to the event's FANUC ER-4iA cell. It is a plan and an interface
contract. The repository has now controlled the physical robot: recorded
rollout 5 camera-verified all three cube placements in `virtual-table` mode.
That result is integration evidence, not proof of force/contact sensing,
complete swept-path clearance, arbitrary objects/layouts, or screw support.
The colour-bucket demo's preserved default mode uses configured simulator
poses; its opt-in camera modes use rendered RGB for decisions but still use
simulator truth only for evaluation. Do not send any Isaac trajectory directly
to the robot. The maintained beginner procedure is
[`PHYSICAL_HARDWARE_TEST_SETUP.md`](PHYSICAL_HARDWARE_TEST_SETUP.md).

The physical interface described here is pinned to the official SIDC Lab client
at commit
[`ff7534e730f2b3db261c7f22d3d9b56a40c00eed`](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026/tree/ff7534e730f2b3db261c7f22d3d9b56a40c00eed).
Re-audit the upstream package before updating that revision.

## Non-negotiable assumptions

> **Object-profile reminder — print this at every run:** the preserved Isaac
> cube task uses **10 mm** geometry. Hardware perception requires an explicit
> **10 mm or 16 mm** cube profile selected from a measured/operator-confirmed
> physical cube. The work-surface height remains **−335.1 mm in UF0** and the
> provisional fingertip pick target **−332.6 mm** under the pinned client's
> **TCP means fingertip** convention. An organizer confirmed on October 26,
> 2026 that the standard fingers touch the table near UF0 Z = −335 mm and the
> former −345 mm value was for another group's shorter fingers. These facts do
> not prove the installed fingers or active UTOOL1/UF0, which still require a
> supervised high, non-contact verification before physical descent.

Keep these values in one configuration object rather than copying numbers into
perception, planning, simulation, and hardware code:

```text
SIM_CUBE_SIZE_MM = 10.0
HARDWARE_VISION_CUBE_SIZE_MM = 10.0 or 16.0 (explicit per run)
SURFACE_Z_BASE_M = -0.0051
UF0_Z_OFFSET_MM = -330.0
SURFACE_Z_UF0_MM = -335.1
PICK_TIP_CLEARANCE_MM = 2.5
PICK_TIP_Z_UF0_MM = -332.6
```

The calculation is `−335.1 + 2.5 = −332.6 mm`. The exact contact relationship
depends on the selected cube and the custom hook geometry. The configured hook
opening is 15.99 mm, so a nominal 16 mm cube is not nominally smaller than the
gap. Rounded cube edges or the physical hook may still permit the demonstrated
grasp, but video cannot establish fit. A 16 mm hover is allowed; 16 mm descent
requires an explicit mechanical fit confirmation. A 10 mm cube also has small
clearances and still requires measured camera/calibration error before contact.

### Resolved FAQ discrepancy and remaining physical check

The pinned client repository calls `obs["tcp_pose"]` the **fingertip**, defines
UF0 as the base frame shifted by −330 mm, and places the work surface at about
−335 mm. Its mock accepts TCP z only in `[−334, 300]` mm. Those facts support
the provisional −332.6 mm value above.

An organizer clarified by email on October 26, 2026 that the FAQ's former −345
mm value had been changed accidentally for another group using shorter fingers.
The FAQ was reverted: the standard fingers still contact the table near −335
mm. That establishes that the −345 mm value does not apply to the standard
fingers. The project continues to use the pinned client's checked-in 0 mm
`tcp_beyond_fingertips_mm` convention; the email alone does not prove the live
action/report frame, so do not infer or add a transform from either FAQ value.

The clarification identifies the intended setup, but software cannot prove
which fingers or controller frame are active on test day. Before the first
contact attempt, an assigned cell supervisor must still confirm:

1. what point `obs["tcp_pose"]` reports;
2. what point a `tcp` action commands;
3. UTOOL1 and UF0 currently loaded on the controller; and
4. the observed fingertip-to-table clearance at a high, non-contact target.

Start well above the table, approach in supervised increments, and stop before
the low pick height. Do not “test” the convention by touching the table. If the
installed hardware differs from the confirmed standard-finger setup, stop and
update the single shared transform and its tests from measured, supervisor-
approved values; do not add an unexplained offset in the hardware adapter.

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
| `obs["images"]` | Dictionary of requested `table`, `side`, and/or `wrist` images. Arrays are `(H, W, 3)` `uint8` **RGB**. The client supports requested sizes; the current policy requests uncropped 1920×1080 table RGB to preserve calibration pixels. Downloaded result videos may be resized/compressed and are not calibration evidence. |
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

The live hackathon site is authoritative for service motion limits. As of
October 30, 2026 it lists 150 mm/s, 15 mm per row, 60 mm queued ahead, and 10°
per row. The archived snapshot in `scilabDocumentation.txt` contains older
100 mm/s, 12 mm, 40 mm, and 8° values. Shared trajectories remain at 5 mm, while
the supervised hardware launcher's `smooth` profile uses the common lower
envelope of 100 mm/s, 10 mm rows and 8° rows. The explicit
`--motion-profile conservative` fallback restores 5 mm / 50 mm/s. Recheck the
live site and current mock before each request rather than increasing project
limits to whatever maximum is advertised.

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
- explicit object-size and provisional tool-point configuration. Isaac's
  preserved cube task remains 10 mm while hardware selects a measured 10/16 mm
  profile; parity means the active size is carried explicitly, not silently
  assumed equal.

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
- Centralize the numeric assumptions shown at the top of this document.
- Print the 10 mm Isaac assumption, explicit 10/16 mm hardware profile,
  resolved FAQ history, and still-unverified installed frame/fit state at every
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
uv run supervised_run.py hover --cube-mm 16 --mock --trials 1 --trial-s 2
```

Success is the literal `✓ Ready for the real robot` verdict with zero rejected
actions. This proves only formats and connection behavior. Mock images are
synthetic, and the mock's initial TCP pose is absent. The motion variable above
is therefore a **mock protocol bootstrap**, not physical authorization.

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

### Gate 4 — supervised stationary hold-command diagnostic

During an approved slot, with the operator in control:

- first ask whether the service supports a true observation-only/no-op mode;
- start the client and inspect camera names, image colour order, timestamps,
  joint convention, TCP pose, and table/bin/cube visibility;
- save calibration overlays without commanding a contact motion;
- confirm available cube/bin colours and exact placement names; and
- measure observation-to-action round trip and actual pose lag.

The checked-in fallback repeats one fresh measured TCP/gripper row. It does not
run perception or planning, but it is not read-only: latency and controller
tracking can still produce a small correction. Do not jog the robot while it is
active, and do not use it as an emergency stop.

Stop if calibration error consumes the available grasp clearance.

### Gate 5 — supervised hover confirmation

- With the operator's approved empty/clear workspace, command only a high,
  non-contact pose.
- Verify the observed TCP response and visible fingertip height.
- Approach in operator-approved increments, remaining well above the cube and
  table.
- Confirm the organizer-resolved standard-finger −335 mm convention against the
  installed fingers and active frames before enabling the provisional −332.6
  mm target.

This gate has been exercised on the physical cell, but it remains mandatory for
every new object size, integration profile, layout, or grasp strategy. The
project still lacks proof of complete full-arm/finger/hose/held-object swept
clearance; a prior successful route is not approval for a changed route.

Any disagreement between expected and observed frames returns the project to
Gate 1. Do not compensate by trial-and-error at table height.

### Gate 6 — one cube, one approved destination

- Use one measured 10 or 16 mm cube at a conservative known position. Prefer a
  clear `virtual-table` zone for the first relocation; use a bin only when the
  approved submission and actual scene contain it.
- Detect it from a real image, execute one pick/place, and independently verify
  lift and final placement.
- Save a trustworthy log containing all observations used for decisions,
  actions, timings, result, and any operator intervention.

This remains the first complete milestone for every materially changed policy.
Only after it succeeds should the team add several cubes, bins, changing
layouts, screws, or a learned scorer.

### Gate 7 — multi-object static sort

- Detect all cubes and bins needed as obstacles, not only the chosen target.
- Reject blocked, unreachable, stale, or ambiguous candidates.
- Compare the geometric baseline with the highest-predicted-complete-success
  selector on matched layouts with identical recovery rules. The current plan
  does not optimize or balance execution time.

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

- A visible cube centroid lies on the cube's top face. Map a top-face pixel at
  `surface_z + selected_cube_edge_m`, not at the table plane. That is +0.010 m
  for the 10 mm profile and +0.016 m for the 16 mm profile.
- The raw detector retains cube-like components across the calibrated image.
  The policy classifies bin observations first, then accepts pick targets only
  in the provisional robot-base pickup region `x=0.30..0.55 m`,
  `y=-0.30..0.30 m`. Non-bin components outside it, including weak ambiguous
  evidence, remain visible to fail-closed outcome verification. Overlay the
  bounds on the real table image, measure
  calibration residual and boundary jitter, and verify them throughout the
  region; projecting robot/hoses onto the cube-top plane makes this a heuristic,
  not a fixture self-mask.
- The shared known-size gate requires both inferred silhouette dimensions to
  reach 75% of the explicitly declared cube edge, while the dimension selected
  as the footprint edge must remain at or below 140% (7.5--14 mm for 10 mm;
  12--22.4 mm for 16 mm). Allowing the other dimension to be longer retains
  oblique views containing a side face. This rejected a compact 5.4 by 8.9 mm
  yellow robot/tool component in the exact Isaac pre-descent image without
  weakening completed-colour checks.
  It can also hide a partial/dim real cube, so validate it on native real images
  across the whole pickup region and keep missing evidence fail-closed. A
  future joint/calibration/CAD-derived robot mask must mark its dilated pixels
  unknown, never clear; an Isaac-only semantic mask is not hardware parity.
- The pickup region does not prove free space for the fingers or arm, does not
  see unknown-colour obstacles, and does not validate the swept motion. Keep
  those as separate collision and physical-integration gates.
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

The live event page says the real arm follows commanded targets “a little
behind,” but does not quantify that lag. Measure it from the actual run rather
than assuming a fixed value. Each returned row normally represents 0.1 seconds;
a long 100-row chunk can therefore create roughly ten seconds without a new
planning observation. Network latency from Brev adds to perception and
replanning time.

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
policy/config/calibration versions and explicit sim/hardware object-size banner
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
2. Implement the wrapper in `policy.py` and pass Gate 2 with the supervised
   localhost-only mock command shown above.
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
7. During the approved slot, use `uv run supervised_run.py hold`, then the
   explicit `hover --cube-mm {10|16}`, and only after review the documented
   `sort` command. The operator homes the robot, prepares each rollout,
   starts/stops it, and records faults/interventions. Plain `uv run run.py`
   invokes only the stationary hold-command diagnostic.
8. Use `uv run run.py --status` for status/results and
   `uv run run.py --download` for the recorded LeRobotDataset.

The standard documentation describes operator-prepared starting layouts, not
objects entering while the arm moves. Dynamic feeding requires explicit
organizer approval and a safe test protocol.

## Missing information checklist

Do not call the hardware integration complete until these are answered:

- [x] Organizer documentation discrepancy: the standard fingers contact the
      table near UF0 Z = −335 mm; the former −345 mm value was for another
      group's shorter fingers.
- [ ] Does the installed cell actually use the standard fingers and the pinned
      client's fingertip action/report convention? Verify active UTOOL1/UF0 at a
      high, non-contact pose.
- [ ] What UTOOL1, UF0, payload, and controller program are active for the
      approved event interface?
- [ ] Where is the custom aluminium hook-finger CAD, including notch dimensions
      and its relation to the reported TCP? The resource page currently exposes
      the standard SMC body model, which is insufficient for final clearance.
- [ ] Which measured 10 or 16 mm red/yellow/blue cubes are available, and are
      physical bins required or is the approved task `virtual-table`?
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

### Future screw hardware gate

The official default scene and the scripted Isaac demo both mention screws,
but the current physical policy is cube-specific. Before a screw submission,
complete camera-derived screw pose, task/destination mapping, multiple grasp
candidates, deterministic finger/approach/held-part/swept-clearance checks,
success-only scoring, required orientation execution, and lift/release
verification. Prove these using recorded real-image replay and camera-derived
Isaac trials, then request a separate one-screw supervised test. Do not relabel
a screw as a cube to bypass the adapter.

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
