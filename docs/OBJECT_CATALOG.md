# Object catalog contract

This file defines the object-local coordinate convention shared by perception,
grasp generation, CAD geometry, and labels. A class is not integration-ready
until its entry is complete. A model class name alone is insufficient.

`Pose3D` represents `T_robot_base_object`: it rotates points/vectors from the
class's object-local frame into the robot-base frame, then translates them by
`position_base_m`. Quaternions are `(x, y, z, w)`. `dimensions_m` are lengths
along the catalog's local X, Y, and Z axes, in that order.

## `cube` (current compatibility class)

- Reference geometry: the configured 10 mm cube baseline. The hardware adapter
  also has a narrow, explicitly selected 16 mm vision profile, but the current
  cube projection/planner still uses configured task heights and clearances
  rather than deriving every value from `dimensions_m`. Treat that profile as
  a confirmed setup choice under the same fingertip/table assumptions, not as
  proof that the stack is generally dimension-aware. Any other measured size
  needs an audited task profile (or a size-specific catalog class) before it is
  mapped deliberately in `CUBE_TASK_CLASS_IDS`.
- Origin: geometric center of the cube.
- Local +Z: outward normal of the face designated as the top in the upright
  table task. In an upright observation it points approximately along robot
  base +Z, away from the table.
- Local +X and +Y: two adjacent square edges, forming a right-handed frame
  where `+X cross +Y = +Z`.
- Symmetry: rotations by 90 degrees about local +Z are equivalent for the
  current square-cube grasp. The compatibility adapter canonicalizes yaw to
  `[-pi/4, pi/4)`.
- Upright restriction: the current cube task rejects a local +Z axis tilted by
  more than 10 degrees from robot-base +Z. It does not silently discard roll or
  pitch.
- Current destination semantics: `sort_label` is one of `red`, `yellow`, or
  `blue`; class and destination are separate fields.
- Current executor support: vertical top-down grasp plus yaw only.

## `screw_iso10642_m6x30` (provisional integration class)

This entry freezes the exact supplied simulation asset and the intended handoff
name. It does **not** make the class hardware-ready. A vision provider must not
emit this `class_id` for robot execution until the object adapter and
feasibility checks described below exist.

### Verified CAD and simulation facts

- Human-readable part: ISO 10642 M6 x 30 countersunk screw, with a 90-degree
  head and nominal 4 mm hex socket. The supplied geometry does not model
  threads.
- Reference STEP: `reference-assets/models/ISO10642-M6x30-countersunk-screw.step`,
  SHA-256 `8a9fb00b8983813814f11c160725aa0e77941434ac7b43b47696a70f6b47a264`.
- Executed simulation mesh:
  `reference-assets/models/ISO10642-M6x30-countersunk-screw.stl`, SHA-256
  `6a842f1de0923083b0eb2a3a209eacae87cc83d4b62e15b768a0a8cd277cec2a`,
  1,910 triangles. Its millimetre bounds are
  `(-6, -5.998135, -30)` to `(6, 5.998135, 0)`.
- Catalog origin: center of the head face at CAD local `(0, 0, 0)`. The tip and
  shaft extend along local `-Z`; local `+X` and `+Y` span the head face and form
  the right-handed source frame. `dimensions_m` describes the local X/Y/Z
  bounding-box lengths: `(0.012, 0.01199627, 0.030)`.
- Current scripted-pose restriction: the screw lies on the table using
  `Rz(yaw) * Ry(-90 degrees)`, so local `-Z` is the horizontal screw axis and
  local `+X` points upward. This is one simulated pose family, not a general
  physical-pose assumption.
- Current simulated grasp family: a vertical pinch of the nominal 6 mm shaft at
  local `(0, 0, -0.016)` m. The configured simulator mass is 7.5 g; it is
  CAD-volume based, not a measurement of an event screw.
- Intended task label: `sort_label="red"`; both scripted screws go to the red
  destination. The mixed demo verifies all transformed mesh vertices inside
  the 35 mm bin pocket. It uses configured object poses, not camera perception.
- Axial symmetry is not yet frozen for training labels. The countersunk head is
  mostly rotationally symmetric but the hex socket may retain 60-degree detail;
  the team must choose and test a canonical convention before producing pose
  labels rather than silently treating axial roll as either unique or
  continuous.

### Unverified physical facts and required integration

- The profile-unset HSV cube provider cannot detect this screw. The opt-in
  hybrid vision profile can emit a restricted experimental observation for an
  isolated, lying, dark screw on the bright table, but its pose is not yet
  physically validated and the shared executor cannot act on it. The scripted
  screw demo still has no camera or training-data mode. Required evidence
  includes a calibrated camera-derived pose, visibility/occlusion quality,
  held-out real replay, and a documented failure threshold.
- The real screw's exact finish, mass, dimensions/tolerances, available poses,
  and match to this CAD have not been measured. CAD truth may label simulation
  but may not repair a real camera result.
- The custom fingers' shaft contact, closure, friction, retention through the
  complete transport, release, and collision clearance have not been validated
  on hardware. The one scripted pinch is a candidate, not proof of feasibility.
- The generic object target/destination adapter and screw-capable executor do
  not yet exist. Do not add this class to `CUBE_TASK_CLASS_IDS`; keep it visible
  as unsupported until those pieces and their tests are implemented.
- The physical red destination (bin versus an approved virtual-table footprint)
  must be frozen for the screw trial. A simulation red-bin placement does not
  choose the physical setup automatically.

## New-class template

Before adding a screw, vial, nail, or another part, record:

1. Stable `class_id` and human-readable part/revision name.
2. Exact CAD/reference file and its hash or version.
3. Object-local origin and the physical meaning of +X, +Y, and +Z.
4. Pose transform direction (`T_robot_base_object`, never its inverse).
5. Which local axes `dimensions_m` describe.
6. Rotational/reflection symmetries and the canonical pose rule used in labels.
7. Whether poses are full 6-DoF, table-upright, or otherwise constrained.
8. Allowed gripper/finger geometry and candidate grasp families.
9. Destination semantics and any allowed `sort_label` values.
10. Minimum perception evidence/quality needed before the object may be picked.
11. Deterministic fit, approach, collision, retention, and release checks.
12. Representative training/evaluation conditions and known unsupported cases.

Do not add an unfinished class to `CUBE_TASK_CLASS_IDS` merely to make it reach
the old cube planner. Keep it visible as unsupported until its task adapter and
feasibility checker implement the catalog entry.
