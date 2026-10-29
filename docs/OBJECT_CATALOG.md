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
