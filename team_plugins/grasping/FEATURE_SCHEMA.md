# Team 8 grasp feature schema v1

Schema ID: `team8-preaction-grasp-features-v1`.

Every value is numeric, finite, and available before robot motion. Collection
and runtime use the exact order below. Changing a definition, unit, or order
requires a new schema and a newly evaluated model.

| Feature | Unit | Definition |
| --- | --- | --- |
| `object_is_screw` | boolean 0/1 | 1 for the catalog M6x30 screw, otherwise 0. |
| `axial_offset_m` | m | Screw pinch distance from the catalog head origin; 0 for cubes. |
| `yaw_offset_abs_rad` | rad | Absolute TipPose finger-length yaw offset from the observed cube yaw; 0 for the current screw family. The parallel-jaw closing axis is perpendicular to this yaw. |
| `required_opening_m` | m | Projected object width plus seating margin. |
| `opening_fraction` | ratio | Required opening divided by the 15.99 mm hook-open limit. |
| `pick_radius_m` | m | Radial distance of the candidate fingertip pick pose from robot base. |
| `release_radius_m` | m | Radial distance of the configured release fingertip pose. |
| `nearest_object_centre_m` | m | Nearest other observed object centre in the table plane. |
| `minimum_footprint_clearance_m` | m | Conservative signed clearance between the top-down finger envelope and observed object bounding circles. Finger width lies on the TipPose yaw axis; opening plus finger thickness lies on the perpendicular closing axis. |
| `neighbour_count` | count | Other objects in the complete generic observation. |
| `unknown_geometry_count` | count | Other objects without dimensions; any nonzero value is rejected before scoring. |
| `confidence` | ratio | Target detector confidence. |
| `pose_quality` | ratio | Target pose-quality score; missing evidence is rejected before scoring. |
| `visibility` | ratio | Target visibility; missing evidence is rejected before scoring. |
| `approach_distance_m` | m | Straight-line candidate approach distance. |

The supported "block" geometry is the configured 10 mm square cube class.
Rectangular, tilted, or uncatalogued block geometry is unsupported and rejected
before scoring. Unsupported generic-scene objects, missing target
geometry/evidence, class mismatches, cube-size disagreement, and
square-symmetry-aware cube pose disagreement are also rejected before scoring. The baseline
`cube_scene_to_observation` adapter is recognized only by its preserved
`legacy_square_yaw_rad` evidence; because that contract has no separate
pose-quality or visibility fields, detector confidence is conservatively used
for both. Other providers must supply the explicit fields.

The footprint clearance is a conservative preflight approximation, not a claim
of full link, hose, bin-rim, swept-volume, or collision-free planning.

The v1 `axial_offset_m` value is a signed/raw position, not distance from a
learned optimum. A fail/success/fail pattern at 12/16/20 mm is therefore
non-monotone in that feature and must trigger a reviewed nonlinear feature or
new schema/model design before training. The 12-attempt target-isolated pilot
is diagnostic and never marks itself trainer-ready.
