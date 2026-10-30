# Golden cross-team scene fixtures

These are **hand-authored, camera-contract-shaped** JSON examples for the
handoff between computer vision, grasping, and lead integration. They use the
exact version-1 `SceneObservation` serialization from
`sorting_core.perception`; they are not camera-inference outputs, training
data, simulator ground truth, or evidence of model performance.

| Fixture | Expected behavior |
|---|---|
| `valid_cube_scene.v1.json` | Three well-formed cube records validate, project into the compatibility cube task, and exercise grasp-planning contracts. |
| `valid_screw_scene.v1.json` | The supplied M6x30 screw validates and round-trips, but remains explicitly unsupported by the current cube task adapter. |
| `mixed_cube_screw_scene.v1.json` | A valid cube and screw coexist; the cube projects, the screw remains unsupported, and the legacy checker rejects execution rather than treating it as clear space. |
| `stale_cube_scene.v1.json` | Structurally valid JSON presented to a newer request; stale-timestamp validation must reject it. |
| `occluded_mixed_scene.v1.json` | A cube plus an unresolved occluded object validate. The unknown object stays visible and blocks the legacy checker. |
| `invalid_bad_quaternion_scene.v1.json` | Deliberately malformed orientation data must fail strict parsing before planning. |

The covered boundary is deliberately narrow: strict JSON parsing and
serialization, request/calibration attribution, generic-to-cube task
projection, preservation of unsupported objects, deterministic feasibility,
and highest-success candidate selection. These tests do **not** run a camera
model, validate calibration accuracy, establish physical reach/collision
clearance, or prove an object can be grasped. The shared test uses the legacy
pipeline and small test doubles; it does not run a selected team's provider or
learned scorer. Each team should add a team-owned test that feeds the relevant
golden fixture through its actual factory when that is possible without model
downloads or GPU inference.

All positions and dimensions are metres in `robot_base`. Orientations are unit
quaternions in `(x, y, z, w)` order. Every detection cites the fictional
`golden-table-calibration-v1`; this exercises attribution only. A fixture
timestamp is backend time, not wall-clock time. The screw origin follows the
catalog head-center convention at table surface + 6.5 mm (`z=0.0014 m`) for
the documented known-valid lying reference pose.

Run the focused handoff test from the repository root:

```bash
uv run --project hardware/fanuc_cell \
  python -m unittest tests.test_team_handoff -v
```

The occluded fixture is blocked because its unresolved object has an unsupported
class and the legacy checker rejects every unsupported object. Its uncertainty
metadata is preserved but is **not** interpreted as geometric uncertainty by
that checker.

When the contract changes, add a new versioned fixture and an explicit
migration test. Do not silently rewrite version-1 files to fit one model.
