# Isaac camera-to-policy bridge

`camera_sort.py` is the non-Isaac-dependent bridge for running the shared
colour sorter from rendered pixels instead of configured cube poses. It accepts
only:

- a raw native-resolution RGB frame;
- the fixed table camera calibration read from the USD stage;
- the shared `SortConfig`; and
- a scene ID and capture timestamp.

It returns the same `SceneDetections` and `PickPlacePlan` types used by the
FANUC hardware policy. Simulator object transforms are intentionally absent
from the runtime API. Ground truth belongs in a separate evaluator.

## Isaac integration hook

After the table camera is initialized and the renderer has produced a frame:

```python
from isaac_bridge import CalibratedPinholeCamera, IsaacCameraSortAdapter
from sorting_core import load_config

# Prefer metadata read from the live USD camera prim, using the same schema as
# digital-twin-setup/capture_scene.py writes to camera.json.
config = load_config()
camera_model = CalibratedPinholeCamera.from_capture_metadata(
    camera_metadata,
    # camera_to_world and table_top_z in capture metadata are USD-world values.
    # This explicit base-frame surface height is required for conversion.
    surface_base_z_m=config.frames.surface_base_z_m,
)
adapter = IsaacCameraSortAdapter(camera_model, config)

rgba = camera.get_rgba()
decision = adapter.observe_and_plan(
    rgba[:, :, :3].astype("uint8"),
    scene_id=f"isaac:{frame_number}",
    capture_time_s=float(world.current_time),
)
```

For a fixed camera, `CalibratedPinholeCamera.from_look_at(...)` can construct
the same pinhole model from the runtime `eye`, `look_at`, `up`, resolution,
focal length, and aperture. Reading the transform from USD is safer because it
also captures parent and stage transforms. Never pass `camera_to_world`
directly as though it were `camera_to_base`: the supplied stage puts the robot
base at a nonzero world Z, which would produce incorrect ray/plane
intersections. The metadata constructor fails closed unless the canonical
surface height needed for that conversion is supplied.

The existing scripted demo remains the default known-good physics/rendering
smoke test. Its opt-in `--camera-perception` mode uses this bridge at initial
observation and again at the pre-descent revalidation gate, then feeds the
camera-derived target pose into the existing checked Isaac execution path.
`--training-data` adds the separate domain-randomization/data contract while
leaving the default path unchanged.

## Deliberate limitations

- This is the narrow red/yellow/blue 10 mm cube baseline, not Cosmos.
- The image must be raw RGB without the demo's text overlay.
- It assumes one unambiguous cube of each configured colour. Missing, merged,
  duplicate, wrongly sized, or unknown detections fail closed in the planner.
- It uses the ideal pinhole model rendered by Isaac. The hardware projection
  still uses the measured real-camera distortion calibration.
- It does not provide a complete swept-volume collision planner, temporal
  tracking, or persistent cross-frame IDs. The demo's physical post-place
  verification uses simulator truth strictly as evaluator output.
- Camera-derived square yaw is only meaningful modulo 90 degrees.
