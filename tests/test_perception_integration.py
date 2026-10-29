import json
import math
from dataclasses import replace
from pathlib import Path
import unittest

from sorting_core.contracts import CubeDetection, SceneDetections
from sorting_core.perception import (
    CameraFrame,
    DetectedObject,
    DepthEncoding,
    PerceptionRequest,
    Pose3D,
    RobotState,
    SceneObservation,
    VisionProvider,
    scene_observation_from_dict,
    scene_observation_to_dict,
    validate_scene_for_request,
)
from sorting_core.task_perception import (
    cube_scene_from_observation,
    cube_scene_to_observation,
)


def _object(
    *,
    track_id="object-1",
    class_id="cube",
    sort_label="red",
    capture_time_s=12.5,
):
    return DetectedObject(
        track_id=track_id,
        class_id=class_id,
        sort_label=sort_label,
        capture_time_s=capture_time_s,
        pose=Pose3D.from_yaw((0.41, -0.10, 0.003), math.radians(17.0)),
        dimensions_m=(0.016, 0.016, 0.016),
        confidence=0.91,
        pose_quality=0.82,
        visibility=0.75,
        source_camera_ids=("table", "side"),
        attributes={"pixel_center_uv": [901.25, 431.5], "occluded": False},
    )


class PerceptionContractTests(unittest.TestCase):
    def test_checked_in_json_schema_requires_runtime_provenance_lists(self):
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "schemas"
            / "scene-observation.v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["calibration_ids"]["minItems"], 1
        )
        self.assertEqual(
            schema["$defs"]["detectedObject"]["properties"]
            ["source_camera_ids"]["minItems"],
            1,
        )

    def test_pose_requires_finite_unit_quaternion_and_constructs_yaw(self):
        pose = Pose3D.from_yaw((0.4, -0.1, 0.02), math.pi / 3.0)
        self.assertEqual(pose.position_base_m, (0.4, -0.1, 0.02))
        self.assertAlmostEqual(pose.orientation_xyzw[2], 0.5)
        self.assertAlmostEqual(pose.orientation_xyzw[3], math.sqrt(3.0) / 2.0)

        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            Pose3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 2.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            Pose3D((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def test_request_preserves_multi_camera_timing_and_same_observation_state(self):
        table_rgb = object()
        side_rgb = object()
        request = PerceptionRequest(
            scene_id="trial-1:scene-2",
            capture_time_s=20.0,
            frames=(
                CameraFrame("table", 19.97, "table-2026-10-07", rgb=table_rgb),
                CameraFrame("side", 19.91, "side-2026-10-08", rgb=side_rgb),
            ),
            robot_state=RobotState(
                arm_joints_rad=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                tcp_pose_base_m_wpr_rad=(
                    0.420,
                    -0.100,
                    0.070,
                    -math.pi,
                    0.0,
                    0.0,
                ),
                gripper_state=0.0,
            ),
            metadata={"observation_source": "official-cell"},
        )
        self.assertIs(request.frame("table").rgb, table_rgb)
        self.assertIs(request.frame("side").rgb, side_rgb)
        self.assertNotEqual(
            request.frame("table").capture_time_s,
            request.frame("side").capture_time_s,
        )
        self.assertEqual(len(request.robot_state.arm_joints_rad), 6)
        self.assertAlmostEqual(
            request.robot_state.tcp_pose_base_m_wpr_rad[0], 0.420
        )
        with self.assertRaises(KeyError):
            request.frame("wrist")
        with self.assertRaisesRegex(ValueError, "duplicate camera"):
            replace(request, frames=(request.frames[0], request.frames[0]))

        with self.assertRaisesRegex(ValueError, "6 finite"):
            RobotState(arm_joints_rad=(0.0,) * 7)

    def test_rgb_and_depth_use_distinct_calibration_ids(self):
        with self.assertRaisesRegex(ValueError, "depth frame requires"):
            CameraFrame(
                "table", 20.0, "table-rgb-cal", rgb=object(), depth=object()
            )
        with self.assertRaisesRegex(ValueError, "DepthEncoding"):
            CameraFrame(
                "table",
                20.0,
                "table-rgb-cal",
                rgb=object(),
                depth=object(),
                depth_calibration_id="table-depth-cal",
            )
        frame = CameraFrame(
            "table",
            20.0,
            "table-rgb-cal",
            rgb=object(),
            depth=object(),
            depth_calibration_id="table-depth-cal",
            depth_encoding=DepthEncoding(
                meters_per_unit=0.001,
                invalid_values=(0.0,),
                aligned_to_rgb=False,
            ),
        )
        self.assertAlmostEqual(frame.depth_encoding.meters_per_unit, 0.001)
        self.assertFalse(frame.depth_encoding.aligned_to_rgb)
        request = PerceptionRequest("scene", 20.0, (frame,))
        scene = SceneObservation(
            scene_id="scene",
            capture_time_s=20.0,
            objects=(
                replace(
                    _object(capture_time_s=20.0),
                    source_camera_ids=("table",),
                ),
            ),
            provider_name="rgbd-provider",
            provider_version="1",
            calibration_ids=("table-rgb-cal", "table-depth-cal"),
        )
        self.assertIs(validate_scene_for_request(request, scene), scene)

    def test_scene_rejects_wrong_frame_duplicate_tracks_and_mixed_timestamps(self):
        first = _object()
        with self.assertRaisesRegex(ValueError, "coordinate_frame"):
            SceneObservation(
                scene_id="scene",
                capture_time_s=12.5,
                coordinate_frame="camera_table",
                objects=(first,),
                provider_name="test",
                provider_version="1",
            )
        with self.assertRaisesRegex(ValueError, "duplicate track"):
            SceneObservation(
                scene_id="scene",
                capture_time_s=12.5,
                objects=(first, first),
                provider_name="test",
                provider_version="1",
            )
        with self.assertRaisesRegex(ValueError, "scene capture timestamp"):
            SceneObservation(
                scene_id="scene",
                capture_time_s=12.5,
                objects=(replace(first, capture_time_s=12.6),),
                provider_name="test",
                provider_version="1",
            )

    def test_scene_json_round_trip_is_strict_and_json_serializable(self):
        scene = SceneObservation(
            scene_id="scene-12",
            capture_time_s=12.5,
            objects=(_object(),),
            provider_name="example-model",
            provider_version="weights-sha-1234",
            calibration_ids=("table-2026-10-07", "side-2026-10-08"),
        )
        encoded = scene_observation_to_dict(scene)
        json.dumps(encoded, allow_nan=False)
        decoded = scene_observation_from_dict(encoded)
        self.assertEqual(scene, decoded)
        self.assertEqual(
            scene_observation_to_dict(decoded),
            encoded,
        )

        encoded["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown unexpected"):
            scene_observation_from_dict(encoded)

        encoded.pop("unexpected")
        encoded["schema_version"] = 1.0
        with self.assertRaisesRegex(ValueError, "schema version"):
            scene_observation_from_dict(encoded)

    def test_nested_json_schema_rejects_unknown_pose_fields(self):
        scene = SceneObservation(
            scene_id="scene-12",
            capture_time_s=12.5,
            objects=(_object(),),
            provider_name="example-model",
            provider_version="1",
        )
        encoded = scene_observation_to_dict(scene)
        encoded["objects"][0]["pose"]["degrees"] = [0, 0, 17]
        with self.assertRaisesRegex(ValueError, "unknown degrees"):
            scene_observation_from_dict(encoded)

    def test_provider_protocol_is_structural(self):
        scene = SceneObservation(
            scene_id="scene-12",
            capture_time_s=12.5,
            objects=(_object(),),
            provider_name="fake",
            provider_version="1",
        )

        class FakeProvider:
            def observe(self, request):
                return scene

        self.assertIsInstance(FakeProvider(), VisionProvider)

    def test_runtime_boundary_rejects_stale_or_misattributed_provider_output(self):
        request = PerceptionRequest(
            scene_id="scene-12",
            capture_time_s=12.5,
            frames=(
                CameraFrame("table", 12.49, "table-cal", rgb=object()),
                CameraFrame("side", 12.47, "side-cal", rgb=object()),
            ),
        )
        scene = SceneObservation(
            scene_id=request.scene_id,
            capture_time_s=request.capture_time_s,
            objects=(_object(),),
            provider_name="fake",
            provider_version="1",
            calibration_ids=("table-cal", "side-cal"),
        )
        self.assertIs(validate_scene_for_request(request, scene), scene)
        with self.assertRaisesRegex(ValueError, "different scene_id"):
            validate_scene_for_request(
                request, replace(scene, scene_id="another-scene")
            )
        with self.assertRaisesRegex(ValueError, "stale capture"):
            validate_scene_for_request(
                request,
                replace(
                    scene,
                    capture_time_s=12.6,
                    objects=(replace(scene.objects[0], capture_time_s=12.6),),
                ),
            )
        with self.assertRaisesRegex(ValueError, "unavailable calibration"):
            validate_scene_for_request(
                request, replace(scene, calibration_ids=("unknown-cal",))
            )
        with self.assertRaisesRegex(ValueError, "identify its calibration"):
            validate_scene_for_request(
                request, replace(scene, calibration_ids=())
            )
        with self.assertRaisesRegex(ValueError, "source camera"):
            validate_scene_for_request(
                request,
                replace(
                    scene,
                    objects=(replace(scene.objects[0], source_camera_ids=()),),
                ),
            )
        side_only = replace(
            scene,
            objects=(
                replace(scene.objects[0], source_camera_ids=("side",)),
            ),
            calibration_ids=("table-cal",),
        )
        with self.assertRaisesRegex(
            ValueError, "side without citing one of its calibrations"
        ):
            validate_scene_for_request(request, side_only)


class CubeTaskPerceptionAdapterTests(unittest.TestCase):
    def test_legacy_cube_scene_round_trip_preserves_every_field(self):
        cubes = (
            CubeDetection(
                track_id="scene:red:0",
                colour="red",
                capture_time_s=33.0,
                position_base_m=(0.41, -0.10, 0.003),
                yaw_rad=math.radians(17.0),
                confidence=0.88,
                pixel_center_uv=(900.5, 430.25),
                mask_area_px=196.0,
            ),
            CubeDetection(
                track_id="scene:blue:0",
                colour="blue",
                capture_time_s=33.0,
                position_base_m=(0.46, -0.16, 0.003),
                yaw_rad=math.radians(-31.0),
                confidence=0.97,
            ),
        )
        legacy = SceneDetections("scene", 33.0, cubes)
        generic = cube_scene_to_observation(
            legacy,
            dimensions_m=0.016,
            class_id="cube_16mm",
            provider_name="hsv-baseline",
            provider_version="current",
            calibration_ids=("table-v1",),
            source_camera_ids=("table",),
        )
        projection = cube_scene_from_observation(
            generic, class_ids=("cube_16mm",)
        )
        self.assertEqual(projection.scene, legacy)
        self.assertEqual(projection.unsupported_objects, ())
        self.assertTrue(
            all(value.dimensions_m == (0.016, 0.016, 0.016) for value in generic.objects)
        )

    def test_legacy_round_trip_preserves_square_yaw_boundary_exactly(self):
        for index, yaw in enumerate(
            (
                -math.pi / 4.0,
                math.nextafter(-math.pi / 4.0, -math.inf),
                math.nextafter(math.pi / 4.0, math.inf),
            )
        ):
            cube = CubeDetection(
                track_id=f"boundary-{index}",
                colour="red",
                capture_time_s=33.0,
                position_base_m=(0.41, -0.10, 0.003),
                yaw_rad=yaw,
            )
            legacy = SceneDetections("scene", 33.0, (cube,))
            recovered = cube_scene_from_observation(
                cube_scene_to_observation(legacy, dimensions_m=0.016)
            ).scene
            self.assertEqual(recovered, legacy)

    def test_tilted_cube_is_not_silently_projected_to_upright_yaw(self):
        tilt = math.radians(20.0)
        tilted = replace(
            _object(track_id="tilted-cube"),
            pose=Pose3D(
                (0.41, -0.10, 0.003),
                (math.sin(tilt / 2.0), 0.0, 0.0, math.cos(tilt / 2.0)),
            ),
        )
        scene = SceneObservation(
            scene_id="tilted-scene",
            capture_time_s=12.5,
            objects=(tilted,),
            provider_name="six-dof-provider",
            provider_version="1",
        )
        with self.assertRaisesRegex(ValueError, "upright-task tilt limit"):
            cube_scene_from_observation(scene)

        tilted_with_legacy_hint = replace(
            scene,
            objects=(
                replace(
                    tilted,
                    attributes={"legacy_square_yaw_rad": 0.0},
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "upright-task tilt limit"):
            cube_scene_from_observation(tilted_with_legacy_hint)

    def test_unsupported_objects_are_returned_not_silently_discarded(self):
        cube = _object(track_id="cube-1", class_id="cube", sort_label="yellow")
        screw = _object(
            track_id="screw-1", class_id="iso10642_m6x30", sort_label="red"
        )
        scene = SceneObservation(
            scene_id="mixed-scene",
            capture_time_s=12.5,
            objects=(cube, screw),
            provider_name="mixed-model",
            provider_version="1",
        )
        projection = cube_scene_from_observation(scene)
        self.assertEqual([value.track_id for value in projection.scene.cubes], ["cube-1"])
        self.assertEqual(projection.unsupported_objects, (screw,))

    def test_supported_object_requires_sort_label(self):
        scene = SceneObservation(
            scene_id="scene",
            capture_time_s=12.5,
            objects=(_object(sort_label=None),),
            provider_name="test",
            provider_version="1",
        )
        with self.assertRaisesRegex(ValueError, "has no sort_label"):
            cube_scene_from_observation(scene)


if __name__ == "__main__":
    unittest.main()
