"""Use rendered Isaac RGB observations for the shared colour-sort policy.

Runtime decisions in this module are derived only from image pixels, a camera
calibration, and the checked-in task configuration.  Simulator object poses do
not appear in the adapter API.  They may be compared with the returned
``SceneDetections`` by a separate evaluator, but must not be used to repair a
missed detection or choose an action.

The matrix convention matches the camera metadata written by the supplied
Isaac capture script: a row-major USD transform where camera-local row vectors
map into the USD world frame.  The capture metadata is *not* already in the
robot-base frame.  :meth:`CalibratedPinholeCamera.from_capture_metadata`
performs that conversion explicitly.  USD cameras look along local ``-Z``
with local ``+Y`` up.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sorting_core import (
    DestinationMaskError,
    PickPlacePlan,
    SceneDetections,
    SortConfig,
    build_colour_sort_plans,
    filter_scene_to_pickup_region,
    ignore_configured_bucket_pixels,
)
from sorting_core.grasping import GraspPlanningPipeline
from sorting_core.perception import (
    CameraFrame,
    PerceptionRequest,
    RobotState,
    SceneObservation,
    VisionProvider,
    validate_scene_for_request,
)
from sorting_core.task_perception import cube_scene_from_observation
from sorting_core.team_integrations import (
    CUBE_TASK_CLASS_IDS,
    create_grasp_pipeline,
    create_vision_provider,
)


class IsaacObservationError(ValueError):
    """Raised when an Isaac image or camera calibration is unusable."""


def _positive_finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise IsaacObservationError(f"{name} must be positive and finite")
    return number


def _resolution(value: Sequence[Any]) -> tuple[int, int]:
    try:
        width, height = value
    except (TypeError, ValueError) as exc:
        raise IsaacObservationError("resolution must contain width and height") from exc
    if isinstance(width, bool) or isinstance(height, bool):
        raise IsaacObservationError("resolution must contain integer pixels")
    if int(width) != width or int(height) != height or width < 2 or height < 2:
        raise IsaacObservationError("resolution must contain integer pixels >= 2")
    return int(width), int(height)


@dataclass(frozen=True)
class CalibratedPinholeCamera:
    """A fixed Isaac camera expressed in the canonical robot-base frame.

    ``camera_to_base_row_major`` uses the same row-vector layout as the 4x4
    matrix emitted by
    ``UsdGeom.Xformable(...).ComputeLocalToWorldTransform(0)``, after any
    required world-to-base conversion. Translation is therefore in row three,
    not column three. This class models the pinhole image Isaac renders; it
    intentionally does not add the real camera's lens distortion or sensor
    noise.
    """

    resolution_px: tuple[int, int]
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    camera_to_base_row_major: tuple[tuple[float, float, float, float], ...]
    default_plane_base_z_m: float

    def __post_init__(self) -> None:
        resolution = _resolution(self.resolution_px)
        focal = _positive_finite(self.focal_length_mm, "focal length")
        horizontal = _positive_finite(
            self.horizontal_aperture_mm, "horizontal aperture"
        )
        vertical = _positive_finite(self.vertical_aperture_mm, "vertical aperture")
        plane_z = float(self.default_plane_base_z_m)
        if not math.isfinite(plane_z):
            raise IsaacObservationError("default plane height must be finite")

        matrix = np.asarray(self.camera_to_base_row_major, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise IsaacObservationError(
                "camera_to_base_row_major must be a finite 4x4 matrix"
            )
        rotation = matrix[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
            raise IsaacObservationError("camera rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
            raise IsaacObservationError("camera rotation must be right-handed")
        if not np.allclose(matrix[:3, 3], 0.0, atol=1e-8) or not np.allclose(
            matrix[3, 3], 1.0, atol=1e-8
        ):
            raise IsaacObservationError(
                "camera matrix must use the USD row-vector transform convention"
            )

        frozen_matrix = tuple(tuple(float(v) for v in row) for row in matrix)
        object.__setattr__(self, "resolution_px", resolution)
        object.__setattr__(self, "focal_length_mm", focal)
        object.__setattr__(self, "horizontal_aperture_mm", horizontal)
        object.__setattr__(self, "vertical_aperture_mm", vertical)
        object.__setattr__(self, "camera_to_base_row_major", frozen_matrix)
        object.__setattr__(self, "default_plane_base_z_m", plane_z)

    @property
    def _matrix(self) -> np.ndarray:
        return np.asarray(self.camera_to_base_row_major, dtype=float)

    @property
    def eye_base_m(self) -> np.ndarray:
        return self._matrix[3, :3].copy()

    @property
    def _tan_half_horizontal(self) -> float:
        return self.horizontal_aperture_mm / (2.0 * self.focal_length_mm)

    @property
    def _tan_half_vertical(self) -> float:
        return self.vertical_aperture_mm / (2.0 * self.focal_length_mm)

    @classmethod
    def from_capture_metadata(
        cls,
        metadata: Mapping[str, Any] | str | Path,
        *,
        surface_base_z_m: float | None = None,
    ) -> "CalibratedPinholeCamera":
        """Build from the camera JSON schema written by ``capture_scene.py``.

        ``capture_scene.py`` records ``camera_to_world``, the table surface's
        world Z in ``table_top_z``, and the robot base's world X/Y in
        ``robot_base_xy``.  The caller must also provide the same surface
        height in the canonical robot-base frame.  These values locate the
        robot-base origin in USD world coordinates, after which this method
        converts the camera translation to robot-base coordinates.  Omitting
        the explicit surface height fails closed rather than silently treating
        world coordinates as robot-base coordinates.

        This conversion relies on the supplied twin's documented aligned
        world/base axes.  Reading the live transform is still preferred to
        recomputing it from nominal ``eye/look_at/up`` values because it catches
        camera-parent and stage-transform changes.
        """

        if isinstance(metadata, (str, Path)):
            try:
                parsed = json.loads(Path(metadata).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise IsaacObservationError(
                    f"cannot read Isaac camera metadata {metadata}: {exc}"
                ) from exc
        else:
            parsed = metadata
        if surface_base_z_m is None:
            raise IsaacObservationError(
                "surface_base_z_m is required to convert camera_to_world "
                "metadata into the robot-base frame"
            )
        try:
            surface_base_z = float(surface_base_z_m)
            table_surface_world_z = float(parsed["table_top_z"])
            robot_base_xy = np.asarray(parsed["robot_base_xy"], dtype=float)
            camera_to_world = np.asarray(parsed["camera_to_world"], dtype=float)
            if not math.isfinite(surface_base_z):
                raise IsaacObservationError("surface_base_z_m must be finite")
            if robot_base_xy.shape != (2,) or not np.all(np.isfinite(robot_base_xy)):
                raise IsaacObservationError(
                    "camera metadata robot_base_xy must contain two finite values"
                )
            if camera_to_world.shape != (4, 4) or not np.all(
                np.isfinite(camera_to_world)
            ):
                raise IsaacObservationError(
                    "camera metadata camera_to_world must be a finite 4x4 matrix"
                )
            if not math.isfinite(table_surface_world_z):
                raise IsaacObservationError(
                    "camera metadata table_top_z must be finite"
                )

            # In this twin the robot-base axes are aligned with USD world.  A
            # point on the table surface has base Z=surface_base_z_m, so the
            # base origin's world Z follows without assuming it is world Z=0.
            base_origin_world = np.array(
                [
                    robot_base_xy[0],
                    robot_base_xy[1],
                    table_surface_world_z - surface_base_z,
                ],
                dtype=float,
            )
            camera_to_base = camera_to_world.copy()
            camera_to_base[3, :3] -= base_origin_world
            return cls(
                resolution_px=tuple(parsed["resolution"]),
                focal_length_mm=parsed["focal_length_mm"],
                horizontal_aperture_mm=parsed["horizontal_aperture_mm"],
                vertical_aperture_mm=parsed["vertical_aperture_mm"],
                camera_to_base_row_major=tuple(
                    tuple(row) for row in camera_to_base
                ),
                default_plane_base_z_m=surface_base_z,
            )
        except IsaacObservationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise IsaacObservationError(
                f"incomplete Isaac camera metadata: missing or invalid {exc}"
            ) from exc

    @classmethod
    def from_look_at(
        cls,
        *,
        eye_base_m: Sequence[float],
        look_at_base_m: Sequence[float],
        up_base: Sequence[float],
        resolution_px: Sequence[int],
        focal_length_mm: float,
        horizontal_aperture_mm: float,
        default_plane_base_z_m: float,
    ) -> "CalibratedPinholeCamera":
        """Construct the fixed camera used by an Isaac ``eye/look_at/up`` spec.

        This is a convenient setup hook.  Production capture should prefer
        :meth:`from_capture_metadata`, which uses the transform actually read
        from the USD stage.
        """

        eye = np.asarray(eye_base_m, dtype=float)
        target = np.asarray(look_at_base_m, dtype=float)
        up_hint = np.asarray(up_base, dtype=float)
        if (
            eye.shape != (3,)
            or target.shape != (3,)
            or up_hint.shape != (3,)
            or not np.all(np.isfinite([*eye, *target, *up_hint]))
        ):
            raise IsaacObservationError("eye, look_at and up must be finite 3-vectors")
        forward = target - eye
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-9:
            raise IsaacObservationError("camera eye and look_at must differ")
        forward /= forward_norm
        right = np.cross(forward, up_hint)
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1e-9:
            raise IsaacObservationError("camera up must not be parallel to view axis")
        right /= right_norm
        up = np.cross(right, forward)

        matrix = np.eye(4)
        matrix[:3, :3] = (right, up, -forward)
        matrix[3, :3] = eye
        width, height = _resolution(resolution_px)
        vertical_aperture = float(horizontal_aperture_mm) * height / width
        return cls(
            resolution_px=(width, height),
            focal_length_mm=focal_length_mm,
            horizontal_aperture_mm=horizontal_aperture_mm,
            vertical_aperture_mm=vertical_aperture,
            camera_to_base_row_major=tuple(tuple(row) for row in matrix),
            default_plane_base_z_m=default_plane_base_z_m,
        )

    def project(self, point_base_m: Sequence[float]) -> np.ndarray | None:
        """Project one robot-base point to continuous ``(u, v)`` pixels."""

        point = np.asarray(point_base_m, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise IsaacObservationError("point must contain three finite coordinates")
        relative = point - self.eye_base_m
        camera = self._matrix[:3, :3] @ relative
        if camera[2] >= -1e-9:  # USD camera looks along local -Z.
            return None
        x_ndc = camera[0] / (-camera[2]) / self._tan_half_horizontal
        y_ndc = camera[1] / (-camera[2]) / self._tan_half_vertical
        width, height = self.resolution_px
        return np.array(
            [(x_ndc + 1.0) * 0.5 * width, (1.0 - y_ndc) * 0.5 * height]
        )

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the base-frame ray through continuous pixel coordinates."""

        u, v = float(u), float(v)
        if not math.isfinite(u) or not math.isfinite(v):
            raise IsaacObservationError("pixel coordinates must be finite")
        width, height = self.resolution_px
        x_ndc = u / width * 2.0 - 1.0
        y_ndc = 1.0 - v / height * 2.0
        direction_camera = np.array(
            [
                x_ndc * self._tan_half_horizontal,
                y_ndc * self._tan_half_vertical,
                -1.0,
            ]
        )
        direction_base = direction_camera @ self._matrix[:3, :3]
        direction_base /= np.linalg.norm(direction_base)
        return self.eye_base_m, direction_base

    def to_plane(self, u: float, v: float, z: float | None = None) -> np.ndarray:
        """Intersect a pixel ray with a horizontal robot-base plane."""

        plane_z = self.default_plane_base_z_m if z is None else float(z)
        if not math.isfinite(plane_z):
            raise IsaacObservationError("plane height must be finite")
        origin, direction = self.ray(u, v)
        if abs(float(direction[2])) <= 1e-9:
            raise IsaacObservationError("camera ray is parallel to the requested plane")
        distance = (plane_z - origin[2]) / direction[2]
        if distance <= 0.0:
            raise IsaacObservationError("requested plane is behind the camera")
        return origin + direction * distance


@dataclass(frozen=True)
class CameraSortDecision:
    """One perception result and the plans derived exclusively from it."""

    raw_scene: SceneDetections
    scene: SceneDetections
    plans: tuple[PickPlacePlan, ...]
    # Appended and optional so existing callers that construct this public
    # compatibility result positionally remain valid.
    observation: SceneObservation | None = None


class IsaacCameraSortAdapter:
    """Convert one native Isaac RGB frame into shared detections and plans."""

    def __init__(
        self,
        camera: CalibratedPinholeCamera,
        config: SortConfig,
        *,
        perception_provider: VisionProvider | None = None,
        grasp_pipeline: GraspPlanningPipeline | None = None,
        vision_checkpoint: str | None = None,
        grasp_checkpoint: str | None = None,
        ignore_bucket_pixels: bool = True,
    ):
        self.camera = camera
        self.config = config.validate()
        if not isinstance(ignore_bucket_pixels, bool):
            raise TypeError("ignore_bucket_pixels must be a bool")
        self.ignore_bucket_pixels = ignore_bucket_pixels
        self.perception_provider = (
            perception_provider
            if perception_provider is not None
            else create_vision_provider(
                config=self.config,
                camera_projectors={"table": camera},
                calibration_ids={"table": "isaac-live-camera-metadata"},
                checkpoint=vision_checkpoint,
            )
        )
        self.grasp_pipeline = (
            grasp_pipeline
            if grasp_pipeline is not None
            else create_grasp_pipeline(
                config=self.config,
                checkpoint=grasp_checkpoint,
            )
        )

    def observe_request(self, request: PerceptionRequest) -> SceneObservation:
        """Run an already assembled single- or multi-camera request.

        This is the backend-neutral seam for team providers that consume more
        than Isaac's legacy single table-camera convenience method.  The
        caller remains responsible for supplying calibrated frames and any
        joint/TCP state needed to transform a moving wrist camera.
        """

        if not isinstance(request, PerceptionRequest):
            raise TypeError("request must be a PerceptionRequest")
        try:
            provider_request = self._request_with_bucket_pixels_ignored(request)
            result = self.perception_provider.observe(provider_request)
            return validate_scene_for_request(provider_request, result)
        except IsaacObservationError:
            raise
        except Exception as exc:
            raise IsaacObservationError(
                f"vision provider failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _request_with_bucket_pixels_ignored(
        self, request: PerceptionRequest
    ) -> PerceptionRequest:
        """Replace only configured-bin pixels in the table frame.

        The Isaac colour-sort scene always contains the configured physical
        bins.  Side/wrist frames and the caller's source arrays remain
        byte-identical.  Evaluator truth is not consumed: the mask comes only
        from the task configuration and the runtime camera calibration.
        """

        if not self.ignore_bucket_pixels:
            return request
        try:
            table_frame = request.frame("table")
        except KeyError:
            return request
        if table_frame.rgb is None:
            return request
        # ``observe_request`` is also the generic plugin seam.  A specialized
        # provider may carry an opaque test/service payload instead of a local
        # decoded array; only the native RGB path can apply this image mask.
        if not isinstance(table_frame.rgb, np.ndarray):
            return request
        image = np.asarray(table_frame.rgb)
        width, height = self.camera.resolution_px
        if image.shape != (height, width, 3):
            raise IsaacObservationError(
                "table RGB resolution does not match camera calibration: "
                f"got {image.shape}, expected {(height, width, 3)}"
            )
        try:
            mask_result = ignore_configured_bucket_pixels(
                image, self.camera, self.config
            )
        except DestinationMaskError as exc:
            raise IsaacObservationError(
                f"configured bucket pixel mask failed: {exc}"
            ) from exc
        frames = tuple(
            replace(frame, rgb=mask_result.rgb)
            if frame.camera_id == "table"
            else frame
            for frame in request.frames
        )
        metadata = dict(request.metadata)
        metadata["configured_bucket_pixel_mask"] = mask_result.metadata()
        return PerceptionRequest(
            scene_id=request.scene_id,
            capture_time_s=request.capture_time_s,
            frames=frames,
            robot_state=request.robot_state,
            metadata=metadata,
        )

    def plan_observation(
        self,
        observation: SceneObservation,
        *,
        robot_state: RobotState | None = None,
    ) -> CameraSortDecision:
        """Project one validated generic observation and plan the cube task."""

        if not isinstance(observation, SceneObservation):
            raise TypeError("observation must be a SceneObservation")
        projection = cube_scene_from_observation(
            observation, class_ids=CUBE_TASK_CLASS_IDS
        )
        raw_scene = projection.scene
        scene = filter_scene_to_pickup_region(raw_scene, self.config)
        return CameraSortDecision(
            raw_scene=raw_scene,
            scene=scene,
            plans=build_colour_sort_plans(
                scene,
                self.config,
                grasp_pipeline=self.grasp_pipeline,
                generic_scene=observation,
                unsupported_objects=projection.unsupported_objects,
                robot_state=robot_state,
            ),
            observation=observation,
        )

    def observe_request_and_plan(
        self, request: PerceptionRequest
    ) -> CameraSortDecision:
        """Run a complete multi-camera perception-to-grasp-planning pass."""

        return self.plan_observation(
            self.observe_request(request), robot_state=request.robot_state
        )

    def observe_generic(
        self,
        rgb: np.ndarray,
        *,
        scene_id: str,
        capture_time_s: float,
    ) -> SceneObservation:
        """Return the provider-neutral scene used at the team integration seam."""

        image = np.asarray(rgb)
        width, height = self.camera.resolution_px
        if image.shape != (height, width, 3):
            raise IsaacObservationError(
                "RGB resolution does not match camera calibration: "
                f"got {image.shape[:2][::-1]}, expected {(width, height)}"
            )
        request = PerceptionRequest(
            scene_id=scene_id,
            capture_time_s=capture_time_s,
            frames=(
                CameraFrame(
                    camera_id="table",
                    capture_time_s=capture_time_s,
                    calibration_id="isaac-live-camera-metadata",
                    rgb=image,
                ),
            ),
        )
        return self.observe_request(request)

    def observe(
        self,
        rgb: np.ndarray,
        *,
        scene_id: str,
        capture_time_s: float,
    ) -> SceneDetections:
        """Return pick-eligible detections from an unannotated Isaac RGB slice."""

        raw_scene = self.observe_raw(
            rgb, scene_id=scene_id, capture_time_s=capture_time_s
        )
        return filter_scene_to_pickup_region(raw_scene, self.config)

    def observe_raw(
        self,
        rgb: np.ndarray,
        *,
        scene_id: str,
        capture_time_s: float,
    ) -> SceneDetections:
        """Return broad cube-like detections before pick-eligibility filtering."""

        observation = self.observe_generic(
            rgb, scene_id=scene_id, capture_time_s=capture_time_s
        )
        return cube_scene_from_observation(
            observation, class_ids=CUBE_TASK_CLASS_IDS
        ).scene

    def observe_and_plan(
        self,
        rgb: np.ndarray,
        *,
        scene_id: str,
        capture_time_s: float,
    ) -> CameraSortDecision:
        """Run the same detector and deterministic planner used by hardware."""

        observation = self.observe_generic(
            rgb, scene_id=scene_id, capture_time_s=capture_time_s
        )
        return self.plan_observation(observation)
