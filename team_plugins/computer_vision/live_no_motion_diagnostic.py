"""Run the selected vision provider on live cell RGB without planning motion.

This is a deliberately narrow adapter around the official SIDC Lab client.  It
selects the requested checked-in vision profile *before* importing ``Policy``, clears the
physical-motion/descent interlocks, lets the normal policy construct its strict
measured-pose hold row, and only then calls the existing private perception
boundary for the trusted table-camera mode.  Side and wrist modes save native-
resolution evidence plus measured robot feedback but deliberately skip provider
inference because their motion-coordinate adapters are not yet validated.  An
opt-in table+side mode saves both fixed-camera views at their shared native
1920x1080 size while still running inference on the table view only.  It never
calls ``_start_next_plan`` or a motion phase.

The official client exposes one global RGB request size.  The fixed table/side
cameras share 1920x1080, while the pinned wrist calibration is 1280x720.  A
mixed-size request is rejected rather than silently rescaling one stream
through another camera's calibration.

The official TCP protocol has no no-op.  Repeating the measured TCP/gripper row
may still produce a small controller correction; this tool is not an emergency
stop and does not replace the cell operator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from sorting_core.perception import (
    SceneObservation,
    scene_observation_to_dict,
    validate_scene_for_request,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARDWARE_DIR = REPOSITORY_ROOT / "hardware" / "fanuc_cell"
HYBRID_PROFILE_PATH = Path(__file__).with_name("profile.hybrid-geometry.v1.json")
YOLO11N_PROFILE_PATH = Path(__file__).with_name("profile.yolo11n-cube.v1.json")
YOLO11N_PICKUP_GATE_PROFILE_PATH = Path(__file__).with_name(
    "profile.yolo11n-cube-pickup-gate.v1.json"
)
# Backward-compatible name used by existing callers/tests. The default remains
# the exact profile that this diagnostic selected before YOLO support existed.
PROFILE_PATH = HYBRID_PROFILE_PATH
CALIBRATION_PATH = HARDWARE_DIR / "calibration" / "cameras.json"

PROFILE_ENV = "ROCKWELL_INTEGRATION_PROFILE"
VISION_CHECKPOINT_ENV = "ROCKWELL_VISION_CHECKPOINT"
CUBE_EDGE_ENV = "ROCKWELL_VISION_CUBE_EDGE_MM"
DESTINATION_ENV = "ROCKWELL_DESTINATION_MODE"
MOTION_ENV = "ROCKWELL_ENABLE_PHYSICAL_MOTION"
DESCENT_ENV = "ROCKWELL_ENABLE_PHYSICAL_DESCENT"
GRASP_FIT_ENV = "ROCKWELL_CONFIRM_CUBE_GRASP_FIT"

_INTERLOCK_ENVIRONMENT = (MOTION_ENV, DESCENT_ENV, GRASP_FIT_ENV)
_MAXIMUM_SNAPSHOTS = 100
_CAPTURE_CAMERA_IDS = ("table", "side", "wrist")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_camera_spec(camera_id: str) -> dict[str, Any]:
    """Return the pinned native RGB capture contract for one physical camera.

    The official client exposes one global RGB size per observation request, so
    cameras with different native sizes cannot be requested together without
    resampling at least one stream.  This helper exposes each pinned size; the
    group validator permits only single-camera capture or compatible table+side.
    """

    if camera_id not in _CAPTURE_CAMERA_IDS:
        raise ValueError(
            f"capture_camera must be one of {', '.join(_CAPTURE_CAMERA_IDS)}"
        )
    try:
        calibration_root = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        calibration = calibration_root[camera_id]
        size = calibration["size"]
        width, height = int(size[0]), int(size[1])
        rms_px = float(calibration["rms_px"])
        solved = str(calibration["solved"])
        transform = str(calibration["transform"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid RGB calibration entry for capture camera {camera_id!r}"
        ) from exc
    if width <= 0 or height <= 0 or not np.isfinite(rms_px):
        raise ValueError(
            f"invalid RGB calibration entry for capture camera {camera_id!r}"
        )
    return {
        "camera_id": camera_id,
        "image_size_wh": (width, height),
        "calibration_id": f"sidclab-cameras.json:{camera_id}",
        "calibration_solved": solved,
        "calibration_rms_px": rms_px,
        "transform_semantics": transform,
        "moving_camera": camera_id == "wrist",
        "trusted_for_motion_coordinates": camera_id == "table",
        "provider_inference_enabled": camera_id == "table",
    }


def _capture_camera_specs(camera_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Return one compatible native-resolution contract per requested camera.

    Single-camera table/side/wrist capture remains supported.  The only
    multi-camera evidence mode is table+side: both fixed cameras have the same
    pinned native size, and only table is allowed to feed the provider.
    """

    selected = tuple(str(camera_id) for camera_id in camera_ids)
    if not selected:
        raise ValueError("at least one capture camera is required")
    if len(set(selected)) != len(selected):
        raise ValueError("capture cameras must be unique")
    if len(selected) > 1 and selected != ("table", "side"):
        raise ValueError(
            "the only supported paired capture is table+side; wrist must be "
            "captured separately at its pinned native resolution"
        )
    specs = {camera_id: _capture_camera_spec(camera_id) for camera_id in selected}
    sizes = {tuple(spec["image_size_wh"]) for spec in specs.values()}
    if len(sizes) != 1:
        raise ValueError(
            "requested cameras have different native RGB sizes; capture them "
            "separately rather than rescaling calibration"
        )
    inference_cameras = tuple(
        camera_id
        for camera_id, spec in specs.items()
        if bool(spec["provider_inference_enabled"])
    )
    if len(selected) > 1 and inference_cameras != ("table",):
        raise ValueError("paired capture must keep table as the sole inference camera")
    return specs


def _git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return None
    value = completed.stdout.strip()
    return value or None


def _repository_provenance(profile_path: Path = PROFILE_PATH) -> dict[str, Any]:
    status_process = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    worktree_dirty = (
        None
        if status_process.returncode
        else bool(status_process.stdout.strip())
    )
    return {
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "worktree_dirty": worktree_dirty,
        "profile_name": profile_path.name,
        "profile_sha256": _sha256_file(profile_path),
        "calibration_path": CALIBRATION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "calibration_sha256": _sha256_file(CALIBRATION_PATH),
        "diagnostic_tool_sha256": _sha256_file(Path(__file__).resolve()),
    }


def configure_no_motion_environment(
    cube_edge_mm: int,
    *,
    profile_path: Path = PROFILE_PATH,
    vision_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Select the provider and remove every inherited motion-enabling flag."""

    if cube_edge_mm not in {10, 16}:
        raise ValueError("cube_edge_mm must be 10 or 16")
    for name in _INTERLOCK_ENVIRONMENT:
        os.environ.pop(name, None)
    selected_profile = profile_path.resolve()
    if not selected_profile.is_file():
        raise ValueError(f"vision profile does not exist: {selected_profile}")
    os.environ[PROFILE_ENV] = str(selected_profile)
    if vision_checkpoint is None:
        os.environ.pop(VISION_CHECKPOINT_ENV, None)
    else:
        os.environ[VISION_CHECKPOINT_ENV] = vision_checkpoint
    os.environ[CUBE_EDGE_ENV] = str(cube_edge_mm)
    os.environ[DESTINATION_ENV] = "virtual-table"
    return {
        "integration_profile": str(selected_profile),
        "vision_checkpoint_configured": vision_checkpoint is not None,
        "cube_edge_mm": cube_edge_mm,
        "destination_mode": "virtual-table",
        "physical_motion_enabled": False,
        "physical_descent_enabled": False,
        "warning": (
            "measured-pose hold is not an emergency stop and may make a small "
            "controller correction"
        ),
    }


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    _write_bytes_atomic(path, payload)


def _prepare_output_directory(path: Path) -> Path:
    destination = path.expanduser().resolve()
    repository = REPOSITORY_ROOT.resolve()
    if destination == repository or destination.is_relative_to(repository):
        raise ValueError(
            "output directory must be outside the repository so hardware/private "
            f"captures cannot enter Git: {destination}"
        )
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"output directory must be new or empty to prevent mixed evidence: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _finite_feedback_vector(
    obs: Mapping[str, Any], key: str, length: int
) -> list[float]:
    try:
        values = np.asarray(obs.get(key), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"obs[{key!r}] must contain {length} finite values") from exc
    if values.shape != (length,) or not np.all(np.isfinite(values)):
        raise ValueError(f"obs[{key!r}] must contain {length} finite values")
    return [float(value) for value in values]


class SnapshotRecorder:
    """Persist bounded decoded inputs and validated provider scenes."""

    def __init__(
        self,
        output_dir: Path,
        *,
        maximum_snapshots: int,
        cube_edge_mm: int,
        provenance: Mapping[str, Any] | None = None,
        capture_camera_spec: Mapping[str, Any] | None = None,
        capture_camera_specs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not 1 <= int(maximum_snapshots) <= _MAXIMUM_SNAPSHOTS:
            raise ValueError(
                f"maximum_snapshots must be in [1, {_MAXIMUM_SNAPSHOTS}]"
            )
        if cube_edge_mm not in {10, 16}:
            raise ValueError("cube_edge_mm must be 10 or 16")
        self.output_dir = _prepare_output_directory(Path(output_dir))
        self.maximum_snapshots = int(maximum_snapshots)
        self.cube_edge_mm = cube_edge_mm
        self.provenance = dict(provenance or _repository_provenance())
        if capture_camera_spec is not None and capture_camera_specs is not None:
            raise ValueError(
                "capture_camera_spec and capture_camera_specs are mutually exclusive"
            )
        if capture_camera_specs is not None:
            self.capture_camera_specs = {
                str(camera_id): dict(spec)
                for camera_id, spec in capture_camera_specs.items()
            }
        elif capture_camera_spec is not None:
            single_spec = dict(capture_camera_spec)
            camera_id = str(single_spec.get("camera_id", ""))
            if not camera_id:
                raise ValueError("capture camera specification requires camera_id")
            self.capture_camera_specs = {camera_id: single_spec}
        else:
            self.capture_camera_specs = {}
        # Preserve the historical attribute for single-camera callers/tests.
        self.capture_camera_spec = (
            next(iter(self.capture_camera_specs.values()))
            if len(self.capture_camera_specs) == 1
            else None
        )
        for camera_id, spec in self.capture_camera_specs.items():
            if str(spec.get("camera_id")) != camera_id:
                raise ValueError(
                    "capture camera specification key does not match camera_id: "
                    f"{camera_id!r} != {spec.get('camera_id')!r}"
                )
        self.provider_inference_camera_ids = tuple(
            camera_id
            for camera_id, spec in self.capture_camera_specs.items()
            if bool(spec.get("provider_inference_enabled", False))
        )
        if self.capture_camera_specs and self.provider_inference_camera_ids not in {
            (),
            ("table",),
        }:
            raise ValueError(
                "only the table camera may be enabled for provider inference"
            )
        self.provider_inference_enabled = bool(
            self.provider_inference_camera_ids
            if self.capture_camera_specs
            else True
        )
        self.attempts = 0
        self.failures = 0
        self._limit_reported = False

    def _save_rgb_frames(self, index: int, policy: Any, obs: Mapping[str, Any]):
        import cv2

        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("obs['images'] must be a mapping")
        configured_camera_ids = tuple(policy.CAMERAS)
        missing_camera_ids = [
            camera_id
            for camera_id in configured_camera_ids
            if camera_id not in images
        ]
        if missing_camera_ids:
            raise ValueError(
                "missing configured RGB cameras: " + ", ".join(missing_camera_ids)
            )
        validated_frames: list[tuple[str, np.ndarray, Mapping[str, Any] | None]] = []
        for camera_id in configured_camera_ids:
            rgb = np.asarray(images[camera_id])
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(
                    f"camera {camera_id!r} must be HxWx3 uint8 RGB, got "
                    f"{rgb.dtype} {rgb.shape}"
                )
            camera_spec = self.capture_camera_specs.get(str(camera_id))
            if self.capture_camera_specs and camera_spec is None:
                raise ValueError(
                    f"missing capture camera specification for {camera_id!r}"
                )
            if camera_spec is not None:
                expected_width, expected_height = camera_spec["image_size_wh"]
                expected_shape = (int(expected_height), int(expected_width), 3)
                if rgb.shape != expected_shape:
                    raise ValueError(
                        f"camera {camera_id!r} response must match its pinned native "
                        f"RGB calibration size {expected_width}x{expected_height}, got "
                        f"{rgb.shape[1]}x{rgb.shape[0]}"
                    )
            validated_frames.append((str(camera_id), rgb, camera_spec))

        records: list[dict[str, Any]] = []
        for camera_id, rgb, camera_spec in validated_frames:
            safe_camera_id = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(camera_id)
            )
            filename = f"snapshot-{index:03d}-{safe_camera_id}.png"
            ok, encoded = cv2.imencode(
                ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )
            if not ok:
                raise OSError(f"failed to encode camera {camera_id!r} as PNG")
            _write_bytes_atomic(self.output_dir / filename, encoded.tobytes())
            frame_record = {
                "camera_id": str(camera_id),
                "file": filename,
                "shape_hwc": [int(value) for value in rgb.shape],
                "dtype": str(rgb.dtype),
                "channel_order": "RGB",
                "pixel_provenance": (
                    "decoded_backend_observation_at_requested_resolution"
                ),
                "sensor_native_unmodified": False,
                "requested_wire_format": str(
                    getattr(policy, "IMAGE_FORMAT", "unknown")
                ),
                "rgb_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
                "png_sha256": _sha256_file(self.output_dir / filename),
            }
            if camera_spec is not None:
                frame_record["calibration"] = {
                    "calibration_id": camera_spec["calibration_id"],
                    "native_size_wh": [
                        int(value) for value in camera_spec["image_size_wh"]
                    ],
                    "solved": camera_spec["calibration_solved"],
                    "rms_px": float(camera_spec["calibration_rms_px"]),
                    "transform_semantics": camera_spec["transform_semantics"],
                    "moving_camera": bool(camera_spec["moving_camera"]),
                    "trusted_for_motion_coordinates": bool(
                        camera_spec["trusted_for_motion_coordinates"]
                    ),
                    "provider_inference_enabled": bool(
                        camera_spec["provider_inference_enabled"]
                    ),
                }
            records.append(frame_record)
        return records

    def capture(
        self,
        policy: Any,
        obs: Mapping[str, Any],
        *,
        act_started_s: float,
        hold_ready_s: float,
    ) -> None:
        """Run only the policy's existing perception boundary and save evidence."""

        if self.attempts >= self.maximum_snapshots:
            if not self._limit_reported:
                print(
                    json.dumps(
                        {
                            "event": "vision_diagnostic_capture_limit_reached",
                            "maximum_snapshots": self.maximum_snapshots,
                        },
                        sort_keys=True,
                    )
                )
                self._limit_reported = True
            return

        index = self.attempts
        self.attempts += 1
        frames = self._save_rgb_frames(index, policy, obs)
        started = time.perf_counter()
        scene: SceneObservation | None = None
        failure: Exception | None = None
        failure_stage: str | None = None
        provider = (
            getattr(policy, "_perception_provider", None)
            if self.provider_inference_enabled
            else None
        )
        proxy = None
        if provider is not None:
            class _CapturingProvider:
                """Preserve a validated provider scene before task partitioning."""

                def __init__(self, delegate: Any) -> None:
                    self.delegate = delegate
                    self.scene: SceneObservation | None = None

                def observe(self, request: Any) -> Any:
                    result = self.delegate.observe(request)
                    self.scene = validate_scene_for_request(request, result)
                    return result

                def __getattr__(self, name: str) -> Any:
                    return getattr(self.delegate, name)

            proxy = _CapturingProvider(provider)
            policy._perception_provider = proxy
        original_policy_cameras = tuple(getattr(policy, "CAMERAS", ()))
        if self.provider_inference_enabled:
            try:
                if self.provider_inference_camera_ids:
                    # The hardware observation may contain capture-only views.
                    # Keep them out of the provider request so the checked-in
                    # table-only profile and provenance remain exact.
                    policy.CAMERAS = self.provider_inference_camera_ids
                # This is the narrow private integration described in the module
                # docstring. It builds/validates the provider request *and* applies
                # the cube task's diagnostic partitioning. A task-partition error can
                # therefore follow valid provider output. It never starts a plan or
                # constructs a motion phase.
                scene = policy._detect_scene_views(obs).observation
                if not isinstance(scene, SceneObservation):
                    raise TypeError(
                        "policy perception boundary returned no SceneObservation"
                    )
            except Exception as exc:  # explicit evidence plus nonzero process exit
                failure = exc
                if proxy is not None and proxy.scene is not None:
                    # The provider result itself passed the strict request boundary;
                    # retain it even when the legacy cube-task partition rejects the
                    # scene afterward (for example because a task-specific colour
                    # condition is absent). This distinction is essential when the
                    # purpose of the run is vision evidence rather than execution.
                    scene = proxy.scene
                    failure_stage = "task_partition_after_valid_provider_scene"
                else:
                    failure_stage = "provider_or_request_boundary"
                self.failures += 1
            finally:
                policy.CAMERAS = original_policy_cameras
                if proxy is not None:
                    policy._perception_provider = provider
        latency_ms = (time.perf_counter() - started) * 1000.0

        trial = obs.get("trial")
        trial_id = trial.get("trial_id") if isinstance(trial, Mapping) else None
        observation_time_s = float(obs.get("t"))
        if not np.isfinite(observation_time_s):
            raise ValueError("obs['t'] must be finite")
        record: dict[str, Any] = {
            "schema_version": 1,
            "snapshot_index": index,
            "status": (
                "error"
                if failure is not None
                else "ok"
                if self.provider_inference_enabled
                else "capture_only"
            ),
            "capture_time_s": observation_time_s,
            "trial_id": trial_id,
            "cube_edge_mm": self.cube_edge_mm,
            "perception_latency_ms": latency_ms,
            "hold_action_generated_before_perception": True,
            "motion_planning_called": False,
            "provider_inference_ran": self.provider_inference_enabled,
            "frames": frames,
            "robot_feedback": {
                "source_timestamp_s": observation_time_s,
                "timestamp_semantics": (
                    "backend_observation_time_not_camera_exposure_time"
                ),
                "joints_deg_and_gripper_0_to_1": _finite_feedback_vector(
                    obs, "joints", 7
                ),
                "tcp_pose_uf0_mm_wpr_deg": _finite_feedback_vector(
                    obs, "tcp_pose", 6
                ),
            },
            "provenance": self.provenance,
        }
        if scene is not None:
            scene_filename = f"snapshot-{index:03d}-scene.json"
            _write_json_atomic(
                self.output_dir / scene_filename, scene_observation_to_dict(scene)
            )
            record.update(
                {
                    "scene_file": scene_filename,
                    "scene_sha256": _sha256_file(self.output_dir / scene_filename),
                    "provider_name": scene.provider_name,
                    "provider_version": scene.provider_version,
                    "calibration_ids": list(scene.calibration_ids),
                    "object_count": len(scene.objects),
                }
            )
        if failure is not None:
            record["error"] = {
                "type": type(failure).__name__,
                "message": str(failure),
                "stage": failure_stage,
            }
        action_delay_lower_bound_ms = (time.perf_counter() - act_started_s) * 1000.0
        record["timing"] = {
            "hold_construction_ms": (hold_ready_s - act_started_s) * 1000.0,
            "perception_and_task_boundary_ms": latency_ms,
            "action_return_delay_lower_bound_ms": action_delay_lower_bound_ms,
            "lower_bound_excludes": "final record write and event print",
        }
        record_filename = f"snapshot-{index:03d}-record.json"
        _write_json_atomic(self.output_dir / record_filename, record)
        print(
            json.dumps(
                {
                    "event": "vision_diagnostic_snapshot",
                    "snapshot_index": index,
                    "status": record["status"],
                    "record_file": record_filename,
                    "perception_latency_ms": round(latency_ms, 3),
                    "action_return_delay_lower_bound_ms": round(
                        action_delay_lower_bound_ms, 3
                    ),
                    "error": record.get("error"),
                },
                sort_keys=True,
            )
        )


def diagnostic_policy_class(
    base_policy_class: type,
    recorder: SnapshotRecorder,
    *,
    capture_camera_id: str = "table",
    capture_camera_ids: Sequence[str] | None = None,
    capture_image_size: tuple[int, int] | None = None,
) -> type:
    """Return a Policy subclass that preserves hold output byte-for-byte."""

    selected_camera_ids = (
        (capture_camera_id,)
        if capture_camera_ids is None
        else tuple(str(camera_id) for camera_id in capture_camera_ids)
    )
    if capture_camera_ids is not None and capture_camera_id != "table":
        raise ValueError(
            "capture_camera_id cannot be combined with capture_camera_ids"
        )
    if not selected_camera_ids or any(
        camera_id not in _CAPTURE_CAMERA_IDS for camera_id in selected_camera_ids
    ):
        raise ValueError(
            f"capture cameras must be selected from {', '.join(_CAPTURE_CAMERA_IDS)}"
        )
    if len(set(selected_camera_ids)) != len(selected_camera_ids):
        raise ValueError("capture cameras must be unique")
    if len(selected_camera_ids) > 1 and selected_camera_ids != ("table", "side"):
        raise ValueError("the only supported paired capture is table+side")
    if capture_image_size is None:
        capture_image_size = tuple(
            int(value)
            for value in getattr(base_policy_class, "IMAGE_SIZE", (1920, 1080))
        )
    if (
        len(capture_image_size) != 2
        or int(capture_image_size[0]) <= 0
        or int(capture_image_size[1]) <= 0
    ):
        raise ValueError("capture_image_size must be positive (width, height)")

    class LiveNoMotionDiagnosticPolicy(base_policy_class):
        def _assert_interlocks_closed(self) -> None:
            if bool(getattr(self, "_motion_enabled", False)) or bool(
                getattr(self, "_descent_enabled", False)
            ):
                raise RuntimeError(
                    "vision diagnostic refuses to start with motion/descent enabled"
                )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._assert_interlocks_closed()

        def reset(self):
            result = super().reset()
            # The official client resets Policy for every rollout.  Re-check
            # after each reset instead of relying only on construction-time
            # environment state.
            self._assert_interlocks_closed()
            return result

        def act(self, obs):
            # Construction/reset checks are not enough if state is modified
            # later. Refuse before even asking the base policy for a row.
            self._assert_interlocks_closed()
            act_started_s = time.perf_counter()
            hold_action = super().act(obs)
            hold_ready_s = time.perf_counter()
            failures_before_capture = recorder.failures
            try:
                recorder.capture(
                    self,
                    obs,
                    act_started_s=act_started_s,
                    hold_ready_s=hold_ready_s,
                )
            except Exception as exc:
                # The hold row already exists. A disk/encoding/provenance
                # failure must be visible, but must not replace that row with an
                # exception after perception was deliberately made non-motion.
                if recorder.failures == failures_before_capture:
                    recorder.failures += 1
                print(
                    json.dumps(
                        {
                            "event": "vision_diagnostic_recording_error",
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                            "hold_action_preserved": True,
                        },
                        sort_keys=True,
                    )
                )
            return hold_action

    LiveNoMotionDiagnosticPolicy.__name__ = "LiveNoMotionDiagnosticPolicy"
    LiveNoMotionDiagnosticPolicy.__qualname__ = "LiveNoMotionDiagnosticPolicy"
    # The official client reads these attributes from the replacement class.
    # Table+side may be requested together because both fixed cameras have the
    # same pinned native size. Wrist remains a separate single-camera capture.
    LiveNoMotionDiagnosticPolicy.CAMERAS = selected_camera_ids
    LiveNoMotionDiagnosticPolicy.IMAGE_SIZE = tuple(
        int(value) for value in capture_image_size
    )
    # If the base hardware policy uses reduced queue-poll previews, keep them
    # on the same explicitly selected stream. In particular, a wrist/side-only
    # approved diagnostic must never inherit the base policy's table preview.
    LiveNoMotionDiagnosticPolicy.QUEUE_POLL_CAMERAS = (selected_camera_ids[0],)
    return LiveNoMotionDiagnosticPolicy


def _validate_loaded_profile(
    team_integrations: ModuleType,
    official_run: ModuleType,
    expected_profile_path: Path = PROFILE_PATH,
) -> None:
    profile = getattr(team_integrations, "ACTIVE_INTEGRATION_PROFILE", None)
    if profile is None:
        raise RuntimeError(
            "integration profile was not active; start this diagnostic in a fresh "
            "Python process"
        )
    loaded_path = Path(str(getattr(profile, "source_path", ""))).resolve()
    if loaded_path != expected_profile_path.resolve():
        raise RuntimeError(
            "a different integration profile was already imported; start this "
            f"diagnostic in a fresh Python process (loaded {loaded_path})"
        )
    expected_hash = _sha256_file(expected_profile_path)
    if str(getattr(profile, "source_sha256", "")) != expected_hash:
        raise RuntimeError("loaded integration profile bytes do not match disk")
    cameras = tuple(getattr(official_run.Policy, "CAMERAS", ()))
    if cameras != ("table",):
        raise RuntimeError(
            f"diagnostic expected only table RGB from the active profile, got {cameras}"
        )


def _load_official_run(expected_profile_path: Path = PROFILE_PATH) -> ModuleType:
    hardware_path = str(HARDWARE_DIR)
    if hardware_path not in sys.path:
        sys.path.insert(0, hardware_path)
    existing = sys.modules.get("run")
    if existing is not None:
        source = Path(getattr(existing, "__file__", "")).resolve()
        if source != (HARDWARE_DIR / "run.py").resolve():
            raise RuntimeError(f"unrelated module named 'run' is already loaded: {source}")
        official_run = existing
    else:
        official_run = importlib.import_module("run")
    team_integrations = importlib.import_module("sorting_core.team_integrations")
    _validate_loaded_profile(
        team_integrations, official_run, expected_profile_path
    )
    return official_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "new or empty directory outside Git for decoded requested-resolution "
            "frames and JSON evidence"
        ),
    )
    parser.add_argument("--cube-mm", required=True, type=int, choices=(10, 16))
    parser.add_argument(
        "--capture-camera",
        choices=_CAPTURE_CAMERA_IDS,
        default="table",
        help=(
            "single RGB camera to save at its pinned native calibration size; "
            "side/wrist are capture-only and are not used for pose or motion"
        ),
    )
    parser.add_argument(
        "--aux-camera",
        choices=("side", "wrist"),
        help=(
            "optional capture-only view; only --capture-camera table with side "
            "is supported because wrist has a different native resolution"
        ),
    )
    parser.add_argument(
        "--vision",
        choices=("hybrid-geometry", "yolo11n", "yolo11n-pickup-gate"),
        default="hybrid-geometry",
        help=(
            "vision provider to exercise (default: hybrid-geometry); YOLO11n "
            "profiles require --vision-checkpoint; pickup-gate is no-motion "
            "evidence only"
        ),
    )
    parser.add_argument(
        "--vision-checkpoint",
        help="external fine-tuned YOLO11n checkpoint; not stored in Git",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        choices=range(1, _MAXIMUM_SNAPSHOTS + 1),
        default=10,
        metavar="1..100",
    )
    parser.add_argument("--url", help="forward a non-default cell URL")
    parser.add_argument("--token", help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint",
        help=(
            "forward the official client's legacy policy checkpoint; do not "
            "use this for YOLO11n (use --vision-checkpoint instead)"
        ),
    )
    parser.add_argument("--ca", help="forward a CA certificate path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print effective non-secret settings without importing the cell client",
    )
    return parser


def _forwarded_arguments(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    for option, value in (
        ("--url", args.url),
        ("--token", args.token),
        ("--checkpoint", args.checkpoint),
        ("--ca", args.ca),
    ):
        if value is not None:
            forwarded.extend((option, str(value)))
    return forwarded


def main(
    argv: Sequence[str] | None = None,
    *,
    official_run_loader: Callable[[], ModuleType] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    capture_camera_ids = (
        (args.capture_camera,)
        if args.aux_camera is None
        else (args.capture_camera, args.aux_camera)
    )
    try:
        camera_specs = _capture_camera_specs(capture_camera_ids)
    except ValueError as exc:
        parser.error(str(exc))
    capture_image_size = tuple(
        next(iter(camera_specs.values()))["image_size_wh"]
    )
    inference_camera_ids = tuple(
        camera_id
        for camera_id, spec in camera_specs.items()
        if bool(spec["provider_inference_enabled"])
    )
    if args.capture_camera != "table" and (
        args.vision != "hybrid-geometry" or args.vision_checkpoint is not None
    ):
        parser.error(
            "--vision/--vision-checkpoint apply only to --capture-camera table; "
            "side and wrist are evidence-only"
        )
    if args.vision in {"yolo11n", "yolo11n-pickup-gate"}:
        if args.vision_checkpoint is None:
            parser.error(
                f"--vision {args.vision} requires --vision-checkpoint PATH"
            )
        profile_path = (
            YOLO11N_PROFILE_PATH
            if args.vision == "yolo11n"
            else YOLO11N_PICKUP_GATE_PROFILE_PATH
        )
    else:
        if args.vision_checkpoint is not None:
            parser.error(
                "--vision-checkpoint requires --vision yolo11n or "
                "yolo11n-pickup-gate"
            )
        profile_path = HYBRID_PROFILE_PATH
    environment_summary = configure_no_motion_environment(
        args.cube_mm,
        profile_path=profile_path,
        vision_checkpoint=args.vision_checkpoint,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "event": "vision_diagnostic_dry_run",
                    **environment_summary,
                    "vision": args.vision,
                    "capture_camera": args.capture_camera,
                    "capture_cameras": list(capture_camera_ids),
                    "auxiliary_capture_only_cameras": [
                        camera_id
                        for camera_id in capture_camera_ids
                        if camera_id not in inference_camera_ids
                    ],
                    "capture_image_size_wh": list(capture_image_size),
                    "provider_inference_enabled": bool(inference_camera_ids),
                    "provider_inference_cameras": list(inference_camera_ids),
                    "trusted_for_motion_coordinate_cameras": [
                        camera_id
                        for camera_id, spec in camera_specs.items()
                        if bool(spec["trusted_for_motion_coordinates"])
                    ],
                    "trusted_for_motion_coordinates": all(
                        bool(spec["trusted_for_motion_coordinates"])
                        for spec in camera_specs.values()
                    ),
                    "maximum_snapshots": args.max_snapshots,
                    "output_dir": str(args.output_dir.expanduser().resolve()),
                },
                sort_keys=True,
            )
        )
        return 0

    recorder = SnapshotRecorder(
        args.output_dir,
        maximum_snapshots=args.max_snapshots,
        cube_edge_mm=args.cube_mm,
        provenance=_repository_provenance(profile_path),
        capture_camera_specs=camera_specs,
    )
    official_run = (
        _load_official_run(profile_path)
        if official_run_loader is None
        else official_run_loader()
    )
    original_policy = official_run.Policy
    official_run.Policy = diagnostic_policy_class(
        original_policy,
        recorder,
        capture_camera_ids=capture_camera_ids,
        capture_image_size=capture_image_size,
    )
    previous_argv = sys.argv
    try:
        sys.argv = [str(HARDWARE_DIR / "run.py"), *_forwarded_arguments(args)]
        official_run.main()
    finally:
        sys.argv = previous_argv
        official_run.Policy = original_policy
    return 2 if recorder.failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


__all__ = [
    "SnapshotRecorder",
    "configure_no_motion_environment",
    "diagnostic_policy_class",
    "main",
]
