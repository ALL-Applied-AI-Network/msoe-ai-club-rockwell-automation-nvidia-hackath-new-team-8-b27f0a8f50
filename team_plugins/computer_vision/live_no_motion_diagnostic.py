"""Run the selected vision provider on live cell RGB without planning motion.

This is a deliberately narrow adapter around the official SIDC Lab client.  It
selects the checked-in hybrid profile *before* importing ``Policy``, clears the
physical-motion/descent interlocks, lets the normal policy construct its strict
measured-pose hold row, and only then calls the existing private perception
boundary.  It never calls ``_start_next_plan`` or a motion phase.

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

from sorting_core.perception import SceneObservation, scene_observation_to_dict


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARDWARE_DIR = REPOSITORY_ROOT / "hardware" / "fanuc_cell"
PROFILE_PATH = Path(__file__).with_name("profile.hybrid-geometry.v1.json")
CALIBRATION_PATH = HARDWARE_DIR / "calibration" / "cameras.json"

PROFILE_ENV = "ROCKWELL_INTEGRATION_PROFILE"
CUBE_EDGE_ENV = "ROCKWELL_VISION_CUBE_EDGE_MM"
DESTINATION_ENV = "ROCKWELL_DESTINATION_MODE"
MOTION_ENV = "ROCKWELL_ENABLE_PHYSICAL_MOTION"
DESCENT_ENV = "ROCKWELL_ENABLE_PHYSICAL_DESCENT"
GRASP_FIT_ENV = "ROCKWELL_CONFIRM_CUBE_GRASP_FIT"

_INTERLOCK_ENVIRONMENT = (MOTION_ENV, DESCENT_ENV, GRASP_FIT_ENV)
_MAXIMUM_SNAPSHOTS = 100


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _repository_provenance() -> dict[str, Any]:
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
        "profile_name": PROFILE_PATH.name,
        "profile_sha256": _sha256_file(PROFILE_PATH),
        "calibration_path": CALIBRATION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "calibration_sha256": _sha256_file(CALIBRATION_PATH),
        "diagnostic_tool_sha256": _sha256_file(Path(__file__).resolve()),
    }


def configure_no_motion_environment(cube_edge_mm: int) -> dict[str, Any]:
    """Select the provider and remove every inherited motion-enabling flag."""

    if cube_edge_mm not in {10, 16}:
        raise ValueError("cube_edge_mm must be 10 or 16")
    for name in _INTERLOCK_ENVIRONMENT:
        os.environ.pop(name, None)
    os.environ[PROFILE_ENV] = str(PROFILE_PATH.resolve())
    os.environ[CUBE_EDGE_ENV] = str(cube_edge_mm)
    os.environ[DESTINATION_ENV] = "virtual-table"
    return {
        "integration_profile": str(PROFILE_PATH.resolve()),
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
        self.attempts = 0
        self.failures = 0
        self._limit_reported = False

    def _save_rgb_frames(self, index: int, policy: Any, obs: Mapping[str, Any]):
        import cv2

        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("obs['images'] must be a mapping")
        records: list[dict[str, Any]] = []
        for camera_id in tuple(policy.CAMERAS):
            if camera_id not in images:
                raise ValueError(f"missing configured RGB camera {camera_id!r}")
            rgb = np.asarray(images[camera_id])
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(
                    f"camera {camera_id!r} must be HxWx3 uint8 RGB, got "
                    f"{rgb.dtype} {rgb.shape}"
                )
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
            records.append(
                {
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
            )
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
        try:
            # This is the narrow private integration described in the module
            # docstring. It builds/validates the provider request *and* applies
            # the cube task's diagnostic partitioning. A task-partition error can
            # therefore follow valid provider output. It never starts a plan or
            # constructs a motion phase.
            scene = policy._detect_scene_views(obs).observation
            if not isinstance(scene, SceneObservation):
                raise TypeError("policy perception boundary returned no SceneObservation")
        except Exception as exc:  # explicit evidence plus nonzero process exit
            failure = exc
            self.failures += 1
        latency_ms = (time.perf_counter() - started) * 1000.0

        trial = obs.get("trial")
        trial_id = trial.get("trial_id") if isinstance(trial, Mapping) else None
        observation_time_s = float(obs.get("t"))
        if not np.isfinite(observation_time_s):
            raise ValueError("obs['t'] must be finite")
        record: dict[str, Any] = {
            "schema_version": 1,
            "snapshot_index": index,
            "status": "error" if failure is not None else "ok",
            "capture_time_s": observation_time_s,
            "trial_id": trial_id,
            "cube_edge_mm": self.cube_edge_mm,
            "perception_latency_ms": latency_ms,
            "hold_action_generated_before_perception": True,
            "motion_planning_called": False,
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


def diagnostic_policy_class(base_policy_class: type, recorder: SnapshotRecorder) -> type:
    """Return a Policy subclass that preserves hold output byte-for-byte."""

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
            recorder.capture(
                self,
                obs,
                act_started_s=act_started_s,
                hold_ready_s=hold_ready_s,
            )
            return hold_action

    LiveNoMotionDiagnosticPolicy.__name__ = "LiveNoMotionDiagnosticPolicy"
    LiveNoMotionDiagnosticPolicy.__qualname__ = "LiveNoMotionDiagnosticPolicy"
    return LiveNoMotionDiagnosticPolicy


def _validate_loaded_profile(team_integrations: ModuleType, official_run: ModuleType) -> None:
    profile = getattr(team_integrations, "ACTIVE_INTEGRATION_PROFILE", None)
    if profile is None:
        raise RuntimeError(
            "integration profile was not active; start this diagnostic in a fresh "
            "Python process"
        )
    loaded_path = Path(str(getattr(profile, "source_path", ""))).resolve()
    if loaded_path != PROFILE_PATH.resolve():
        raise RuntimeError(
            "a different integration profile was already imported; start this "
            f"diagnostic in a fresh Python process (loaded {loaded_path})"
        )
    expected_hash = _sha256_file(PROFILE_PATH)
    if str(getattr(profile, "source_sha256", "")) != expected_hash:
        raise RuntimeError("loaded integration profile bytes do not match disk")
    cameras = tuple(getattr(official_run.Policy, "CAMERAS", ()))
    if cameras != ("table",):
        raise RuntimeError(
            f"diagnostic expected only table RGB from the active profile, got {cameras}"
        )


def _load_official_run() -> ModuleType:
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
    _validate_loaded_profile(team_integrations, official_run)
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
        "--max-snapshots",
        type=int,
        choices=range(1, _MAXIMUM_SNAPSHOTS + 1),
        default=10,
        metavar="1..100",
    )
    parser.add_argument("--url", help="forward a non-default cell URL")
    parser.add_argument("--token", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help="forward the vision checkpoint path")
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
    official_run_loader: Callable[[], ModuleType] = _load_official_run,
) -> int:
    args = _parser().parse_args(argv)
    environment_summary = configure_no_motion_environment(args.cube_mm)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "event": "vision_diagnostic_dry_run",
                    **environment_summary,
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
    )
    official_run = official_run_loader()
    original_policy = official_run.Policy
    official_run.Policy = diagnostic_policy_class(original_policy, recorder)
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
