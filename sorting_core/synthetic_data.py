"""Reproducible synthetic-data configuration, manifests, and image effects.

This module deliberately has no Isaac dependency.  Isaac-side code samples one
manifest per episode, applies only the parameters it actually supports, and
records that application status.  A sampled value is not evidence that a USD
attribute or physics property changed.

Raw renderer output and derived sensor-augmented images are separate artifacts.
Never burn presentation text into either training input.  Simulator truth is
written under ``evaluator_only`` so it cannot silently become a runtime feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC_DATA_CONFIG_PATH = (
    REPO_ROOT / "config" / "synthetic_data_generation.json"
)


class SyntheticDataError(ValueError):
    """Raised when synthetic-data configuration or labels are unsafe/invalid."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SyntheticDataError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise SyntheticDataError(f"{name} must be finite")
    return number


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntheticDataError(f"{name} must be an object")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Return a stable UTF-8 JSON representation for hashes and manifests."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _repo_relative_file(path_value: str, name: str) -> tuple[Path, str]:
    relative = PurePosixPath(str(path_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise SyntheticDataError(f"{name} must be a repository-relative path")
    resolved = (REPO_ROOT / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SyntheticDataError(f"{name} escapes the repository") from exc
    if not resolved.is_file():
        raise SyntheticDataError(f"{name} does not exist: {relative.as_posix()}")
    return resolved, relative.as_posix()


def _artifact_relative_path(path_value: str, name: str) -> str:
    relative = PurePosixPath(str(path_value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SyntheticDataError(f"{name} must be a safe run-relative path")
    return relative.as_posix()


@dataclass(frozen=True)
class CapturePolicy:
    """Camera and file rules for one training-data stream."""

    camera_name: str
    resolution: tuple[int, int]
    colour_space: str
    image_encoding: str
    save_raw_frames: bool
    burn_in_overlay: bool
    calibration_file: str
    calibration_entry: str
    calibration_sha256: str
    apply_calibrated_radial_distortion: bool
    maximum_state_skew_s: float

    def validate(self) -> "CapturePolicy":
        if self.camera_name != "table":
            raise SyntheticDataError("the checked-in capture policy must use table camera")
        if self.resolution != (1920, 1080):
            raise SyntheticDataError(
                "table training capture must retain native calibrated 1920x1080"
            )
        if self.colour_space != "RGB":
            raise SyntheticDataError("training capture colour space must be RGB")
        if self.image_encoding != "png":
            raise SyntheticDataError("raw training frames must use lossless PNG")
        if not self.save_raw_frames:
            raise SyntheticDataError("raw renderer frames must be retained")
        if self.burn_in_overlay:
            raise SyntheticDataError("training frames must not contain text overlays")
        if self.maximum_state_skew_s <= 0 or not math.isfinite(
            self.maximum_state_skew_s
        ):
            raise SyntheticDataError("maximum_state_skew_s must be positive")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "resolution": list(self.resolution),
            "colour_space": self.colour_space,
            "image_encoding": self.image_encoding,
            "save_raw_frames": self.save_raw_frames,
            "burn_in_overlay": self.burn_in_overlay,
            "calibration_file": self.calibration_file,
            "calibration_entry": self.calibration_entry,
            "calibration_sha256": self.calibration_sha256,
            "apply_calibrated_radial_distortion": (
                self.apply_calibrated_radial_distortion
            ),
            "maximum_state_skew_s": self.maximum_state_skew_s,
        }


@dataclass(frozen=True)
class ParameterSpec:
    """One bounded random variable and the component responsible for applying it."""

    name: str
    consumer: str
    distribution: str
    minimum: float
    maximum: float
    nominal: float
    unit: str
    status: str

    def validate(self) -> "ParameterSpec":
        if not self.name or "." not in self.name:
            raise SyntheticDataError("randomization parameter names must be namespaced")
        if self.consumer not in {"isaac_stage", "sensor_postprocess"}:
            raise SyntheticDataError(f"unsupported consumer for {self.name}")
        if self.distribution not in {"uniform", "randint"}:
            raise SyntheticDataError(f"unsupported distribution for {self.name}")
        if self.minimum > self.maximum:
            raise SyntheticDataError(f"reversed bounds for {self.name}")
        if not self.minimum <= self.nominal <= self.maximum:
            raise SyntheticDataError(f"nominal is outside bounds for {self.name}")
        if self.distribution == "randint":
            for label, value in (
                ("minimum", self.minimum),
                ("maximum", self.maximum),
                ("nominal", self.nominal),
            ):
                if not float(value).is_integer():
                    raise SyntheticDataError(
                        f"{self.name} {label} must be integral for randint"
                    )
        if not self.unit.strip() or not self.status.strip():
            raise SyntheticDataError(f"unit/status missing for {self.name}")
        return self

    def sample(self, episode_seed: int) -> int | float:
        # A per-parameter seed makes an existing parameter stable if a later
        # config revision adds another parameter or changes JSON key order.
        digest = hashlib.sha256(
            f"rockwell-synthetic-v1:{episode_seed}:{self.name}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big", signed=False))
        if self.distribution == "randint":
            return rng.randint(int(self.minimum), int(self.maximum))
        return rng.uniform(self.minimum, self.maximum)

    def as_dict(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer,
            "distribution": self.distribution,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "nominal": self.nominal,
            "unit": self.unit,
            "status": self.status,
        }


@dataclass(frozen=True)
class SyntheticDataSpec:
    schema_version: int
    purpose: str
    range_status: str
    range_warning: str
    capture: CapturePolicy
    parameters: Mapping[str, ParameterSpec]
    source_path: str
    source_sha256: str

    def validate(self) -> "SyntheticDataSpec":
        if self.schema_version != 1:
            raise SyntheticDataError(
                f"unsupported synthetic-data schema {self.schema_version}"
            )
        if not self.purpose.strip() or not self.range_warning.strip():
            raise SyntheticDataError("purpose and provisional-range warning are required")
        if self.range_status != "provisional_unmeasured":
            raise SyntheticDataError(
                "ranges must remain explicitly provisional until physical measurements exist"
            )
        self.capture.validate()
        if not self.parameters:
            raise SyntheticDataError("at least one randomization parameter is required")
        for name, spec in self.parameters.items():
            if name != spec.name:
                raise SyntheticDataError(f"parameter key/name mismatch for {name}")
            spec.validate()
        static_name = "physics.object_static_friction"
        dynamic_name = "physics.object_dynamic_friction"
        try:
            static_friction = self.parameters[static_name]
            dynamic_friction = self.parameters[dynamic_name]
        except KeyError as exc:
            raise SyntheticDataError(
                "static and dynamic object-friction ranges are both required"
            ) from exc
        if dynamic_friction.maximum > static_friction.minimum:
            raise SyntheticDataError(
                "dynamic-friction maximum must not exceed static-friction minimum"
            )
        return self


def load_synthetic_data_spec(
    path: str | Path | None = None,
) -> SyntheticDataSpec:
    """Load and cross-check capture policy against checked-in calibration."""

    config_path = (
        Path(path).resolve()
        if path is not None
        else DEFAULT_SYNTHETIC_DATA_CONFIG_PATH.resolve()
    )
    try:
        raw_bytes = config_path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticDataError(f"cannot load {config_path}: {exc}") from exc
    root = _mapping(raw, "synthetic-data config")
    capture_raw = _mapping(root.get("capture"), "capture")
    status_raw = _mapping(root.get("range_status"), "range_status")
    parameters_raw = _mapping(root.get("parameters"), "parameters")

    try:
        width, height = (int(v) for v in capture_raw["resolution"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SyntheticDataError("capture.resolution must contain two integers") from exc

    calibration_path, calibration_relpath = _repo_relative_file(
        str(capture_raw.get("calibration_file", "")), "capture.calibration_file"
    )
    try:
        calibration_root = _mapping(
            json.loads(calibration_path.read_text(encoding="utf-8")),
            "camera calibration",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticDataError(f"cannot load camera calibration: {exc}") from exc
    calibration_entry = str(capture_raw.get("calibration_entry", ""))
    camera_calibration = _mapping(
        calibration_root.get(calibration_entry),
        f"camera calibration entry {calibration_entry!r}",
    )
    try:
        calibrated_size = tuple(int(v) for v in camera_calibration["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SyntheticDataError("calibrated camera size is invalid") from exc
    if calibrated_size != (width, height):
        raise SyntheticDataError(
            f"capture resolution {(width, height)} does not match calibrated "
            f"size {calibrated_size}"
        )

    capture = CapturePolicy(
        camera_name=str(capture_raw.get("camera_name", "")),
        resolution=(width, height),
        colour_space=str(capture_raw.get("colour_space", "")),
        image_encoding=str(capture_raw.get("image_encoding", "")).lower(),
        save_raw_frames=bool(capture_raw.get("save_raw_frames", False)),
        burn_in_overlay=bool(capture_raw.get("burn_in_overlay", True)),
        calibration_file=calibration_relpath,
        calibration_entry=calibration_entry,
        calibration_sha256=_sha256_bytes(calibration_path.read_bytes()),
        apply_calibrated_radial_distortion=bool(
            capture_raw.get("apply_calibrated_radial_distortion", False)
        ),
        maximum_state_skew_s=_finite(
            capture_raw.get("maximum_state_skew_s"), "maximum_state_skew_s"
        ),
    )

    range_status = str(status_raw.get("name", ""))
    parameters: dict[str, ParameterSpec] = {}
    for name, value in parameters_raw.items():
        parameter = _mapping(value, f"parameters.{name}")
        parameters[str(name)] = ParameterSpec(
            name=str(name),
            consumer=str(parameter.get("consumer", "")),
            distribution=str(parameter.get("distribution", "")),
            minimum=_finite(parameter.get("minimum"), f"{name}.minimum"),
            maximum=_finite(parameter.get("maximum"), f"{name}.maximum"),
            nominal=_finite(parameter.get("nominal"), f"{name}.nominal"),
            unit=str(parameter.get("unit", "")),
            status=range_status,
        )

    try:
        source_path = config_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source_path = str(config_path)
    return SyntheticDataSpec(
        schema_version=int(root.get("schema_version", -1)),
        purpose=str(root.get("purpose", "")),
        range_status=range_status,
        range_warning=str(status_raw.get("warning", "")),
        capture=capture,
        parameters=parameters,
        source_path=source_path,
        source_sha256=_sha256_bytes(raw_bytes),
    ).validate()


@dataclass(frozen=True)
class RandomizationManifest:
    """Exact requested episode values, before Isaac application is asserted."""

    schema_version: int
    episode_id: str
    episode_seed: int
    spec_path: str
    spec_sha256: str
    capture: Mapping[str, Any]
    range_warning: str
    parameters: Mapping[str, Mapping[str, Any]]
    sensor_noise_seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "generator": "sorting_core.synthetic_data.sample_randomization",
            "spec_path": self.spec_path,
            "spec_sha256": self.spec_sha256,
            "capture": dict(self.capture),
            "range_warning": self.range_warning,
            "sensor_noise_seed": self.sensor_noise_seed,
            "parameters": {
                name: dict(value) for name, value in sorted(self.parameters.items())
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.as_dict())

    def value(self, name: str) -> int | float:
        try:
            return self.parameters[name]["requested_value"]
        except KeyError as exc:
            raise SyntheticDataError(f"manifest has no parameter {name}") from exc


def sample_randomization(
    spec: SyntheticDataSpec,
    *,
    episode_seed: int,
    episode_id: str,
) -> RandomizationManifest:
    """Sample exact bounded values without claiming they were applied in Isaac."""

    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
        raise SyntheticDataError("episode_seed must be an integer")
    episode_name = str(episode_id).strip()
    if not episode_name:
        raise SyntheticDataError("episode_id must not be empty")
    parameters: dict[str, dict[str, Any]] = {}
    for name in sorted(spec.parameters):
        parameter = spec.parameters[name]
        parameters[name] = {
            **parameter.as_dict(),
            "requested_value": parameter.sample(episode_seed),
            "application_status": "pending",
        }
    noise_digest = hashlib.sha256(
        f"rockwell-synthetic-noise-v1:{episode_seed}".encode("utf-8")
    ).digest()
    return RandomizationManifest(
        schema_version=1,
        episode_id=episode_name,
        episode_seed=episode_seed,
        spec_path=spec.source_path,
        spec_sha256=spec.source_sha256,
        capture=spec.capture.as_dict(),
        range_warning=spec.range_warning,
        parameters=parameters,
        sensor_noise_seed=int.from_bytes(noise_digest[:4], "big", signed=False),
    )


def finalize_application_status(
    manifest: RandomizationManifest,
    *,
    applied: Iterable[str],
    skipped: Mapping[str, str],
) -> dict[str, Any]:
    """Classify every requested parameter as applied or skipped with a reason.

    The returned dictionary is suitable for ``randomization_manifest.json``.
    It refuses partial accounting so a run cannot claim randomization merely
    because values were sampled.
    """

    applied_set = {str(name) for name in applied}
    skipped_map = {str(name): str(reason).strip() for name, reason in skipped.items()}
    known = set(manifest.parameters)
    unknown = (applied_set | set(skipped_map)) - known
    overlap = applied_set & set(skipped_map)
    missing = known - applied_set - set(skipped_map)
    if unknown:
        raise SyntheticDataError(f"unknown application parameters: {sorted(unknown)}")
    if overlap:
        raise SyntheticDataError(f"parameters both applied and skipped: {sorted(overlap)}")
    if missing:
        raise SyntheticDataError(f"unclassified application parameters: {sorted(missing)}")
    if any(not reason for reason in skipped_map.values()):
        raise SyntheticDataError("every skipped parameter requires a reason")

    result = manifest.as_dict()
    for name, entry in result["parameters"].items():
        if name in applied_set:
            entry["application_status"] = "applied"
        else:
            entry["application_status"] = "skipped"
            entry["skip_reason"] = skipped_map[name]
    result["manifest_sha256_before_application_status"] = manifest.sha256
    result["all_parameters_classified"] = True
    return result


def load_capture_calibration(spec: SyntheticDataSpec) -> Mapping[str, Any]:
    """Load the exact calibrated camera entry referenced by the capture policy."""

    path, _ = _repo_relative_file(
        spec.capture.calibration_file, "capture.calibration_file"
    )
    if _sha256_bytes(path.read_bytes()) != spec.capture.calibration_sha256:
        raise SyntheticDataError("camera calibration changed after spec was loaded")
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "calibration")
    return _mapping(root.get(spec.capture.calibration_entry), "camera calibration")


def radial_distortion_maps(
    width: int,
    height: int,
    fx: float,
    k1: float,
):
    """Build OpenCV remap arrays that add the calibrated one-term distortion.

    The input is an undistorted/pinhole render.  Output pixels use the supplied
    calibration model with principal point ``(width/2, height/2)``.  The
    inverse is solved iteratively because ``cv2.remap`` maps output to input.
    """

    import numpy as np

    width = int(width)
    height = int(height)
    focal = _finite(fx, "fx")
    radial = _finite(k1, "k1")
    if width < 2 or height < 2 or focal <= 0:
        raise SyntheticDataError("invalid distortion image size/focal length")
    yy, xx = np.indices((height, width), dtype=np.float64)
    xd = (xx - width / 2.0) / focal
    yd = (yy - height / 2.0) / focal
    xu = xd.copy()
    yu = yd.copy()
    for _ in range(6):
        scale = 1.0 + radial * (xu * xu + yu * yu)
        xu = xd / scale
        yu = yd / scale
    map_x = (xu * focal + width / 2.0).astype(np.float32)
    map_y = (yu * focal + height / 2.0).astype(np.float32)
    return map_x, map_y


def apply_radial_distortion(image, *, fx: float, k1: float, nearest: bool = False):
    """Apply the same radial transform to RGB, masks, or other image labels."""

    import cv2
    import numpy as np

    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise SyntheticDataError("distortion input must be HxW or HxWxC")
    height, width = array.shape[:2]
    map_x, map_y = radial_distortion_maps(width, height, fx, k1)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(
        array,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def distort_points_uv(
    points_uv: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    fx: float,
    k1: float,
) -> tuple[tuple[float, float], ...]:
    """Forward-distort point labels with the checked-in one-term camera model."""

    focal = _finite(fx, "fx")
    radial = _finite(k1, "k1")
    if focal <= 0:
        raise SyntheticDataError("fx must be positive")
    result = []
    for index, point in enumerate(points_uv):
        if len(point) != 2:
            raise SyntheticDataError(f"point {index} must have u and v")
        u = _finite(point[0], f"point {index} u")
        v = _finite(point[1], f"point {index} v")
        x = (u - width / 2.0) / focal
        y = (v - height / 2.0) / focal
        scale = 1.0 + radial * (x * x + y * y)
        result.append((x * scale * focal + width / 2.0, y * scale * focal + height / 2.0))
    return tuple(result)


@dataclass(frozen=True)
class SensorAugmentationResult:
    image_rgb: Any
    applied_parameters: tuple[str, ...]
    processing: Mapping[str, Any]


def augment_sensor_rgb(
    image_rgb,
    manifest: RandomizationManifest,
    *,
    frame_index: int,
    camera_calibration: Mapping[str, Any] | None = None,
) -> SensorAugmentationResult:
    """Create a deterministic derived RGB image; never overwrite the raw PNG.

    This approximates visible sensor artifacts in display RGB.  It is useful for
    robustness testing, but it is not a replacement for measurements of the
    real camera response, motion blur, auto-exposure, or rolling shutter.
    """

    import cv2
    import numpy as np

    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise SyntheticDataError("frame_index must be a non-negative integer")
    if frame_index < 0:
        raise SyntheticDataError("frame_index must be a non-negative integer")
    frame_seed_digest = hashlib.sha256(
        (
            "rockwell-sensor-noise-frame-v1:"
            f"{manifest.sensor_noise_seed}:{frame_index}"
        ).encode("utf-8")
    ).digest()
    frame_noise_seed = int.from_bytes(frame_seed_digest[:8], "big", signed=False)

    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise SyntheticDataError("sensor input must be HxWx3 uint8 RGB")
    expected_width, expected_height = manifest.capture["resolution"]
    if (image.shape[1], image.shape[0]) != (expected_width, expected_height):
        raise SyntheticDataError(
            "sensor input must use manifest native capture resolution "
            f"{expected_width}x{expected_height}"
        )
    sensor_names = tuple(
        name
        for name, entry in manifest.parameters.items()
        if entry["consumer"] == "sensor_postprocess"
    )
    required = {
        "sensor.exposure_stops",
        "sensor.white_balance_r",
        "sensor.white_balance_g",
        "sensor.white_balance_b",
        "sensor.gamma",
        "sensor.gaussian_noise_std_dn",
        "sensor.gaussian_blur_sigma_px",
        "sensor.vignette_strength",
        "sensor.jpeg_quality",
    }
    if set(sensor_names) != required:
        raise SyntheticDataError(
            "sensor manifest parameter set changed; update augmentation atomically"
        )

    working = image
    distortion_applied = False
    if bool(manifest.capture["apply_calibrated_radial_distortion"]):
        calibration = _mapping(camera_calibration, "camera_calibration")
        calibration_size = tuple(int(v) for v in calibration.get("size", ()))
        if calibration_size != (expected_width, expected_height):
            raise SyntheticDataError("camera calibration size does not match image")
        working = apply_radial_distortion(
            working,
            fx=_finite(calibration.get("fx"), "camera fx"),
            k1=_finite(calibration.get("k1"), "camera k1"),
        )
        distortion_applied = True

    values = {name: manifest.value(name) for name in required}
    floating = working.astype(np.float32) / 255.0
    floating *= 2.0 ** float(values["sensor.exposure_stops"])
    floating *= np.asarray(
        [
            values["sensor.white_balance_r"],
            values["sensor.white_balance_g"],
            values["sensor.white_balance_b"],
        ],
        dtype=np.float32,
    ).reshape((1, 1, 3))
    floating = np.clip(floating, 0.0, 1.0)
    floating = np.power(floating, 1.0 / float(values["sensor.gamma"]))

    strength = float(values["sensor.vignette_strength"])
    if strength > 0:
        height, width = floating.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float32)
        xx = (xx - width / 2.0) / (width / 2.0)
        yy = (yy - height / 2.0) / (height / 2.0)
        radial = np.clip((xx * xx + yy * yy) / 2.0, 0.0, 1.0)
        floating *= (1.0 - strength * radial)[..., None]

    noise_std = float(values["sensor.gaussian_noise_std_dn"]) / 255.0
    if noise_std > 0:
        rng = np.random.default_rng(frame_noise_seed)
        floating += rng.normal(0.0, noise_std, size=floating.shape).astype(np.float32)
    working = np.clip(np.rint(floating * 255.0), 0, 255).astype(np.uint8)

    blur_sigma = float(values["sensor.gaussian_blur_sigma_px"])
    if blur_sigma >= 0.05:
        radius = max(1, int(math.ceil(3.0 * blur_sigma)))
        kernel = radius * 2 + 1
        working = cv2.GaussianBlur(working, (kernel, kernel), blur_sigma)

    quality = int(values["sensor.jpeg_quality"])
    if quality < 100:
        bgr = cv2.cvtColor(working, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise SyntheticDataError("OpenCV could not encode JPEG augmentation")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise SyntheticDataError("OpenCV could not decode JPEG augmentation")
        working = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    return SensorAugmentationResult(
        image_rgb=working,
        applied_parameters=tuple(sorted(required)),
        processing={
            "input": "raw lossless pinhole RGB",
            "output": "derived sensor-augmented RGB",
            "calibrated_radial_distortion_applied": distortion_applied,
            "sensor_noise_base_seed": manifest.sensor_noise_seed,
            "sensor_noise_frame_seed": frame_noise_seed,
            "frame_index": frame_index,
            "warning": (
                "Display-RGB approximation only; physical camera response, motion "
                "blur, auto-exposure and rolling shutter remain unmeasured."
            ),
        },
    )


def _finite_vector(value: Any, size: int, name: str) -> list[float]:
    try:
        result = [_finite(v, name) for v in value]
    except TypeError as exc:
        raise SyntheticDataError(f"{name} must contain {size} values") from exc
    if len(result) != size:
        raise SyntheticDataError(f"{name} must contain {size} values")
    return result


def _gripper_state(
    *,
    jaw_half_opening_m: Any,
    closure_fraction: Any,
    name: str,
) -> dict[str, float]:
    opening = _finite(jaw_half_opening_m, f"{name}.jaw_half_opening_m")
    closure = _finite(closure_fraction, f"{name}.closure_fraction")
    if opening < 0.0:
        raise SyntheticDataError(f"{name}.jaw_half_opening_m must be non-negative")
    if not 0.0 <= closure <= 1.0:
        raise SyntheticDataError(f"{name}.closure_fraction must be in [0, 1]")
    return {
        "jaw_half_opening_m": opening,
        # Dimensionless position over the configured full mechanical stroke:
        # 0=open and 1=closed. This is not the seventh protocol command.
        "closure_fraction": closure,
    }


_AMODAL_BBOX_STATUSES = {
    "valid",
    "valid_clipped",
    "invalid_behind_camera",
    "invalid_outside_image",
    "not_generated",
}


def _amodal_bbox(
    raw: Mapping[str, Any],
    *,
    prefix: str,
    name: str,
    resolution: tuple[int, int],
) -> tuple[list[float] | None, str]:
    bbox_raw = raw.get(f"{prefix}_amodal_bbox_xyxy_px")
    bbox = (
        None
        if bbox_raw is None
        else _finite_vector(bbox_raw, 4, f"{name}.{prefix}_amodal_bbox")
    )
    status = str(raw.get(f"{prefix}_amodal_bbox_status", "")).strip()
    if status not in _AMODAL_BBOX_STATUSES:
        raise SyntheticDataError(
            f"{name}.{prefix}_amodal_bbox_status must be one of "
            f"{sorted(_AMODAL_BBOX_STATUSES)}"
        )
    is_valid = status in {"valid", "valid_clipped"}
    if is_valid != (bbox is not None):
        raise SyntheticDataError(
            f"{name}.{prefix}_amodal_bbox status/value disagree"
        )
    if bbox is not None and (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        raise SyntheticDataError(
            f"{name}.{prefix}_amodal_bbox must have positive width and height"
        )
    if bbox is not None:
        width, height = resolution
        if not (
            0.0 <= bbox[0] <= width
            and 0.0 <= bbox[2] <= width
            and 0.0 <= bbox[1] <= height
            and 0.0 <= bbox[3] <= height
        ):
            raise SyntheticDataError(
                f"{name}.{prefix}_amodal_bbox must be clipped to the image"
            )
        if status == "valid_clipped" and not (
            bbox[0] == 0.0
            or bbox[1] == 0.0
            or bbox[2] == width
            or bbox[3] == height
        ):
            raise SyntheticDataError(
                f"{name}.{prefix}_amodal_bbox marked clipped but touches no image edge"
            )
    return bbox, status


def _validate_object_label(
    raw: Mapping[str, Any], index: int, resolution: tuple[int, int]
) -> dict[str, Any]:
    name = f"objects[{index}]"
    track_id = str(raw.get("track_id", "")).strip()
    class_name = str(raw.get("class_name", "")).strip()
    colour = str(raw.get("colour", "")).strip().lower()
    if not track_id or not class_name or not colour:
        raise SyntheticDataError(f"{name} requires track_id, class_name and colour")
    visible_raw = raw.get("visible_fraction")
    visible_fraction = (
        None
        if visible_raw is None
        else _finite(visible_raw, f"{name}.visible_fraction")
    )
    if visible_fraction is not None and not 0.0 <= visible_fraction <= 1.0:
        raise SyntheticDataError(f"{name}.visible_fraction must be in [0, 1]")
    visibility_status = str(
        raw.get(
            "visibility_status",
            "unknown_not_annotated" if visible_fraction is None else "measured",
        )
    ).strip()
    if visibility_status not in {"measured", "unknown_not_annotated"}:
        raise SyntheticDataError(
            f"{name}.visibility_status must be measured or unknown_not_annotated"
        )
    bbox_type = str(raw.get("amodal_bbox_type", "")).strip()
    if bbox_type != "projected_3d_cuboid_clipped_to_image":
        raise SyntheticDataError(
            f"{name}.amodal_bbox_type must explicitly identify the projected 3D cuboid"
        )
    raw_bbox_value, raw_bbox_status = _amodal_bbox(
        raw, prefix="raw", name=name, resolution=resolution
    )
    derived_bbox_value, derived_bbox_status = _amodal_bbox(
        raw, prefix="derived", name=name, resolution=resolution
    )
    mask = raw.get("instance_mask_relpath")
    mask_value = None if mask is None else _artifact_relative_path(mask, f"{name}.mask")
    return {
        "track_id": track_id,
        "class_name": class_name,
        "colour": colour,
        "position_base_m": _finite_vector(
            raw.get("position_base_m"), 3, f"{name}.position_base_m"
        ),
        "quaternion_base_wxyz": _finite_vector(
            raw.get("quaternion_base_wxyz"), 4, f"{name}.quaternion_base_wxyz"
        ),
        "dimensions_m": _finite_vector(
            raw.get("dimensions_m"), 3, f"{name}.dimensions_m"
        ),
        "visible_fraction": visible_fraction,
        "visibility_status": visibility_status,
        "occluded": (
            None
            if visible_fraction is None
            else bool(raw.get("occluded", visible_fraction < 0.999))
        ),
        # These are amodal projections of the complete simulator-truth 3D
        # cuboid.  They are not visible-pixel boxes and do not imply visibility.
        "amodal_bbox_type": bbox_type,
        "raw_amodal_bbox_xyxy_px": raw_bbox_value,
        "raw_amodal_bbox_status": raw_bbox_status,
        "derived_amodal_bbox_xyxy_px": derived_bbox_value,
        "derived_amodal_bbox_status": derived_bbox_status,
        "instance_mask_relpath": mask_value,
    }


def build_synchronized_frame_record(
    spec: SyntheticDataSpec,
    manifest: RandomizationManifest,
    *,
    frame_index: int,
    raw_image_relpath: str,
    capture_time_s: float,
    robot_state_time_s: float,
    scene_state_time_s: float,
    measured_arm_joints_deg: Sequence[float],
    measured_gripper_half_opening_m: float,
    measured_gripper_closure_fraction: float,
    measured_tcp_pose_uf0_mm_deg: Sequence[float],
    commanded_arm_joints_deg: Sequence[float],
    commanded_gripper_half_opening_m: float,
    commanded_gripper_closure_fraction: float,
    commanded_tcp_pose_uf0_mm_deg: Sequence[float],
    scene_version: str,
    phase: str,
    objects: Sequence[Mapping[str, Any]],
    derived_image_relpath: str | None = None,
) -> dict[str, Any]:
    """Build a leakage-aware, time-synchronized per-frame label record."""

    if manifest.spec_sha256 != spec.source_sha256:
        raise SyntheticDataError("manifest was sampled from a different spec")
    if frame_index < 0:
        raise SyntheticDataError("frame_index must be non-negative")
    capture_time = _finite(capture_time_s, "capture_time_s")
    robot_time = _finite(robot_state_time_s, "robot_state_time_s")
    scene_time = _finite(scene_state_time_s, "scene_state_time_s")
    skew = max(abs(robot_time - capture_time), abs(scene_time - capture_time))
    if skew > spec.capture.maximum_state_skew_s:
        raise SyntheticDataError(
            f"state/image skew {skew:.6f}s exceeds "
            f"{spec.capture.maximum_state_skew_s:.6f}s"
        )
    scene_name = str(scene_version).strip()
    phase_name = str(phase).strip()
    if not scene_name or not phase_name:
        raise SyntheticDataError("scene_version and phase must not be empty")
    raw_path = _artifact_relative_path(raw_image_relpath, "raw_image_relpath")
    if PurePosixPath(raw_path).suffix.lower() != ".png":
        raise SyntheticDataError("raw training image must be PNG")
    derived_path = (
        None
        if derived_image_relpath is None
        else _artifact_relative_path(derived_image_relpath, "derived_image_relpath")
    )
    object_labels = [
        _validate_object_label(
            _mapping(value, f"objects[{index}]"),
            index,
            spec.capture.resolution,
        )
        for index, value in enumerate(objects)
    ]
    return {
        "schema_version": 2,
        "episode_id": manifest.episode_id,
        "frame_index": int(frame_index),
        "capture_time_s": capture_time,
        "camera": {
            **spec.capture.as_dict(),
            "raw_image_relpath": raw_path,
            "derived_sensor_image_relpath": derived_path,
        },
        "synchronization": {
            "robot_state_time_s": robot_time,
            "scene_state_time_s": scene_time,
            "maximum_observed_skew_s": skew,
            "allowed_skew_s": spec.capture.maximum_state_skew_s,
            "within_tolerance": True,
        },
        "policy_input": {
            "measured_robot_state": {
                "source": "Isaac articulation readback after image capture",
                "arm_joint_positions_deg": _finite_vector(
                    measured_arm_joints_deg,
                    6,
                    "measured_arm_joints_deg",
                ),
                "gripper": _gripper_state(
                    jaw_half_opening_m=measured_gripper_half_opening_m,
                    closure_fraction=measured_gripper_closure_fraction,
                    name="measured_gripper",
                ),
                "tcp_pose_uf0_mm_deg": _finite_vector(
                    measured_tcp_pose_uf0_mm_deg,
                    6,
                    "measured_tcp_pose_uf0_mm_deg",
                ),
            },
            "scene_version": scene_name,
            "phase": phase_name,
        },
        "commanded_robot_state": {
            "source": "last Isaac articulation target; TCP derived through FK",
            "arm_joint_targets_deg": _finite_vector(
                commanded_arm_joints_deg,
                6,
                "commanded_arm_joints_deg",
            ),
            "gripper": _gripper_state(
                jaw_half_opening_m=commanded_gripper_half_opening_m,
                closure_fraction=commanded_gripper_closure_fraction,
                name="commanded_gripper",
            ),
            "tcp_target_uf0_mm_deg": _finite_vector(
                commanded_tcp_pose_uf0_mm_deg,
                6,
                "commanded_tcp_pose_uf0_mm_deg",
            ),
        },
        "evaluator_only": {
            "warning": (
                "Simulator truth/annotations are labels and evaluation data, "
                "never policy inputs or runtime pose repairs."
            ),
            "objects": object_labels,
        },
        "randomization": {
            "manifest_sha256": manifest.sha256,
            "episode_seed": manifest.episode_seed,
        },
    }


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write canonical JSON without leaving a partial manifest after failure."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(destination)


__all__ = [
    "CapturePolicy",
    "DEFAULT_SYNTHETIC_DATA_CONFIG_PATH",
    "ParameterSpec",
    "RandomizationManifest",
    "SensorAugmentationResult",
    "SyntheticDataError",
    "SyntheticDataSpec",
    "apply_radial_distortion",
    "augment_sensor_rgb",
    "build_synchronized_frame_record",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "distort_points_uv",
    "finalize_application_status",
    "load_capture_calibration",
    "load_synthetic_data_spec",
    "radial_distortion_maps",
    "sample_randomization",
    "write_json_atomic",
]
