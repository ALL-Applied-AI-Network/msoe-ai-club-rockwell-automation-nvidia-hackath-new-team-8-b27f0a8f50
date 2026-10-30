"""Strict, lazy plugin loading for team-owned vision and grasp systems.

``ROCKWELL_INTEGRATION_PROFILE`` selects a UTF-8 JSON profile. Relative
paths resolve from the repository root. Reading a profile never imports model
code; the referenced module is imported only when its factory is invoked.
Plugin modules are trusted executable code. Validation prevents ambiguous
configuration, not malicious Python.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


INTEGRATION_PROFILE_ENV = "ROCKWELL_INTEGRATION_PROFILE"
INTEGRATION_PROFILE_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MAX_PROFILE_BYTES = 1_000_000


class IntegrationLoadError(RuntimeError):
    """Raised before execution when an integration cannot be loaded safely."""


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationLoadError(f"{name} must be a non-empty string")
    return value.strip()


def _strict_object(
    value: Any,
    name: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IntegrationLoadError(f"{name} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise IntegrationLoadError(
            f"{name} contains unknown keys: {', '.join(sorted(unknown))}"
        )
    missing = required - set(value)
    if missing:
        raise IntegrationLoadError(
            f"{name} is missing required keys: {', '.join(sorted(missing))}"
        )
    return value


def _string_tuple(value: Any, name: str, *, nonempty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise IntegrationLoadError(f"{name} must be a JSON array")
    result = tuple(_nonempty_string(item, f"{name} item") for item in value)
    if nonempty and not result:
        raise IntegrationLoadError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise IntegrationLoadError(f"{name} must not contain duplicates")
    return result


def _context_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Freeze a directly constructed context's declared camera IDs."""

    if not isinstance(value, (list, tuple)):
        raise IntegrationLoadError(f"{name} must be a list or tuple")
    result = tuple(_nonempty_string(item, f"{name} item") for item in value)
    if len(set(result)) != len(result):
        raise IntegrationLoadError(f"{name} must not contain duplicates")
    return result


def _string_mapping(value: Any, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise IntegrationLoadError(f"{name} must be a JSON object")
    result = {
        _nonempty_string(key, f"{name} key"): _nonempty_string(
            item, f"{name}.{key}"
        )
        for key, item in value.items()
    }
    return MappingProxyType(result)


def _freeze_json(value: Any, name: str = "options") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IntegrationLoadError(f"{name} must not contain non-finite numbers")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _nonempty_string(key, f"{name} key"): _freeze_json(
                    item, f"{name}.{key}"
                )
                for key, item in value.items()
            }
        )
    raise IntegrationLoadError(f"{name} contains a non-JSON value")


def _factory_reference(value: Any, name: str) -> str:
    reference = _nonempty_string(value, name)
    module_name, separator, attribute = reference.partition(":")
    if (
        separator != ":"
        or not attribute
        or ":" in attribute
        or not all(part.isidentifier() for part in module_name.split("."))
        or not attribute.isidentifier()
    ):
        raise IntegrationLoadError(
            f"{name} must use the form 'package.module:factory_name'"
        )
    return reference


@dataclass(frozen=True)
class VisionPluginSpec:
    factory: str
    camera_ids: tuple[str, ...]
    depth_camera_ids: tuple[str, ...]
    calibration_ids: Mapping[str, str]
    depth_calibration_ids: Mapping[str, str]
    options: Mapping[str, Any]


@dataclass(frozen=True)
class GraspPluginSpec:
    factory: str
    options: Mapping[str, Any]


@dataclass(frozen=True)
class IntegrationProfile:
    profile_id: str
    source_path: str
    source_sha256: str
    vision: VisionPluginSpec | None = None
    grasp: GraspPluginSpec | None = None

    @property
    def provenance(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": INTEGRATION_PROFILE_SCHEMA_VERSION,
                "profile_id": self.profile_id,
                "source_path": self.source_path,
                "source_sha256": self.source_sha256,
                "vision_factory": None if self.vision is None else self.vision.factory,
                "grasp_factory": None if self.grasp is None else self.grasp.factory,
            }
        )


@dataclass(frozen=True)
class VisionFactoryContext:
    config: Any
    camera_projectors: Mapping[str, Any]
    calibration_ids: Mapping[str, str]
    expected_cube_edge_m: float | None
    checkpoint: str | None
    detector: Any | None
    options: Mapping[str, Any]
    camera_ids: tuple[str, ...]
    depth_camera_ids: tuple[str, ...]
    declared_calibration_ids: Mapping[str, str]
    declared_depth_calibration_ids: Mapping[str, str]
    profile_id: str
    profile_path: str
    profile_sha256: str

    def __post_init__(self) -> None:
        camera_ids = _context_string_tuple(self.camera_ids, "camera_ids")
        depth_camera_ids = _context_string_tuple(
            self.depth_camera_ids, "depth_camera_ids"
        )
        if not camera_ids and not depth_camera_ids:
            raise IntegrationLoadError(
                "vision context must request at least one RGB or depth camera"
            )
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "depth_camera_ids", depth_camera_ids)
        if not isinstance(self.camera_projectors, Mapping):
            raise IntegrationLoadError("camera_projectors must be a mapping")
        object.__setattr__(
            self, "camera_projectors", MappingProxyType(dict(self.camera_projectors))
        )
        object.__setattr__(
            self,
            "calibration_ids",
            _string_mapping(self.calibration_ids, "calibration_ids"),
        )
        if set(self.calibration_ids) != set(camera_ids):
            raise IntegrationLoadError(
                "calibration_ids keys must exactly match camera_ids"
            )
        if not isinstance(self.options, Mapping):
            raise IntegrationLoadError("options must be a mapping")
        object.__setattr__(self, "options", _freeze_json(self.options))
        object.__setattr__(
            self,
            "declared_calibration_ids",
            _string_mapping(
                self.declared_calibration_ids, "declared_calibration_ids"
            ),
        )
        object.__setattr__(
            self,
            "declared_depth_calibration_ids",
            _string_mapping(
                self.declared_depth_calibration_ids,
                "declared_depth_calibration_ids",
            ),
        )
        if set(self.declared_calibration_ids) != set(camera_ids):
            raise IntegrationLoadError(
                "declared_calibration_ids keys must exactly match camera_ids"
            )
        if set(self.declared_depth_calibration_ids) != set(depth_camera_ids):
            raise IntegrationLoadError(
                "declared_depth_calibration_ids keys must exactly match "
                "depth_camera_ids"
            )
        object.__setattr__(
            self, "profile_id", _nonempty_string(self.profile_id, "profile_id")
        )
        object.__setattr__(
            self,
            "profile_path",
            _nonempty_string(self.profile_path, "profile_path"),
        )
        profile_sha256 = _nonempty_string(
            self.profile_sha256, "profile_sha256"
        ).lower()
        if len(profile_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in profile_sha256
        ):
            raise IntegrationLoadError(
                "profile_sha256 must be a 64-character hexadecimal digest"
            )
        object.__setattr__(self, "profile_sha256", profile_sha256)


@dataclass(frozen=True)
class GraspFactoryContext:
    config: Any
    checkpoint: str | None
    options: Mapping[str, Any]
    profile_id: str
    profile_path: str
    profile_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.options, Mapping):
            raise IntegrationLoadError("options must be a mapping")
        object.__setattr__(self, "options", _freeze_json(self.options))
        object.__setattr__(
            self, "profile_id", _nonempty_string(self.profile_id, "profile_id")
        )
        object.__setattr__(
            self,
            "profile_path",
            _nonempty_string(self.profile_path, "profile_path"),
        )
        profile_sha256 = _nonempty_string(
            self.profile_sha256, "profile_sha256"
        ).lower()
        if len(profile_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in profile_sha256
        ):
            raise IntegrationLoadError(
                "profile_sha256 must be a 64-character hexadecimal digest"
            )
        object.__setattr__(self, "profile_sha256", profile_sha256)


def _reject_constant(value: str) -> None:
    raise IntegrationLoadError(f"profile contains invalid JSON constant {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrationLoadError(f"profile contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_vision(value: Any) -> VisionPluginSpec:
    raw = _strict_object(
        value,
        "vision",
        allowed=frozenset(
            {
                "factory",
                "camera_ids",
                "depth_camera_ids",
                "calibration_ids",
                "depth_calibration_ids",
                "options",
            }
        ),
        required=frozenset({"factory", "camera_ids", "calibration_ids"}),
    )
    camera_ids = _string_tuple(
        raw["camera_ids"], "vision.camera_ids", nonempty=False
    )
    depth_camera_ids = _string_tuple(
        raw.get("depth_camera_ids", []),
        "vision.depth_camera_ids",
        nonempty=False,
    )
    calibration_ids = _string_mapping(
        raw["calibration_ids"], "vision.calibration_ids"
    )
    depth_calibration_ids = _string_mapping(
        raw.get("depth_calibration_ids", {}),
        "vision.depth_calibration_ids",
    )
    if not camera_ids and not depth_camera_ids:
        raise IntegrationLoadError(
            "vision must request at least one RGB or depth camera"
        )
    if set(calibration_ids) != set(camera_ids):
        raise IntegrationLoadError(
            "vision.calibration_ids keys must exactly match vision.camera_ids"
        )
    if set(depth_calibration_ids) != set(depth_camera_ids):
        raise IntegrationLoadError(
            "vision.depth_calibration_ids keys must exactly match "
            "vision.depth_camera_ids"
        )
    options_raw = raw.get("options", {})
    if not isinstance(options_raw, dict):
        raise IntegrationLoadError("vision.options must be a JSON object")
    return VisionPluginSpec(
        factory=_factory_reference(raw["factory"], "vision.factory"),
        camera_ids=camera_ids,
        depth_camera_ids=depth_camera_ids,
        calibration_ids=calibration_ids,
        depth_calibration_ids=depth_calibration_ids,
        options=_freeze_json(options_raw, "vision.options"),
    )


def _parse_grasp(value: Any) -> GraspPluginSpec:
    raw = _strict_object(
        value,
        "grasp",
        allowed=frozenset({"factory", "options"}),
        required=frozenset({"factory"}),
    )
    options_raw = raw.get("options", {})
    if not isinstance(options_raw, dict):
        raise IntegrationLoadError("grasp.options must be a JSON object")
    return GraspPluginSpec(
        factory=_factory_reference(raw["factory"], "grasp.factory"),
        options=_freeze_json(options_raw, "grasp.options"),
    )


def load_integration_profile(path_value: str | os.PathLike[str]) -> IntegrationProfile:
    """Read and validate one profile without importing either plugin module."""

    raw_path = Path(path_value).expanduser()
    source = raw_path if raw_path.is_absolute() else REPOSITORY_ROOT / raw_path
    try:
        source = source.resolve(strict=True)
        if not source.is_file():
            raise IntegrationLoadError(f"integration profile is not a file: {source}")
        payload = source.read_bytes()
    except IntegrationLoadError:
        raise
    except (OSError, RuntimeError) as exc:
        raise IntegrationLoadError(
            f"cannot read integration profile {source}: {exc}"
        ) from exc
    if len(payload) > _MAX_PROFILE_BYTES:
        raise IntegrationLoadError(
            f"integration profile exceeds {_MAX_PROFILE_BYTES} bytes: {source}"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except IntegrationLoadError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrationLoadError(
            f"integration profile is not strict UTF-8 JSON: {source}: {exc}"
        ) from exc
    raw = _strict_object(
        value,
        "profile",
        allowed=frozenset({"schema_version", "profile_id", "vision", "grasp"}),
        required=frozenset({"schema_version", "profile_id"}),
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != INTEGRATION_PROFILE_SCHEMA_VERSION
    ):
        raise IntegrationLoadError("profile.schema_version must be the integer 1")
    if "vision" not in raw and "grasp" not in raw:
        raise IntegrationLoadError("profile must configure vision, grasp, or both")
    return IntegrationProfile(
        profile_id=_nonempty_string(raw["profile_id"], "profile.profile_id"),
        source_path=str(source),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        vision=None if "vision" not in raw else _parse_vision(raw["vision"]),
        grasp=None if "grasp" not in raw else _parse_grasp(raw["grasp"]),
    )


def load_active_integration_profile() -> IntegrationProfile | None:
    """Load the environment-selected profile, or preserve the baseline."""

    path_value = os.environ.get(INTEGRATION_PROFILE_ENV, "").strip()
    if not path_value:
        return None
    return load_integration_profile(path_value)


def load_factory(reference: str, component: str) -> Callable[[Any], Any]:
    """Lazily import one already-validated ``module:factory`` reference."""

    checked_reference = _factory_reference(reference, f"{component} factory")
    checked_component = _nonempty_string(component, "component")
    module_name, attribute = checked_reference.split(":", 1)
    try:
        module = import_module(module_name)
        factory = getattr(module, attribute)
    except Exception as exc:
        raise IntegrationLoadError(
            f"cannot load {checked_component} factory {checked_reference}: {exc}"
        ) from exc
    if not callable(factory):
        raise IntegrationLoadError(
            f"{checked_component} factory {checked_reference} is not callable"
        )
    return factory


def create_profile_vision_provider(
    profile: IntegrationProfile,
    context: VisionFactoryContext,
) -> Any:
    """Invoke a selected vision factory and validate its structural contract."""

    if profile.vision is None:
        raise IntegrationLoadError("active profile does not configure vision")
    factory = load_factory(profile.vision.factory, "vision")
    try:
        provider = factory(context)
    except Exception as exc:
        raise IntegrationLoadError(
            f"vision factory {profile.vision.factory} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    from .perception import VisionProvider

    if not isinstance(provider, VisionProvider):
        raise IntegrationLoadError(
            f"vision factory {profile.vision.factory} did not return a "
            "VisionProvider"
        )
    return provider


def create_profile_grasp_pipeline(
    profile: IntegrationProfile,
    context: GraspFactoryContext,
) -> Any:
    """Invoke a selected grasp factory and require the shared pipeline type."""

    if profile.grasp is None:
        raise IntegrationLoadError("active profile does not configure grasp")
    factory = load_factory(profile.grasp.factory, "grasp")
    try:
        pipeline = factory(context)
    except Exception as exc:
        raise IntegrationLoadError(
            f"grasp factory {profile.grasp.factory} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    from .grasping import GraspPlanningPipeline

    if not isinstance(pipeline, GraspPlanningPipeline):
        raise IntegrationLoadError(
            f"grasp factory {profile.grasp.factory} did not return a "
            "GraspPlanningPipeline"
        )
    return pipeline


__all__ = [
    "GraspFactoryContext",
    "GraspPluginSpec",
    "INTEGRATION_PROFILE_ENV",
    "INTEGRATION_PROFILE_SCHEMA_VERSION",
    "IntegrationLoadError",
    "IntegrationProfile",
    "VisionFactoryContext",
    "VisionPluginSpec",
    "create_profile_grasp_pipeline",
    "create_profile_vision_provider",
    "load_active_integration_profile",
    "load_factory",
    "load_integration_profile",
]
