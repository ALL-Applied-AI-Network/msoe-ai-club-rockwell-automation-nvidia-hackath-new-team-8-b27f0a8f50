"""Privacy-aware real-camera profiles and deployment-preview augmentation.

The real evidence consists of compressed, downsampled rollout videos.  This
module therefore keeps native lossless Isaac renders as the authoritative
synthetic source and creates a *separate* 640x360 deployment-like preview.  It
does not claim to reproduce the physical camera response or an AV1 bitstream.

Human/setup curation is default-deny.  Frames are eligible for an appearance
fit only when they are explicitly present in the checked-in fit split and do
not overlap the excluded split.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_PROFILE_PATH = REPO_ROOT / "config" / "real_camera_profiles.json"
DEFAULT_FRAME_CURATION_PATH = REPO_ROOT / "config" / "hardware_frame_curation.json"


class CameraAppearanceError(ValueError):
    """Raised when camera evidence or preview inputs are unsafe or malformed."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CameraAppearanceError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise CameraAppearanceError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CameraAppearanceError(f"{name} must be an integer") from exc
    if result != value:
        raise CameraAppearanceError(f"{name} must be an integer")
    return result


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraAppearanceError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise CameraAppearanceError(f"{name} must be finite")
    return result


def _resolution(value: Any, name: str) -> tuple[int, int]:
    try:
        width, height = (_integer(item, name) for item in value)
    except (TypeError, ValueError) as exc:
        raise CameraAppearanceError(f"{name} must contain width and height") from exc
    if width < 2 or height < 2:
        raise CameraAppearanceError(f"{name} must be at least 2x2")
    return width, height


def _rgb_triplet(value: Any, name: str) -> tuple[float, float, float]:
    try:
        result = tuple(_finite(item, name) for item in value)
    except TypeError as exc:
        raise CameraAppearanceError(f"{name} must contain three values") from exc
    if len(result) != 3 or any(item < 0.0 or item > 1.0 for item in result):
        raise CameraAppearanceError(f"{name} must contain three values in [0, 1]")
    return result  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_file(path_value: Any, name: str) -> tuple[Path, str]:
    relative = PurePosixPath(str(path_value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise CameraAppearanceError(f"{name} must be a repository-relative path")
    path = (REPO_ROOT / Path(*relative.parts)).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise CameraAppearanceError(f"{name} escapes the repository") from exc
    if not path.is_file():
        raise CameraAppearanceError(f"{name} does not exist: {relative.as_posix()}")
    return path, relative.as_posix()


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraAppearanceError(f"cannot load {path}: {exc}") from exc


@dataclass(frozen=True)
class CuratedRange:
    dataset: str
    episode: int
    start_frame: int
    end_frame_exclusive: int
    tags: tuple[str, ...]

    def contains(self, dataset: str, episode: int, frame_index: int) -> bool:
        return (
            dataset == self.dataset
            and episode == self.episode
            and self.start_frame <= frame_index < self.end_frame_exclusive
        )


@dataclass(frozen=True)
class FrameCuration:
    source_path: str
    source_sha256: str
    default_policy: str
    privacy_rule: str
    splits: Mapping[str, tuple[CuratedRange, ...]]

    def approved(
        self,
        *,
        dataset: str,
        episode: int,
        frame_index: int,
        split: str = "appearance_fit",
    ) -> bool:
        """Return eligibility under an explicit split and privacy exclusions."""

        if frame_index < 0:
            return False
        candidates = self.splits.get(split, ())
        included = any(
            item.contains(dataset, episode, frame_index) for item in candidates
        )
        excluded = any(
            item.contains(dataset, episode, frame_index)
            for item in self.splits.get("excluded_human_or_setup", ())
        )
        return bool(included and not excluded)


def _parse_curated_ranges(value: Any, split: str) -> tuple[CuratedRange, ...]:
    if not isinstance(value, list):
        raise CameraAppearanceError(f"curation split {split} must be a list")
    parsed: list[CuratedRange] = []
    for entry_index, raw_value in enumerate(value):
        raw = _mapping(raw_value, f"curation {split}[{entry_index}]")
        dataset = str(raw.get("dataset", "")).strip()
        episode = _integer(raw.get("episode"), f"curation {split} episode")
        tags_raw = raw.get("tags", [])
        if not dataset or episode < 0 or not isinstance(tags_raw, list):
            raise CameraAppearanceError(f"invalid curation entry in split {split}")
        tags = tuple(str(tag).strip() for tag in tags_raw)
        if any(not tag for tag in tags):
            raise CameraAppearanceError(f"empty curation tag in split {split}")
        ranges = raw.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            raise CameraAppearanceError(f"curation {split} ranges must be non-empty")
        for range_index, pair in enumerate(ranges):
            try:
                start, end = (
                    _integer(item, f"curation {split} range {range_index}")
                    for item in pair
                )
            except (TypeError, ValueError) as exc:
                raise CameraAppearanceError(
                    f"curation {split} range {range_index} must have two integers"
                ) from exc
            if start < 0 or end <= start:
                raise CameraAppearanceError(
                    f"curation {split} range {range_index} is invalid"
                )
            parsed.append(CuratedRange(dataset, episode, start, end, tags))
    return tuple(parsed)


def _overlap(first: CuratedRange, second: CuratedRange) -> bool:
    return (
        first.dataset == second.dataset
        and first.episode == second.episode
        and first.start_frame < second.end_frame_exclusive
        and second.start_frame < first.end_frame_exclusive
    )


def load_frame_curation(path: str | Path | None = None) -> FrameCuration:
    source = Path(path).resolve() if path is not None else DEFAULT_FRAME_CURATION_PATH
    raw = _load_json(source, "frame curation")
    if _integer(raw.get("schema_version"), "curation schema_version") != 1:
        raise CameraAppearanceError("unsupported frame-curation schema")
    default_policy = str(raw.get("default_policy", "")).strip()
    if default_policy != "deny":
        raise CameraAppearanceError("frame curation must remain default-deny")
    privacy_rule = str(raw.get("privacy_rule", "")).strip()
    if not privacy_rule:
        raise CameraAppearanceError("frame curation requires a privacy rule")
    archives = _mapping(raw.get("archives"), "curation archives")
    for archive_id, archive_value in archives.items():
        archive = _mapping(archive_value, f"archive {archive_id}")
        digest = str(archive.get("sha256", "")).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise CameraAppearanceError(f"archive {archive_id} has invalid SHA-256")
    split_raw = _mapping(raw.get("splits"), "curation splits")
    required = {
        "appearance_fit",
        "wrist_hose_occlusion_stress",
        "held_out_validation",
        "excluded_human_or_setup",
    }
    if not required.issubset(split_raw):
        raise CameraAppearanceError(
            f"curation is missing splits: {sorted(required - set(split_raw))}"
        )
    splits = {
        str(name): _parse_curated_ranges(value, str(name))
        for name, value in split_raw.items()
    }
    for protected_split in ("appearance_fit", "held_out_validation"):
        for candidate in splits[protected_split]:
            for excluded in splits["excluded_human_or_setup"]:
                if _overlap(candidate, excluded):
                    raise CameraAppearanceError(
                        f"{protected_split} overlaps privacy exclusion for "
                        f"{candidate.dataset} episode {candidate.episode}"
                    )
    for fit in splits["appearance_fit"]:
        for held_out in splits["held_out_validation"]:
            if _overlap(fit, held_out):
                raise CameraAppearanceError(
                    "appearance-fit and held-out validation ranges overlap"
                )
    try:
        source_path = source.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source_path = str(source)
    return FrameCuration(
        source_path=source_path,
        source_sha256=_sha256(source),
        default_policy=default_policy,
        privacy_rule=privacy_rule,
        splits=splits,
    )


@dataclass(frozen=True)
class HoseOcclusionProfile:
    enabled: bool
    application_probability: float = 0.0
    curve_count: tuple[int, int] = (0, 0)
    width_px: tuple[int, int] = (0, 0)
    blue_rgb: tuple[int, int, int] = (0, 0, 0)
    shadow_rgb: tuple[int, int, int] = (0, 0, 0)
    shadow_offset_px: tuple[int, int] = (0, 0)
    ignore_mask_dilation_px: int = 0


@dataclass(frozen=True)
class CameraAppearanceProfile:
    camera_id: str
    native_resolution: tuple[int, int]
    calibration_file: str
    calibration_entry: str
    latest_archive_rgb_mean_01: tuple[float, float, float]
    older_archive_rgb_mean_01: tuple[float, float, float]
    curated_fit_sampled_frames: int
    fit_neutral_rgb_median_8bit: tuple[int, int, int]
    fit_neutral_rgb_p05_8bit: tuple[int, int, int]
    fit_neutral_rgb_p95_8bit: tuple[int, int, int]
    fit_neutral_luma_p05_median_p95_8bit: tuple[int, int, int]
    heldout_neutral_rgb_median_8bit: tuple[int, int, int]
    heldout_neutral_luma_p05_median_p95_8bit: tuple[int, int, int]
    profile_status: str
    hose_occlusion: HoseOcclusionProfile
    fixed_training_rgb_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fixed_training_gain_status: str = "identity fallback"


@dataclass(frozen=True)
class CameraAppearanceProfiles:
    source_path: str
    source_sha256: str
    curation: FrameCuration
    deployment_resolution: tuple[int, int]
    deployment_fps: int
    cameras: Mapping[str, CameraAppearanceProfile]
    deployment_pixel_format: str = "yuv420p"
    codec_evidence: str = "unknown"
    crf_evidence: int | None = None
    preview_approximation: str = "not specified"

    def camera(self, camera_id: str) -> CameraAppearanceProfile:
        name = str(camera_id).strip().lower()
        try:
            return self.cameras[name]
        except KeyError as exc:
            raise CameraAppearanceError(f"unknown camera {camera_id!r}") from exc


def _rgb8(value: Any, name: str) -> tuple[int, int, int]:
    try:
        result = tuple(_integer(item, name) for item in value)
    except TypeError as exc:
        raise CameraAppearanceError(f"{name} must contain three integers") from exc
    if len(result) != 3 or any(item < 0 or item > 255 for item in result):
        raise CameraAppearanceError(f"{name} must contain three values in [0, 255]")
    return result  # type: ignore[return-value]


def _bounded_rgb_gain(value: Any, name: str) -> tuple[float, float, float]:
    try:
        result = tuple(_finite(item, name) for item in value)
    except TypeError as exc:
        raise CameraAppearanceError(f"{name} must contain three values") from exc
    if len(result) != 3 or any(
        item < 2.0 / 3.0 or item > 1.5 for item in result
    ):
        raise CameraAppearanceError(
            f"{name} must contain three values in [2/3, 1.5]"
        )
    return result  # type: ignore[return-value]


def _integer_pair(value: Any, name: str, *, minimum: int = 0) -> tuple[int, int]:
    try:
        low, high = (_integer(item, name) for item in value)
    except (TypeError, ValueError) as exc:
        raise CameraAppearanceError(f"{name} must contain two integers") from exc
    if low < minimum or high < low:
        raise CameraAppearanceError(f"{name} has invalid bounds")
    return low, high


def _parse_hose(value: Any, camera_id: str) -> HoseOcclusionProfile:
    raw = _mapping(value, f"camera {camera_id} hose occlusion")
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return HoseOcclusionProfile(enabled=False)
    if camera_id != "wrist":
        raise CameraAppearanceError("synthetic hose occlusion may only target wrist")
    probability = _finite(
        raw.get("application_probability"), "hose application_probability"
    )
    if not 0.0 <= probability <= 1.0:
        raise CameraAppearanceError("hose application_probability must be in [0, 1]")
    curve_count = _integer_pair(raw.get("curve_count"), "hose curve_count", minimum=1)
    width = _integer_pair(
        raw.get("width_px_at_deployment_resolution"),
        "hose width_px_at_deployment_resolution",
        minimum=1,
    )
    offset = _integer_pair(raw.get("shadow_offset_px"), "hose shadow_offset_px")
    dilation = _integer(
        raw.get("ignore_mask_dilation_px", 0), "hose ignore_mask_dilation_px"
    )
    if not 0 <= dilation <= 32:
        raise CameraAppearanceError(
            "hose ignore_mask_dilation_px must be in [0, 32]"
        )
    return HoseOcclusionProfile(
        enabled=True,
        application_probability=probability,
        curve_count=curve_count,
        width_px=width,
        blue_rgb=_rgb8(raw.get("blue_rgb"), "hose blue_rgb"),
        shadow_rgb=_rgb8(raw.get("shadow_rgb"), "hose shadow_rgb"),
        shadow_offset_px=offset,
        ignore_mask_dilation_px=dilation,
    )


def load_camera_appearance_profiles(
    path: str | Path | None = None,
) -> CameraAppearanceProfiles:
    source = Path(path).resolve() if path is not None else DEFAULT_CAMERA_PROFILE_PATH
    raw = _load_json(source, "camera appearance profiles")
    if _integer(raw.get("schema_version"), "camera profile schema_version") != 1:
        raise CameraAppearanceError("unsupported camera-profile schema")
    curation_path, curation_relpath = _repo_file(
        raw.get("curation_manifest"), "curation_manifest"
    )
    curation = load_frame_curation(curation_path)
    deployment = _mapping(raw.get("deployment_stream"), "deployment_stream")
    deployment_resolution = _resolution(
        deployment.get("resolution"), "deployment resolution"
    )
    if any(value % 2 for value in deployment_resolution):
        raise CameraAppearanceError("YUV420 deployment dimensions must be even")
    deployment_pixel_format = str(
        deployment.get("pixel_format", "")
    ).strip().lower()
    if deployment_pixel_format != "yuv420p":
        raise CameraAppearanceError("deployment pixel format must be yuv420p")
    deployment_fps = _integer(deployment.get("fps"), "deployment fps")
    if deployment_fps <= 0:
        raise CameraAppearanceError("deployment fps must be positive")
    codec_evidence = str(deployment.get("codec_evidence", "")).strip().lower()
    if not codec_evidence:
        raise CameraAppearanceError("deployment codec_evidence is required")
    crf_evidence = _integer(deployment.get("crf_evidence"), "deployment crf_evidence")
    if not 0 <= crf_evidence <= 63:
        raise CameraAppearanceError("deployment crf_evidence must be in [0, 63]")
    preview_approximation = str(
        deployment.get("preview_approximation", "")
    ).strip()
    if not preview_approximation:
        raise CameraAppearanceError("deployment preview_approximation is required")
    camera_raw = _mapping(raw.get("cameras"), "cameras")
    if set(camera_raw) != {"table", "side", "wrist"}:
        raise CameraAppearanceError("profiles must define table, side, and wrist")
    cameras: dict[str, CameraAppearanceProfile] = {}
    for camera_id, raw_value in camera_raw.items():
        entry = _mapping(raw_value, f"camera {camera_id}")
        native = _resolution(entry.get("native_resolution"), f"{camera_id} native")
        calibration_path, calibration_relpath = _repo_file(
            entry.get("calibration_file"), f"{camera_id} calibration_file"
        )
        calibration = _load_json(calibration_path, "camera calibration")
        calibration_entry = str(entry.get("calibration_entry", "")).strip()
        calibrated = _mapping(
            calibration.get(calibration_entry),
            f"camera calibration entry {calibration_entry!r}",
        )
        if _resolution(calibrated.get("size"), f"{camera_id} calibrated size") != native:
            raise CameraAppearanceError(
                f"{camera_id} profile resolution does not match calibration"
            )
        status = str(entry.get("profile_status", "")).strip()
        if not status:
            raise CameraAppearanceError(f"{camera_id} profile_status is required")
        sampled_frames = _integer(
            entry.get("curated_fit_sampled_frames"),
            f"{camera_id} curated_fit_sampled_frames",
        )
        if sampled_frames < 1:
            raise CameraAppearanceError(
                f"{camera_id} curated_fit_sampled_frames must be positive"
            )
        cameras[camera_id] = CameraAppearanceProfile(
            camera_id=camera_id,
            native_resolution=native,
            calibration_file=calibration_relpath,
            calibration_entry=calibration_entry,
            latest_archive_rgb_mean_01=_rgb_triplet(
                entry.get("latest_archive_rgb_mean_01"),
                f"{camera_id} latest_archive_rgb_mean_01",
            ),
            older_archive_rgb_mean_01=_rgb_triplet(
                entry.get("older_archive_rgb_mean_01"),
                f"{camera_id} older_archive_rgb_mean_01",
            ),
            curated_fit_sampled_frames=sampled_frames,
            fit_neutral_rgb_median_8bit=_rgb8(
                entry.get("fit_neutral_rgb_median_8bit"),
                f"{camera_id} fit_neutral_rgb_median_8bit",
            ),
            fit_neutral_rgb_p05_8bit=_rgb8(
                entry.get("fit_neutral_rgb_p05_8bit"),
                f"{camera_id} fit_neutral_rgb_p05_8bit",
            ),
            fit_neutral_rgb_p95_8bit=_rgb8(
                entry.get("fit_neutral_rgb_p95_8bit"),
                f"{camera_id} fit_neutral_rgb_p95_8bit",
            ),
            fit_neutral_luma_p05_median_p95_8bit=_rgb8(
                entry.get("fit_neutral_luma_p05_median_p95_8bit"),
                f"{camera_id} fit_neutral_luma_p05_median_p95_8bit",
            ),
            heldout_neutral_rgb_median_8bit=_rgb8(
                entry.get("heldout_neutral_rgb_median_8bit"),
                f"{camera_id} heldout_neutral_rgb_median_8bit",
            ),
            heldout_neutral_luma_p05_median_p95_8bit=_rgb8(
                entry.get("heldout_neutral_luma_p05_median_p95_8bit"),
                f"{camera_id} heldout_neutral_luma_p05_median_p95_8bit",
            ),
            profile_status=status,
            hose_occlusion=_parse_hose(
                entry.get("synthetic_hose_occlusion"), camera_id
            ),
            fixed_training_rgb_gain=_bounded_rgb_gain(
                entry.get("fixed_training_rgb_gain", (1.0, 1.0, 1.0)),
                f"{camera_id} fixed_training_rgb_gain",
            ),
            fixed_training_gain_status=str(
                entry.get(
                    "fixed_training_gain_status",
                    "identity fallback for a legacy profile without a fitted gain",
                )
            ).strip(),
        )
        if not cameras[camera_id].fixed_training_gain_status:
            raise CameraAppearanceError(
                f"{camera_id} fixed_training_gain_status is required"
            )
    try:
        source_path = source.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source_path = str(source)
    # Keep the normalized repository path represented in load evidence even
    # though the curation object independently hashes its content.
    if curation.source_path != curation_relpath:
        raise CameraAppearanceError("loaded curation path does not match profile path")
    return CameraAppearanceProfiles(
        source_path=source_path,
        source_sha256=_sha256(source),
        curation=curation,
        deployment_resolution=deployment_resolution,
        deployment_fps=deployment_fps,
        cameras=cameras,
        deployment_pixel_format=deployment_pixel_format,
        codec_evidence=codec_evidence,
        crf_evidence=crf_evidence,
        preview_approximation=preview_approximation,
    )


@dataclass(frozen=True)
class DeploymentPreviewResult:
    image_rgb: Any
    occluder_mask: Any
    processing: Mapping[str, Any]
    ignore_mask: Any = None


def camera_neutral_region_masks(
    width: int, height: int, camera_id: str
) -> Mapping[str, Any]:
    """Return the normalized neutral regions used for fit and preview matching.

    Saturation filtering is applied separately.  Keeping these masks in the
    shared core prevents the deployment-preview gain from being estimated on a
    different part of the image than the real-camera evidence target.
    """

    import cv2
    import numpy as np

    if isinstance(width, bool) or isinstance(height, bool):
        raise CameraAppearanceError("mask dimensions must be integers")
    width, height = int(width), int(height)
    if width < 2 or height < 2:
        raise CameraAppearanceError("mask dimensions must be at least 2x2")
    camera = str(camera_id).strip().lower()
    if camera not in {"table", "side", "wrist"}:
        raise CameraAppearanceError(f"unsupported camera {camera_id!r}")

    if camera == "side":
        wall = np.zeros((height, width), dtype=np.uint8)
        wall[
            int(0.08 * height) : int(0.55 * height),
            int(0.04 * width) : int(0.70 * width),
        ] = 255
        table = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(
            table,
            np.asarray(
                [
                    (0.02 * width, 0.60 * height),
                    (0.70 * width, 0.56 * height),
                    (0.95 * width, 0.98 * height),
                    (0.00 * width, 0.98 * height),
                ],
                dtype=np.int32,
            ),
            255,
        )
        return {"neutral_wall": wall, "neutral_table": table}

    if camera == "wrist":
        workspace = np.zeros((height, width), dtype=np.uint8)
        workspace[
            int(0.08 * height) : int(0.92 * height),
            int(0.08 * width) : int(0.92 * width),
        ] = 255
        return {"neutral_workspace": workspace}

    table = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(
        table,
        np.asarray(
            [
                (0.20 * width, 0.24 * height),
                (0.80 * width, 0.24 * height),
                (0.97 * width, 0.93 * height),
                (0.03 * width, 0.93 * height),
            ],
            dtype=np.int32,
        ),
        255,
    )
    table[: int(0.62 * height), int(0.36 * width) : int(0.64 * width)] = 0
    background = np.zeros((height, width), dtype=np.uint8)
    background[: int(0.17 * height), : int(0.34 * width)] = 255
    background[: int(0.17 * height), int(0.66 * width) :] = 255
    return {"neutral_table": table, "neutral_background": background}


def _bezier(control: Sequence[Sequence[float]], samples: int = 72):
    import numpy as np

    points = np.asarray(control, dtype=np.float64)
    if points.shape != (4, 2):
        raise CameraAppearanceError("hose curve requires four 2D control points")
    t = np.linspace(0.0, 1.0, samples)[:, None]
    curve = (
        (1.0 - t) ** 3 * points[0]
        + 3.0 * (1.0 - t) ** 2 * t * points[1]
        + 3.0 * (1.0 - t) * t**2 * points[2]
        + t**3 * points[3]
    )
    return np.rint(curve).astype(np.int32).reshape((-1, 1, 2))


def _add_wrist_hoses(image_rgb, profile: HoseOcclusionProfile, rng):
    import cv2
    import numpy as np

    image = np.asarray(image_rgb).copy()
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    count = int(rng.integers(profile.curve_count[0], profile.curve_count[1] + 1))
    widths = []
    for index in range(count):
        line_width = int(rng.integers(profile.width_px[0], profile.width_px[1] + 1))
        widths.append(line_width)
        # Real wrist footage shows both near-vertical and diagonal blue loops.
        # Keep endpoints outside the image so the augmentation resembles an
        # entering cable rather than a painted floating segment.
        top_x = float(rng.uniform(-0.08, 0.38) * width)
        bottom_x = float(rng.uniform(-0.10, 1.08) * width)
        bend = float(rng.uniform(-0.25, 0.35) * width)
        control = (
            (top_x, -0.08 * height),
            (top_x + bend, 0.24 * height),
            (bottom_x - bend, 0.68 * height),
            (bottom_x, 1.08 * height),
        )
        curve = _bezier(control)
        shadow_offset = int(
            rng.integers(
                profile.shadow_offset_px[0], profile.shadow_offset_px[1] + 1
            )
        )
        shadow_curve = curve + np.asarray([[[shadow_offset, shadow_offset]]])
        cv2.polylines(
            image,
            [shadow_curve],
            False,
            profile.shadow_rgb,
            line_width + max(2, shadow_offset),
            cv2.LINE_AA,
        )
        cv2.polylines(
            mask,
            [shadow_curve],
            False,
            255,
            line_width + max(2, shadow_offset),
            cv2.LINE_8,
        )
        cv2.polylines(
            image,
            [curve],
            False,
            profile.blue_rgb,
            line_width,
            cv2.LINE_AA,
        )
        cv2.polylines(mask, [curve], False, 255, line_width, cv2.LINE_8)
    return image, mask, widths


def _match_neutral_median(image_rgb, target_rgb_8bit, camera_id: str):
    """Broadly align neutral pixels to a curated compressed-video target.

    This intentionally adjusts only a derived preview.  Gains are bounded so
    coloured parts cannot be arbitrarily remapped by an unrepresentative
    frame, and every requested/applied gain is returned for audit.
    """

    import cv2
    import numpy as np

    image = np.asarray(image_rgb)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    region_masks = camera_neutral_region_masks(
        int(image.shape[1]), int(image.shape[0]), camera_id
    )
    combined_region = np.zeros(image.shape[:2], dtype=np.uint8)
    for region_mask in region_masks.values():
        combined_region = cv2.bitwise_or(combined_region, region_mask)
    neutral = (hsv[..., 1] <= 48) & (combined_region > 0)
    if int(np.count_nonzero(neutral)) < 1024:
        return image.copy(), {
            "applied": False,
            "reason": "fewer than 1024 neutral pixels",
        }
    measured = np.median(image[neutral].astype(np.float32), axis=0)
    target = np.asarray(target_rgb_8bit, dtype=np.float32)
    requested = target / np.maximum(measured, 1.0)
    applied = np.clip(requested, 2.0 / 3.0, 1.5)
    floating = image.astype(np.float32) * applied.reshape((1, 1, 3))
    result = np.clip(np.rint(floating), 0.0, 255.0).astype(np.uint8)
    return result, {
        "applied": True,
        "neutral_rule": "OpenCV HSV saturation <= 48",
        "neutral_region_names": sorted(region_masks),
        "source_neutral_pixel_count": int(np.count_nonzero(neutral)),
        "source_neutral_rgb_median_8bit": measured.tolist(),
        "target_neutral_rgb_median_8bit": target.tolist(),
        "requested_rgb_gain": requested.tolist(),
        "applied_rgb_gain": applied.tolist(),
        "gain_bounds": [2.0 / 3.0, 1.5],
        "limitation": (
            "Broad display-RGB alignment only; this is not a physical camera "
            "response, exposure, or material calibration."
        ),
    }


def _apply_fixed_profile_gain(image_rgb, profile: CameraAppearanceProfile):
    """Apply one configured gain without conditioning on the current frame.

    Unlike per-frame neutral matching, this preserves intentional exposure and
    white-balance variation between synthetic training samples.  The gains are
    deliberately bounded and their provisional evidence status travels with
    every derived image.
    """

    import numpy as np

    image = np.asarray(image_rgb)
    gain = np.asarray(profile.fixed_training_rgb_gain, dtype=np.float32)
    floating = image.astype(np.float32) * gain.reshape((1, 1, 3))
    result = np.clip(np.rint(floating), 0.0, 255.0).astype(np.uint8)
    return result, {
        "applied": True,
        "mode": "fixed_profile",
        "applied_rgb_gain": gain.tolist(),
        "gain_bounds": [2.0 / 3.0, 1.5],
        "profile_status": profile.fixed_training_gain_status,
        "frame_statistics_used": False,
        "limitation": (
            "Fixed display-RGB alignment only; this is not a physical camera "
            "response, exposure, or material calibration."
        ),
    }


def create_deployment_preview(
    image_rgb,
    profiles: CameraAppearanceProfiles,
    *,
    camera_id: str,
    episode_seed: int,
    frame_index: int,
    add_wrist_occlusion: bool = False,
    match_curated_neutral_median: bool = True,
    photometric_mode: str | None = None,
    compression_domain: str = "archive_yuv420",
) -> DeploymentPreviewResult:
    """Create a deterministic deployment-like view without changing raw RGB.

    The returned ``occluder_mask`` covers the synthetic cable and shadow
    geometry before chroma subsampling.  ``ignore_mask`` dilates that geometry
    to conservatively include antialiasing and chroma spill.  Neither is an
    exact changed-pixel mask after the I420 round trip.  Callers must update
    visibility labels conservatively or mark the view as unlabeled robustness
    imagery.  Table and side views never receive this augmentation.

    ``photometric_mode=None`` preserves the legacy
    ``match_curated_neutral_median`` switch.  Training collection should use
    ``fixed_profile`` so sample-specific exposure/white-balance variation is
    not normalized away.  ``archive_yuv420`` preserves the legacy I420 round
    trip; ``uncompressed_preview`` retains resized display RGB.
    """

    import cv2
    import numpy as np

    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
        raise CameraAppearanceError("episode_seed must be an integer")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise CameraAppearanceError("frame_index must be a non-negative integer")
    profile = profiles.camera(camera_id)
    source = np.asarray(image_rgb)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise CameraAppearanceError("camera image must be HxWx3 uint8 RGB")
    expected_width, expected_height = profile.native_resolution
    if (source.shape[1], source.shape[0]) != (expected_width, expected_height):
        raise CameraAppearanceError(
            f"{profile.camera_id} image must be native "
            f"{expected_width}x{expected_height}"
        )
    width, height = profiles.deployment_resolution
    preview = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    if photometric_mode is None:
        selected_photometric_mode = (
            "per_frame_neutral" if match_curated_neutral_median else "none"
        )
    else:
        selected_photometric_mode = str(photometric_mode).strip().lower()
    if selected_photometric_mode not in {
        "per_frame_neutral",
        "fixed_profile",
        "none",
    }:
        raise CameraAppearanceError(
            "photometric_mode must be per_frame_neutral, fixed_profile, or none"
        )
    if selected_photometric_mode == "per_frame_neutral":
        preview, photometric = _match_neutral_median(
            preview, profile.fit_neutral_rgb_median_8bit, profile.camera_id
        )
        photometric = {"mode": "per_frame_neutral", **photometric}
    elif selected_photometric_mode == "fixed_profile":
        preview, photometric = _apply_fixed_profile_gain(preview, profile)
    else:
        photometric = {
            "applied": False,
            "mode": "none",
            "reason": "disabled by caller",
        }
    mask = np.zeros((height, width), dtype=np.uint8)
    digest = hashlib.sha256(
        (
            "rockwell-camera-preview-v1:"
            f"{profiles.source_sha256}:{profile.camera_id}:"
            f"{episode_seed}:{frame_index}"
        ).encode("utf-8")
    ).digest()
    preview_seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(preview_seed)
    hose_applied = False
    hose_widths: list[int] = []
    hose = profile.hose_occlusion
    if (
        add_wrist_occlusion
        and profile.camera_id == "wrist"
        and hose.enabled
        and float(rng.random()) < hose.application_probability
    ):
        preview, mask, hose_widths = _add_wrist_hoses(preview, hose, rng)
        hose_applied = True

    ignore_mask = mask.copy()
    if hose_applied and hose.ignore_mask_dilation_px > 0:
        radius = hose.ignore_mask_dilation_px
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        ignore_mask = cv2.dilate(mask, kernel)

    # I420 is a planar 4:2:0 representation.  The round trip reproduces the
    # main chroma-resolution loss of the downloaded stream without pretending
    # to reproduce its AV1 encoder or rate control.
    selected_compression_domain = str(compression_domain).strip().lower()
    if selected_compression_domain not in {
        "archive_yuv420",
        "uncompressed_preview",
    }:
        raise CameraAppearanceError(
            "compression_domain must be archive_yuv420 or uncompressed_preview"
        )
    if selected_compression_domain == "archive_yuv420":
        i420 = cv2.cvtColor(np.ascontiguousarray(preview), cv2.COLOR_RGB2YUV_I420)
        preview = cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)
        chroma_approximation = "RGB -> planar I420 -> RGB"
        compression_intended_use = (
            "decoded archive-video robustness preview; not raw-camera training input"
        )
    else:
        chroma_approximation = "disabled; resized display RGB retained"
        compression_intended_use = (
            "uncompressed resized preview; no archive compression approximation"
        )
    coverage = float(np.count_nonzero(mask) / mask.size)
    ignore_coverage = float(np.count_nonzero(ignore_mask) / ignore_mask.size)
    return DeploymentPreviewResult(
        image_rgb=preview,
        occluder_mask=mask,
        ignore_mask=ignore_mask,
        processing={
            "camera_id": profile.camera_id,
            "profile_path": profiles.source_path,
            "profile_sha256": profiles.source_sha256,
            "curation_path": profiles.curation.source_path,
            "curation_sha256": profiles.curation.source_sha256,
            "input": "native lossless RGB",
            "input_resolution": list(profile.native_resolution),
            "output": "deployment-like RGB preview",
            "output_resolution": list(profiles.deployment_resolution),
            "resize_interpolation": "cv2.INTER_AREA",
            "chroma_approximation": chroma_approximation,
            "curated_neutral_median_match": photometric,
            "photometric_alignment": photometric,
            "photometric_alignment_mode": selected_photometric_mode,
            "av1_bitstream_reproduced": False,
            "compression_domain": {
                "mode": selected_compression_domain,
                "pixel_format": profiles.deployment_pixel_format,
                "codec_evidence": profiles.codec_evidence,
                "crf_evidence": profiles.crf_evidence,
                "preview_approximation": profiles.preview_approximation,
                "archive_codec_bitstream_reproduced": False,
                "intended_use": compression_intended_use,
            },
            "preview_seed": preview_seed,
            "synthetic_wrist_hose": {
                "applied": hose_applied,
                "widths_px": hose_widths,
                "mask_nonzero_fraction": coverage,
                "geometry_mask_nonzero_fraction": coverage,
                "ignore_mask_nonzero_fraction": ignore_coverage,
                "ignore_mask_dilation_px": (
                    hose.ignore_mask_dilation_px if hose_applied else 0
                ),
                "ignore_mask_contains_geometry": bool(
                    np.all((mask == 0) | (ignore_mask > 0))
                ),
                "label_rule": (
                    "exclude the returned conservative ignore_mask from "
                    "supervised pixel loss and recompute visibility from valid "
                    "pixels, or store this as unlabeled robustness imagery; do "
                    "not infer visible fraction from mask area alone"
                ),
            },
        },
    )


__all__ = [
    "CameraAppearanceError",
    "CameraAppearanceProfile",
    "CameraAppearanceProfiles",
    "CuratedRange",
    "DEFAULT_CAMERA_PROFILE_PATH",
    "DEFAULT_FRAME_CURATION_PATH",
    "DeploymentPreviewResult",
    "FrameCuration",
    "HoseOcclusionProfile",
    "create_deployment_preview",
    "load_camera_appearance_profiles",
    "load_frame_curation",
]
