"""Apply one checked, replayable synthetic-data episode to the Isaac scene.

This module is deliberately outside the supplied ``digital-twin-setup`` tree.
It mutates only the disposable runtime modules/stage for the current process.
The default scripted demo never calls these helpers, so its validated scene
and motion path remain available as a regression test.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from sorting_core.synthetic_data import RandomizationManifest, SyntheticDataError


COLOURS = ("red", "yellow", "blue")

# Most authored USD numeric attributes used below are Sdf ``float`` values.
# Isaac therefore returns IEEE-754 binary32 values even though Python supplied
# a binary64 float.  Keep readback fail-closed for meaningful mismatches while
# accepting the unavoidable serialization error at large light intensities.
USD_FLOAT32_READBACK_RTOL = float(np.finfo(np.float32).eps * 4.0)

CAMERA_PARAMETER_NAMES = frozenset(
    {
        "camera.translation_x_m",
        "camera.translation_y_m",
        "camera.translation_z_m",
        "camera.rotation_roll_deg",
        "camera.rotation_pitch_deg",
        "camera.rotation_yaw_deg",
        "camera.focal_length_scale",
    }
)
MATERIAL_PARAMETER_NAMES = frozenset(
    {
        "materials.coloured_plastic_roughness",
        "materials.coloured_plastic_albedo_scale",
        "materials.cube_hue_jitter_deg",
        "materials.cube_saturation_scale",
        "materials.table_roughness",
    }
)
DESTINATION_MATERIAL_PARAMETER_NAMES = frozenset(
    {
        "materials.destination_plastic_roughness",
        "materials.destination_albedo_scale",
        "materials.destination_hue_jitter_deg",
        "materials.destination_saturation_scale",
    }
)
LIGHT_COLOR_PARAMETER_NAMES = frozenset(
    {
        "lighting.color_r_scale",
        "lighting.color_g_scale",
        "lighting.color_b_scale",
    }
)

# The logical destination labels deliberately remain red/yellow/blue so the
# sorting policy and ground-truth schema do not change.  Only their rendered
# appearances follow the physical orange/yellow/light-blue containers.  The
# observed hue/saturation values are stable medians from manually isolated
# regions in table_frame0.png and table_frame400.png (compressed 640x360
# rollout video).  They are display-RGB evidence, not measured material
# albedos.  The nominal diffuse values preserve those hue/saturation targets
# at conservative brightnesses suitable for the existing lighting rig.
DESTINATION_APPEARANCE_TARGETS: Mapping[str, Mapping[str, Any]] = {
    "red": {
        "appearance_name": "orange",
        "observed_display_hue_deg": 16.5,
        "observed_display_saturation": 0.760,
        "nominal_diffuse_rgb": (0.640, 0.288, 0.154),
    },
    "yellow": {
        "appearance_name": "yellow",
        "observed_display_hue_deg": 46.0,
        "observed_display_saturation": 0.724,
        "nominal_diffuse_rgb": (0.780, 0.648, 0.215),
    },
    "blue": {
        "appearance_name": "light_blue",
        "observed_display_hue_deg": 208.0,
        "observed_display_saturation": 0.520,
        "nominal_diffuse_rgb": (0.326, 0.515, 0.680),
    },
}

TRAINING_APPEARANCE_PROFILE = {
    "name": "real-table-video-broad-fit-v1",
    "table_reference_asset": "reference-assets/table/manipbench-table-surface.png",
    "fit_rollout": "../HardwareResults/HardwareResultVideos/Rollout5TableCam.mp4",
    "held_out_rollout": "../HardwareResults/HardwareResultVideos/rollout4TableCam.mp4",
    "fit_scope": "broad exposure, colour cast, vignetting, and tabletop appearance",
    "limitation": (
        "Compressed 640x360 rollout video cannot calibrate native sensor noise, "
        "precise blur, camera response, or small-object edge quality."
    ),
}


def apply_training_appearance_profile(scene_config: Any) -> dict[str, Any]:
    """Enable the measured-cell tabletop only for opt-in training runs.

    The starter scene places a pristine white paper backdrop slightly above its
    optional scan of the real work surface. Leaving both enabled hides the
    reference scan. Training mode therefore enables the scan and disables the
    paper layer before ``cell.build(stage)``. Default simulation and hardware
    paths never call this helper.
    """

    for name in ("MAT_MARKINGS", "SHOW_BACKDROP"):
        if not hasattr(scene_config, name):
            raise SyntheticDataError(
                f"training appearance profile requires scene_config.{name}"
            )
    previous = {
        "mat_markings": bool(scene_config.MAT_MARKINGS),
        "clean_paper_backdrop": bool(scene_config.SHOW_BACKDROP),
    }
    scene_config.MAT_MARKINGS = True
    scene_config.SHOW_BACKDROP = False
    return {
        **TRAINING_APPEARANCE_PROFILE,
        "previous": previous,
        "effective": {
            "mat_markings": True,
            "clean_paper_backdrop": False,
        },
    }


def _with_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe mapping carrying a hash of its unhashed contents."""

    result = dict(payload)
    encoded = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    result["sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def _assert_close(actual: Any, expected: Any, name: str, *, atol: float = 1e-6) -> None:
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array,
        expected_array,
        rtol=USD_FLOAT32_READBACK_RTOL,
        atol=atol,
    ):
        raise SyntheticDataError(
            f"Isaac readback mismatch for {name}: got {actual_array.tolist()}, "
            f"expected {expected_array.tolist()}"
        )


def _set_and_verify(attribute: Any, value: float, name: str) -> float:
    if not attribute.IsValid():
        raise SyntheticDataError(f"Isaac attribute is missing: {name}")
    if attribute.Set(float(value)) is False:
        raise SyntheticDataError(f"Isaac refused to set {name}")
    readback = attribute.Get()
    if readback is None:
        raise SyntheticDataError(f"Isaac returned no readback for {name}")
    _assert_close(float(readback), float(value), name)
    return float(readback)


def _value(manifest: RandomizationManifest, name: str) -> float:
    return float(manifest.value(name))


def _vary_rgb(
    rgb: Any,
    *,
    hue_jitter_deg: float,
    saturation_scale: float,
    value_scale: float,
) -> tuple[float, float, float]:
    """Apply bounded HSV appearance variation to one linear RGB triplet.

    HSV is used only to keep the independently sampled hue and saturation
    changes interpretable.  The result is still an authored USD diffuse RGB
    value, not a claim that display RGB is a physical reflectance model.
    """

    source = np.asarray(rgb, dtype=float)
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise SyntheticDataError(
            "material diffuse colour must be three finite values"
        )
    if np.any(source < 0.0) or np.any(source > 1.0):
        raise SyntheticDataError("material diffuse colour must be inside [0, 1]")
    if saturation_scale <= 0.0 or value_scale <= 0.0:
        raise SyntheticDataError("material appearance scales must be positive")
    hue, saturation, value = colorsys.rgb_to_hsv(*(float(v) for v in source))
    hue = (hue + float(hue_jitter_deg) / 360.0) % 1.0
    saturation = float(np.clip(saturation * saturation_scale, 0.0, 1.0))
    value = float(np.clip(value * value_scale, 0.0, 1.0))
    return tuple(float(v) for v in colorsys.hsv_to_rgb(hue, saturation, value))


def _rgb_evidence(rgb: Any) -> dict[str, Any]:
    values = tuple(float(value) for value in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(*values)
    return {
        "rgb": list(values),
        "hue_deg": float(hue * 360.0),
        "saturation": float(saturation),
        "value": float(value),
    }


def _rotate(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a vector around a normalized axis using Rodrigues' formula."""

    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise SyntheticDataError("camera randomization produced a zero rotation axis")
    axis /= norm
    vector = np.asarray(vector, dtype=float)
    angle = math.radians(float(angle_deg))
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - math.cos(angle))
    )


def _randomized_camera(camera: Mapping[str, Any], manifest: RandomizationManifest):
    result = dict(camera)
    eye = np.asarray(camera["eye"], dtype=float)
    target = np.asarray(camera["look_at"], dtype=float)
    up = np.asarray(camera["up"], dtype=float)
    translation = np.array(
        [
            _value(manifest, "camera.translation_x_m"),
            _value(manifest, "camera.translation_y_m"),
            _value(manifest, "camera.translation_z_m"),
        ]
    )
    eye += translation
    target += translation

    distance = float(np.linalg.norm(target - eye))
    forward = (target - eye) / distance
    up /= np.linalg.norm(up)
    forward = _rotate(
        forward, np.array([0.0, 0.0, 1.0]), _value(manifest, "camera.rotation_yaw_deg")
    )
    up = _rotate(
        up, np.array([0.0, 0.0, 1.0]), _value(manifest, "camera.rotation_yaw_deg")
    )
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    pitch = _value(manifest, "camera.rotation_pitch_deg")
    forward = _rotate(forward, right, pitch)
    up = _rotate(up, right, pitch)
    up = _rotate(up, forward, _value(manifest, "camera.rotation_roll_deg"))
    up -= forward * np.dot(up, forward)
    up /= np.linalg.norm(up)

    result["eye"] = tuple(float(v) for v in eye)
    result["look_at"] = tuple(float(v) for v in eye + forward * distance)
    result["up"] = tuple(float(v) for v in up)
    result["focal_length"] = float(camera["focal_length"]) * _value(
        manifest, "camera.focal_length_scale"
    )
    return result


def prepare_prebuild_randomization(
    scene_config: Any,
    cell_module: Any,
    manifest: RandomizationManifest,
    base_spawns: Mapping[str, tuple[float, float, float]],
) -> tuple[dict[str, tuple[float, float, float]], set[str]]:
    """Prepare parameters that must exist before ``cell.build(stage)``.

    Returns the episode's cube spawn dictionary and an intentionally empty
    applied set.  Changing a Python configuration object is not proof that the
    USD stage consumed it.  :func:`apply_postbuild_randomization` adds these
    names only after checking the resulting stage values.
    """

    applied: set[str] = set()
    if hasattr(scene_config, "_TRAINING_RANDOMIZATION_BASE"):
        raise SyntheticDataError(
            "prebuild randomization was already prepared in this runtime process"
        )
    scene_config._TRAINING_RANDOMIZATION_BASE = {
        "cube_mass_kg": float(scene_config.CUBE_MASS),
        "camera": dict(scene_config.TABLE_CAM),
        "spawns": {
            colour: tuple(float(value) for value in base_spawns[colour])
            for colour in COLOURS
        },
        "materials": {
            name: (
                tuple(float(value) for value in cell_module._MATERIAL_SPEC[name][0]),
                float(cell_module._MATERIAL_SPEC[name][1]),
                float(cell_module._MATERIAL_SPEC[name][2]),
            )
            for name in (
                "plastic_red",
                "plastic_yellow",
                "plastic_blue",
                "table_white",
            )
        },
    }
    spawns: dict[str, tuple[float, float, float]] = {}
    for colour in COLOURS:
        x, y, yaw = (float(v) for v in base_spawns[colour])
        names = {
            "x": f"scene.{colour}_cube_x_offset_m",
            "y": f"scene.{colour}_cube_y_offset_m",
            "yaw": f"scene.{colour}_cube_yaw_offset_deg",
        }
        spawns[colour] = (
            x + _value(manifest, names["x"]),
            y + _value(manifest, names["y"]),
            yaw + _value(manifest, names["yaw"]),
        )

    minimum = 0.012
    for index, first in enumerate(COLOURS):
        for second in COLOURS[index + 1 :]:
            separation = math.dist(spawns[first][:2], spawns[second][:2])
            if separation < minimum:
                raise SyntheticDataError(
                    f"randomized {first}/{second} centres are only "
                    f"{separation * 1000.0:.1f} mm apart"
                )

    scene_config.CUBE_MASS = float(scene_config.CUBE_MASS) * _value(
        manifest, "physics.cube_mass_scale"
    )

    scene_config.TABLE_CAM = _randomized_camera(scene_config.TABLE_CAM, manifest)

    roughness = _value(manifest, "materials.coloured_plastic_roughness")
    albedo_scale = _value(manifest, "materials.coloured_plastic_albedo_scale")
    hue_jitter_deg = _value(manifest, "materials.cube_hue_jitter_deg")
    saturation_scale = _value(manifest, "materials.cube_saturation_scale")
    for name in ("plastic_red", "plastic_yellow", "plastic_blue"):
        diffuse, metallic, _ = cell_module._MATERIAL_SPEC[name]
        varied = _vary_rgb(
            diffuse,
            hue_jitter_deg=hue_jitter_deg,
            saturation_scale=saturation_scale,
            value_scale=albedo_scale,
        )
        cell_module._MATERIAL_SPEC[name] = (varied, metallic, roughness)
    table_diffuse, table_metallic, _ = cell_module._MATERIAL_SPEC["table_white"]
    cell_module._MATERIAL_SPEC["table_white"] = (
        table_diffuse,
        table_metallic,
        _value(manifest, "materials.table_roughness"),
    )
    return spawns, applied


def apply_postbuild_randomization(
    stage: Any, cell_module: Any, manifest: RandomizationManifest
) -> tuple[set[str], dict[str, Any]]:
    """Apply stage values, verify USD/PhysX readback, and return evidence.

    A sampled or Python-side value is never counted as applied.  This function
    returns a parameter name only after the task prim/material/attribute that
    consumes it has been read back and compared with the requested value.
    Camera values are verified separately by :func:`readback_effective_camera`
    after the Isaac camera sensor is initialized.
    """

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    applied: set[str] = set()
    base_values = getattr(cell_module.C, "_TRAINING_RANDOMIZATION_BASE", None)
    if not isinstance(base_values, Mapping):
        raise SyntheticDataError(
            "postbuild randomization is missing its prebuild baseline"
        )
    readback: dict[str, Any] = {
        "schema_version": 1,
        "source": "USD/PhysX readback after cell.build(stage)",
        "requested_manifest_sha256": manifest.sha256,
    }

    # The cube asset is a referenced USD with its own colour-variant material.
    # Merely changing cell._MATERIAL_SPEC does not affect that referenced mesh.
    # Author a distinct local material on every task cube's actual Die mesh,
    # bind it more strongly than the referenced relationship, then ask USD which
    # material is truly bound and read the exact shader inputs back.
    UsdGeom.Scope.Define(stage, "/World/Looks/TrainingData")
    material_rows: dict[str, Any] = {}
    for index, colour in enumerate(COLOURS):
        diffuse, metallic, roughness = cell_module._MATERIAL_SPEC[
            f"plastic_{colour}"
        ]
        base_diffuse, base_metallic, _ = base_values["materials"][
            f"plastic_{colour}"
        ]
        expected_diffuse = _vary_rgb(
            base_diffuse,
            hue_jitter_deg=_value(manifest, "materials.cube_hue_jitter_deg"),
            saturation_scale=_value(
                manifest, "materials.cube_saturation_scale"
            ),
            value_scale=_value(
                manifest, "materials.coloured_plastic_albedo_scale"
            ),
        )
        expected_roughness = _value(
            manifest, "materials.coloured_plastic_roughness"
        )
        _assert_close(diffuse, expected_diffuse, f"{colour} configured diffuse")
        _assert_close(metallic, base_metallic, f"{colour} configured metallic")
        _assert_close(
            roughness, expected_roughness, f"{colour} configured roughness"
        )
        material_path = f"/World/Looks/TrainingData/SortCube{index}_{colour}"
        shader_path = material_path + "/PreviewSurface"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        diffuse_input = shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        )
        metallic_input = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
        roughness_input = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        diffuse_input.Set(Gf.Vec3f(*(float(value) for value in diffuse)))
        metallic_input.Set(float(metallic))
        roughness_input.Set(float(roughness))
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        mesh_path = f"/World/SortCube{index}/Visual/Geom/Die"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid():
            raise SyntheticDataError(
                f"referenced task-cube visual mesh is missing: {mesh_path}"
            )
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        binding_api.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound_result = binding_api.ComputeBoundMaterial()
        bound_material = (
            bound_result[0] if isinstance(bound_result, tuple) else bound_result
        )
        if (
            not bound_material
            or bound_material.GetPrim().GetPath() != material.GetPrim().GetPath()
        ):
            bound_path = (
                None
                if not bound_material
                else str(bound_material.GetPrim().GetPath())
            )
            raise SyntheticDataError(
                f"task-cube material binding did not take for {mesh_path}: "
                f"got {bound_path}, expected {material_path}"
            )
        diffuse_readback = [float(value) for value in diffuse_input.Get()]
        metallic_readback = float(metallic_input.Get())
        roughness_readback = float(roughness_input.Get())
        _assert_close(
            diffuse_readback, expected_diffuse, f"{colour} cube diffuse colour"
        )
        _assert_close(metallic_readback, base_metallic, f"{colour} cube metallic")
        _assert_close(
            roughness_readback, expected_roughness, f"{colour} cube roughness"
        )
        material_rows[colour] = {
            "semantic_label": colour,
            "appearance_name": colour,
            "task_prim_path": f"/World/SortCube{index}",
            "referenced_visual_mesh_path": mesh_path,
            "effective_material_path": material_path,
            "binding_strength": "strongerThanDescendants",
            "shader_id": str(shader.GetIdAttr().Get()),
            "diffuse_color": diffuse_readback,
            "base_appearance": _rgb_evidence(base_diffuse),
            "effective_appearance": _rgb_evidence(diffuse_readback),
            "metallic": metallic_readback,
            "roughness": roughness_readback,
        }

    # Make table roughness equally auditable.  The runtime's optional printed
    # artwork is a separate overlay and is intentionally not replaced here.
    table_diffuse, table_metallic, table_roughness = cell_module._MATERIAL_SPEC[
        "table_white"
    ]
    base_table_diffuse, base_table_metallic, _ = base_values["materials"][
        "table_white"
    ]
    expected_table_roughness = _value(manifest, "materials.table_roughness")
    _assert_close(table_diffuse, base_table_diffuse, "configured table diffuse")
    _assert_close(table_metallic, base_table_metallic, "configured table metallic")
    _assert_close(
        table_roughness, expected_table_roughness, "configured table roughness"
    )
    table_material_path = "/World/Looks/TrainingData/TableWhite"
    table_material = UsdShade.Material.Define(stage, table_material_path)
    table_shader = UsdShade.Shader.Define(
        stage, table_material_path + "/PreviewSurface"
    )
    table_shader.CreateIdAttr("UsdPreviewSurface")
    table_diffuse_input = table_shader.CreateInput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f
    )
    table_metallic_input = table_shader.CreateInput(
        "metallic", Sdf.ValueTypeNames.Float
    )
    table_roughness_input = table_shader.CreateInput(
        "roughness", Sdf.ValueTypeNames.Float
    )
    table_diffuse_input.Set(
        Gf.Vec3f(*(float(value) for value in table_diffuse))
    )
    table_metallic_input.Set(float(table_metallic))
    table_roughness_input.Set(float(table_roughness))
    table_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    table_material.CreateSurfaceOutput().ConnectToSource(
        table_shader.ConnectableAPI(), "surface"
    )
    table_bindings = []
    for table_path in ("/World/Table", "/World/Mat"):
        table_prim = stage.GetPrimAtPath(table_path)
        if not table_prim.IsValid():
            raise SyntheticDataError(f"table visual prim is missing: {table_path}")
        table_binding = UsdShade.MaterialBindingAPI.Apply(table_prim)
        table_binding.Bind(
            table_material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound_result = table_binding.ComputeBoundMaterial()
        bound_material = (
            bound_result[0] if isinstance(bound_result, tuple) else bound_result
        )
        if (
            not bound_material
            or bound_material.GetPrim().GetPath()
            != table_material.GetPrim().GetPath()
        ):
            raise SyntheticDataError(f"table material binding did not take: {table_path}")
        table_bindings.append(table_path)
    table_roughness_readback = float(table_roughness_input.Get())
    _assert_close(
        table_roughness_readback,
        expected_table_roughness,
        "table material roughness",
    )
    printed_overlay_row = None
    printed_overlay = stage.GetPrimAtPath("/World/MatPrint")
    if printed_overlay.IsValid():
        printed_material = cell_module.MATS.get("mat_print")
        if printed_material is None or not printed_material.GetPrim().IsValid():
            raise SyntheticDataError(
                "printed table overlay exists without its expected material"
            )
        overlay_roughness_input = None
        overlay_shader_path = None
        for shader_prim in Usd.PrimRange(printed_material.GetPrim()):
            shader = UsdShade.Shader(shader_prim)
            if not shader:
                continue
            for input_name in (
                "reflection_roughness_constant",
                "reflection_roughness",
                "roughness",
            ):
                candidate = shader.GetInput(input_name)
                if candidate and candidate.GetAttr().IsValid():
                    overlay_roughness_input = candidate
                    overlay_shader_path = str(shader_prim.GetPath())
                    break
            if overlay_roughness_input is not None:
                break
        if overlay_roughness_input is None:
            raise SyntheticDataError(
                "cannot find the printed table overlay's roughness shader input"
            )
        overlay_roughness = _set_and_verify(
            overlay_roughness_input.GetAttr(),
            expected_table_roughness,
            "printed table overlay roughness",
        )
        printed_overlay_row = {
            "prim_path": "/World/MatPrint",
            "material_path": str(printed_material.GetPrim().GetPath()),
            "shader_path": overlay_shader_path,
            "roughness": overlay_roughness,
        }
    readback["materials"] = {
        "task_cubes": material_rows,
        "table": {
            "effective_material_path": table_material_path,
            "bound_prim_paths": table_bindings,
            "printed_artwork_overlay_replaced": False,
            "printed_artwork_overlay": printed_overlay_row,
            "diffuse_color": [float(value) for value in table_diffuse_input.Get()],
            "metallic": float(table_metallic_input.Get()),
            "roughness": table_roughness_readback,
        },
    }
    applied.update(MATERIAL_PARAMETER_NAMES)

    # Verify that scene-layout offsets and mass reached the actual task rigid
    # bodies.  This prevents a renamed config field from silently producing a
    # manifest full of false "applied" claims.
    object_rows: dict[str, Any] = {}
    for index, (x, y, colour) in enumerate(cell_module.C.SORT_CUBES):
        base_x, base_y, base_yaw = base_values["spawns"][colour]
        expected_x = base_x + _value(
            manifest, f"scene.{colour}_cube_x_offset_m"
        )
        expected_y = base_y + _value(
            manifest, f"scene.{colour}_cube_y_offset_m"
        )
        expected_yaw = base_yaw + _value(
            manifest, f"scene.{colour}_cube_yaw_offset_deg"
        )
        _assert_close([x, y], [expected_x, expected_y], f"{colour} config spawn")
        _assert_close(
            cell_module.C.SORT_CUBE_YAWS_DEG[index],
            expected_yaw,
            f"{colour} config yaw",
        )
        prim_path = f"/World/SortCube{index}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise SyntheticDataError(f"randomized task cube is missing: {prim_path}")
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        position = [float(matrix[3][axis]) for axis in range(3)]
        expected_yaw = float(expected_yaw)
        actual_yaw = math.degrees(
            math.atan2(float(matrix[0][1]), float(matrix[0][0]))
        )
        yaw_error = (actual_yaw - expected_yaw + 180.0) % 360.0 - 180.0
        _assert_close(position[:2], [float(x), float(y)], f"{colour} cube spawn")
        _assert_close(yaw_error, 0.0, f"{colour} cube yaw", atol=1e-4)
        mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
        if not mass_attr.IsValid() or mass_attr.Get() is None:
            raise SyntheticDataError(f"task-cube mass is missing: {prim_path}")
        mass = float(mass_attr.Get())
        expected_mass = float(base_values["cube_mass_kg"]) * _value(
            manifest, "physics.cube_mass_scale"
        )
        _assert_close(
            float(cell_module.C.CUBE_MASS), expected_mass, "configured cube mass"
        )
        _assert_close(mass, expected_mass, f"{colour} cube mass")
        object_rows[str(colour)] = {
            "prim_path": prim_path,
            "position_world_m": position,
            "yaw_world_deg": actual_yaw,
            "mass_kg": mass,
        }
        applied.update(
            {
                f"scene.{colour}_cube_x_offset_m",
                f"scene.{colour}_cube_y_offset_m",
                f"scene.{colour}_cube_yaw_offset_deg",
            }
        )
    readback["task_objects"] = object_rows
    applied.add("physics.cube_mass_scale")

    light_scales = {
        "/World/DomeLight": "lighting.dome_intensity_scale",
        "/World/KeyLight": "lighting.key_intensity_scale",
        "/World/FillLight": "lighting.fill_intensity_scale",
        "/World/RimLight": "lighting.rim_intensity_scale",
    }
    for path, parameter in light_scales.items():
        prim = stage.GetPrimAtPath(path)
        attribute = prim.GetAttribute("inputs:intensity") if prim.IsValid() else None
        if attribute is None or not attribute.IsValid() or not attribute.HasAuthoredValue():
            raise SyntheticDataError(f"Isaac light is missing intensity: {path}")
        original = float(attribute.Get())
        actual = _set_and_verify(
            attribute,
            original * _value(manifest, parameter),
            f"{path} intensity",
        )
        readback.setdefault("lighting", {})[path] = {
            "intensity": actual,
        }
        applied.add(parameter)

    # A small RGB tint variation covers plausible changes in the cell's mixed
    # illumination without pretending that the compressed rollout calibrated a
    # physical colour temperature.  Normalize when needed so a white light does
    # not clip one channel and silently erase a sampled scale.
    color_scale = np.asarray(
        [
            _value(manifest, "lighting.color_r_scale"),
            _value(manifest, "lighting.color_g_scale"),
            _value(manifest, "lighting.color_b_scale"),
        ],
        dtype=float,
    )
    for path in light_scales:
        prim = stage.GetPrimAtPath(path)
        attribute = prim.GetAttribute("inputs:color") if prim.IsValid() else None
        if attribute is None or not attribute.IsValid():
            raise SyntheticDataError(f"Isaac light is missing color: {path}")
        original_value = attribute.Get()
        if original_value is None:
            raise SyntheticDataError(f"Isaac light returned no color: {path}")
        original_color = np.asarray(original_value, dtype=float)
        if original_color.shape != (3,) or not np.all(np.isfinite(original_color)):
            raise SyntheticDataError(f"Isaac light color is invalid: {path}")
        expected_color = original_color * color_scale
        maximum = float(np.max(expected_color))
        if maximum > 1.0:
            expected_color /= maximum
        if attribute.Set(Gf.Vec3f(*(float(v) for v in expected_color))) is False:
            raise SyntheticDataError(f"Isaac refused to set {path} color")
        actual_color = [float(value) for value in attribute.Get()]
        _assert_close(actual_color, expected_color, f"{path} color")
        readback["lighting"][path].update(
            {
                "original_color": [float(value) for value in original_color],
                "sampled_color_scale": [float(value) for value in color_scale],
                "color": actual_color,
            }
        )
    applied.update(LIGHT_COLOR_PARAMETER_NAMES)

    size_scale = _value(manifest, "lighting.source_size_scale")
    for path in ("/World/KeyLight", "/World/FillLight", "/World/RimLight"):
        prim = stage.GetPrimAtPath(path)
        for attribute_name in ("inputs:width", "inputs:height"):
            attribute = prim.GetAttribute(attribute_name)
            if not attribute.IsValid() or not attribute.HasAuthoredValue():
                raise SyntheticDataError(
                    f"Isaac area light is missing {attribute_name}: {path}"
                )
            original = float(attribute.Get())
            actual = _set_and_verify(
                attribute,
                original * size_scale,
                f"{path} {attribute_name}",
            )
            readback["lighting"][path][attribute_name.removeprefix("inputs:")] = actual
    applied.add("lighting.source_size_scale")

    physics_prim = stage.GetPrimAtPath(cell_module.PHYS_MAT_PATH)
    if not physics_prim.IsValid():
        raise SyntheticDataError("Isaac object physics material is missing")
    physics = UsdPhysics.MaterialAPI(physics_prim)
    for parameter, getter in (
        ("physics.object_static_friction", physics.GetStaticFrictionAttr),
        ("physics.object_dynamic_friction", physics.GetDynamicFrictionAttr),
        ("physics.object_restitution", physics.GetRestitutionAttr),
    ):
        attribute = getter()
        if not attribute.IsValid():
            raise SyntheticDataError(f"Isaac physics attribute is missing: {parameter}")
        actual = _set_and_verify(attribute, _value(manifest, parameter), parameter)
        readback.setdefault("physics_material", {})[parameter] = actual
        applied.add(parameter)
    return applied, _with_sha256(readback)


def apply_destination_material_randomization(
    stage: Any,
    manifest: RandomizationManifest,
    *,
    stage_readback: Mapping[str, Any] | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Bind audited orange/yellow/light-blue materials after bins exist.

    The scripted demo creates destination meshes after ``cell.build(stage)``,
    so this is intentionally separate from :func:`apply_postbuild_randomization`.
    Logical labels and prim paths remain red/yellow/blue; only the appearance is
    changed.  Passing the earlier stage readback merges and re-hashes the
    destination evidence so the caller can keep one authoritative artifact.
    """

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    if stage_readback is None:
        readback: dict[str, Any] = {
            "schema_version": 1,
            "source": "USD/PhysX readback after training scene construction",
            "requested_manifest_sha256": manifest.sha256,
        }
    else:
        readback = dict(stage_readback)
        supplied_sha256 = readback.pop("sha256", None)
        expected_sha256 = _with_sha256(readback)["sha256"]
        if supplied_sha256 != expected_sha256:
            raise SyntheticDataError(
                "stage randomization readback changed before destination binding"
            )
        if readback.get("requested_manifest_sha256") != manifest.sha256:
            raise SyntheticDataError(
                "destination material manifest does not match stage readback"
            )

    materials_readback = dict(readback.get("materials", {}))
    if "destinations" in materials_readback:
        raise SyntheticDataError("destination materials were already applied")

    UsdGeom.Scope.Define(stage, "/World/Looks/TrainingData")
    roughness = _value(manifest, "materials.destination_plastic_roughness")
    hue_jitter_deg = _value(manifest, "materials.destination_hue_jitter_deg")
    saturation_scale = _value(
        manifest, "materials.destination_saturation_scale"
    )
    albedo_scale = _value(manifest, "materials.destination_albedo_scale")
    rows: dict[str, Any] = {}
    for semantic_label in COLOURS:
        target = DESTINATION_APPEARANCE_TARGETS[semantic_label]
        appearance_name = str(target["appearance_name"])
        base_diffuse = tuple(float(v) for v in target["nominal_diffuse_rgb"])
        expected_diffuse = _vary_rgb(
            base_diffuse,
            hue_jitter_deg=hue_jitter_deg,
            saturation_scale=saturation_scale,
            value_scale=albedo_scale,
        )
        material_path = (
            f"/World/Looks/TrainingData/Destination_{semantic_label}_"
            f"{appearance_name}"
        )
        shader_path = material_path + "/PreviewSurface"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        diffuse_input = shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        )
        metallic_input = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
        roughness_input = shader.CreateInput(
            "roughness", Sdf.ValueTypeNames.Float
        )
        diffuse_input.Set(Gf.Vec3f(*expected_diffuse))
        metallic_input.Set(0.0)
        roughness_input.Set(roughness)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        prim_path = f"/World/ColourBin_{semantic_label}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise SyntheticDataError(
                f"destination visual prim is missing; call after add_bin: {prim_path}"
            )
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound_result = binding_api.ComputeBoundMaterial()
        bound_material = (
            bound_result[0] if isinstance(bound_result, tuple) else bound_result
        )
        if (
            not bound_material
            or bound_material.GetPrim().GetPath() != material.GetPrim().GetPath()
        ):
            bound_path = (
                None
                if not bound_material
                else str(bound_material.GetPrim().GetPath())
            )
            raise SyntheticDataError(
                f"destination material binding did not take for {prim_path}: "
                f"got {bound_path}, expected {material_path}"
            )

        diffuse_readback = [float(value) for value in diffuse_input.Get()]
        metallic_readback = float(metallic_input.Get())
        roughness_readback = float(roughness_input.Get())
        _assert_close(
            diffuse_readback,
            expected_diffuse,
            f"{semantic_label}/{appearance_name} destination diffuse colour",
        )
        _assert_close(metallic_readback, 0.0, "destination metallic")
        _assert_close(
            roughness_readback,
            roughness,
            f"{semantic_label} destination roughness",
        )
        rows[semantic_label] = {
            "semantic_label": semantic_label,
            "appearance_name": appearance_name,
            "prim_path": prim_path,
            "effective_material_path": material_path,
            "binding_strength": "strongerThanDescendants",
            "shader_id": str(shader.GetIdAttr().Get()),
            "evidence_source_files": [
                "../HardwareResults/HardwareResultVideos/table_frame0.png",
                "../HardwareResults/HardwareResultVideos/table_frame400.png",
            ],
            "observed_compressed_display_hue_deg": float(
                target["observed_display_hue_deg"]
            ),
            "observed_compressed_display_saturation": float(
                target["observed_display_saturation"]
            ),
            "evidence_limitation": (
                "display-RGB median from compressed 640x360 video; not a "
                "measured physical albedo"
            ),
            "base_appearance": _rgb_evidence(base_diffuse),
            "effective_appearance": _rgb_evidence(diffuse_readback),
            "metallic": metallic_readback,
            "roughness": roughness_readback,
        }

    materials_readback["destinations"] = rows
    readback["materials"] = materials_readback
    return set(DESTINATION_MATERIAL_PARAMETER_NAMES), _with_sha256(readback)


def readback_effective_camera(
    stage: Any,
    *,
    camera_prim_path: str,
    resolution_px: tuple[int, int],
    robot_base_origin_world_m: tuple[float, float, float],
    expected_camera: Mapping[str, Any],
    base_camera: Mapping[str, Any],
    manifest: RandomizationManifest,
) -> tuple[dict[str, Any], set[str]]:
    """Read the camera actually authored in USD and verify randomized values.

    USD and the robot policy share axis directions in this cell, but not their
    origins.  Both transforms are recorded explicitly so a world-frame camera
    translation cannot later be mistaken for robot-base coordinates.
    """

    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim.IsValid():
        raise SyntheticDataError(f"Isaac camera prim is missing: {camera_prim_path}")
    camera = UsdGeom.Camera(prim)
    sampled_expected_camera = _randomized_camera(base_camera, manifest)
    for key in ("eye", "look_at", "up"):
        _assert_close(
            expected_camera[key],
            sampled_expected_camera[key],
            f"randomized camera config {key}",
        )
    _assert_close(
        expected_camera["focal_length"],
        sampled_expected_camera["focal_length"],
        "randomized camera config focal length",
    )
    focal = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
    width, height = (int(value) for value in resolution_px)
    expected_width, expected_height = (
        int(value) for value in expected_camera["resolution"]
    )
    if (width, height) != (expected_width, expected_height):
        raise SyntheticDataError(
            "Isaac render resolution does not match randomized camera config: "
            f"{(width, height)} != {(expected_width, expected_height)}"
        )
    _assert_close(focal, float(expected_camera["focal_length"]), "camera focal length")
    _assert_close(
        horizontal_aperture,
        float(expected_camera["horiz_aperture"]),
        "camera horizontal aperture",
    )
    _assert_close(
        vertical_aperture,
        horizontal_aperture * height / width,
        "camera vertical aperture",
    )

    usd_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    matrix_world = np.asarray(
        [[float(usd_matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype=float,
    )
    if matrix_world.shape != (4, 4):
        raise SyntheticDataError("Isaac camera readback is not a 4x4 transform")
    eye_expected = np.asarray(expected_camera["eye"], dtype=float)
    target_expected = np.asarray(expected_camera["look_at"], dtype=float)
    up_hint = np.asarray(expected_camera["up"], dtype=float)
    forward_expected = target_expected - eye_expected
    forward_expected /= np.linalg.norm(forward_expected)
    right_expected = np.cross(forward_expected, up_hint)
    right_expected /= np.linalg.norm(right_expected)
    up_expected = np.cross(right_expected, forward_expected)
    rotation_expected = np.stack(
        [right_expected, up_expected, -forward_expected], axis=0
    )
    _assert_close(matrix_world[3, :3], eye_expected, "camera eye in USD world")
    _assert_close(
        matrix_world[:3, :3],
        rotation_expected,
        "camera orientation in USD world",
        atol=1e-5,
    )

    base_origin = np.asarray(robot_base_origin_world_m, dtype=float)
    if base_origin.shape != (3,) or not np.all(np.isfinite(base_origin)):
        raise SyntheticDataError("robot-base world origin must be a finite 3-vector")
    matrix_base = matrix_world.copy()
    matrix_base[3, :3] -= base_origin
    fx_px = focal / horizontal_aperture * width
    fy_px = focal / vertical_aperture * height
    payload = {
        "schema_version": 1,
        "source": "USD stage readback after randomized camera construction",
        "camera_prim_path": camera_prim_path,
        "requested_manifest_sha256": manifest.sha256,
        "sampled_camera_parameters": {
            name: float(manifest.value(name)) for name in sorted(CAMERA_PARAMETER_NAMES)
        },
        "intrinsics": {
            "resolution_px": [width, height],
            "resolution_source": (
                "initialized Isaac Camera wrapper configuration; USD camera "
                "prims do not store raster resolution"
            ),
            "focal_length_mm": focal,
            "horizontal_aperture_mm": horizontal_aperture,
            "vertical_aperture_mm": vertical_aperture,
            "fx_px": fx_px,
            "fy_px": fy_px,
            "cx_px": width / 2.0,
            "cy_px": height / 2.0,
            "principal_point_source": "ideal centered Isaac pinhole assumption",
            "lens_model": "ideal Isaac pinhole; physical radial distortion is a separate derived-image transform",
        },
        "extrinsics": {
            "matrix_convention": (
                "row-major USD row-vector local-to-parent transform; translation is row 3"
            ),
            "camera_local_axes": "+X right, +Y up, -Z optical/view direction",
            "usd_world": {
                "camera_to_world_row_major": matrix_world.tolist(),
                "eye_world_m": matrix_world[3, :3].tolist(),
            },
            "robot_base": {
                "relationship_to_usd_world": (
                    "axes aligned; robot-base origin translated from USD world by "
                    "robot_base_origin_world_m"
                ),
                "robot_base_origin_world_m": base_origin.tolist(),
                "camera_to_robot_base_row_major": matrix_base.tolist(),
                "eye_robot_base_m": matrix_base[3, :3].tolist(),
            },
        },
    }
    return _with_sha256(payload), set(CAMERA_PARAMETER_NAMES)


__all__ = [
    "apply_destination_material_randomization",
    "apply_postbuild_randomization",
    "prepare_prebuild_randomization",
    "readback_effective_camera",
]
