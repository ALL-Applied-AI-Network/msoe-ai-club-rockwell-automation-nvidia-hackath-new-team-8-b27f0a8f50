"""Matched realistic-domain randomization for the mixed-object benchmark.

The existing training-data helper was intentionally written for exactly one
red, yellow, and blue cube.  This module keeps that proven default untouched
while applying the same measured ranges to an arbitrary benchmark manifest.
Every value is sampled before a policy cell is chosen, so all four cells see
the same scene, appearance, physics, and sensor condition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from sorting_core.synthetic_data import (
    RandomizationManifest,
    SyntheticDataSpec,
    sample_randomization,
)


SCENE_PARAMETER_PREFIX = "scene."
SENSOR_PARAMETER_PREFIX = "sensor."
PHYSICS_PARAMETER_PREFIX = "physics."


@dataclass(frozen=True)
class MatchedRandomization:
    """One combined manifest plus the independently sampled source manifests."""

    combined: RandomizationManifest
    appearance: RandomizationManifest
    physics: RandomizationManifest
    sensor: RandomizationManifest

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "combination_rule": (
                "materials/lighting/camera from appearance seed; physics from "
                "physics seed; sensor postprocess from sensor seed; scene poses "
                "come only from the benchmark condition manifest"
            ),
            "combined": self.combined.as_dict(),
            "combined_sha256": self.combined.sha256,
            "source_manifest_sha256": {
                "appearance": self.appearance.sha256,
                "physics": self.physics.sha256,
                "sensor": self.sensor.sha256,
            },
            "source_episode_seed": {
                "appearance": self.appearance.episode_seed,
                "physics": self.physics.episode_seed,
                "sensor": self.sensor.episode_seed,
            },
        }


def build_matched_randomization(
    spec: SyntheticDataSpec,
    *,
    condition_id: str,
    appearance_seed: int,
    physics_seed: int,
    sensor_seed: int,
) -> MatchedRandomization:
    """Sample independent domains and combine them without resampling."""

    appearance = sample_randomization(
        spec,
        episode_seed=int(appearance_seed),
        episode_id=f"{condition_id}:appearance",
    )
    physics = sample_randomization(
        spec,
        episode_seed=int(physics_seed),
        episode_id=f"{condition_id}:physics",
    )
    sensor = sample_randomization(
        spec,
        episode_seed=int(sensor_seed),
        episode_id=f"{condition_id}:sensor",
    )
    parameters: dict[str, Mapping[str, Any]] = {}
    for name, appearance_entry in appearance.parameters.items():
        source = appearance
        if name.startswith(PHYSICS_PARAMETER_PREFIX):
            source = physics
        elif name.startswith(SENSOR_PARAMETER_PREFIX):
            source = sensor
        parameters[name] = dict(source.parameters[name])
    # ``episode_seed`` is provenance only; the exact values and all three
    # source hashes above are authoritative.  Derive a stable composite value
    # rather than pretending one source seed generated every parameter.
    seed_digest = hashlib.sha256(
        f"mixed-benchmark-v1:{appearance_seed}:{physics_seed}:{sensor_seed}".encode()
    ).digest()
    combined_seed = int.from_bytes(seed_digest[:8], "big", signed=False)
    combined = RandomizationManifest(
        schema_version=appearance.schema_version,
        episode_id=f"{condition_id}:matched-realism",
        episode_seed=combined_seed,
        spec_path=appearance.spec_path,
        spec_sha256=appearance.spec_sha256,
        capture=appearance.capture,
        range_warning=appearance.range_warning,
        parameters=parameters,
        sensor_noise_seed=sensor.sensor_noise_seed,
    )
    return MatchedRandomization(
        combined=combined,
        appearance=appearance,
        physics=physics,
        sensor=sensor,
    )


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def generalized_prebuild(
    scene_config: Any,
    cell_module: Any,
    randomization: RandomizationManifest,
) -> dict[str, Any]:
    """Apply the existing realistic camera/material baseline before build.

    ``prepare_prebuild_randomization`` owns the measured colour fit and camera
    perturbation.  Its returned fixed-three-cube positions are deliberately
    ignored; the immutable benchmark condition supplies all object poses.
    """

    from isaac_bridge.training_episode import (
        apply_training_appearance_profile,
        prepare_prebuild_randomization,
    )

    profile = apply_training_appearance_profile(scene_config)
    base_spawns = {
        colour: tuple(scene_config_value)
        for colour, scene_config_value in {
            "red": (0.44, -0.19, 0.0),
            "yellow": (0.44, 0.0, 22.0),
            "blue": (0.44, 0.19, -28.0),
        }.items()
    }
    _unused_spawns, applied = prepare_prebuild_randomization(
        scene_config, cell_module, randomization, base_spawns
    )
    if applied:
        raise RuntimeError(
            "prebuild helper claimed USD application before stage construction"
        )
    return profile


def generalized_postbuild(
    stage: Any,
    cell_module: Any,
    randomization: RandomizationManifest,
    cube_instances: Sequence[Mapping[str, Any]],
) -> tuple[set[str], dict[str, Any]]:
    """Bind/read back realistic materials, lights, masses, and physics.

    This is benchmark-only generalization of ``apply_postbuild_randomization``.
    It supports repeated colours and arbitrary cube counts while retaining the
    same USD/PhysX readback checks and parameter ranges.
    """

    import math
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaac_bridge import training_episode as training

    applied: set[str] = set()
    readback: dict[str, Any] = {
        "schema_version": 1,
        "source": "benchmark generalized USD/PhysX readback",
        "requested_manifest_sha256": randomization.sha256,
    }
    base_values = getattr(cell_module.C, "_TRAINING_RANDOMIZATION_BASE", None)
    if not isinstance(base_values, Mapping):
        raise RuntimeError("benchmark prebuild baseline is missing")

    UsdGeom.Scope.Define(stage, "/World/Looks/TrainingData")
    material_rows: dict[str, Any] = {}
    for index, row in enumerate(cube_instances):
        colour = str(row["colour"])
        instance_id = str(row["instance_id"])
        diffuse, metallic, roughness = cell_module._MATERIAL_SPEC[
            f"plastic_{colour}"
        ]
        material_path = f"/World/Looks/TrainingData/SortCube{index}_{colour}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
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
            raise RuntimeError(f"cube visual mesh is missing: {mesh_path}")
        binding = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        binding.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound = binding.ComputeBoundMaterial()
        bound_material = bound[0] if isinstance(bound, tuple) else bound
        if not bound_material or bound_material.GetPrim().GetPath() != material.GetPrim().GetPath():
            raise RuntimeError(f"cube material binding failed: {instance_id}")
        material_rows[instance_id] = {
            "prim_path": f"/World/SortCube{index}",
            "mesh_path": mesh_path,
            "colour": colour,
            "material_path": material_path,
            "diffuse_color": [float(value) for value in diffuse_input.Get()],
            "metallic": float(metallic_input.Get()),
            "roughness": float(roughness_input.Get()),
        }
    applied.update(training.MATERIAL_PARAMETER_NAMES - {"materials.table_roughness"})

    # The printed work-surface overlay is the visible tabletop in realistic
    # mode.  Vary its actual roughness rather than only an obscured base layer.
    expected_table_roughness = float(randomization.value("materials.table_roughness"))
    printed_overlay = stage.GetPrimAtPath("/World/MatPrint")
    overlay_row = None
    if printed_overlay.IsValid():
        printed_material = cell_module.MATS.get("mat_print")
        roughness_input = None
        shader_path = None
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
                    roughness_input = candidate
                    shader_path = str(shader_prim.GetPath())
                    break
            if roughness_input is not None:
                break
        if roughness_input is None:
            raise RuntimeError("printed tabletop has no writable roughness input")
        value = training._set_and_verify(
            roughness_input.GetAttr(),
            expected_table_roughness,
            "benchmark printed-table roughness",
        )
        overlay_row = {"shader_path": shader_path, "roughness": value}
    applied.add("materials.table_roughness")

    object_rows: dict[str, Any] = {}
    for index, row in enumerate(cube_instances):
        prim_path = f"/World/SortCube{index}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"benchmark cube is missing: {prim_path}")
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        position = [float(matrix[3][axis]) for axis in range(3)]
        expected_xy = [float(row["position_base_m"][0]), float(row["position_base_m"][1])]
        training._assert_close(position[:2], expected_xy, f"{row['instance_id']} spawn")
        expected_yaw = float(row["yaw_deg"])
        actual_yaw = math.degrees(math.atan2(float(matrix[0][1]), float(matrix[0][0])))
        yaw_error = (actual_yaw - expected_yaw + 180.0) % 360.0 - 180.0
        training._assert_close(yaw_error, 0.0, f"{row['instance_id']} yaw", atol=1e-4)
        mass = float(UsdPhysics.MassAPI(prim).GetMassAttr().Get())
        training._assert_close(mass, float(cell_module.C.CUBE_MASS), f"{row['instance_id']} mass")
        object_rows[str(row["instance_id"])] = {
            "prim_path": prim_path,
            "position_world_m": position,
            "yaw_world_deg": actual_yaw,
            "mass_kg": mass,
        }
    applied.add("physics.cube_mass_scale")

    light_scales = {
        "/World/DomeLight": "lighting.dome_intensity_scale",
        "/World/KeyLight": "lighting.key_intensity_scale",
        "/World/FillLight": "lighting.fill_intensity_scale",
        "/World/RimLight": "lighting.rim_intensity_scale",
    }
    light_rows: dict[str, Any] = {}
    for path, parameter in light_scales.items():
        prim = stage.GetPrimAtPath(path)
        attribute = prim.GetAttribute("inputs:intensity") if prim.IsValid() else None
        if attribute is None or not attribute.IsValid():
            raise RuntimeError(f"Isaac light is missing: {path}")
        original = float(attribute.Get())
        intensity = training._set_and_verify(
            attribute,
            original * float(randomization.value(parameter)),
            f"{path} intensity",
        )
        light_rows[path] = {"intensity": intensity}
        applied.add(parameter)
    color_scale = np.asarray(
        [
            randomization.value("lighting.color_r_scale"),
            randomization.value("lighting.color_g_scale"),
            randomization.value("lighting.color_b_scale"),
        ],
        dtype=float,
    )
    for path in light_scales:
        attribute = stage.GetPrimAtPath(path).GetAttribute("inputs:color")
        original = np.asarray(attribute.Get(), dtype=float)
        expected = original * color_scale
        if float(np.max(expected)) > 1.0:
            expected /= float(np.max(expected))
        attribute.Set(Gf.Vec3f(*(float(value) for value in expected)))
        actual = [float(value) for value in attribute.Get()]
        training._assert_close(actual, expected, f"{path} color")
        light_rows[path]["color"] = actual
    applied.update(training.LIGHT_COLOR_PARAMETER_NAMES)
    size_scale = float(randomization.value("lighting.source_size_scale"))
    for path in ("/World/KeyLight", "/World/FillLight", "/World/RimLight"):
        for attribute_name in ("inputs:width", "inputs:height"):
            attribute = stage.GetPrimAtPath(path).GetAttribute(attribute_name)
            original = float(attribute.Get())
            value = training._set_and_verify(
                attribute,
                original * size_scale,
                f"{path} {attribute_name}",
            )
            light_rows[path][attribute_name.removeprefix("inputs:")] = value
    applied.add("lighting.source_size_scale")

    physics_prim = stage.GetPrimAtPath(cell_module.PHYS_MAT_PATH)
    physics_api = UsdPhysics.MaterialAPI(physics_prim)
    physics_rows: dict[str, float] = {}
    for parameter, getter in (
        ("physics.object_static_friction", physics_api.GetStaticFrictionAttr),
        ("physics.object_dynamic_friction", physics_api.GetDynamicFrictionAttr),
        ("physics.object_restitution", physics_api.GetRestitutionAttr),
    ):
        value = training._set_and_verify(
            getter(), float(randomization.value(parameter)), parameter
        )
        physics_rows[parameter] = value
        applied.add(parameter)
    readback.update(
        {
            "materials": {
                "task_cubes": material_rows,
                "printed_table_overlay": overlay_row,
            },
            "task_objects": object_rows,
            "lighting": light_rows,
            "physics_material": physics_rows,
        }
    )
    return applied, training._with_sha256(readback)


def classify_application(
    randomization: RandomizationManifest,
    *,
    applied: Sequence[str],
) -> dict[str, Any]:
    """Produce a complete application ledger for the mixed benchmark."""

    from sorting_core.synthetic_data import finalize_application_status

    applied_set = set(applied)
    skipped: dict[str, str] = {}
    for name in randomization.parameters:
        if name in applied_set:
            continue
        if name.startswith(SCENE_PARAMETER_PREFIX):
            skipped[name] = (
                "replaced by the immutable arbitrary-count benchmark scene manifest"
            )
        else:
            skipped[name] = "not applied because the episode failed before this consumer"
    return finalize_application_status(
        randomization,
        applied=applied_set,
        skipped=skipped,
    )


__all__ = [
    "MatchedRandomization",
    "build_matched_randomization",
    "classify_application",
    "generalized_postbuild",
    "generalized_prebuild",
    "write_json_atomic",
]
