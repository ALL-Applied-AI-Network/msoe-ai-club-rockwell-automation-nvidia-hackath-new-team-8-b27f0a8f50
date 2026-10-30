from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import MappingProxyType
import unittest
from unittest import mock

from sorting_core.integration_plugins import (
    GraspFactoryContext,
    IntegrationLoadError,
    IntegrationProfile,
    VisionFactoryContext,
    load_active_integration_profile,
    load_factory,
    load_integration_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ENV = "ROCKWELL_INTEGRATION_PROFILE"


def _valid_profile(*, vision: bool = True, grasp: bool = True) -> dict:
    result: dict = {
        "schema_version": 1,
        "profile_id": "test-team-profile",
    }
    if vision:
        result["vision"] = {
            "factory": "external_team_plugin:create_vision",
            "camera_ids": ["table", "side"],
            "depth_camera_ids": ["wrist"],
            "calibration_ids": {
                "table": "test-cal:table",
                "side": "test-cal:side",
            },
            "depth_calibration_ids": {"wrist": "test-cal:wrist-depth"},
            "options": {"threshold": 0.73, "labels": ["cube", "screw"]},
        }
    if grasp:
        result["grasp"] = {
            "factory": "external_team_plugin:create_grasp",
            "options": {"selector": "highest-success", "candidate_count": 4},
        }
    return result


def _write_json(directory: Path, value: object, name: str = "profile.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@contextmanager
def _temporary_module(source: str, name: str = "external_team_plugin"):
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        (directory / f"{name}.py").write_text(
            textwrap.dedent(source), encoding="utf-8"
        )
        sys.path.insert(0, str(directory))
        importlib.invalidate_caches()
        sys.modules.pop(name, None)
        try:
            yield directory, name
        finally:
            sys.modules.pop(name, None)
            try:
                sys.path.remove(str(directory))
            except ValueError:
                pass
            importlib.invalidate_caches()


def _run_python(source: str, *, profile_path: Path, extra_pythonpath: Path | None = None):
    env = os.environ.copy()
    env[PROFILE_ENV] = str(profile_path)
    search_paths = [str(REPO_ROOT)]
    if extra_pythonpath is not None:
        search_paths.insert(0, str(extra_pythonpath))
    existing = env.get("PYTHONPATH")
    if existing:
        search_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(search_paths)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class IntegrationProfileParsingTests(unittest.TestCase):
    def test_absent_profile_is_the_dependency_free_baseline(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(load_active_integration_profile())

    def test_environment_selects_strict_profile_and_preserves_options(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            path = _write_json(Path(raw_directory), _valid_profile())
            with mock.patch.dict(
                os.environ, {PROFILE_ENV: str(path)}, clear=True
            ):
                profile = load_active_integration_profile()

        self.assertIsInstance(profile, IntegrationProfile)
        self.assertEqual(profile.profile_id, "test-team-profile")
        self.assertEqual(profile.source_path, str(path.resolve()))
        self.assertEqual(len(profile.source_sha256), 64)
        self.assertEqual(profile.vision.factory, "external_team_plugin:create_vision")
        self.assertEqual(profile.vision.camera_ids, ("table", "side"))
        self.assertEqual(profile.vision.depth_camera_ids, ("wrist",))
        self.assertEqual(
            dict(profile.vision.calibration_ids),
            {"table": "test-cal:table", "side": "test-cal:side"},
        )
        self.assertEqual(
            dict(profile.vision.depth_calibration_ids),
            {"wrist": "test-cal:wrist-depth"},
        )
        self.assertEqual(profile.vision.options["threshold"], 0.73)
        self.assertEqual(profile.grasp.options["candidate_count"], 4)

    def test_depth_only_vision_profile_parses_without_an_rgb_camera(self):
        value = _valid_profile(grasp=False)
        value["vision"].update(
            {
                "camera_ids": [],
                "depth_camera_ids": ["wrist"],
                "calibration_ids": {},
                "depth_calibration_ids": {"wrist": "test-cal:wrist-depth"},
            }
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            path = _write_json(Path(raw_directory), value)
            profile = load_integration_profile(path)

        self.assertEqual(profile.vision.camera_ids, ())
        self.assertEqual(profile.vision.depth_camera_ids, ("wrist",))
        self.assertEqual(dict(profile.vision.calibration_ids), {})
        self.assertEqual(
            dict(profile.vision.depth_calibration_ids),
            {"wrist": "test-cal:wrist-depth"},
        )

    def test_relative_profile_path_is_resolved_from_repository_root(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as raw_directory:
            directory = Path(raw_directory)
            path = _write_json(directory, _valid_profile(vision=False))
            relative = path.relative_to(REPO_ROOT)
            profile = load_integration_profile(relative)
        self.assertEqual(profile.source_path, str(path.resolve()))

    def test_profile_must_enable_at_least_one_component(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            path = _write_json(
                Path(raw_directory),
                {"schema_version": 1, "profile_id": "empty"},
            )
            with self.assertRaisesRegex(IntegrationLoadError, "vision|grasp"):
                load_integration_profile(path)

    def test_profile_rejects_invalid_json_and_non_object_root(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            invalid = directory / "invalid.json"
            invalid.write_text("{not json", encoding="utf-8")
            with self.assertRaises(IntegrationLoadError):
                load_integration_profile(invalid)
            non_object = _write_json(directory, ["not", "an", "object"], "list.json")
            with self.assertRaisesRegex(IntegrationLoadError, "object"):
                load_integration_profile(non_object)

    def test_profile_rejects_duplicate_keys_and_nonfinite_json_numbers(self):
        invalid_payloads = (
            '{"schema_version":1,"profile_id":"one",'
            '"profile_id":"two","grasp":{"factory":"a:b"}}',
            '{"schema_version":1,"profile_id":"nan",'
            '"grasp":{"factory":"a:b","options":{"value":NaN}}}',
            '{"schema_version":1,"profile_id":"infinity",'
            '"grasp":{"factory":"a:b","options":{"value":Infinity}}}',
        )
        for index, payload in enumerate(invalid_payloads):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as raw_directory:
                path = Path(raw_directory) / "profile.json"
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(IntegrationLoadError):
                    load_integration_profile(path)

    def test_options_are_recursively_frozen_after_validation(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            path = _write_json(Path(raw_directory), _valid_profile())
            profile = load_integration_profile(path)
        self.assertEqual(profile.vision.options["labels"], ("cube", "screw"))
        with self.assertRaises(TypeError):
            profile.vision.options["threshold"] = 0.1
        with self.assertRaises(TypeError):
            profile.grasp.options["candidate_count"] = 1

    def test_profile_rejects_unknown_keys_and_schema_versions(self):
        mutations = (
            (lambda value: value.update({"unexpected": True}), "unknown|unexpected"),
            (lambda value: value.update({"schema_version": 2}), "schema"),
            (lambda value: value["vision"].update({"surprise": 1}), "unknown|surprise"),
            (lambda value: value["grasp"].update({"surprise": 1}), "unknown|surprise"),
        )
        for index, (mutate, pattern) in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as raw_directory:
                value = _valid_profile()
                mutate(value)
                path = _write_json(Path(raw_directory), value)
                with self.assertRaisesRegex(IntegrationLoadError, pattern):
                    load_integration_profile(path)

    def test_profile_rejects_bad_factory_specs_and_options(self):
        mutations = (
            lambda value: value["vision"].update({"factory": "no-colon"}),
            lambda value: value["grasp"].update({"factory": "module:"}),
            lambda value: value["vision"].update({"options": []}),
            lambda value: value["grasp"].update({"options": "not-an-object"}),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as raw_directory:
                value = _valid_profile()
                mutate(value)
                path = _write_json(Path(raw_directory), value)
                with self.assertRaises(IntegrationLoadError):
                    load_integration_profile(path)

    def test_profile_rejects_ambiguous_camera_and_calibration_metadata(self):
        mutations = (
            lambda value: value["vision"].update({"camera_ids": "table"}),
            lambda value: value["vision"].update({"camera_ids": ["table", "table"]}),
            lambda value: value["vision"].update({"camera_ids": [""]}),
            lambda value: value["vision"].update({"calibration_ids": {"table": "x"}}),
            lambda value: value["vision"].update({"depth_camera_ids": ["wrist", "wrist"]}),
            lambda value: value["vision"].update({"depth_calibration_ids": {}}),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as raw_directory:
                value = _valid_profile()
                mutate(value)
                path = _write_json(Path(raw_directory), value)
                with self.assertRaises(IntegrationLoadError):
                    load_integration_profile(path)


class FactoryContextValidationTests(unittest.TestCase):
    def _vision_context(self, **overrides):
        values = {
            "config": object(),
            "camera_projectors": {"wrist": object()},
            "calibration_ids": {},
            "expected_cube_edge_m": None,
            "checkpoint": "vision.ckpt",
            "detector": None,
            "options": {
                "model": {"labels": ["cube", {"enabled": True}]},
            },
            "camera_ids": [],
            "depth_camera_ids": ["wrist"],
            "declared_calibration_ids": {},
            "declared_depth_calibration_ids": {
                "wrist": "test-cal:wrist-depth",
            },
            "profile_id": "direct-context",
            "profile_path": "profiles/direct-context.json",
            "profile_sha256": "A" * 64,
        }
        values.update(overrides)
        return VisionFactoryContext(**values)

    def _grasp_context(self, **overrides):
        values = {
            "config": object(),
            "checkpoint": "grasp.ckpt",
            "options": {
                "selector": {"weights": [0.8, {"fallback": False}]},
            },
            "profile_id": "direct-context",
            "profile_path": "profiles/direct-context.json",
            "profile_sha256": "B" * 64,
        }
        values.update(overrides)
        return GraspFactoryContext(**values)

    def test_direct_contexts_freeze_nested_inputs_and_normalize_camera_lists(self):
        vision = self._vision_context()
        grasp = self._grasp_context()

        self.assertEqual(vision.camera_ids, ())
        self.assertEqual(vision.depth_camera_ids, ("wrist",))
        self.assertIsInstance(vision.camera_projectors, MappingProxyType)
        self.assertIsInstance(vision.calibration_ids, MappingProxyType)
        self.assertIsInstance(vision.declared_calibration_ids, MappingProxyType)
        self.assertIsInstance(
            vision.declared_depth_calibration_ids, MappingProxyType
        )
        self.assertIsInstance(vision.options, MappingProxyType)
        self.assertIsInstance(vision.options["model"], MappingProxyType)
        self.assertEqual(
            vision.options["model"]["labels"],
            ("cube", MappingProxyType({"enabled": True})),
        )
        self.assertEqual(vision.profile_sha256, "a" * 64)

        self.assertIsInstance(grasp.options, MappingProxyType)
        self.assertIsInstance(grasp.options["selector"], MappingProxyType)
        self.assertEqual(
            grasp.options["selector"]["weights"],
            (0.8, MappingProxyType({"fallback": False})),
        )
        self.assertEqual(grasp.profile_sha256, "b" * 64)

        with self.assertRaises(TypeError):
            vision.options["model"] = {}
        with self.assertRaises(TypeError):
            vision.options["model"]["labels"][1]["enabled"] = False
        with self.assertRaises(TypeError):
            grasp.options["selector"]["weights"][1]["fallback"] = True

    def test_direct_vision_context_rejects_no_cameras(self):
        with self.assertRaisesRegex(
            IntegrationLoadError, "at least one RGB or depth camera"
        ):
            self._vision_context(
                camera_ids=[],
                depth_camera_ids=[],
                declared_calibration_ids={},
                declared_depth_calibration_ids={},
            )

    def test_direct_vision_context_requires_exact_declared_calibration_keys(self):
        cases = (
            {
                "camera_ids": ["table"],
                "calibration_ids": {"table": "runtime-cal:table"},
                "declared_calibration_ids": {},
            },
            {
                "camera_ids": [],
                "declared_calibration_ids": {"table": "unexpected"},
            },
            {
                "depth_camera_ids": ["wrist"],
                "declared_depth_calibration_ids": {},
            },
            {
                "camera_ids": ["table"],
                "calibration_ids": {"table": "runtime-cal:table"},
                "declared_calibration_ids": {
                    "table": "test-cal:table",
                },
                "depth_camera_ids": [],
                "declared_depth_calibration_ids": {
                    "wrist": "unexpected",
                },
            },
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                IntegrationLoadError, "keys must exactly match"
            ):
                self._vision_context(**overrides)

    def test_direct_vision_context_requires_exact_effective_calibration_keys(self):
        with self.assertRaisesRegex(
            IntegrationLoadError,
            "calibration_ids keys must exactly match camera_ids",
        ):
            self._vision_context(
                camera_ids=["table"],
                calibration_ids={},
                declared_calibration_ids={"table": "test-cal:table"},
                depth_camera_ids=[],
                declared_depth_calibration_ids={},
            )

    def test_direct_contexts_reject_invalid_profile_hashes(self):
        invalid_hashes = ("", "0" * 63, "0" * 65, "g" * 64)
        for hash_value in invalid_hashes:
            with self.subTest(context="vision", hash_value=hash_value):
                with self.assertRaisesRegex(IntegrationLoadError, "profile_sha256"):
                    self._vision_context(profile_sha256=hash_value)
            with self.subTest(context="grasp", hash_value=hash_value):
                with self.assertRaisesRegex(IntegrationLoadError, "profile_sha256"):
                    self._grasp_context(profile_sha256=hash_value)


class FactoryLoadingTests(unittest.TestCase):
    def test_profile_parsing_does_not_import_heavy_team_module(self):
        source = """
            from pathlib import Path
            Path(__file__).with_suffix('.imported').write_text('imported')
            def create_vision(context):
                return object()
        """
        with _temporary_module(source) as (directory, _), tempfile.TemporaryDirectory() as raw:
            profile_path = _write_json(Path(raw), _valid_profile(grasp=False))
            profile = load_integration_profile(profile_path)
            marker = directory / "external_team_plugin.imported"
            self.assertFalse(marker.exists())
            factory = load_factory(profile.vision.factory, "vision")
            self.assertTrue(callable(factory))
            self.assertTrue(marker.exists())

    def test_factory_loader_accepts_external_package_callable(self):
        source = """
            def create_vision(context):
                return context
        """
        with _temporary_module(source):
            factory = load_factory("external_team_plugin:create_vision", "vision")
            sentinel = object()
            self.assertIs(factory(sentinel), sentinel)

    def test_factory_loader_fails_closed_for_import_attribute_and_type_errors(self):
        source = """
            not_callable = 3
        """
        with _temporary_module(source):
            cases = (
                "missing_team_package:create",
                "external_team_plugin:missing",
                "external_team_plugin:not_callable",
            )
            for spec in cases:
                with self.subTest(spec=spec), self.assertRaises(IntegrationLoadError):
                    load_factory(spec, "vision")

    def test_factory_loader_wraps_exception_raised_during_module_import(self):
        source = """
            raise RuntimeError('native runtime failed during import')
        """
        with _temporary_module(source):
            with self.assertRaisesRegex(
                IntegrationLoadError, "native runtime failed during import"
            ):
                load_factory("external_team_plugin:create_vision", "vision")


class TeamIntegrationActivationTests(unittest.TestCase):
    PLUGIN_SOURCE = """
        from sorting_core.grasping import GraspPlanningPipeline

        last_grasp_context = None

        class Provider:
            def __init__(self, context):
                self.context = context

            def observe(self, request):
                raise AssertionError('not needed by activation test')

        def create_vision(context):
            return Provider(context)

        def create_grasp(context):
            global last_grasp_context
            last_grasp_context = context
            return GraspPlanningPipeline.legacy_default(context.config)
    """

    def test_external_profile_controls_import_time_cameras_and_factory_contexts(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "external_team_plugin.py").write_text(
                textwrap.dedent(self.PLUGIN_SOURCE), encoding="utf-8"
            )
            profile = _write_json(directory, _valid_profile())
            result = _run_python(
                """
                import json
                from sorting_core.config import load_config
                import sorting_core.team_integrations as integrations
                import external_team_plugin

                config = load_config()
                provider = integrations.create_vision_provider(
                    config=config,
                    camera_projectors={"table": object(), "side": object()},
                    expected_cube_edge_m=0.016,
                    checkpoint="vision.ckpt",
                )
                pipeline = integrations.create_grasp_pipeline(
                    config=config,
                    checkpoint="grasp.ckpt",
                )
                grasp_context = external_team_plugin.last_grasp_context
                print(json.dumps({
                    "profile_id": integrations.ACTIVE_INTEGRATION_PROFILE.profile_id,
                    "camera_ids": list(integrations.PERCEPTION_CAMERA_IDS),
                    "depth_camera_ids": list(integrations.PERCEPTION_DEPTH_CAMERA_IDS),
                    "calibration_ids": dict(integrations.PERCEPTION_CALIBRATION_IDS),
                    "depth_calibration_ids": dict(integrations.PERCEPTION_DEPTH_CALIBRATION_IDS),
                    "vision_checkpoint": provider.context.checkpoint,
                    "vision_options": dict(provider.context.options),
                    "vision_profile_id": provider.context.profile_id,
                    "vision_profile_path": provider.context.profile_path,
                    "vision_profile_sha256": provider.context.profile_sha256,
                    "grasp_checkpoint": grasp_context.checkpoint,
                    "grasp_options": dict(grasp_context.options),
                    "grasp_profile_id": grasp_context.profile_id,
                    "grasp_profile_path": grasp_context.profile_path,
                    "grasp_profile_sha256": grasp_context.profile_sha256,
                    "pipeline_type": type(pipeline).__name__,
                }, sort_keys=True))
                """,
                profile_path=profile,
                extra_pythonpath=directory,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["profile_id"], "test-team-profile")
        self.assertEqual(payload["camera_ids"], ["table", "side"])
        self.assertEqual(payload["depth_camera_ids"], ["wrist"])
        self.assertEqual(
            payload["calibration_ids"],
            {"side": "test-cal:side", "table": "test-cal:table"},
        )
        self.assertEqual(
            payload["depth_calibration_ids"],
            {"wrist": "test-cal:wrist-depth"},
        )
        self.assertEqual(payload["vision_checkpoint"], "vision.ckpt")
        self.assertEqual(payload["grasp_checkpoint"], "grasp.ckpt")
        self.assertEqual(payload["vision_options"]["threshold"], 0.73)
        self.assertEqual(payload["grasp_options"]["candidate_count"], 4)
        self.assertEqual(payload["vision_profile_id"], "test-team-profile")
        self.assertEqual(payload["grasp_profile_id"], "test-team-profile")
        self.assertEqual(payload["vision_profile_path"], str(profile.resolve()))
        self.assertEqual(payload["grasp_profile_path"], str(profile.resolve()))
        self.assertEqual(len(payload["vision_profile_sha256"]), 64)
        self.assertEqual(
            payload["vision_profile_sha256"], payload["grasp_profile_sha256"]
        )
        self.assertEqual(payload["pipeline_type"], "GraspPlanningPipeline")

    def test_depth_only_profile_drives_import_time_camera_constants(self):
        value = _valid_profile(grasp=False)
        value["vision"].update(
            {
                "camera_ids": [],
                "depth_camera_ids": ["wrist"],
                "calibration_ids": {},
                "depth_calibration_ids": {"wrist": "test-cal:wrist-depth"},
            }
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            profile = _write_json(Path(raw_directory), value)
            result = _run_python(
                """
                import sys
                from pathlib import Path
                import sorting_core.team_integrations as integrations

                root = Path.cwd()
                sys.path.insert(0, str(root / 'hardware' / 'fanuc_cell'))
                from policy import Policy

                assert integrations.PERCEPTION_CAMERA_IDS == (), (
                    integrations.PERCEPTION_CAMERA_IDS
                )
                assert integrations.PERCEPTION_DEPTH_CAMERA_IDS == ('wrist',), (
                    integrations.PERCEPTION_DEPTH_CAMERA_IDS
                )
                assert dict(integrations.PERCEPTION_CALIBRATION_IDS) == {}
                assert dict(integrations.PERCEPTION_DEPTH_CALIBRATION_IDS) == {
                    'wrist': 'test-cal:wrist-depth',
                }
                assert Policy.CAMERAS == [], Policy.CAMERAS
                assert Policy.DEPTH == ['wrist'], Policy.DEPTH
                print('depth-only-camera-contract')
                """,
                profile_path=profile,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("depth-only-camera-contract", result.stdout)

    def test_hardware_policy_declares_profile_cameras_before_construction(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "external_team_plugin.py").write_text(
                textwrap.dedent(self.PLUGIN_SOURCE), encoding="utf-8"
            )
            profile = _write_json(directory, _valid_profile())
            result = _run_python(
                """
                import sys
                from pathlib import Path
                root = Path.cwd()
                sys.path.insert(0, str(root / 'hardware' / 'fanuc_cell'))
                from policy import Policy
                assert Policy.CAMERAS == ['table', 'side'], Policy.CAMERAS
                assert Policy.DEPTH == ['wrist'], Policy.DEPTH
                print('policy-camera-contract')
                """,
                profile_path=profile,
                extra_pythonpath=directory,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("policy-camera-contract", result.stdout)

    def test_active_profile_import_keeps_selected_model_module_lazy(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            marker = directory / "selected-module-imported"
            (directory / "external_team_plugin.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('imported')\n"
                "class Provider:\n"
                "    def observe(self, request):\n"
                "        raise AssertionError('not used')\n"
                "def create_vision(context):\n"
                "    return Provider()\n",
                encoding="utf-8",
            )
            profile = _write_json(directory, _valid_profile(grasp=False))
            result = _run_python(
                """
                import sorting_core.team_integrations as integrations
                assert integrations.ACTIVE_INTEGRATION_PROFILE is not None
                assert integrations.PERCEPTION_CAMERA_IDS == ('table', 'side')
                print('profile-only')
                """,
                profile_path=profile,
                extra_pythonpath=directory,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("profile-only", result.stdout)
            self.assertFalse(marker.exists())

    def test_baseline_import_does_not_import_unselected_team_module(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            marker = directory / "unexpected-import"
            (directory / "external_team_plugin.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('imported')\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.pop(PROFILE_ENV, None)
            env["PYTHONPATH"] = os.pathsep.join((str(directory), str(REPO_ROOT)))
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sorting_core.team_integrations as t; "
                    "assert t.PERCEPTION_CAMERA_IDS == ('table',); "
                    "assert t.ACTIVE_INTEGRATION_PROFILE is None",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists())

    def test_factory_constructor_failure_is_wrapped_and_never_falls_back(self):
        source = """
            def create_vision(context):
                raise RuntimeError('accelerator unavailable')
        """
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "external_team_plugin.py").write_text(
                textwrap.dedent(source), encoding="utf-8"
            )
            profile = _write_json(directory, _valid_profile(grasp=False))
            result = _run_python(
                """
                from sorting_core.config import load_config
                from sorting_core.integration_plugins import IntegrationLoadError
                import sorting_core.team_integrations as integrations
                try:
                    integrations.create_vision_provider(
                        config=load_config(),
                        camera_projectors={"table": object(), "side": object()},
                    )
                except IntegrationLoadError as error:
                    assert 'vision' in str(error).lower()
                    assert isinstance(error.__cause__, RuntimeError)
                    print('wrapped')
                else:
                    raise AssertionError('broken plugin silently fell back')
                """,
                profile_path=profile,
                extra_pythonpath=directory,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("wrapped", result.stdout)

    def test_wrong_factory_return_types_fail_at_the_activation_boundary(self):
        source = """
            def create_vision(context):
                return object()
            def create_grasp(context):
                return object()
        """
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            (directory / "external_team_plugin.py").write_text(
                textwrap.dedent(source), encoding="utf-8"
            )
            profile = _write_json(directory, _valid_profile())
            result = _run_python(
                """
                from sorting_core.config import load_config
                from sorting_core.integration_plugins import IntegrationLoadError
                import sorting_core.team_integrations as integrations

                config = load_config()
                failures = []
                for name, call in (
                    ('vision', lambda: integrations.create_vision_provider(
                        config=config,
                        camera_projectors={"table": object(), "side": object()},
                    )),
                    ('grasp', lambda: integrations.create_grasp_pipeline(config=config)),
                ):
                    try:
                        call()
                    except IntegrationLoadError:
                        failures.append(name)
                assert failures == ['vision', 'grasp'], failures
                print('fail-closed')
                """,
                profile_path=profile,
                extra_pythonpath=directory,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fail-closed", result.stdout)


if __name__ == "__main__":
    unittest.main()
