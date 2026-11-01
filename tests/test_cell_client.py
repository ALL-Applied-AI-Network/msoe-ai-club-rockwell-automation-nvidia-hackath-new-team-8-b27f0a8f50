from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CELL_ROOT = ROOT / "hardware" / "fanuc_cell"
if str(CELL_ROOT) not in sys.path:
    sys.path.insert(0, str(CELL_ROOT))

from cell_client import _observe_policy, _use_queue_preview, run_trial


class _RecordingCell:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def observe(
        self,
        cameras,
        size=(640, 360),
        fmt="png",
        quality=90,
        depth=(),
    ):
        self.calls.append(
            {
                "cameras": tuple(cameras),
                "size": tuple(size),
                "format": fmt,
                "quality": quality,
                "depth": tuple(depth),
            }
        )
        return {"queued": 0}


class _TrialCell(_RecordingCell):
    def __init__(self, observations=None) -> None:
        super().__init__()
        self._observations = iter(
            observations
            or (
                ("running", 0),
                ("running", 2),
                ("running", 0),
                ("running", 0),
                ("ended", 0),
            )
        )
        self.actions: list[list[list[float]]] = []
        self.action_observation_times: list[float | None] = []

    def clock(self):
        return 0.0, 0.0

    def start(self, trial_id: str) -> None:
        self.started_trial = trial_id

    def observe(
        self,
        cameras,
        size=(640, 360),
        fmt="png",
        quality=90,
        depth=(),
    ):
        super().observe(cameras, size, fmt, quality, depth)
        state, queued = next(self._observations)
        running = state == "running"
        return {
            "t": float(len(self.calls)),
            "status": "running" if running else "ended",
            "trial": {"status": "running"} if running else None,
            "queued": queued,
            "images": {"request_size": tuple(size)},
        }

    def act_tcp(self, rows, dt=0.1, obs_t=None) -> None:
        self.actions.append(rows.tolist())
        self.action_observation_times.append(obs_t)


class _PreviewPolicy:
    CAMERAS = ["table"]
    IMAGE_SIZE = (1920, 1080)
    IMAGE_FORMAT = "png"
    QUEUE_POLL_IMAGE_SIZE = (160, 90)
    QUEUE_POLL_IMAGE_FORMAT = "jpeg"
    QUEUE_POLL_IMAGE_QUALITY = 50
    QUEUE_POLL_MIN_ROWS = 4
    ACTION = "tcp"
    PREFETCH = False
    DT = 0.1

    def __init__(self) -> None:
        self.observation_sizes: list[tuple[int, int]] = []

    def reset(self) -> None:
        pass

    def act(self, obs):
        self.observation_sizes.append(obs["images"]["request_size"])
        count = 4 if len(self.observation_sizes) == 1 else 1
        return [[0.0] * 7 for _ in range(count)]


class CellClientQueuePreviewTests(unittest.TestCase):
    def test_policy_without_preview_keeps_original_full_observation(self) -> None:
        class Policy:
            CAMERAS = ["table"]
            IMAGE_SIZE = (1920, 1080)
            IMAGE_FORMAT = "png"
            DEPTH = ("table",)

        cell = _RecordingCell()
        _observe_policy(cell, Policy(), queue_poll=True)

        self.assertEqual(
            cell.calls,
            [
                {
                    "cameras": ("table",),
                    "size": (1920, 1080),
                    "format": "png",
                    "quality": 90,
                    "depth": ("table",),
                }
            ],
        )

    def test_queue_preview_uses_declared_small_lossy_request(self) -> None:
        class Policy:
            CAMERAS = ["table"]
            IMAGE_SIZE = (1920, 1080)
            IMAGE_FORMAT = "png"
            DEPTH = ("table",)
            QUEUE_POLL_CAMERAS = ("table",)
            QUEUE_POLL_IMAGE_SIZE = (160, 90)
            QUEUE_POLL_IMAGE_FORMAT = "jpeg"
            QUEUE_POLL_IMAGE_QUALITY = 50
            QUEUE_POLL_DEPTH = ()

        cell = _RecordingCell()
        _observe_policy(cell, Policy(), queue_poll=True)
        _observe_policy(cell, Policy(), queue_poll=False)

        self.assertEqual(
            cell.calls,
            [
                {
                    "cameras": ("table",),
                    "size": (160, 90),
                    "format": "jpeg",
                    "quality": 50,
                    "depth": (),
                },
                {
                    "cameras": ("table",),
                    "size": (1920, 1080),
                    "format": "png",
                    "quality": 90,
                    "depth": ("table",),
                },
            ],
        )

    def test_run_trial_never_passes_queue_preview_to_policy(self) -> None:
        cell = _TrialCell()
        policy = _PreviewPolicy()

        run_trial(cell, policy, {"trial_id": "test-t00", "layout": "L1"})

        self.assertEqual(policy.observation_sizes, [(1920, 1080), (1920, 1080)])
        self.assertEqual(
            [call["size"] for call in cell.calls],
            [
                (1920, 1080),
                (160, 90),
                (160, 90),
                (1920, 1080),
                (1920, 1080),
            ],
        )
        self.assertEqual(len(cell.actions), 2)

    def test_short_chunks_skip_the_extra_preview_round_trip(self) -> None:
        policy = _PreviewPolicy()

        self.assertFalse(_use_queue_preview(policy, 0))
        self.assertFalse(_use_queue_preview(policy, 1))
        self.assertFalse(_use_queue_preview(policy, 3))
        self.assertTrue(_use_queue_preview(policy, 4))

    def test_prefetch_replacement_uses_full_observation_timestamp(self) -> None:
        class PrefetchPolicy(_PreviewPolicy):
            PREFETCH = True

            def act(self, obs):
                self.observation_sizes.append(obs["images"]["request_size"])
                return [[0.0] * 7 for _ in range(10)]

        cell = _TrialCell(
            observations=(
                ("running", 0),
                ("running", 2),
                ("running", 0),
                ("ended", 0),
            )
        )
        policy = PrefetchPolicy()

        run_trial(cell, policy, {"trial_id": "test-t00", "layout": "L1"})

        self.assertEqual(policy.observation_sizes, [(1920, 1080), (1920, 1080)])
        self.assertEqual(cell.action_observation_times, [None, 3.0])


if __name__ == "__main__":
    unittest.main()
