from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CELL_ROOT = ROOT / "hardware" / "fanuc_cell"
if str(CELL_ROOT) not in sys.path:
    sys.path.insert(0, str(CELL_ROOT))

import cell_client
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


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _TimedTrialCell(_TrialCell):
    def __init__(
        self,
        clock: _FakeClock,
        *,
        observations,
        observation_delays,
        post_delays,
    ) -> None:
        super().__init__(observations=observations)
        self.fake_clock = clock
        self._observation_delays = iter(observation_delays)
        self._post_delays = iter(post_delays)

    def observe(
        self,
        cameras,
        size=(640, 360),
        fmt="png",
        quality=90,
        depth=(),
    ):
        self.fake_clock.advance(next(self._observation_delays))
        return super().observe(cameras, size, fmt, quality, depth)

    def act_tcp(self, rows, dt=0.1, obs_t=None) -> None:
        self.fake_clock.advance(next(self._post_delays))
        super().act_tcp(rows, dt, obs_t)


class _TimedPrefetchPolicy(_PreviewPolicy):
    PREFETCH = True
    DT = 0.125

    def __init__(self, clock: _FakeClock, inference_delays) -> None:
        super().__init__()
        self.clock = clock
        self._inference_delays = iter(inference_delays)

    def act(self, obs):
        self.clock.advance(next(self._inference_delays))
        self.observation_sizes.append(obs["images"]["request_size"])
        return [[0.0] * 7 for _ in range(32)]


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
        self.assertEqual(cell.action_observation_times, [None, None])

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

    def test_adaptive_prefetch_times_preview_full_inference_and_post(self) -> None:
        clock = _FakeClock()
        cell = _TimedTrialCell(
            clock,
            observations=(
                ("running", 0),
                ("running", 5),
                ("running", 4),
                ("running", 6),
                ("running", 5),
                ("ended", 0),
            ),
            observation_delays=(0.125, 0.125, 0.25, 0.125, 0.25, 0.125),
            post_delays=(0.125, 0.125, 0.125),
        )
        policy = _TimedPrefetchPolicy(clock, (0.125, 0.125, 0.125))
        telemetry: list[dict[str, object]] = []

        def record(event: str, **fields) -> None:
            telemetry.append({"event": event, **fields})

        with (
            patch("cell_client.time.monotonic", new=clock.monotonic),
            patch("cell_client._emit_client_event", side_effect=record),
            patch.dict(
                cell_client.PREFETCH_STATS,
                {"chunks": 0, "ahead": 0, "short": 0},
                clear=True,
            ),
        ):
            run_trial(cell, policy, {"trial_id": "test-t00", "layout": "L1"})

        self.assertEqual(
            [call["size"] for call in cell.calls],
            [
                (1920, 1080),
                (160, 90),
                (1920, 1080),
                (160, 90),
                (1920, 1080),
                (160, 90),
            ],
        )
        self.assertEqual(cell.action_observation_times, [None, 3.0, 5.0])
        self.assertEqual(
            [event["prefetch_lead_rows"] for event in telemetry], [0, 5, 6]
        )
        self.assertEqual(
            [event["queue_preview_used"] for event in telemetry],
            [False, True, True],
        )
        self.assertEqual(
            [event["preview_queue_depth"] for event in telemetry],
            [None, 5, 6],
        )
        self.assertEqual(
            [event["full_observation_queue_depth"] for event in telemetry],
            [0, 4, 5],
        )
        self.assertEqual(
            [event["dispatch_pipeline_s"] for event in telemetry],
            [0.375, 0.625, 0.625],
        )
        self.assertEqual(
            [event["dispatch_pipeline_ewma_s"] for event in telemetry],
            [0.375, 0.45, 0.5025],
        )

    def test_failed_post_does_not_enable_preview_or_increment_stats(self) -> None:
        class FailingOnceCell(_TrialCell):
            def __init__(self) -> None:
                super().__init__(
                    observations=(
                        ("running", 0),
                        ("running", 0),
                        ("ended", 0),
                    )
                )
                self.fail_next_post = True

            def act_tcp(self, rows, dt=0.1, obs_t=None) -> None:
                if self.fail_next_post:
                    self.fail_next_post = False
                    raise RuntimeError("simulated rejected POST")
                super().act_tcp(rows, dt, obs_t)

        class LongChunkPolicy(_PreviewPolicy):
            PREFETCH = True

            def act(self, obs):
                self.observation_sizes.append(obs["images"]["request_size"])
                return [[0.0] * 7 for _ in range(32)]

        cell = FailingOnceCell()
        policy = LongChunkPolicy()
        telemetry: list[dict[str, object]] = []

        with (
            patch(
                "cell_client._emit_client_event",
                side_effect=lambda event, **fields: telemetry.append(
                    {"event": event, **fields}
                ),
            ),
            patch.dict(
                cell_client.PREFETCH_STATS,
                {"chunks": 0, "ahead": 0, "short": 0},
                clear=True,
            ),
        ):
            run_trial(cell, policy, {"trial_id": "test-t00", "layout": "L1"})
            stats = dict(cell_client.PREFETCH_STATS)

        self.assertEqual(
            [call["size"] for call in cell.calls],
            [(1920, 1080), (1920, 1080), (160, 90)],
        )
        self.assertEqual(len(cell.actions), 1)
        self.assertEqual(len(telemetry), 1)
        self.assertEqual(stats, {"chunks": 1, "ahead": 0, "short": 0})

    def test_numeric_prefetch_remains_fixed(self) -> None:
        class FixedPrefetchPolicy(_PreviewPolicy):
            PREFETCH = 3

        cell = _TrialCell(observations=(("running", 3), ("ended", 0)))
        policy = FixedPrefetchPolicy()

        with patch("cell_client._emit_client_event"):
            run_trial(cell, policy, {"trial_id": "test-t00", "layout": "L1"})

        self.assertEqual(cell.action_observation_times, [1.0])


if __name__ == "__main__":
    unittest.main()
