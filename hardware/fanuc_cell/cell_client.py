"""Client for the FANUC ER-4iA cell. You should not need to edit this file."""
from __future__ import annotations

import base64
import json
import logging
import time

import cv2
import numpy as np
import requests

log = logging.getLogger("cell")
OPEN = ("waiting", "preparing", "ready", "running")


class CellClient:
    def __init__(self, url: str, token: str, ca: str | None = None):
        """ca: a certificate file to trust (the relay's self-signed one), else the system CAs."""
        self.api = url.rstrip("/") + "/api/v1"
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"
        if ca:
            self.s.verify = ca

    def _call(self, method: str, path: str, **kw) -> dict:
        # GETs are safe to retry after a connection failure.  POSTs are not:
        # the server may have applied a command even if its response was lost,
        # and this API exposes no idempotency key or action sequence number.
        attempts = 3 if method.upper() == "GET" else 1
        for attempt in range(attempts):
            try:
                r = self.s.request(method, f"{self.api}/{path}", timeout=10, **kw)
                break
            except requests.ConnectionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(1)
        try:
            body = r.json()
        except ValueError:
            body = {"error": f"HTTP {r.status_code}"}
        if r.status_code == 502:
            raise RuntimeError("the cell is offline; try again later")
        if r.status_code != 200:
            raise RuntimeError(f"{method} /{path}: {body.get('error', body)}")
        return body

    def submit(self, submission: dict) -> dict:
        """Send submission.yaml; returns {"request_id", "secret"}. The operator approves it into a run."""
        return self._call("POST", "submit", json={"submission": submission})

    def request_status(self, request_id: str, secret: str) -> dict:
        """{"status": pending | approved | rejected, "reason"}, and {"run_id", "token"} once approved."""
        return self._call("GET", f"requests/{request_id}", headers={"X-Request-Secret": secret})

    def trials(self) -> list[dict]:
        return self._call("GET", "run")["trials"]

    def start(self, trial_id: str) -> None:
        self._call("POST", f"trials/{trial_id}/start")

    def clock(self, n: int = 5) -> tuple[float, float]:
        """(cell clock - local clock, round trip) in seconds, median of n /time calls."""
        offs, rtts = [], []
        for _ in range(n):
            t1 = time.time()
            t = float(self._call("GET", "time")["t"])
            t2 = time.time()
            offs.append(t - (t1 + t2) / 2)
            rtts.append(t2 - t1)
        return float(np.median(offs)), float(np.median(rtts))

    def observe(self, cameras, size=(640, 360), fmt: str = "png", quality: int = 90, depth=()) -> dict:
        """{"t", "status", "trial", "queued", "joints" (7,), "tcp_pose", "images": {camera: HxWx3 RGB uint8}}
        and, with depth=["table"], "depth": {"table": 400x640 uint16 mm, 0 = no depth}.

        fmt "jpeg" cuts the transfer ~10x (lossy); "png" is lossless. Depth is always lossless."""
        params = {"cameras": ",".join(cameras), "width": size[0], "height": size[1], "format": fmt, "quality": quality}
        if depth:
            params["depth"] = ",".join(depth)
        obs = self._call("GET", "observation", params=params)
        obs["joints"] = np.asarray(obs["joints"], dtype=np.float32)
        obs["images"] = {c: cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_COLOR),
                                         cv2.COLOR_BGR2RGB) for c, b in obs["images"].items()}
        if "depth" in obs:
            obs["depth"] = {c: cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_UNCHANGED)
                            for c, b in obs["depth"].items()}
        return obs

    def act(self, joints, dt: float = 0.1, obs_t: float | None = None) -> None:
        """obs_t: the observation this chunk was planned from, while the previous one still played (PREFETCH):
        the cell replaces what is still queued and skips the rows whose time has passed."""
        joints = np.asarray(joints, dtype=float)
        if not np.all(np.isfinite(joints)):
            raise RuntimeError("POST /actions: joints contain NaN or inf")
        self._call("POST", "actions", json={"joints": joints.tolist(), "dt": dt, **_replace(obs_t)})

    def act_tcp(self, rows, dt: float = 0.1, obs_t: float | None = None) -> None:
        """rows: (N, 7) [x, y, z mm, w, p, r deg, gripper], UTOOL1 (fingertip) in UF0, the frame of obs["tcp_pose"]."""
        rows = np.asarray(rows, dtype=float)
        if not np.all(np.isfinite(rows)):
            raise RuntimeError("POST /actions: tcp contains NaN or inf")
        self._call("POST", "actions", json={"tcp": rows.tolist(), "dt": dt, **_replace(obs_t)})


PREFETCH_STATS = {"chunks": 0, "ahead": 0, "short": 0}  # for run.py --mock's summary


def _replace(obs_t: float | None) -> dict:
    return {} if obs_t is None else {"obs_t": obs_t, "replace": True}


def _observe_policy(cell: CellClient, policy, *, queue_poll: bool) -> dict:
    """Observe the cell, optionally using a policy-declared queue preview.

    A queue preview is never passed to ``Policy.act``.  It exists only to poll
    trial status and the number of queued action rows without repeatedly
    transferring the policy's full-resolution perception image while the arm
    is still executing a chunk.  Policies that do not declare
    ``QUEUE_POLL_IMAGE_SIZE`` retain the original full-observation behavior.
    """

    image_size = getattr(policy, "IMAGE_SIZE", (640, 360))
    image_format = getattr(policy, "IMAGE_FORMAT", "png")
    image_quality = getattr(policy, "IMAGE_QUALITY", 90)
    cameras = policy.CAMERAS
    depth = getattr(policy, "DEPTH", ())
    if queue_poll and getattr(policy, "QUEUE_POLL_IMAGE_SIZE", None) is not None:
        image_size = policy.QUEUE_POLL_IMAGE_SIZE
        image_format = getattr(policy, "QUEUE_POLL_IMAGE_FORMAT", image_format)
        image_quality = getattr(policy, "QUEUE_POLL_IMAGE_QUALITY", image_quality)
        cameras = getattr(policy, "QUEUE_POLL_CAMERAS", cameras)
        depth = getattr(policy, "QUEUE_POLL_DEPTH", ())
    return cell.observe(
        cameras,
        image_size,
        image_format,
        quality=image_quality,
        depth=depth,
    )


def _use_queue_preview(policy, last_chunk_rows: int) -> bool:
    """Whether a completed action chunk is long enough to justify a preview GET."""

    if getattr(policy, "QUEUE_POLL_IMAGE_SIZE", None) is None:
        return False
    minimum_rows = max(1, int(getattr(policy, "QUEUE_POLL_MIN_ROWS", 1)))
    return int(last_chunk_rows) >= minimum_rows


def _emit_client_event(event: str, **fields) -> None:
    """Emit machine-readable client telemetry beside the policy event stream."""

    print(
        json.dumps(
            {
                "event": event,
                "event_wall_time_s": time.time(),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_trial(cell: CellClient, policy, trial: dict) -> None:
    offset, rtt = cell.clock()
    log.info("cell clock offset %+.3f s, round trip %.0f ms", offset, rtt * 1000)
    cell.start(trial["trial_id"])
    policy.reset()
    log.info("trial %s (layout %s) running", trial["trial_id"], trial["layout"])
    dt = getattr(policy, "DT", 0.1)
    prefetch = getattr(policy, "PREFETCH", False)
    # Measured end-to-end time for an action-producing iteration: final queue
    # preview (when used), full policy observation, inference, and action POST.
    # Queue-only polling iterations are intentionally excluded.
    took = None
    last = 0  # actions in the last chunk
    while True:
        pipeline_started = time.monotonic()
        used_queue_preview = _use_queue_preview(policy, last)
        obs = _observe_policy(cell, policy, queue_poll=used_queue_preview)
        if obs["trial"] is None or obs["trial"]["status"] != "running":
            log.info("trial %s ended", trial["trial_id"])
            return
        # PREFETCH: plan the next actions while the last ones still play. True
        # starts when what is left covers the measured end-to-end dispatch
        # pipeline; a number starts when that many actions are left.
        if prefetch is True:
            lead = 0 if took is None else int(np.ceil(took / dt)) + 2
            if last < 2 * lead:  # too few actions to cover act(): planning ahead would only cut them short
                lead = 0
        else:
            lead = int(prefetch or 0)
        preview_queue_depth = int(obs["queued"]) if used_queue_preview else None
        if obs["status"] != "running" or obs["queued"] > lead:
            time.sleep(0.01)  # wait until the last actions have (almost) played out
            continue
        if used_queue_preview:
            # Policy.act must always receive the complete calibrated observation,
            # never the low-resolution queue preview.  Recheck state because the
            # trial or queue can change during the full-resolution request.
            obs = _observe_policy(cell, policy, queue_poll=False)
            if obs["trial"] is None or obs["trial"]["status"] != "running":
                log.info("trial %s ended", trial["trial_id"])
                return
            if obs["status"] != "running" or obs["queued"] > lead:
                time.sleep(0.01)
                continue
        chunk = np.asarray(policy.act(obs), dtype=float)
        if chunk.ndim == 1:  # one target
            chunk = chunk[None]
        obs_t = obs["t"] if lead else None  # the cell drops what is left and skips rows already in the past
        had_timing_sample = took is not None
        try:
            if getattr(policy, "ACTION", "joints") == "tcp":
                cell.act_tcp(chunk, dt, obs_t)
            else:
                cell.act(chunk, dt, obs_t)
        except (RuntimeError, requests.RequestException) as e:
            # Policy.act() is stateful: it can advance dwell/phase counters and
            # gripper intent before the POST is acknowledged.  The protocol has
            # no transaction or idempotency token, so invoking it again could
            # skip a command or duplicate an action whose response was lost.
            failure = (
                "action POST failed after Policy.act; refusing to retry with "
                "advanced policy state. Start a fresh process so Policy.reset() "
                "runs before another hardware action"
            )
            try:
                trials = cell.trials()
                if not isinstance(trials, list):
                    raise ValueError("GET /run did not return a trial list")
                matches = [
                    candidate
                    for candidate in trials
                    if isinstance(candidate, dict)
                    and candidate.get("trial_id") == trial["trial_id"]
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"GET /run returned {len(matches)} matches for "
                        f"trial {trial['trial_id']}"
                    )
                status = matches[0].get("status")
                if not isinstance(status, str) or not status:
                    raise ValueError("matching trial has no valid status")
            except Exception as probe_error:
                log.error(
                    "trial %s action failed and its read-only status probe failed: %s",
                    trial["trial_id"],
                    probe_error,
                )
                raise RuntimeError(f"{failure}; trial status could not be confirmed") from e
            if status != "running":
                # The normal end-of-trial race rejects a final chunk after the
                # server has already made the trial non-running.  It is safe to
                # return, but still unsafe to call Policy.act again.
                log.info(
                    "trial %s became %s while its final action was posted; "
                    "not retrying the action",
                    trial["trial_id"],
                    status,
                )
                return
            raise RuntimeError(f"{failure}; trial is still running") from e
        pipeline_elapsed = time.monotonic() - pipeline_started
        took = (
            pipeline_elapsed
            if took is None
            else 0.7 * took + 0.3 * pipeline_elapsed
        )
        last = len(chunk)
        PREFETCH_STATS["chunks"] += 1
        PREFETCH_STATS["ahead"] += bool(lead)
        PREFETCH_STATS["short"] += bool(
            prefetch is True and had_timing_sample and lead == 0
        )
        _emit_client_event(
            "client_action_dispatch",
            action_rows=last,
            action_interval_s=float(dt),
            dispatch_pipeline_ewma_s=round(float(took), 6),
            dispatch_pipeline_s=round(float(pipeline_elapsed), 6),
            full_observation_queue_depth=int(obs["queued"]),
            prefetch_lead_rows=int(lead),
            preview_queue_depth=preview_queue_depth,
            queue_preview_used=bool(used_queue_preview),
            replacement_used=obs_t is not None,
        )


def run_all(cell: CellClient, policy) -> None:
    """Run every trial of the run as the operator marks it ready. Returns when every trial is over and scored (an
    ended trial not scored yet may still be voided and run again)."""
    while True:
        trials = cell.trials()
        if not any(t["status"] in OPEN or (t.get("scored") is False and t["status"] != "cancelled") for t in trials):
            log.info("all trials finished")
            return
        for t in trials:
            if t["status"] == "ready":
                run_trial(cell, policy, t)
        time.sleep(0.5)
