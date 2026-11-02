"""Serve pinned YOLO11n inference to Isaac over localhost HTTP.

Run this with the plugin-owned YOLO Python environment, not Isaac's Python.
The server binds to 127.0.0.1 by default and transports lossless PNG images.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
import json
import math
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

from experiments.randomized_sort_benchmark.vision import (
    PINNED_YOLO11N_SHA256,
    YOLO_CLASS_NAMES,
)


MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    host = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(host)


class YoloState:
    def __init__(self, checkpoint: Path) -> None:
        digest = _sha256_file(checkpoint)
        if digest != PINNED_YOLO11N_SHA256:
            raise ValueError(
                f"checkpoint hash mismatch: {digest} != {PINNED_YOLO11N_SHA256}"
            )
        from ultralytics import YOLO

        self.model = YOLO(str(checkpoint), task="detect")
        names = {
            int(key): str(value)
            for key, value in dict(getattr(self.model, "names", {})).items()
        }
        if names != YOLO_CLASS_NAMES:
            raise ValueError(
                f"checkpoint class map mismatch: {names!r} != {YOLO_CLASS_NAMES!r}"
            )
        self.checkpoint_sha256 = digest
        self.names = names
        self.lock = threading.Lock()
        self.ultralytics_version = metadata.version("ultralytics")
        self.torch_version = metadata.version("torch")

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        image_size: int,
        confidence: float,
        iou: float,
    ) -> list[dict[str, Any]]:
        with self.lock:
            results = tuple(
                self.model.predict(
                    source=image_bgr,
                    imgsz=image_size,
                    conf=confidence,
                    iou=iou,
                    verbose=False,
                )
            )
        if len(results) != 1:
            raise RuntimeError("single-image inference did not return one result")
        raw_boxes = getattr(results[0], "boxes", None)
        if raw_boxes is None:
            return []
        xyxy = _as_numpy(getattr(raw_boxes, "xyxy", ()))
        scores = _as_numpy(getattr(raw_boxes, "conf", ()))
        classes = _as_numpy(getattr(raw_boxes, "cls", ()))
        if xyxy.size == scores.size == classes.size == 0:
            return []
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise RuntimeError("model returned malformed boxes")
        rows: list[dict[str, Any]] = []
        for coordinates, score, class_id in zip(xyxy, scores, classes, strict=True):
            class_number = float(class_id)
            values = [float(value) for value in coordinates]
            confidence_value = float(score)
            if (
                not class_number.is_integer()
                or int(class_number) not in self.names
                or not all(math.isfinite(value) for value in (*values, confidence_value))
            ):
                raise RuntimeError("model returned invalid numeric box data")
            rows.append(
                {
                    "class_id": int(class_number),
                    "class_name": self.names[int(class_number)],
                    "confidence": confidence_value,
                    "xyxy": values,
                }
            )
        return rows


class Handler(BaseHTTPRequestHandler):
    state: YoloState

    def log_message(self, format: str, *args: Any) -> None:
        print("[yolo-sidecar] " + (format % args), flush=True)

    def _write_json(self, status: int, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/health":
            self._write_json(404, {"error": "not found"})
            return
        self._write_json(
            200,
            {
                "schema_version": "randomized-yolo-sidecar-v1",
                "checkpoint_sha256": self.state.checkpoint_sha256,
                "names": self.state.names,
                "ultralytics_version": self.state.ultralytics_version,
                "torch_version": self.state.torch_version,
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/predict":
            self._write_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_IMAGE_BYTES:
                raise ValueError("invalid image length")
            if self.headers.get("Content-Type") != "image/png":
                raise ValueError("only image/png is accepted")
            image_size = int(self.headers["X-Yolo-Image-Size"])
            confidence = float(self.headers["X-Yolo-Confidence"])
            iou = float(self.headers["X-Yolo-Iou"])
            if image_size <= 0 or not 0.0 <= confidence <= 1.0 or not 0.0 <= iou <= 1.0:
                raise ValueError("invalid inference settings")
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("truncated image body")
            import cv2

            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("invalid PNG image")
            boxes = self.state.predict(
                image,
                image_size=image_size,
                confidence=confidence,
                iou=iou,
            )
            self._write_json(
                200,
                {
                    "schema_version": "randomized-yolo-prediction-v1",
                    "checkpoint_sha256": self.state.checkpoint_sha256,
                    "input_png_sha256": hashlib.sha256(payload).hexdigest(),
                    "boxes": boxes,
                },
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._write_json(400, {"error": str(exc)})
        except Exception as exc:
            self._write_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("the benchmark YOLO sidecar may only bind to localhost")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    Handler.state = YoloState(checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "event": "yolo_sidecar_ready",
                "endpoint": f"http://{args.host}:{args.port}",
                "checkpoint_sha256": Handler.state.checkpoint_sha256,
                "names": Handler.state.names,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
