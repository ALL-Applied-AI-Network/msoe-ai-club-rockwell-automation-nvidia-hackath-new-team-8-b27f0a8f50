# Shadow replay comparison

`compare_replays.py` compares an authoritative replay with a candidate replay
over the exact same sampled table-camera pixels. It is offline: it never loads
a model, contacts the cell, or issues a robot command.

Run `replay_camera` twice from the same clean commit, using identical capture,
calibration, cube-size, and sampling arguments. Use the existing hybrid
geometry profile for the baseline and the experimental profile/checkpoint for
the candidate. For a strict provider, pass `--record-vision-errors` so an
abstention remains an explicit error instead of ending the complete replay.
Then run:

```bash
uv run --project hardware/fanuc_cell python -m \
  team_plugins.computer_vision.compare_replays \
  /path/to/baseline-replay /path/to/candidate-replay \
  --output /path/to/new-shadow-comparison.json
```

The tool accepts only clean, provenance-stable replay outputs. It requires the
capture, calibration, project commit, replay tool, cube-size profile, sampling
arguments, source-frame indices, timestamps, and every per-frame RGB hash to
match. It re-hashes both `report.json` and `scenes.jsonl` before and after the
comparison and creates an optional output file exclusively.

With `--output`, the complete report is written only to that new file. Standard
output contains a compact JSON receipt with the sanitized output filename,
output SHA-256, processed/accepted/abstained counts, longest error streak,
error-category count, and evidence limitation. It never prints an absolute
output path. Without `--output`, stdout retains the complete JSON report for
pipelines that intentionally consume it there.

Every baseline frame must contain a contract-valid `SceneObservation`. A
candidate frame must contain either one contract-valid scene or one explicit
`VisionError` abstention—never both and never neither. The report includes:

- candidate acceptance and abstention rates;
- compact error-category counts, bounded representative frame/hash examples,
  and
  consecutive processed-frame streaks;
- object-count and semantic-label disagreement;
- deterministic matched-object robot-base position differences;
- raw quaternion-frame orientation disagreement; and
- provider latency summaries.

Comparison schema version 2 deliberately does not repeat a full provider error
message in every frame or streak. Messages often contain per-frame track IDs,
coordinates, and box values, which made a long replay report needlessly large
without adding a new failure category. Each abstention frame instead stores its
error type, deterministic category ID, and exact message SHA-256. Category rows
retain total/first/last frame evidence, the number of distinct message hashes,
and at most three representative frame/hash pairs. A fallback category stores
only a SHA-256 of its bounded message prefix, never the prefix text. This keeps
machine-specific paths or identifiers out of the comparison and stdout
receipt. Streaks retain their exact source-frame boundaries and category
counts. The unabridged per-frame messages remain in the candidate replay's
immutable `scenes.jsonl`, whose SHA-256 is recorded as
`availability.full_error_detail_source`. Use that file when an individual
digest needs to be investigated.

Object matching first uses identical `track_id`, `class_id`, and `sort_label`.
Remaining objects are paired greedily by smallest robot-base Euclidean
distance, but only inside identical `(class_id, sort_label)` groups. There is no
distance acceptance threshold and no ground-truth identity claim. Raw
quaternion deltas account for the equivalent signs `q` and `-q`, but are not
reduced by cube, screw, or other catalog symmetries. They therefore show
provider disagreement only—not a physical orientation error.

The baseline is not ground truth. High agreement can show that a candidate
preserves current behavior; it cannot prove that either provider is correct,
that positions are accurate in millimetres, or that a grasp will succeed.
