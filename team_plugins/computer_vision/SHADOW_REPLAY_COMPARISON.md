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

Every baseline frame must contain a contract-valid `SceneObservation`. A
candidate frame must contain either one contract-valid scene or one explicit
`VisionError` abstention—never both and never neither. The report includes:

- candidate acceptance and abstention rates;
- error reason counts and consecutive processed-frame streaks;
- object-count and semantic-label disagreement;
- deterministic matched-object robot-base position differences;
- raw quaternion-frame orientation disagreement; and
- provider latency summaries.

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
