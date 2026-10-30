# Measured hardware-pose shadow replay

This opt-in diagnostic renders the existing table, side, and wrist cameras at
curated joint states downloaded from a hardware run. It does **not** replay
recorded actions, simulate controller dynamics, or connect to the physical
robot.

First convert one episode outside the repository:

```bash
uv run --with pyarrow scripts/prepare_hardware_replay.py \
  --dataset /path/to/r20261029-88a87b \
  --episode 0 --split appearance_fit --sample-count 20 \
  --output /workspace/runs/hardware-replay-input.json
```

The converter uses `config/hardware_frame_curation.json` by default. That
manifest is default-deny: the requested dataset/episode must belong to the
named reviewed split, and ranges marked for human or setup activity are not
eligible. Do not use `--ignore-curation` for fitting, training, demos, or
publication. The current review sampled every 20 seconds, so perform a finer
privacy scrub before using a larger set of real frames.

The converter checks the measured J1--J6 values against the recorded TCP in
both XYZ and full FANUC W/P/R orientation. On the curated 12-pose smoke input,
the local residuals are about 0.0054 mm median / 0.0076 mm maximum and 0.0020°
median / 0.0040° maximum. This catches wrist-joint sign or offset errors that a
position-only check could miss.

Then, on the configured Brev instance, load the disposable runtime and run:

```bash
source "/workspace/rockwell-runtime/$(git rev-parse HEAD)/env.sh"
cd "$ROCKWELL_TWIN_RUNTIME"
"$ISAAC/python.sh" \
  /workspace/rockwell-hackathon/demos/hardware_pose_replay/render_pose_replay.py \
  --replay /workspace/runs/hardware-replay-input.json \
  --run /workspace/runs/hardware-pose-replay-01 \
  --camera-matched-appearance
```

Each selected hardware row produces native lossless PNGs and 640x360
deployment previews for all three cameras. `replay_report.json` records camera
transforms per frame; this is particularly important for the moving wrist
camera. The renderer also fails if an arm joint differs from its requested
measured pose by more than 0.02°; the 12-pose L40S smoke run reached 0.0093°
maximum. The derived previews use the camera-specific curated neutral targets
and an I420 chroma round trip; raw renders remain untouched. Add
`--wrist-hose-stress` only for an explicitly unlabeled occlusion stress set.
Its conservative cable-and-shadow geometry masks are saved beside the previews;
I420 chroma spill can extend beyond those masks, so the augmented views remain
explicitly unlabeled robustness images.

`raw/<camera>/` is the untouched Isaac render. `deployment/<camera>/` is a
derived comparison preview, not a replacement for raw data. Camera-specific
neutral matching is bounded and recorded, and the I420 round trip approximates
only the broad chroma loss of the downloaded stream. It does not recover the
real sensor response, AV1 artifacts, exposure control, lens distortion,
sharpness, motion blur, rolling shutter, or flexible-hose physics.

The twin's default task objects are hidden because their poses do not match the
recorded scene. `--show-default-task-objects` is available for a visual scene
check, but it must not be used for pixel-aligned hardware comparison.

Use robot/pedestal silhouettes and fixed background landmarks for alignment.
Mask cubes, humans, laptops, and flexible hoses during camera fitting. Treat
hoses as a separate occlusion stress test. The downloaded 640x360 compressed
videos are useful for broad appearance and framing, but they do not replace
native camera calibration images.
