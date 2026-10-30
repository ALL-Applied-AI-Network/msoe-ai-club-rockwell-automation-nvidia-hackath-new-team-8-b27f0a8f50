# Computer-vision team setup

This is the first file a computer-vision teammate or coding agent should read.
The detailed interface and validation rules are in this directory's
`AGENTS.md`. If a term or command here is unclear, ask the AI to explain it
before continuing; questions are expected.

## Assignment

Build a vision plugin that finds every supported cube and screw visible in the
requested camera frames and returns a camera-derived, robot-base scene. Prefer
adapting a maintained pretrained model or combining a model with calibrated
geometry over training a large model from scratch.

The handoff is **not** a prose string. Runtime code returns the typed
`sorting_core.perception.SceneObservation`. A file or service may carry the
validated JSON serialization of that same object. Each detection needs, when
observable:

- stable `track_id` and catalog `class_id`;
- separate `sort_label` or destination class when applicable;
- position in robot-base metres;
- orientation quaternion `(x, y, z, w)` with documented symmetry;
- dimensions in metres;
- detection confidence and separate pose quality/visibility;
- source camera IDs, reference timestamp, calibration and provider provenance.

Vision does not choose a grasp or command the robot.

## Work boundary and branch

Your normal edit boundary is `team_plugins/computer_vision/**`, and your branch
is `computer-vision`. Do not edit `sorting_core`, `isaac_bridge`,
`hardware/fanuc_cell`, shared configuration, or the grasp team's directory to
make a result pass. If the contract cannot represent something, create a small
failing fixture and ask the lead for a reviewed interface change.

For a fresh clone:

```bash
git fetch origin
git switch --track origin/computer-vision
```

At the start of a session, before an expensive Brev run, and before handoff:

```bash
git switch computer-vision
git status --short
# Continue only when the status is empty.
git fetch origin
git merge --ff-only origin/computer-vision
git merge --no-edit origin/main
```

If the status is not empty or Git reports a conflict, stop and ask the AI/lead.
Do not reset, discard, or stash somebody else's work. After tests pass, push
only this branch:

```bash
git push origin computer-vision
```

## Environment setup

Contract and replay work normally runs in WSL Ubuntu with the checked-in
hardware-client environment:

```bash
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_integration_plugins.py' -v
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_perception_integration.py' -v
```

Use Brev only when the experiment actually needs Isaac/GPU rendering or model
inference. Follow the repository-root `BREV_SETUP.md`; do not install packages
ad hoc into Isaac. Put heavy/conflicting model dependencies in a pinned
team-owned environment or bounded local service.

Before handoff, run the one-command checker described in
[`docs/TEAM_PLUGIN_CHECKER.md`](../../docs/TEAM_PLUGIN_CHECKER.md) with your
actual versioned vision profile, not the example:

```bash
uv run --project hardware/fanuc_cell python scripts/check_team_plugin.py \
  --profile team_plugins/computer_vision/my_profile.v1.json
```

The handoff run requires a clean tree. `--allow-dirty` is only for an
in-progress development check.

## Cube and screw scope

The cube contract is already implemented and is the first integration target.
The supplied simulation screw is the ISO 10642 M6×30 countersunk screw used by
`demos/scripted_cube_screw_bucket_sort`. Its configured geometry is 30 mm long,
with a 12 mm head and 6 mm shaft; the CAD-local shaft points from the head along
local `-Z`. Treat head/tip polarity and shaft-axis symmetry explicitly.

Important limitation: the existing cube-and-screw demo uses configured poses
and currently has no camera/training-data mode. It does not provide honest
screw perception training data. The lead must integrate the screw into a
camera-derived synthetic-data scene before the team claims realistic synthetic
screw coverage. Physical screw type, finish, dimensions, pose distribution,
and gripper fit also require measurement.

Do not label a screw as `cube` to reach the old planner. Report it with a real
catalog class and let unsupported execution fail visibly until the shared
screw task/grasp path is ready.

## Realistic synthetic testing

In this repository, “realistic mode” means the cube demo's opt-in
`--training-data` mode. It uses native calibrated camera resolutions, material
and lighting variation, sensor effects, clutter, synchronized labels, and
multiple cameras. It is domain randomization—not a guarantee of photorealism
or physical accuracy.

After clean-clone Brev setup and sourcing the generated `env.sh`, run from the
repository checkout with a new external output directory:

```bash
"$ISAAC/python.sh" \
  /workspace/rockwell-hackathon/demos/scripted_colour_bucket_sort/colour_bucket_demo.py \
  --run "$ROCKWELL_RUNS_ROOT/vision_training_seed_7" \
  --training-data --training-cameras table,side,wrist \
  --seed 7 --dataset-stride 15
```

For an explicit wrist-occlusion stress set, add `--wrist-hose-stress`. Its
ignore mask must not become a supervised object label. Use native `raw_frames`
and `sensor_frames` plus `frame_labels`; do not train on presentation frames
with overlays. Simulator truth is allowed for labels/evaluation but must never
repair runtime detections.

The table camera is currently the only camera trusted for motion coordinates.
Side and wrist views are auxiliary until timing and cross-camera calibration
are physically validated. Read `docs/SYNTHETIC_DATA.md` before using generated
data and preserve train/validation/test separation by scene seed or sequence.

## Recommended implementation order

1. **Trace the baseline.** Load one saved frame, run the current provider, and
   inspect a valid `SceneObservation` plus overlay.
2. **Freeze the catalog and metrics.** Start with cubes and the exact screw
   revision. Define symmetry-aware pose errors, detection recall, false
   positives, dropout and latency before choosing a model.
3. **Try the simplest credible baseline.** Color/geometry remains useful for
   cubes. For broader objects, benchmark a pretrained detector/segmenter plus
   calibrated depth or CAD/keypoint pose estimation before fine-tuning.
4. **Write the adapter first.** Add a new provider module and independent JSON
   profile in this directory. Convert all outputs into `SceneObservation`.
5. **Replay before Isaac.** Test saved real and synthetic frames, including
   clutter, hose occlusion, hands, blur, touching objects and missing objects.
6. **Run realistic synthetic evaluation.** Use held-out seeds and camera-derived
   pixels. Never use configured spawn poses to repair a decision.
7. **Hand off a small fixture.** Give grasping and lead a serialized scene they
   can use without importing the vision model or dataset.

## Manageable ideas worth testing

- A maintained pretrained segmentation detector plus calibrated depth/plane
  intersection may be enough; test it before training from scratch.
- For screws, compare mask principal-axis orientation with a small keypoint or
  CAD/PnP pose method. The smallest experiment is a held-out axis-angle error
  plot on isolated screws.
- Fuse cameras only after measuring a specific table-camera failure. A second
  view adds calibration/timing failure modes and is not automatically better.
- Reject ambiguous/occluded objects instead of hallucinating a precise pose.
  A useful “cannot safely estimate” output is better than a confident error.

Avoid reinforcement learning, a learned world model, or arbitrary pile
perception until separated cube-and-screw scenes work end to end.

## Required deliverables

- provider module and independently selectable profile in this directory;
- pinned dependency/model retrieval notes and immutable artifact hash;
- completed cube and provisional screw catalog assumptions;
- preprocessing and coordinate-frame documentation;
- small valid/invalid serialized scene fixtures and overlays;
- contract, replay, held-out synthetic and held-out real-camera results;
- per-class recall/false positives, symmetry-aware pose error and latency;
- known failure cases and explicit unsupported conditions;
- no committed weights, datasets, generated frames, secrets or personal paths.

## Done criteria for team handoff

The vision component is ready for lead integration only when:

1. its profile loads without shared-file edits;
2. every output validates through `validate_scene_for_request()`;
3. identical replay input gives understood/reproducible output;
4. cube results and any claimed screw results meet agreed held-out thresholds;
5. failures are explicit rather than returned as false clear space;
6. the baseline still passes with `ROCKWELL_INTEGRATION_PROFILE` unset;
7. the full handoff checklist in `AGENTS.md` is complete.

If you are unsure what to try next, show the AI the latest metric, overlay and
failure example and ask for the smallest experiment that distinguishes the
leading explanations.
