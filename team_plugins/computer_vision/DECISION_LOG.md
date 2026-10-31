# Computer-vision decision log

## 2026-10-31 — Preserve the cube baseline

- Hypothesis: replacing the successful cube HSV/geometry path with a learned
  detector would improve the final hardware session.
- Evidence: rollout-4 compressed replay already showed 121/123 sampled frames
  with exactly one pickable cube of each color under the 16 mm profile.
- Conclusion: not supported. Keep that path as the control and preserve the
  profile-unset behavior.
- Next experiment: native measured cube placements, including lighting and
  cable/hand negatives.

## 2026-10-31 — FoundationPose prerequisite audit

- Hypothesis: FoundationPose can directly replace current perception from RGB.
- Evidence: official APIs require RGB-D, intrinsics, CAD, and a registration
  mask. Archived data has RGB only. FoundationPose is not a detector.
- Conclusion: rejected as the immediate path; gated on representative aligned
  depth and upstream segmentation.
- Next experiment: during the hardware session, capture one raw native RGB plus
  aligned depth frame for an isolated cube and screw and audit valid target
  depth support.

## 2026-10-31 — Geometry-first screw control

- Hypothesis: the known 30 mm length, 12 mm head, and 6 mm shaft can distinguish
  an isolated dark screw and recover the grasp-relevant head/axis pose.
- Setup: synthetic images across polarities/rotations plus archived table video.
- Result: 26 provider edge-case/contract tests pass; real-video screw-like parts are detected with
  plausible head axes. Absolute pose truth is absent.
- Failure: dark details on the blue container initially produced three false
  screw candidates in reviewed frame 620.
- Conclusion: promising only with conservative context/ambiguity gates.
- Next experiment: native annotated isolated screws and silver/black finish
  variations.

## 2026-10-31 — Colored-container context guard

- Hypothesis: an isolated screw should be surrounded predominantly by the
  neutral table, unlike a dark bucket rim/label.
- Setup: require at least 70% neutral bright context around a candidate.
- Result: reviewed frame 620 changed from four accepted screw-class detections to
  one screw plus three explicit unknown obstacles.
- Tradeoff: screws very near/overlapping colored parts will be rejected.
- Conclusion: accept this recall loss for the first no-motion hardware
  perception capture; screw motion remains unsupported.
- Next experiment: measure the accepted-coverage versus false-positive curve on
  manually annotated native images.

## 2026-10-31 — YOLOE visual-prompt branch gate

- Hypothesis: a pretrained visual-prompt segmenter recognizes the exact black
  screw without training.
- Setup: Ultralytics 8.4.194, YOLOE-26s-seg, CPU, one reference box in archived
  frame 620, target frame 990, input size 640.
- Result: import/model/inference succeeded. The isolated experiment is pushed as
  branch `vision/yoloe-visual-prompt`, commit
  `6c06c1a64aa5029f4447a6c34a36c7dd9e773a04`; 13/13 branch tests and the
  258/258 repository baseline passed. At confidence 0.25, archived CPU replay
  returned zero detections in about 17.1 seconds. At 0.05, it returned one
  unverified 0.0606-confidence candidate in about 24.6--45.5 seconds.
- Conclusion: the offline harness passes the technical branch gate, but the
  model has not passed the usefulness gate. Test native ROI tiling before any
  integration.
- Risks: tiny targets, multi-gigabyte runtime stack, and AGPL/enterprise
  licensing.
- Next experiment: several reviewed visual exemplars on native-resolution
  pickup-region tiles; compare masks to the geometry control.

## 2026-10-31 — Sparse sampling falsified the first temporal fix

- Hypothesis: requiring three matched observations removes robot/tool/cable
  screw false positives.
- Setup: cube-only archive
  `236ce2b78e541f4c3511f2cf938d2bf72f75743e73d9ff3c4df50e60f6420787`.
  A sparse every-20-frame replay produced zero screw-class outputs.
- Falsification: replaying source frames 300--560 at every frame (the live 10 Hz
  cadence) produced 34 executable screw-class observations from robot/tool/cable
  shapes.
- Conclusion: sparse replay cannot validate a temporal gate. Three frames alone
  did not establish screw identity.
- Action: require strictly newer source-frame timestamps, tighten projected CAD
  size, and require cross-frame length/width/axis/head-ratio consistency.

## 2026-10-31 — Strict CAD/shape gate replay

- Setup: projected length 26--34 mm, width 9--14 mm; three distinct source
  frames; maximum cross-frame changes of 4 mm length, 3 mm width, 15 degrees
  directed axis, and 0.5 head-width ratio.
- Negative result: the same cube-only frames 300--560 at every frame produced
  zero executable screws and 89 explicit unknown obstacles, down from 34
  executable screw outputs. A separate cube-only hand/tool window, frames
  3640--3800 at every frame, produced zero screw outputs; it also demonstrated
  that general hands are not represented as unknown obstacles.
- Coverage signal: `table.mp4` frames 550--700 at every frame produced one
  accepted track from frames 562--700, 133 screw-class observations, and 210
  unknown obstacles.
- Evidence limit: both recordings are compressed/upscaled and unannotated. The
  positive track is only screw-like; these counts do not prove identity,
  precision, recall, pose accuracy, or grasp suitability. The gates were tuned
  after seeing development recordings and are not held-out evidence.
- Conclusion: retain this as an offline control and native-capture candidate,
  not a motion-ready provider.

## 2026-10-31 — Dirty-tree post-guard replay and provenance gate

- Hypothesis: the conservative guard remains operational across a longer
  compressed recording and the evidence can be reproduced.
- Setup: 199 frames sampled every ten frames from `table.mp4`, 16 mm cube
  profile, qualitative nearest-neighbour upscale to 1920x1080.
- Result: 2,071 raw cube observations, 148 screw-class observations, and 326
  unknown obstacles; vision-pipeline latency mean 328.20 ms and p95 478.46 ms.
  These are
  unannotated output counts, not accuracy.
- Repeatability signal: the longest accepted screw track covered 111 sampled
  frames and deviated by at most 2.23 mm from its own mean XY. No annotation
  proves it remained the same stationary screw, so this cannot be called pose
  error.
- Improvement: replay now validates the shared scene contract and records
  capture, calibration, provider, profile, commit, arguments, and frame indices
  and emits a schema-valid evaluation artifact only from a clean committed
  worktree. This dirty run correctly withheld that standardized artifact.
- Conclusion: suitable for regression triage, not physical pose validation.
- Next experiment: native, independently measured cube/screw poses.

## 2026-10-31 — Clean compressed replay persistence signal

- Setup: committed clean replay of `table.mp4`, sampling every tenth 10 Hz
  source frame with the 16 mm profile and qualitative nearest-neighbour upscale.
- Result: one real-looking fastener track was emitted as screw in 109 of 111
  sampled observations from source frames 560--1660. A second screw-like track
  remained `unknown_obstacle` because its projected footprint crossed the
  pickup boundary.
- Evidence limit: the video is compressed, resized, and unannotated. Persistence
  is not a true-positive label, identity proof, millimetre accuracy result, or
  grasp-suitability result. Boundary rejection could be correct or a calibration
  artifact.
- Conclusion: useful qualitative persistence/abstention evidence only. Retain
  the provider for native no-motion measurement, not screw motion.

## 2026-10-31 — Live no-motion perception seam

- Problem: the checked-in `hold` mode returns before perception, while `hover`
  can start approach motion after a stable scene.
- Action: add a vision-owned bounded diagnostic that selects the hybrid profile,
  clears motion/descent/grasp-fit flags, verifies the policy remains unarmed
  after each reset, runs only `_detect_scene_views`, and returns the exact hold
  action unchanged.
- Evidence: focused tests cover environment clearing, exact hold preservation,
  decoded requested-resolution PNG/scene/provenance output, failure recording,
  snapshot bounds,
  reset-time interlock checking, import ordering, dry-run isolation, and refusal
  to write hardware/private captures anywhere inside the repository.
- Limitation: `_detect_scene_views` is a private lead-owned seam. The official
  boundary also applies cube-task partitioning, so an error can follow valid
  provider output. The TCP contract still has no true no-op; measured-pose
  hold can make a small controller correction and is not an emergency stop.
- Next experiment: collect 5--10 decoded full-frame observations for each
  measured layout, verify whether the backend output is sensor-native, then
  evaluate with the strict annotation workflow before enabling any new provider
  for motion.

## 2026-10-31 — Brev availability

- Hypothesis: a fresh L40S launchable can run the GPU experiments.
- Setup: non-destructive restart of `isaac-launchable-9ff01e`; new AWS
  `g6e.4xlarge` and Crusoe `l40s-48gb.1x` launch attempts.
- Result: existing instance stayed `FAILURE / NOT READY`; both new provisions
  returned Brev `Internal Server Error`.
- Conclusion: external compute blocker, not a model result. Do not reset the
  existing VM blindly because reset persistence is path-limited.
- Next experiment: retry only after Brev service recovers or organizer support
  confirms instance state.
