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

## 2026-10-31 — Held-out L40S neural comparison

- Hypothesis: a pretrained neural segmenter can replace or materially improve
  the deterministic cube detector without task-specific training.
- Setup: isolated Ultralytics 8.4.194 environments on one dedicated Brev L40S;
  nine held-out realistic-simulator frames (27 exact cube boxes) and seven
  provisional, unreviewed real-camera frames (17 cube boxes). The real frames
  are compressed 640x360 video decoded and nearest-neighbour-upscaled, so this
  is an exploratory 2D detection test, not robot-base pose validation.
- YOLOE result: with synthetic visual prompts and confidence 0.01, it scored
  27/27 synthetic true positives and no false positives (mean matched box IoU
  0.916), but 0/17 on real frames plus one false positive. Giving it one real
  visual-prompt frame produced only 1/17 true positives plus one false positive.
- FastSAM result: at confidence 0.10 its class-agnostic proposals covered 27/27
  synthetic boxes and 16/17 real boxes, but the real set contained 176 proposals
  of which 160 were unmatched. The fixed HSV/shape post-filter scored 15 TP,
  9 FP, and 2 FN (62.5% precision, 88.2% recall).
- Conclusion: YOLOE's excellent simulator score did not transfer to the camera.
  FastSAM transfers better as a proposal generator, but the tested filter does
  not beat the calibrated geometry control and is not safe to promote into the
  robot loop. Preserve both as reproducible negative/ablation evidence.
- Next experiment: fine-tune a small pretrained two-class detector on grouped
  synthetic plus independent archived data, select its threshold only on
  synthetic validation, and evaluate the untouched hardware capture once.
- License: both harnesses use Ultralytics, so AGPL-3.0/Enterprise deployment
  terms remain a factory-deployment gate.

## 2026-10-31 — Fine-tuned YOLO11n pilot

- Hypothesis: a small pretrained detector fine-tuned on grouped synthetic plus
  independent archived data transfers better than zero/few-shot prompting.
- Frozen split: train = eight synthetic frames plus six frames from capture
  `0aa194...` (72 cube and 8 screw boxes); validation = two synthetic frames
  (6 cubes); final test = all nine frames from independent capture `236ce2...`
  kept together (35 cubes). No capture ID or image hash crosses splits.
- Setup: Ultralytics 8.4.194 `yolo11n.pt`, 100 epochs, seeds 53/54/55, 960-pixel
  input, L40S. Each operating threshold was selected only from synthetic
  validation, then applied once to the provisional real test capture.
- Result: seed recalls were 71.4%, 68.6%, and 82.9%; precisions were 89.3%,
  88.9%, and 85.3%; F1 scores were 79.4%, 77.4%, and 84.1%. The three-seed
  means were 74.3% recall, 87.8% precision, and 80.3% F1. Mean warm inference
  was about 7.02 ms per cropped frame on the L40S.
- Conclusion: this is the strongest neural detector tested, but it is not ready
  to replace the existing runtime provider. Labels are provisional compressed-
  video boxes; no robot-base pose or orientation was evaluated. The final test
  is cube-only, so screw recall is unmeasured.
- Reproducibility note: the first frozen run used Ultralytics RAM caching, which
  emitted a nondeterminism warning. Preserve and report these exact checkpoints
  rather than rerunning until the test result improves; use disk/no cache in a
  future pre-registered experiment.
- Next experiment: human-review native-resolution cube and screw boxes from a
  new capture, keep complete episodes grouped, then connect detections to the
  existing calibrated geometry only if selective held-out accuracy improves.

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

## 2026-10-31 — Final clean compressed replay

- Setup: clean commit `85be0d0`, 199 frames sampled every ten frames from
  `table.mp4`, explicit 16 mm cube profile, qualitative nearest-neighbour
  upscale, capture SHA-256 `0aa1940f...3d58`.
- Result: 2,071 raw cube observations, 126 screw-candidate observations, and
  217 unknown obstacles; latency mean 278.12 ms, median 276.08 ms, p95
  387.29 ms, maximum 490.17 ms. The run emitted a schema-valid standardized
  artifact.
- Regression check: all 199 structured scene records matched the preceding
  clean provider replay after excluding runtime-only latency; the overlay-only
  change did not alter detections.
- Evidence limit: no reviewed references or independent physical pose truth
  exist in this compressed video, so these counts are not accuracy and cannot
  justify motion.
- Next experiment: use the no-motion diagnostic on measured native layouts and
  evaluate the resulting reviewed annotation manifest.

## 2026-10-31 — Lossless annotation bridge

- Problem: the strict evaluator worked, but beginners had to hand-copy nested
  provider predictions and could not export an irregular frame ledger in one
  attributable step.
- Action: add an offline bundle tool that selects explicit replay frame indices,
  verifies capture/replay/calibration/provider/profile/catalog/Git provenance,
  saves decoded and native-calibration PNGs losslessly, prefills predictions,
  and creates separate non-authoritative review overlays.
- Correctness fix: use the detector's stored image centroid for matching. A
  review found that reprojecting the lower 3D cube center shifted frame-620 cube
  centers by 3.04--6.71 pixels (median 4.07 pixels).
- Evidence: seven focused tests cover schema validity, exact pixels, replay-RGB
  identity, preview separation, strict selections, missing detector centers,
  hash mismatch, report/scene mismatch, and stale output refusal. A development smoke exported
  frames 580 and 620 from the archived replay; it remains pilot evidence only.
- Limitation: the starter manifest has no human references or independent pose
  truth. It cannot produce accuracy until those are added and reviewed.

## 2026-10-31 — Initial Brev outage (superseded)

- Hypothesis: a fresh L40S launchable can run the GPU experiments.
- Setup: non-destructive restart of `isaac-launchable-9ff01e`; new AWS
  `g6e.4xlarge` and Crusoe `l40s-48gb.1x` launch attempts.
- Result: existing instance stayed `FAILURE / NOT READY`; both new provisions
  returned Brev `Internal Server Error`.
- Conclusion at that time: external compute blocker, not a model result. Do not
  reset the existing VM blindly because reset persistence is path-limited.
- Later resolution: dedicated instance `isaac-realism-eval-1031` reached
  `RUNNING / READY` with one L40S and produced the held-out YOLOE and FastSAM
  results recorded above. The failed instance was not repaired or reset.

## 2026-10-31 — Clean cube pilot and wrist-assisted coverage

- Hypothesis: a deliberately different wrist viewpoint can recover a persistent
  table-camera miss without weakening the table detector.
- Setup: hand-free hardware frames 3800/3820 from one static nine-cube layout;
  table/wrist RGB at archived 640x360; provisional table labels; recorded
  joints; checked-in FK and wrist calibration; unchanged 16 mm HSV detector.
- Table result: 14/18 reference rows, one scored false positive. One yellow
  cube is missed twice.
- Wrist result: all six labeled cubes in the wrist FOV are detected in both
  frames, including the table dropout. Provisional table+wrist presence union
  is 17/18 and covers 8/8 clearly in-pickup repeated rows.
- Blocker: corresponding wrist/table base XY differs by 6.5--10.8 mm and there
  are no independent pose truths or per-camera exposure timestamps.
- Decision: continue stop-and-look wrist **presence/coverage** experiments; do
  not use wrist coordinates for grasping or average camera poses until an
  independently measured calibration/timing test passes.

## 2026-10-31 — Cube aspect-only recovery rejected

- Hypothesis: widening the 1.5 color-component aspect gate can recover the
  persistent yellow miss without materially increasing false targets.
- Result: aspect 1.8 recovers the raw contour but it remains below the 0.2
  downstream confidence threshold; full-replay candidates rise 22.3% and
  confidence-eligible false positives worsen. Aspect 2.0 clears the missed
  cube but triples clean-frame false positives and raises candidates 24.8%.
- Decision: do not promote the experiment. Keep the selected 1.5 profile and
  the rejected ablation isolated on `vision/cube-recall-aspect`.

## 2026-10-31 — Opt-in YOLO11n hardware evidence seam

- Hypothesis: the validation-selected seed-55 detector can be exercised by the
  physical client without replacing calibrated cube pose geometry or changing
  the profile-unset baseline.
- Implementation: add a pinned, external-checkpoint YOLO11n profile and a
  component-specific `--vision yolo11n --vision-checkpoint PATH` launcher
  option. Inside the 1000x650 crop, YOLO and full-frame calibrated geometry
  must agree one-to-one; disagreement and screw predictions abstain.
- Verification: 43 focused integration tests and all 263 profile-unset tests
  pass. The repository profile validator passes. The exact checkpoint loads
  through the real hardware `Policy`; both motion flags remain false when not
  explicitly armed.
- Local runtime measurement: WSL CPU, CUDA unavailable; checkpoint load about
  1.1 s, cold inference 11.8 s, then 150 ms and 128 ms. One warm-up pass still
  left the first real-frame replay at 2.48 s, so the provider now runs three
  passes during policy construction. A later construction took 7.25 s and its
  first reconstructed real-frame observation took 591 ms.
- Held-out integration result: reconstructed full-frame replay 003800
  abstained in 2.77 s because one YOLO box had no matching calibrated geometry
  detection. The official mock also returned no action because its synthetic
  image contains none of the required real cubes; that is not an accuracy gate.
- Decision: retain the option for live no-motion evidence collection. Do not
  proceed to hover/contact motion until native live records show complete,
  stable one-to-one detections and calibrated poses within the grasp budget.

## 2026-11-01 — Full Rollout 5 YOLO11n shadow replay

- Hypothesis: the seed-55 YOLO11n profile can corroborate the existing
  calibrated-geometry provider continuously enough to justify promotion from
  no-motion evidence collection.
- Setup: replay every one of 3,389 frames from the hand-free compressed
  `Rollout5TableCam.mp4` recording through both providers at clean commit
  `7538013`. Both runs used the same capture SHA-256, calibration SHA-256,
  decoded frame indices, timestamps, 16 mm cube profile, and per-frame RGB
  hashes. The YOLO replay pinned checkpoint SHA-256
  `010bb71b...788fbb` and recorded provider disagreements as explicit
  abstentions.
- Result: the baseline returned a validated scene on 3,389/3,389 frames. The
  strict YOLO+geometry profile accepted 2,199/3,389 frames (64.89%) and
  abstained on 1,190 (35.11%), across 125 streaks with a maximum streak of 86
  consecutive processed frames (8.6 seconds at the decoded 10 Hz rate).
- Timing: local CPU replay latency was 194.78 ms mean for the baseline and
  104.30 ms mean for the YOLO profile; YOLO median/p95 were 80.32/196.36 ms.
  These timings exclude camera/network/controller latency and do not establish
  accuracy.
- Provider disagreement: on accepted frames, the two providers had different
  object counts on 124 frames and left 126 objects unmatched. Matched position
  deltas were exactly zero because the YOLO profile deliberately retains the
  existing calibrated geometry for pose; this is not independent pose
  validation.
- Baseline caution: the cube-only rollout also produced 75 screw-class
  observations, all outside the pickup region, plus substantial off-region
  component/track churn. The current cube task adapter excludes those screw
  observations, so they did not become executable targets.
- Evidence limit: the input is a resized/compressed visualization video with
  no independent object or robot-base pose truth. Acceptance is availability,
  not recall, and provider agreement is not correctness.
- Decision: do not promote YOLO11n to physical motion authority. Keep the
  geometry baseline as the current hardware default and YOLO11n as an
  experimental shadow/no-motion profile. Before reconsidering, evaluate native
  lossless frames with reviewed cube references and a separate real-screw
  positive/negative set, then measure dropout duration and robot-base pose
  error against independent truth.
