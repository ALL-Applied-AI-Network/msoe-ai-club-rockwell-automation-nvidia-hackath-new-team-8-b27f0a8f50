# Vision approach evaluation

Status date: 2026-10-31. This document separates verified capabilities,
measurements, engineering judgments, and missing evidence.

## Recommendation

Use the hybrid calibrated-geometry profile as the current **offline
experimental control**, not as a motion-ready provider:

- HSV/known-size geometry for red, yellow, and blue cubes;
- dark instance silhouette plus known M6x30 dimensions for isolated screws;
- three strictly newer source-frame confirmations plus projected CAD-size and
  cross-frame shape/axis consistency; and
- explicit rejection for boundary contact, merged shapes, colored-container
  context, invalid calibration/resolution, and ambiguous head/tip polarity.

Its screw scope is isolated, separated parts on the bright neutral table.
Touching/stacked screws abstain. General occlusion is not estimated. Bucket
detection and general unknown-object or unknown-region segmentation are not
implemented.

Keep the completed `vision/yoloe-visual-prompt` branch as an offline experiment;
it has not passed the usefulness gate.
Do not replace geometric coordinate conversion, symmetry handling, or catalog
validation with a neural model. FoundationPose is gated on representative
registered depth and should not be treated as the default.

## Evidence available

### Verified repository/hardware-recording facts

- Official runtime RGB is expected to be uncropped table-camera RGB at
  1920x1080.
- The archived LeRobot recordings contain 69,678 encoded camera frames across
  table, side, and wrist views, but only at 640x360 and 10 Hz.
- The recordings contain no depth, object masks, object poses, per-camera
  exposure timestamps, or measured part ground truth.
- The table-camera calibration reports a 0.465-pixel calibration fit residual,
  but that is not an end-to-end robot-base pose error.
- Side-camera calibration fit is materially worse; the project currently
  permits only the table camera for motion coordinates until the side/wrist
  extrinsics and timing are revalidated.
- Real black screw-like parts are visible in early table views. Their exact
  count, catalog identity, and physical poses are not annotated.
- Hardware logs used a 16 mm cube profile, while archive task text says 10 mm.
  Compressed-video size gating strongly favors 16 mm, but the physical part has
  still not been measured. Keep the size explicit.

### Measured software results

- Before this experiment, rollout-4 compressed replay under the 16 mm cube
  profile found exactly one pickable cube of each color on 121/123 sampled
  frames. That is a stability statistic, not recall or millimetre accuracy.
- Twenty-six hybrid-provider, twelve annotation-evaluator, nine replay-tool,
  fourteen bounded-live-diagnostic, and nine annotation-bundle tests pass (70
  computer-vision-team tests total).
- A 199-frame sampled compressed replay completed with no inference exceptions.
  Before the colored-context guard it produced 278 screw observations and 46
  ambiguous obstacles; these are observations, not true positives.
- Manual review exposed three false screw candidates on/inside the blue
  container in frame 620. Raising the white-table context gate changed that
  frame from four screws to one screw plus three explicit unknown obstacles.
- Four reviewed frames after the guard produced 5 screw observations and 7
  unknown obstacles. There is still no annotation set from which to calculate
  precision/recall.
- A historical dirty-tree post-guard replay sampled 199 compressed/upscaled frames and
  produced 2,071 raw cube observations, 148 screw-class observations, and 326
  unknown obstacles. Raw cube observations intentionally include destination
  and outside-pickup color components needed by the existing policy. These are
  outputs, not true/false-positive counts.
- That historical replay's vision-pipeline CPU latency was mean 328.20 ms, median 318.31
  ms, p95 478.46 ms, and maximum 834.77 ms. The tool times resize, RGB
  conversion, request construction, provider execution, and contract
  validation, while excluding decode, overlay rendering, disk I/O, and
  camera/service/network latency.
- The replay tool now records capture/calibration/provider/profile hashes,
  selected frame indices, and a schema-valid evaluation result. The source was
  `HardwareResults/HardwareResultVideos/table.mp4`, SHA-256
  `0aa1940f3ed93a8f9cef11386946677f0b1216fbe8010c5d55fc0b68505d3d58`.
  The historical dirty-tree run correctly withheld the standardized artifact.
- A clean replay at commit `85be0d0` sampled the same 199 frames and emitted a
  schema-valid evaluation artifact. It produced 2,071 raw cube observations,
  126 screw-candidate observations, and 217 explicit unknown obstacles. CPU
  latency was mean 278.12 ms, median 276.08 ms, p95 387.29 ms, and maximum
  490.17 ms. All 199 structured scene records were semantically identical to
  the preceding clean provider replay after excluding runtime-only latency.
  These are reproducibility/output counts, not precision, recall, or pose
  accuracy: the compressed recording still has no reviewed reference labels or
  independently measured pose truth.
- The longest accepted screw track spanned 111 sampled frames (source frames
  560--1660) and had a maximum projected XY deviation of 2.23 mm from its own
  mean. This is compressed-video repeatability only: identity and physical
  stationarity are not independently labeled, so it is not pose accuracy.
- Sparse sampling initially hid a serious error: the cube-only archive
  `236ce2...0787` produced zero screw outputs every 20 frames, but source frames
  300--560 replayed at every frame produced 34 executable screw observations
  from robot/tool/cable shapes. Three observations alone were not proof of
  identity.
- The tightened development gate uses projected length 26--34 mm, width 9--14
  mm, three distinct source frames, and maximum cross-frame changes of 4 mm
  length, 3 mm width, 15 degrees directed axis, and 0.5 head-width ratio. The
  same 261-frame negative window then produced zero executable screws and 89
  unknown obstacles. Latency was mean 428.09 ms, median 431.40, p95 497.57,
  maximum 606.82.
- A separate cube-only hand/tool window (frames 3640--3800 at every frame)
  produced zero screws but also no general unknown obstacles, demonstrating the
  remaining hand/unknown-segmentation gap.
- An HSV largest-blue-region proxy sampled episode 6 and held-out episode 10 at
  1 Hz. Wrist median area was 9.77%/9.03% and exceeded 2% in 100%/60.47% of
  frames; table and side medians stayed below 0.4% and never exceeded 2%.
  This is not a semantic cable mask—blue objects or hardware can contribute—and
  does not validate multi-camera fusion. It supports keeping the table camera
  as the coordinate source and reobserving from an arm-clear pose until
  side/wrist calibration and exposure timing are measured.
- As a coverage signal, `table.mp4` frames 550--700 at every frame produced one
  screw-like accepted track from frames 562--700, 133 screw observations, and
  210 unknown obstacles. Latency was mean 406.22 ms, median 387.40, p95 530.95,
  maximum 821.43. The track is unannotated and cannot be called a true positive.
- These timings include resize, RGB conversion, request construction, provider
  execution, and contract validation; they exclude decode, overlays, disk I/O,
  service/network delay, and camera latency. The CAD/shape gates were tuned
  after viewing development recordings, so those recordings are not held-out
  final evidence.

### What remains unmeasured

- absolute robot-base position error in millimetres;
- symmetry-aware orientation error against measured physical poses;
- native-resolution precision/recall for either class;
- physical screw finish and exact match to the CAD;
- grasp success from any new vision output;
- confidence calibration;
- camera-to-robot time skew; and
- side/wrist cross-camera bias and synchronization;
- detection of arbitrary unknown objects/occluded regions; and
- the bias from projecting a nonplanar screw silhouette at one assumed
  `surface_z + 6.5 mm` plane and treating its longitudinal extreme as the
  head-face origin.

The provider's 0.15 s frame-age gate cannot resolve the timing item above on
today's hardware adapter because that adapter copies one observation timestamp
into all camera frames. Independent exposure timestamps or a measured timing
experiment are still required.

## Candidate comparison

| Approach | Required inputs | What it can contribute | Main risk/cost | Current decision |
| --- | --- | --- | --- | --- |
| HSV cube + screw silhouette/width profile | Native RGB, fixed table calibration, known catalog geometry | Cube XY/yaw; approximate screw head origin and directed shaft axis; explicit geometric rejection | Dark lookalikes; black/silver finish changes; planar-projection bias; no general 6D pose | Leading offline control; opt-in profile |
| YOLOE-26s segmentation + calibrated geometry | Native tiled RGB; text or visual prompts; GPU preferred | Pretrained instance masks under appearance variation; could bootstrap labels | Tiny targets; AGPL/enterprise licensing; large runtime; zero-shot masks unverified | Offline branch/harness complete; usefulness not demonstrated |
| FoundationPose | RGB, RGB-aligned metric depth, intrinsics, per-instance mask, and either CAD mesh or model-free RGB-D-mask references; NVIDIA GPU stack | Register/track object-to-camera pose plus model score | No archived depth; small/reflective screw; upstream mask source still required; current runtime compatibility unverified | Blocked pending depth/runtime audit |
| FastSAM mask proposals | RGB plus optional box/point/text prompt; GPU preferred | Class-agnostic/single-class instance-mask proposals; possible unknown-object or mask bootstrap | No catalog identity or pose; tiny-object behavior unverified; license ambiguity/Ultralytics terms | Separate experiment only; not a target provider by itself |
| Closed-set segmenter/keypoint model | Human-reviewed masks/keypoints, episode-level split, training GPU | Likely strongest eventual screw mask/head-tip model for the fixed cell | Requires reliable labels and held-out data; overfit risk | Only after label pilot/learning curve |
| GigaPose RGB+CAD | Detection/mask/ROI, CAD templates, older pinned stack | RGB-only novel-object pose | Still needs detection; unnecessary unique 6D pose; integration burden | No branch before planar model fails |

## FoundationPose: verified prerequisite gate

NVIDIA's current standalone inference library documents:

- RGB `HWC uint8`;
- depth `HW float32` in metres;
- 3x3 intrinsics;
- an instance mask for registration;
- OBJ/PLY CAD for model-based use, or RGB-D-mask reference views for its
  model-free mode; and
- a 4x4 object-to-camera output plus a model score.

Registration needs a mask; tracking starts from a previous pose. NVIDIA's Isaac
ROS graph subscribes to color, depth, camera info, and segmentation. It also
documents a minimum point-cloud-support heuristic for drift rejection and says
the generated mask must match RGB dimensions.

Sources:

- <https://github.com/nvidia-isaac/foundation-pose-inference-library>
- <https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html>
- <https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.pdf>
- <https://github.com/nvidia-isaac/foundationpose_perception_pipeline#8-licenses>

The standalone library currently documents an x86-64 NVIDIA container stack
with driver 580 or newer, CUDA 13.2/TensorRT 10.16, Docker Compose 2.30 or
newer, NVIDIA Container Toolkit, and GPU compute capability 7.5 or newer. The
Isaac ROS release is a separate integration with its own ROS/runtime
requirements and an approximately 7 GB documented peak FP32 memory estimate.
Neither stack has been shown compatible with this project's failed/not-ready
Brev instance. Main standalone code is Apache-2.0, while weights and optional
dependencies retain separate terms.

Consequences for this project:

1. Archived RGB video cannot run or validate FoundationPose.
2. The physical depth stream must be proven aligned, synchronized, and dense on
   a 6 mm shaft before model installation is worthwhile.
3. Registration still requires an upstream per-instance mask source, typically
   a detector/segmenter but not necessarily a learned one.
4. FoundationPose outputs camera-frame pose; the plugin still must apply
   `T_base_camera`, catalog axes, symmetry, timestamps, and provenance.
5. A uniform cube's unique orientation and a screw's axial roll remain
   physically ambiguous even if a model emits a quaternion.
6. The documented model score has not been shown to be a calibrated grasp-
   success probability or physical visibility measure.

Kill gate: if representative raw depth does not provide a stable, well-aligned
target footprint, stop this approach. Do not upsample archived video and call it
RGB-D.

## YOLOE smoke

Official YOLOE documentation says the released segmentation checkpoints support
text prompts, visual-example prompts, masks, training, export, and tracking:
<https://docs.ultralytics.com/models/yoloe/>.

That statement applies to the documented `*-seg.pt` text/visual-prompt family.
Prompt-free `*-seg-pf.pt` checkpoints use a built-in vocabulary and are not
trainable through the documented prompt-free mode. Visual-prompt outputs use
generic labels such as `object0`, so an integration must preserve the mapping to
catalog classes rather than treating that label as semantics.

Measured smoke:

- `ultralytics==8.4.194` imported successfully.
- `YOLOE` loaded `yoloe-26s-seg.pt` (29.6 MB).
- The isolated offline harness is pushed on branch
  `vision/yoloe-visual-prompt` at commit
  `6c06c1a64aa5029f4447a6c34a36c7dd9e773a04`; 13/13 branch tests and the
  258/258 repository baseline passed.
- At confidence 0.25, archived CPU inference returned zero detections and took
  about 17.1 seconds.
- At confidence 0.05, it returned one unverified candidate at confidence
  0.0606; CPU inference ranged from about 24.6 to 45.5 seconds.

This passes the import/inference and reproducible-harness gates, not the
usefulness gate. The outputs are unannotated and threshold-sensitive. The next
experiment must use calibrated native-resolution ROI tiles and several reviewed
visual exemplars. Stop if isolated screws still drop out.

Ultralytics publishes its code/models under AGPL unless an Enterprise license
is obtained for proprietary commercial deployment:
<https://www.ultralytics.com/license>. Resolve licensing before presenting this
as a deployable factory product.

## FastSAM prerequisite gate

Ultralytics documents FastSAM as a single-class, class-agnostic segmentation
model: it proposes masks, then box/point/text prompts select them. It does not
produce cube/screw catalog identity, robot-base pose, or calibrated uncertainty;
all of those remain downstream work:
<https://docs.ultralytics.com/models/fast-sam>.

The original repository recommends CUDA and lists CLIP only for text prompting:
<https://github.com/CASIA-LMC-Lab/FastSAM>. Its README and checked-in license
text have conflicting Apache/AGPL statements, and using the Ultralytics wrapper
also invokes Ultralytics licensing. Treat deployment licensing as unresolved
until the organizers/team obtain a clear determination.

Measured smoke on branch `vision/fastsam-mask-provider`, commit
`146043e3ab8ea1edc3f592f7dc81faf8353d5b1d`, used archived frame SHA-256
`2d8a09f3a3c70eec8b9b2212075f21a9405b809193ba5258d5acb5eae7e0dc9d`,
FastSAM-s model SHA-256
`c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844`,
and `ultralytics==8.4.194`. At confidence 0.05 it returned 214 masks; nine
covered the inspected point, including one compact confidence-0.512 proposal
with box `[200, 297, 212, 303]` and area 69 pixels. Cold CPU calls varied from
17.34 to 82.49 s; a repeated same-process call took 3.36 s. The branch passed
7 focused, 27 integration, 14 perception, and 258 profile-unset baseline tests.

This is unannotated proposal coverage, not object identity, base pose, mask
accuracy, or hardware readiness. Giant/nested proposals demonstrate
over-segmentation; semantic classification, calibrated geometry, GPU latency,
runtime compatibility, and AGPL/Enterprise licensing remain blockers. Do not
promote this branch into the runtime provider from this evidence.

## Clean archived cube pilot and wrist-view audit

The strongest current hand-free table-camera pilot uses source frames 3800 and
3820 from hardware rollout `r20261029-88a87b`. It contains nine physical cubes
in one static layout, observed twice. The provisional, unreviewed labels report
14/18 true-positive reference rows, four dropouts, and one scored false
positive: 77.8% row-level recall and 93.3% scored precision. A declared
lower-right staging region excludes nine unresolved predictions, so this is a
pickup-task/ROI-conditioned result, not whole-image precision. One yellow cube
is missed in both frames; two other cubes flicker. There is still no independent
base-pose truth.

An offline wrist-view audit used the same frames, archived joint state, checked-
in FK and `T_flange_cam`, the unchanged known-size HSV detector, and the 16 mm
observation profile. The wrist detector returned all six labeled cubes visible
in its field of view in both frames, including the persistent table-camera
yellow-cube miss at confidence 0.509 and 0.519. A provisional class/presence
union covers 17/18 rows (94.4%) and all 8/8 clearly in-pickup repeated rows.
This is two correlated frames, not a standardized multi-camera accuracy result.

Corresponding wrist/table base-XY estimates disagree by about 6.5--10.8 mm
(about 8.6 mm mean), and the dataset has nominally aligned row indices rather
than measured per-camera exposure timestamps. Therefore:

- keep table RGB as the global pose source;
- use a settled wrist view only as presence/coverage evidence and to trigger
  reobservation;
- do not average or command a wrist-only grasp pose; and
- validate wrist extrinsics, exposure/joint timing, and end-to-end base error
  against independently taught/measured points before coordinate fusion.

Reproducible evidence is outside Git under
`HardwareResults/VisionExperiments/wrist-active-perception-audit/` and is
hash-indexed by `ARTIFACTS.sha256`.

## Cube aspect-gate ablation — rejected

The persistent yellow miss has an approximately 18x30-pixel mask in the
1920x1080 annotation image, so the baseline 1.5 component-aspect gate rejects
it. The isolated branch `vision/cube-recall-aspect` at `9e74555` swept limits
1.50--2.00 behind an explicit experimental profile; defaults remained
unchanged.

- Baseline: 14 TP / 1 FP and 1,562 cube candidates over 192 replay frames.
- Aspect 1.8: raw 16 TP / 3 FP, but confidence-eligible (>=0.2) remains
  14 TP / 2 FP; 1,910 candidates (+22.3%).
- Aspect 2.0: confidence-eligible 16 TP / 3 FP; 1,949 candidates (+24.8%).
- On the seven-frame progression pilot, 1.8 keeps 17/17 recall but increases
  confidence-eligible false positives from 15 to 18.

No tested limit improves operational recall without materially worsening
false-target evidence. Reject the aspect-only change and do not select the
experimental profile for hardware. Prefer calibrated stop-and-look multi-view
coverage or a shape/instance method that beats the same held-out labels and
clutter cases.

## Pose semantics

### Cube

- Origin: cube center.
- Local +Z: table normal.
- Observable yaw: modulo 90 degrees.
- Do not evaluate raw quaternion distance as though every square-equivalent yaw
  were different.

### Lying screw

- Origin: head-face center.
- Local -Z: head-to-tip shaft axis.
- Local +X: an experimental canonical base +Z representative in this restricted
  lying-pose family.
- Observable: head/tip polarity and shaft axis.
- Unobservable at current scale: roll around the shaft. The catalog's physical
  axial symmetry remains unresolved, so this is not a frozen full-pose label.

PnP is not the first solution here. A thin screw does not provide four reliable,
non-collinear named correspondences; OpenCV's general PnP/IPPE requirements do
not make a mask centroid into a valid 6D pose:
<https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html>.

## Stop/continue criteria

Continue the hybrid control when:

- native separated parts are detected on repeated frames;
- no bucket/robot/cable/human detail becomes an executable screw;
- head/tip direction is stable;
- measured pose errors fit the actual grasp clearance; and
- total observation age remains below the hardware limit.

Until those conditions are measured on native captures, “continue” means
offline evaluation/no-motion capture only.

Change approach when:

- screw finish is not dark enough for stable segmentation;
- touching screws routinely merge;
- head/tip polarity is unstable;
- camera calibration dominates error; or
- a pretrained/fine-tuned segmenter measurably improves held-out selective
  accuracy enough to justify its runtime and license.

Do not continue any model because it is fashionable. The winner is the simplest
provider with the best held-out accepted-pose accuracy and abstention behavior.
