# YOLO11n closed-set detector pilot

Status: completed offline benchmark plus an explicit experimental runtime
profile. It is **not** hardware-validated pose estimation and is not the
default physical-motion provider.

## Question

Does a small pretrained detector fine-tuned for the project's closed object
catalog transfer from grouped synthetic/archived examples to a different real
hardware capture better than zero/few-shot neural prompting?

The experiment outputs 2D boxes and semantic class IDs only. Cube color remains
a deterministic HSV decision after detection. Calibrated table projection,
catalog geometry, symmetry handling, tracking, uncertainty, and the
`SceneObservation` contract remain separate downstream stages.

## Frozen data split

| Split | Source group | Images | Cube boxes | Screw boxes |
| --- | --- | ---: | ---: | ---: |
| Train | synthetic seeds 101, 211, 307, 401, 509, 601, 709, 811 plus capture `0aa194...` | 14 | 72 | 8 |
| Validation | synthetic seeds 907 and 1009 | 2 | 6 | 0 |
| Final real test | all progression and new-layout frames from capture `236ce2...` | 9 | 35 | 0 |

No cropped-image hash, source-image hash, or capture ID crosses splits. Keeping
the entire final rollout together prevents adjacent-frame/camera-session
leakage. The fixed table crop is `[500, 400, 1500, 1050]`, producing 1000x650
images. Real boxes are provisional and unreviewed; their source is compressed
640x360 video nearest-neighbour-upscaled by the existing replay workflow.

## Frozen training configuration

- Ultralytics `8.4.194`, PyTorch `2.15.0+cu132`, CUDA 13.2;
- pretrained `yolo11n.pt`, detection classes `cube` and
  `screw_iso10642_m6x30`;
- 100 epochs, seeds 53, 54, and 55, image size 960, batch 8;
- AdamW, initial learning rate 0.001, weight decay 0.0005, cosine schedule;
- hue augmentation disabled so downstream color semantics are not deliberately
  corrupted; modest saturation/value and geometric augmentation;
- each deployment threshold selected from the synthetic validation F1 curve;
  and
- final real test evaluated once without post-test threshold tuning.

## Result on the untouched real capture

| Seed | Threshold | TP | FP | FN | Precision | Recall | F1 | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 53 | 0.7628 | 25 | 3 | 10 | 89.3% | 71.4% | 79.4% | 0.767 | 0.314 |
| 54 | 0.7017 | 24 | 3 | 11 | 88.9% | 68.6% | 77.4% | 0.839 | 0.300 |
| 55 | 0.6597 | 29 | 5 | 6 | 85.3% | 82.9% | 84.1% | 0.794 | 0.307 |
| Mean | — | — | — | — | 87.8% | 74.3% | 80.3% | 0.800 | 0.307 |

Warm L40S inference averaged 7.02 ms per cropped frame across the three seeds.
All final-test predictions were class `cube`; the cube-only final set can count
erroneous screw predictions but cannot measure screw recall.

The validation set is only six synthetic cubes, so its 0.995 mAP50 is not a
generalization claim. The first frozen run used Ultralytics `cache="ram"`, which
emitted a nondeterminism warning. Preserve the reported checkpoints and do not
rerun against the same final set until the score looks better. A future
pre-registered experiment should use disk/no cache before viewing test results.

## Experimental runtime integration

`profile.yolo11n-cube.v1.json` selects the exact seed-55 artifact by SHA-256.
The provider uses YOLO only for 2D proposals; the existing calibrated
HSV/known-size geometry still supplies color, robot-base position, and cube yaw.
The strict profile requires one-to-one agreement for every geometry detection
inside the neural crop. Disagreement, unvalidated screw predictions, or
incomplete pick-region crop coverage causes an explicit abstention. It remains
the profile selected by
`supervised_run.py --vision yolo11n --vision-checkpoint PATH`; profile-unset
behavior remains unchanged.

`profile.yolo11n-cube-pickup-gate.v1.json` is a separate replay/no-motion
experiment. It still requires every pickup-eligible geometry detection to
match YOLO, rejects every unmatched YOLO box, rejects screw predictions, and
rejects pickup-eligible geometry outside crop coverage. Geometry outside the
pickup region is retained as scene/destination evidence with explicit
`inside_inference_crop_outside_pickup_geometry_only` provenance instead of
forcing an abstention. Required pickup geometry receives assignment priority
so an optional off-region artifact cannot consume its only candidate box. The
no-motion diagnostic exposes it as `--vision yolo11n-pickup-gate`; the motion
launcher's explicit `--vision` choices do not. The existing advanced
`ROCKWELL_INTEGRATION_PROFILE` override can load arbitrary profiles, so do not
use that override to grant this experiment motion authority.

On all 3,389 frames of the same compressed Rollout 5 table video, the strict
profile accepted 2,199 (64.89%). The pickup gate accepted 2,958 (87.28%), an
increase of exactly 759 frames that the strict error decomposition attributed
only to unmatched off-pickup geometry. It abstained on 431 frames in 48 streaks;
the longest was 50 frames, or 5.0 seconds at decoded 10 Hz. Median/p95 local CPU
latency was 90.52/309.14 ms. Large host-scheduling outliers made the mean and
maximum unsuitable as runtime guarantees. These are agreement-gate
availability measurements on compressed video, not recall, pose accuracy, or
physical-grasp evidence. The baseline remains the motion default.

On the project laptop's WSL CPU, the exact checkpoint loaded in about 1.1 s.
One cold inference took 11.8 s; two subsequent passes took 150 ms and 128 ms.
The provider now runs three warm-up passes during `Policy` construction before
receiving a live frame. A measured three-pass construction took 7.25 s and the
first reconstructed real-frame observation then took 591 ms. The reconstructed
full-frame replay of held-out frame 003800 still abstained because YOLO
produced one box with no matching
calibrated geometry detection. That is the intended fail-closed behavior and
also evidence that this profile must begin with the live no-motion diagnostic,
not hover or contact motion. See `HARDWARE_VALIDATION.md`.

## Decision

This detector transfers much better than the tested YOLOE visual prompts and is
fast enough to remain a candidate. The experimental runtime deliberately
combines it with the calibrated pipeline; it cannot supply rotation or
robot-base position by itself. Do not use the new option for physical motion
from this evidence alone.

Advance only after a new native-resolution, human-reviewed capture provides:

1. held-out cubes and screws from complete independent episodes;
2. precision/recall plus failure overlays, including cable/robot/hand negatives;
3. measured robot-base pose error after calibrated geometry;
4. explicit abstention when a box is occluded or outside usable clearance; and
5. acceptable Ultralytics AGPL-3.0 or Enterprise deployment terms.

The complete private/external reports, checkpoints, plots, and overlays are
under `HardwareResults/VisionExperiments/yolo-detector-finetune-pilot-brev-results/`.
They are intentionally not committed to Git.
