# YOLO11n closed-set detector pilot

Status: completed offline benchmark, **not** a runtime provider and **not**
hardware-ready pose estimation.

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

## Decision

This detector transfers much better than the tested YOLOE visual prompts and is
fast enough to remain a candidate. It does not yet beat the complete calibrated
runtime pipeline, and it cannot supply rotation or robot-base position by
itself. Do not wire it into physical motion from this evidence.

Advance only after a new native-resolution, human-reviewed capture provides:

1. held-out cubes and screws from complete independent episodes;
2. precision/recall plus failure overlays, including cable/robot/hand negatives;
3. measured robot-base pose error after calibrated geometry;
4. explicit abstention when a box is occluded or outside usable clearance; and
5. acceptable Ultralytics AGPL-3.0 or Enterprise deployment terms.

The complete private/external reports, checkpoints, plots, and overlays are
under `HardwareResults/VisionExperiments/yolo-detector-finetune-pilot-brev-results/`.
They are intentionally not committed to Git.
