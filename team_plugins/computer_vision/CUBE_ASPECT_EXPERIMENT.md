# Cube aspect-ratio experiment — rejected for runtime promotion

## Question

Would widening the cube colour-component aspect-ratio gate recover the
persistently missed clean yellow cube without materially increasing false
targets?

The baseline rejects a contour when its minimum-area rectangle has aspect
ratio `>= 1.5`. The missed yellow-C appearance is approximately 18x30 pixels
in the 1920x1080 annotation image (about 6x10 source pixels), so its aspect
ratio is above the baseline gate. This experiment changes no shared config or
default behavior. Only the explicitly selected
`profile.hybrid-cube-screw-aspect18.experimental.v1.json` applies 1.8.

## Evidence and provenance

- experiment commit: `c29c3a46c4df276b28a1ccdfcc96f04a7248020a`;
- capture SHA-256:
  `236ce2b78e541f4c3511f2cf938d2bf72f75743e73d9ff3c4df50e60f6420787`;
- table calibration SHA-256:
  `7a8df1da65f238a1445319d720c1a2ce143e344754d44e42ff2f82d6d68a29f3`;
- experimental profile SHA-256:
  `4242f2033e4b55adc7f00f66a4a69367d7286f343afa3c1be79604faf30dbe43`;
- 192 table frames sampled every 20 source frames, with the same 16 mm cube
  observation profile used by the frozen baseline replay;
- two provisional, unreviewed, hand-free frames of one static nine-cube
  layout; and
- a separate seven-frame rollout-progression pilot.

Generated evidence is outside Git under `HardwareResults/VisionExperiments`:

- `cube-aspect-sweep-c29c3a4.py`, SHA-256
  `75902c5b1009e836c0cb310a59875f4e6d5f81740495cd215b3250360244c266`;
- `cube-aspect-sweep-c29c3a4.json`, SHA-256
  `1d59fa2a4bb95e79356054309f157d241b1ac9dab625b202a7824eb6b75880dc`;
- `hybrid-r20261029-table-aspect18-c29c3a4-clean/`;
- `annotation-bundle-r20261029-new-layout-aspect18-c29c3a4-clean/`; and
- `annotation-bundle-r20261029-progression-aspect18-c29c3a4-clean/`.

## Matched clean-frame sweep

`Raw` counts every supported provider output. `>=0.2` also applies the checked-in
minimum detection confidence used downstream. `Full candidates` is the raw
cube-output count across all 192 replay frames; an increase is a sensitivity
warning, not proof that every added output is false.

| Aspect limit | Raw TP/FP | >=0.2 TP/FP | Full candidates | Delta from 1.5 |
| --- | ---: | ---: | ---: | ---: |
| 1.50 | 14 / 1 | 14 / 1 | 1562 | 0 |
| 1.55 | 14 / 2 | 14 / 1 | 1757 | +195 (+12.5%) |
| 1.60 | 14 / 2 | 14 / 1 | 1759 | +197 (+12.6%) |
| 1.65 | 14 / 2 | 14 / 1 | 1820 | +258 (+16.5%) |
| 1.70 | 14 / 2 | 14 / 1 | 1823 | +261 (+16.7%) |
| 1.75 | 16 / 3 | 14 / 2 | 1884 | +322 (+20.6%) |
| 1.80 | 16 / 3 | 14 / 2 | 1910 | +348 (+22.3%) |
| 1.90 | 16 / 3 | 14 / 2 | 1947 | +385 (+24.6%) |
| 2.00 | 16 / 3 | 16 / 3 | 1949 | +387 (+24.8%) |

At limits 1.75--1.90, yellow C is present in the raw provider result but has
confidence 0.05, 0.10, and 0.185 respectively. It remains below 0.2, so those
rows are not an operational recovery. At 2.0 its confidence is about 0.252 and
it clears the threshold, but scored false positives increase from one to three
and the full replay emits 24.8% more raw cube candidates.

The independent seven-frame progression set already had 17/17 recall. Raising
the limit to 1.8 preserved 17/17 but increased raw scored false positives from
15 to 21; after the 0.2 confidence threshold, false positives still increased
from 15 to 18. Region exclusions rose from 29 to 35 raw outputs.

## Decision

**Reject the aspect-only change for runtime promotion.** No tested setting
improves confidence-eligible recall without worsening false-target evidence
and substantially increasing full-replay candidate count. Keep the production
hybrid profile at 1.5. The explicit experimental profile remains only so the
result is reproducible; do not select it for hardware or present it as the
winning provider.

A better next experiment is instance segmentation or a cube-shape test that
uses projected top/side-face geometry rather than accepting every elongated
colour component. That experiment must beat the same clean labels and clutter
frames before promotion.

These labels are correlated, provisional, and contain no independent
robot-base pose truth. They compare detection behavior only; they do not prove
millimetre pose accuracy or grasp success.
