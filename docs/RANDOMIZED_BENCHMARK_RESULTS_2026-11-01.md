# Randomized realistic sorting pilot — 2026-11-01

## Result in one sentence

In this matched Isaac Sim pilot, camera geometry with deterministic grasp ranking (DD) was the most reliable tested policy: it correctly sorted 45/55 objects (81.8%) and completed 6/10 scenes, outperforming the guarded pretrained-AI grasp ranker and both strict YOLO-corroborated policies.

## Experimental design

- Git commit: `9149d96ca5993f150561bbf62ed439fae402f47b`
- Ten preregistered randomized scenes, one at each object count from 1 through 10.
- The same ten scenes were replayed under four policies, for 40 end-to-end runs and 55 reference-object opportunities per policy.
- Supported objects were 10 mm red/yellow/blue cubes and the supplied ISO 10642 M6x30 screw; screws and red cubes used the red destination.
- Camera pose, lighting, materials, sensor effects, object pose, and physics were randomized in realistic mode.
- Runtime decisions used rendered camera observations, not simulator object truth. Simulator truth was evaluator-only.
- All 40 expected reports completed. There were zero missing runs and zero episode-level infrastructure failures.
- The comparison passed the benchmark's matched-design, realistic-mode, randomization, runtime-provenance, and submission-eligibility gates.

This is an eligible matched pilot, not a claim of broad physical-robot generalization.

## Policies

| Code | Vision | Grasp ranking |
| --- | --- | --- |
| DD | Calibrated deterministic color/geometry | Deterministic feasible-grasp ranking |
| YD | YOLO11n corroboration gate plus calibrated geometry | Deterministic feasible-grasp ranking |
| DA | Calibrated deterministic color/geometry | Guarded pretrained Cosmos3-Nano ranking |
| YA | YOLO11n corroboration gate plus calibrated geometry | Guarded pretrained Cosmos3-Nano ranking |

Deterministic geometry generated and filtered grasp candidates in every policy. Cosmos ranked only already-feasible candidates. YOLO acted as a strict corroboration gate; calibrated geometry still supplied metric pose.

## Results

| Policy | Objects correctly sorted | Complete scenes | Sorted / simulated min | Vision abstention | Cube completion | Screw completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DD | 45/55 (81.8%; cluster 95% CI 69.6–95.5%) | 6/10 (60.0%) | 4.97 | 0.0% | 29/34 (85.3%) | 16/21 (76.2%) |
| DA | 38/55 (69.1%; 57.1–81.8%) | 3/10 (30.0%) | 3.76 | 0.0% | 26/34 (76.5%) | 12/21 (57.1%) |
| YD | 14/55 (25.5%; 11.6–42.6%) | 1/10 (10.0%) | 4.28 | 27.5% | 10/34 (29.4%) | 4/21 (19.0%) |
| YA | 9/55 (16.4%; 7.0–28.8%) | 1/10 (10.0%) | 3.09 | 42.4% | 7/34 (20.6%) | 2/21 (9.5%) |

Shorter simulated times for YD/YA are not a benefit by themselves: those policies frequently abstained early while leaving objects unresolved. Correct completion is the primary outcome.

## Matched comparisons

All differences below compare the same ten scenes. Confidence intervals use scene-condition cluster resampling.

- DA minus DD object completion: -15.2 percentage points (paired 95% CI -29.9 to -3.3). DD also exceeded DA throughput by 1.11 correctly sorted objects per simulated minute (0.49–1.84).
- YD minus DD object completion: -52.4 points (-71.0 to -34.1).
- YA minus DD object completion: -64.6 points (-83.6 to -44.2).
- YA minus DA object completion: -49.4 points (-69.5 to -29.3).
- YD versus YA was inconclusive for object completion because its interval crossed zero.
- The 2x2 interaction intervals crossed zero, so this pilot does not support a special vision-by-grasp synergy claim.

The guarded DA policy made 54 AI requests: 41 bound to a validated Cosmos choice and 13 fell back after the response violated the selection contract (`selected_not_top_ranked`). Cosmos changed the deterministic choice 21 times; 9/21 of those changed choices succeeded. Cosmos confidence is not a calibrated grasp-success probability.

## Capacity and failure findings

DD completed 27/28 objects (96.4%) in the seven ordinary scenes, versus 18/27 (66.7%) in the three capacity-stress-candidate scenes. Those stress tags identify commanded-slot reuse or high red-destination load, not proven physical bin capacity. With only three stress scenes, this is a strong follow-up signal rather than a firm population estimate.

Other observed failure modes included unreachable poses, obstructed top-down screw grasps, unknown/ambiguous obstacles, target movement or loss during revalidation, failed transport/containment, YOLO/geometry disagreement, and AI response-contract rejection.

## Engineering decision

Use DD as the current demonstration/default policy. The tested pretrained-AI ranker did not improve the reliable deterministic baseline, and the strict YOLO corroboration policy reduced executable-target recall substantially. Preserve both plugin interfaces for future, task-specific models, but require held-out evidence before enabling them by default.

This does not establish that “YOLO is bad” or that learned grasp ranking cannot help. It establishes that these specific guarded policies, checkpoint, prompts, and gates did not improve this matched randomized pilot. Physical-camera calibration, representative real recordings, screw-specific training evidence, probability calibration, and supervised FANUC trials remain necessary.

## Evidence provenance

- Suite index SHA-256: `4cfd0b46afb6e53ed3a2f169aaa2f6563986b6ef9fc3ccb64f3143c330e86e8d`
- Frozen runtime comparison key SHA-256: `2938a2fb899163b416dcdbed5845a28eb76404899e7ead8713e0d24c9e67cae6`
- Compact local evidence archive SHA-256: `45b176706695e10eb76c73cbfea05d29d83f60cb5f7d727cee5b26751680abee`
- The compact archive contains the suite manifests, all reports/provenance without large rendered media, `comparison.json`, `cell_summary.csv`, and the generated comparison README.

