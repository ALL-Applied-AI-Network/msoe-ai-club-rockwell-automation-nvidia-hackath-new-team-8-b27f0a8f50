# Grasp success model notes

## Objective and label

A calibrated scorer is intended to estimate complete-grasp success after
deterministic rejection.
A positive label requires the intended object to be acquired, retained through
lift and transport, released at the intended destination, and still pass final
scene verification. Contact, jaw closure, or a brief lift alone is not success.
Cancelled, crashed, and unobserved episodes are unresolved and are not silently
relabeled as failures.

The project's coloured "blocks" are the configured 10 mm square cube class.
Arbitrary rectangular, tilted, or otherwise uncatalogued blocks are unsupported
and fail closed rather than inheriting cube geometry.

Training and held-out splits must be grouped by layout/arrival sequence rather
than adjacent frames. Report Brier score and log loss alongside placement
success and candidate-ranking accuracy.

## Pretrained-model decision

This branch supports two separately pinned pretrained providers for isolated
simulation. Neither provider is a calibrated complete-grasp success model, and
neither is reachable through the standard executable factory.

NVIDIA Cosmos3-Nano is used as a constrained-JSON ranker over candidates that
already passed deterministic feasibility. The validated snapshot revision is
e59a53c25979a090fa8706c9acc0c254a6e89b92 under OpenMDW 1.1. A vLLM 0.21.0
reasoner service returned a schema-complete ranking in 2.024 seconds on the
primary L40S. A live guarded one-candidate planner smoke bound a valid response
and wrote a durable receipt; a separate contradictory three-candidate response
failed semantic validation and used the deterministic fallback. These are
integration/fallback checks, not grasp-quality results. Model confidence remains
separate from GraspScore.success_probability.

NVIDIA GraspGenX is the evaluated geometric proposal source. The validated
Apache-2.0 source commit is
b9429097728cb1c430dd78b92edf17ba318aad03; released checkpoints use the NVIDIA
Open Model License. Its loopback service returned finite 6-DoF poses, including
276 valid infer_object proposals in a live service-only smoke. The repository
adapter never executes those poses directly: it validates SE(3), frame,
calibration, approach axis, and Robotiq descriptor depth, then maps only to an
existing feasible catalog candidate. Model output contains no jaw width, so
catalog finger commands remain authoritative.

The repository does not yet include the capture resolver/ZMQ proposal source that
turns upstream msgpack replies into the request-bound project envelope. GraspGenX
is therefore not an end-to-end planner path until that capture, calibration, and
wire-protocol bridge exists and is tested.

AnyGrasp was not selected because the official SDK requires machine licensing.
Contact-GraspNet and the GraspNet baseline use older conflicting stacks.
FoundationPose estimates object pose rather than grasp quality.

See AI_SELECTION.md for exact hashes, frame conventions, gates, fallback
behavior, durable receipt fields, live evidence, and the no-hardware boundary.

Official references:

- https://github.com/NVlabs/GraspGenX
- https://github.com/graspnet/anygrasp_sdk
- https://github.com/NVlabs/contact_graspnet
- https://github.com/graspnet/graspnet-baseline
- https://github.com/NVlabs/FoundationPose
- https://github.com/NVIDIA/cosmos

## Current scorers

`team8-simulation-ranking-prior-v1` is transparent and explicitly
untrained. It is exposed only through the direct `create_simulation` factory
and the non-executable offline screw planner; no standard integration profile
references it. It exercises candidate generation, rejection, logging, and
matched collection. Its API value is a ranking signal forced through the shared
`success_probability` field, not a calibrated success estimate. The standard
`create` factory refuses it and accepts only a hash-pinned checkpoint scorer.

`team8-json-logistic-v1` loads a strict JSON logistic model whose SHA-256 must
match the integration profile. The loader rejects executable pickle/joblib/
PyTorch artifacts, duplicate keys, nonfinite values, schema drift, feature
reordering, and hash mismatches. Model and dataset artifacts remain outside Git.
Use `train_success_model.py` on grouped Isaac episodes. Dataset rows require
`group_id` and `candidate_id`; groups crossing train/test are rejected and
group-level selection metrics are reported. Model versions bind the trainer
version, feature schema, full dataset hash, and hyperparameters; the separate
checkpoint SHA-256 binds the final bytes. For supervised FANUC composition,
pass the external model and its independently recorded digest through
`supervised_run.py --grasp multigrasp --grasp-checkpoint PATH
--grasp-checkpoint-sha256 HEX`. The launcher verifies the bytes and materializes
a complete temporary vision+grasp profile. Direct Isaac or advanced profile
users may still set `ROCKWELL_GRASP_CHECKPOINT` and provide a matching versioned
profile.

The 12-attempt isolated screw matrix is a diagnostic pilot only and its
collector always reports `trainer_ready: false`. In particular, a
fail/success/fail pattern over the 12/16/20 mm offsets is not monotone in the v1
raw axial-offset feature. Such a result requires a reviewed nonlinear feature
or schema/model revision plus a larger locked collection; it must not be
promoted by training the existing v1 logistic scorer.

## Current execution boundary

Cube decisions satisfy the shared executable `GraspPlanningPipeline`
contract, but the standard camera demo still reconstructs its own grasp. The
grasping-owned `isolated_sim_runner` now wraps the validated mixed Isaac demo,
generates a requested cube or screw candidate from configured target-only truth,
transforms it onto the settled simulator target pose, and verifies the exact
parallel-jaw axis used by the motion segments. It is a
simulation-only experiment, is not referenced by an integration profile, and
uses simulator truth in runtime control.

Cube reports from that runner remain smoke labels because the mixed demo does
not record synchronized cube retention truth throughout carry. A screw report
receives a target-only complete-grasp label only when the recorded screw grasp
point remains within 8 mm of the commanded lift, carry, and lower endpoints and
all revalidation, placement, and final-scene checks resolve. Missing phase or
command evidence is unresolved.

That label does not evaluate final poses or contact safety for the four
non-target obstacles, so it is explicitly ineligible as a collision-safe label.
Pilot feature recomputation uses settled truth for the target but configured
poses for neighboring objects. The public screw decision remains offline
because the shared target is still `CubeDetection`; camera-derived screw
selection and hardware execution still require a reviewed generic target and
executor hook.
