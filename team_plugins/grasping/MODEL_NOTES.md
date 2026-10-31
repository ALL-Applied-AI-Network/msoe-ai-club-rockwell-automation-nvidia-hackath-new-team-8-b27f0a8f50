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

The Brev setup pins and validates the pretrained NVIDIA Cosmos3-Nano model,
which is available to the vision integration for visual/semantic perception.
The grasp-only simulation profile does not itself enable a vision plugin, and
quantitative geometry remains camera-derived. We
reviewed general pretrained grasp models before implementing this selector:

- AnyGrasp is the most plausible future proposal source, but its official SDK
  consumes a calibrated point cloud, returns general 6-DoF grasps, and requires
  machine licensing. The current FANUC executor supports top-down XYZ plus yaw.
- Contact-GraspNet and the GraspNet baseline also consume RGB-D/point clouds and
  return 6-DoF proposals through older, conflicting CUDA/Python stacks.
- FoundationPose estimates object pose; it is not a grasp-success selector.

For those reasons no general 6-DoF checkpoint is wired directly into motion.
A future proposal service may be benchmarked, but every proposal must still be
adapted to the supported top-down family, deterministically checked, and
calibrated against complete project outcomes.

Official references:

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
checkpoint SHA-256 binds the final bytes. Then set
`ROCKWELL_GRASP_CHECKPOINT` and create a matching calibrated profile.

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
