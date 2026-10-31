# Grasp success model notes

## Objective and label

A calibrated scorer is intended to estimate complete-grasp success after
deterministic rejection.
A positive label requires the intended object to be acquired, retained through
lift and transport, released at the intended destination, and still pass final
scene verification. Contact, jaw closure, or a brief lift alone is not success.
Cancelled, crashed, and unobserved episodes are unresolved and are not silently
relabeled as failures.

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

## Current execution boundary

Cube decisions satisfy the shared executable `GraspPlanningPipeline`
contract. The current camera demo reconstructs a grasp instead of consuming the
selected candidate pose, so no closed-loop cube-yaw execution claim is made.
Screw 12/16/20 mm shaft decisions are auditable but offline because the shared public
target is still `CubeDetection` and the executor is cube-specific. The
scripted screw demo can provide matched physics labels, but it does not prove a
camera-derived or closed-loop screw selector. A lead-reviewed generic target
and executor hook is required before screw execution may be claimed.
