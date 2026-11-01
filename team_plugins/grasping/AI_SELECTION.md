# Pretrained AI grasp selection

This branch adds simulation-only AI ranking over the existing deterministic cube
pipeline and the non-executable offline screw planner. It does not add a hardware
execution path. The standard create() factory is unchanged and still requires a
calibrated, hash-pinned success model.

## Safety boundary

The order of operations is fixed:

1. Generate the catalog grasp family.
2. Reject candidates with the deterministic scene-envelope checker.
3. Extract the existing versioned numeric features and baseline score.
4. Send only the remaining EvaluatedGrasp records to a pretrained provider.
5. Validate the complete response, request hash, model identity, candidate set,
   and numeric bounds; normalize valid score pairs by descending score and
   candidate ID before checking the selected top, confidence, ties, and margin.
6. Select an existing feasible evaluation or explicitly use the deterministic
   HighestSuccessSelector fallback.
7. Fsync a hash-chained receipt before returning a simulation plan.

The AI cannot invent or mutate a catalog pose or catalog jaw command, restore a
rejected candidate, or write model confidence into
GraspScore.success_probability. The public entry points are
build_ai_selected_simulation_planner for 10 mm blocks and
build_ai_selected_offline_screw_planner for the supported M6x30 screw. Both
expose planning only and import no robot client, executor, hardware plugin, or
physical-motion API. Screw decisions remain execution_supported=False. Binding
additionally requires the private isolated-simulation capability held by that
module. The normal integration factory has no route to it.

Scene IDs supplied to this layer must be unique per capture. The synchronous
response is SHA-256-bound to a fresh 128-bit decision nonce, the exact scene ID,
target ID, feasible pose set, features, feasibility metrics, baseline scores,
and canonical request context. That context binds the profile identity,
simulation scope, object family, and, for screws, execution_supported=False.
Capture timestamp, point-cloud digest, and observation digest are not yet bound,
so unique-per-capture scene IDs remain a caller requirement. A future GraspGenX
source must additionally bind an application-supplied calibration ID/hash.

## Modes

shadow is the default. A valid model recommendation is recorded as
proposed_candidate_id, but the deterministic fallback remains the executed
candidate.

simulation_binding may select the recommended feasible candidate only when all
validation and gating checks pass. It is unavailable through the standard
factory and has no actuation method. The screw facade remains non-executable even
when a model recommendation is bound.

These outcomes always use the deterministic fallback and record why:

- provider exception or total request deadline;
- malformed JSON, duplicate keys, markdown, trailing data, or oversized reply;
- wrong schema, request hash, model/revision, frame, calibration, or scene;
- missing, duplicate, unknown, infeasible, or incompletely ranked candidate;
- boolean, numeric-string, NaN, infinity, or out-of-range score/confidence;
- selected candidate that is not the unique numeric top after normalization;
- abstention, tied top scores, low confidence, or insufficient score margin.

Array order alone is not trusted: policy v2 deterministically sorts valid
(candidate_id, score) pairs by descending score and then candidate ID before
top/tie/margin checks. The original response remains unchanged and hash-bound in
the receipt. A receipt-write failure raises from selection, rolls back to the
previous file size when possible, and permanently poisons that sink, so no plan
is returned without durable decision evidence.

## Cosmos3-Nano provider

Cosmos3OpenAIProvider is a lightweight client for a separately pinned
OpenAI-compatible service. It requests strict JSON Schema output and then
independently applies every semantic check above. Candidate JSON is explicitly
untrusted prompt data. Endpoint credentials and query strings are rejected,
optional bearer tokens are read from an environment variable, redirects are
disabled, response size is bounded, and one monotonic deadline spans URL open
and response-body reads. The remaining budget is applied to the urllib socket
before each read. The OpenAI envelope's actual model must match the requested
served-model alias.

Validated remote setup on isaac-launchable-83f295:

- model: nvidia/Cosmos3-Nano, 16B parameters, OpenMDW 1.1;
- Hugging Face snapshot revision:
  e59a53c25979a090fa8706c9acc0c254a6e89b92;
- local config.json SHA-256:
  c32f2468a54542c21946bc8eab6172b911dcec9a7193a94c023ea2d4073bcda6;
- vLLM 0.21.0;
- vllm-cosmos3 and transformers-cosmos3 0.1.0, built from NVIDIA
  cosmos-framework commit
  55f0554d0bf167603432041c794a8babbac759b9;
- BF16 reasoner service with stable alias nvidia/Cosmos3-Nano.

Cosmos is a general physical-AI reasoner, not a calibrated grasp-success model.
Its ranking remains advisory and simulation-only. Model-card warnings about
spatial mistakes and hallucinations are why deterministic checks and fallback
remain authoritative.

## GraspGenX provider

GraspGenXCatalogProvider adapts 6-DoF proposals from NVIDIA GraspGenX without
executing them directly. The external inference envelope must bind the exact
request hash, scene, input frame, camera-to-base calibration, model version, and
gripper descriptor fingertip depth.
The repository currently stops at this validated catalog matcher. It does not
yet contain the capture resolver or ZMQ/msgpack proposal source that calls the
upstream service and constructs team8-graspgenx-proposals-v1. The live service
evidence below is therefore service-only, not an end-to-end planner result.


The released model returns a gripper-base transform in the segmented point
cloud frame. Rotation column +Z is the approach direction toward the fingertip;
+X is the parallel-jaw closing direction; +Y completes a right-handed frame.
The adapter validates a finite homogeneous, orthonormal, right-handed SE(3),
transforms it into robot_base, shifts the gripper base to the fingertip using
the pinned descriptor depth, rejects excessive approach tilt, and matches
translation plus closing-axis yaw to an existing feasible candidate. Ambiguous
or unmatched proposals abstain. Parallel-jaw yaw is compared modulo 180
degrees.

The model does not emit a per-grasp jaw width. Sweep-volume extents condition
the network but are not runtime finger commands. Catalog opening and finger
commands remain authoritative.

Validated remote setup on ai-grasp-dev-1101:

- source: NVlabs/GraspGenX at
  b9429097728cb1c430dd78b92edf17ba318aad03, Apache-2.0;
- checkpoint repository commit
  7c834043c11a11417e31d6d5ea9355801e40a2c1, NVIDIA Open Model License;
- gripper-description repository commit
  19a03c00d19aeaf052d0f6801f0041982d676e8a;
- generator SHA-256:
  8b55f31cdb8340a573b4df27b027c15cff326bd6debcb389bf631d2aaab7ac44;
- discriminator SHA-256:
  cbf3f3bdb2e4c03fca8486ed24de0e6a8a859e6bd22bce2f1434a610335abd3e;
- Robotiq 2F-140 config.json SHA-256:
  357cc0b70064c24cf0f3aa3325a7b0839823f8cc1e3a30bfe0cea0f9d2c41127;
- Robotiq 2F-140 fingertip depth 0.195 m;
- direct segmented-point-cloud inference produced finite 4x4 poses and scores;
- loopback infer_object returned 276 valid proposals: 60 diffusion and 216 OBB,
  with scores from 0.002079 to 0.975086;
- upstream non-end-to-end suite: 99 passed, 22 skipped, 3 deselected.

GraspGenX discriminator scores are ranking signals, not calibrated complete
grasp-success probabilities. The upstream service does not perform the project's
scene-collision and catalog checks. Deterministic frame, reach, opening, approach,
obstacle, and catalog checks remain required.

## Durable receipt

Each JSONL record includes:

- policy, request, response, prompt, provider, and model versions;
- request and response SHA-256;
- mode, status, recommendation, executed ID, fallback ID, and reason;
- confidence, computed margin, gate thresholds, timeout, and latency;
- full canonical request and response payloads;
- integration profile ID, path, and SHA-256;
- UTC time, sequence number, previous-record hash, and record hash.

The receipt path must not already exist. Records are append-only for the
session, opened without following symlinks when the platform supports it, and
fsynced before selection returns. Initial file creation also fsyncs the parent
directory. A failed append attempts to truncate back to the original byte count
and permanently poisons the sink so it cannot be reused. A new run must use a
new external path; receipts, weights, datasets, and run artifacts do not belong
in Git.

## Remote evidence

On the L40S primary instance, the restarted Cosmos reasoner used
VLLM_USE_FLASHINFER_SAMPLER=0 because the FlashInfer JIT sampler required nvcc,
which is not installed in the runtime container. With the native sampler:

- health and model-list endpoints returned HTTP 200;
- plain text inference returned the exact requested token sequence in 0.650 s;
- strict JSON Schema inference returned a complete two-candidate ranking in
  2.024 s;
- both responses had finish_reason stop and no reasoning field content;
- resident GPU memory was 39,647 MiB of 46,068 MiB.
A live guarded planner smoke then exercised the public receipt-writing facade
against that service:

- policy team8-pretrained-ai-selection-policy-v2 and prompt
  team8-cosmos3-feasible-grasp-ranker-v1;
- one-feasible-candidate block request bound with confidence 0.9 and margin 1.0;
- request SHA-256
  6eeaae9ed4ee90c25dd357efc8ac79a45ac3287fc35cdca26a34793de471820a;
- response SHA-256
  a6707f70ebcb246692935a7985e9778c8bc0caae88c15672493102c9bda6601e;
- receipt SHA-256
  a8d7466373a656e3f9bb0d179061a8e490db9059749e43a7f3635bbbd9a6b124;
- a separate three-candidate reply contradicted its numeric scores and selected
  ID, so policy v2 returned the deterministic fallback with
  invalid_response:selected_not_top_ranked.

The bound single-candidate smoke proves the guarded provider-to-receipt path, not
multi-candidate grasp quality. The contradictory response proves the semantic
fallback remains active.

Together these checks prove transport, structured output, guarded planner
integration, and fallback behavior. They do not prove grasp accuracy or safety.
