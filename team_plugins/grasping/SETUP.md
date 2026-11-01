# Grasping team setup

This is the first file a grasping teammate or coding agent should read. The
detailed contracts and validation rules are in this directory's `AGENTS.md`.
If a term, score, geometry rule, or command is unclear, ask the AI to explain it
before continuing; questions are part of the workflow.

## Assignment

Build a grasp plugin that:

1. consumes the shared camera-derived scene;
2. generates multiple candidate grasps for the selected cube or screw;
3. deterministically rejects grasps that violate known fit, reach, clearance,
   collision, scene-age or object-identity constraints;
4. predicts complete-grasp success probability for each remaining candidate;
5. selects the candidate with the highest predicted success probability.

For this project, success means acquiring the intended object, retaining it
through the entire lift and transport, and releasing it only at the intended
destination. Touching, closing, or briefly lifting is not success.

Do **not** train an execution-time model or balance predicted time against
success. Actual planning/execution/recovery time should still be logged for
diagnostics and judging, but it does not affect candidate ranking.

## Input and output

Do not accept an informal string such as “red cube at x,y.” The shared input is
the typed `SceneObservation` or its schema-validated JSON serialization. The
current compatibility layer then provides a scene-bound target and planning
context to `GraspPlanningPipeline`.

Output is an auditable `GraspPlanningDecision`: selected executable plan,
selected candidate, all accepted candidate scores/features, and every rejected
candidate with reasons and metrics. A model score may rank feasible grasps; it
may never make an impossible or rejected grasp executable.

## Work boundary and branch

Your normal edit boundary is `team_plugins/grasping/**`, and your branch is
`grasping`. Do not edit perception, hardware, Isaac, shared contracts, or the
vision team's directory to make a test pass. If the contract or executor lacks
a required pose/field, create a small failing fixture and ask the lead for a
reviewed interface change.

For a fresh clone:

```bash
git fetch origin
git switch --track origin/grasping
```

At the start of a session, before an expensive run, and before handoff:

```bash
git switch grasping
git status --short
# Continue only when the status is empty.
git fetch origin
git merge --ff-only origin/grasping
git merge --no-edit origin/main
```

If the status is not empty or Git reports a conflict, stop and ask the AI/lead.
Do not reset, discard, or stash somebody else's work. After tests pass, push
only this branch:

```bash
git push origin grasping
```

## Environment setup

Most grasp candidate, feature and scorer work is CPU-testable in WSL Ubuntu:

```bash
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_integration_plugins.py' -v
uv run --project hardware/fanuc_cell \
  python -m unittest discover -s tests -p 'test_grasping_integration.py' -v
```

Use small serialized `SceneObservation` fixtures so the grasp tests do not need
the vision model, Isaac or the physical robot. Use Brev only for a test that
requires Isaac physics/rendering, following root `BREV_SETUP.md`. Keep weights,
datasets and generated runs outside Git.

Before handoff, run the one-command checker described in
[`docs/TEAM_PLUGIN_CHECKER.md`](../../docs/TEAM_PLUGIN_CHECKER.md) with your
actual versioned grasp profile, not the example:

```bash
uv run --project hardware/fanuc_cell python scripts/check_team_plugin.py \
  --profile team_plugins/grasping/my_profile.v1.json
```

The handoff run requires a clean tree. `--allow-dirty` is only for an
in-progress development check.

## Isolated pretrained services

Pretrained selection is optional, simulation-only, and separate from the normal
plugin environment. Do not install either model into the repository runtime,
commit weights, or expose the services to an unauthenticated network.

For Cosmos3-Nano, use the model snapshot and a dedicated Python 3.13 vLLM
environment under the external workspace. The validated runtime is vLLM 0.21.0
plus NVIDIA cosmos-framework commit
55f0554d0bf167603432041c794a8babbac759b9:

    /workspace/envs/uv-bootstrap/bin/uv venv \
      --python 3.13 --seed --managed-python \
      /workspace/envs/cosmos3-vllm

    /workspace/envs/uv-bootstrap/bin/uv pip install \
      --python /workspace/envs/cosmos3-vllm/bin/python \
      --torch-backend=cu130 \
      vllm==0.21.0 \
      'vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git@55f0554d0bf167603432041c794a8babbac759b9#subdirectory=packages/vllm-cosmos3' \
      'transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git@55f0554d0bf167603432041c794a8babbac759b9#subdirectory=packages/transformers-cosmos3' \
      openai

Download the exact model snapshot and verify its configuration before serving:

    /workspace/envs/uv-bootstrap/bin/uvx --from huggingface-hub hf download \
      nvidia/Cosmos3-Nano \
      --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 \
      --local-dir /workspace/models/Cosmos3-Nano

    printf '%s  %s\n' \
      c32f2468a54542c21946bc8eab6172b911dcec9a7193a94c023ea2d4073bcda6 \
      /workspace/models/Cosmos3-Nano/config.json | sha256sum -c -


Start the reasoner inside the GPU container with a stable served-model alias.
The native sampler flag is required on the validated container because it has
no nvcc for FlashInfer JIT:

    CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    /workspace/envs/cosmos3-vllm/bin/vllm serve \
      /workspace/models/Cosmos3-Nano \
      --served-model-name nvidia/Cosmos3-Nano \
      --hf-overrides '{"architectures":["Cosmos3ReasonerForConditionalGeneration"]}' \
      --tensor-parallel-size 1 \
      --mm-encoder-tp-mode data \
      --async-scheduling \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.90 \
      --host 127.0.0.1 \
      --port 8001

Keep the endpoint loopback-only. Use Cosmos3OpenAIProvider with the complete
chat-completions URL, exact served alias, snapshot revision, total deadline,
and no API key for loopback. If authentication is added, supply only the
environment-variable name; query-string credentials and redirects are refused.
A minimal construction is shown below. factory_context must be the validated
simulation-only profile context, and receipt_path must be a new external file:

    from team_plugins.grasping.ai_providers.cosmos3 import Cosmos3OpenAIProvider
    from team_plugins.grasping.ai_selected_sim_runner import (
        build_ai_selected_simulation_planner,
    )

    provider = Cosmos3OpenAIProvider(
        endpoint="http://127.0.0.1:8001/v1/chat/completions",
        model="nvidia/Cosmos3-Nano",
        model_revision="e59a53c25979a090fa8706c9acc0c254a6e89b92",
    )
    planner = build_ai_selected_simulation_planner(
        factory_context,
        provider,
        receipt_path=receipt_path,
        mode="shadow",
        timeout_s=30.0,
    )
    decision = planner.decide_for_target(scene, target, planning_context)

Use build_ai_selected_offline_screw_planner with the same provider for supported
M6x30 screw candidates; that facade always returns execution_supported=False.


For GraspGenX, use a separate L40S instance and the official repository:

    git clone https://github.com/NVlabs/GraspGenX.git /home/ubuntu/GraspGenX
    cd /home/ubuntu/GraspGenX
    git checkout --detach b9429097728cb1c430dd78b92edf17ba318aad03
    git lfs install
    git lfs pull
    ~/.local/bin/uv sync
    git -C ext/graspgenx_checkpoints checkout --detach \
      7c834043c11a11417e31d6d5ea9355801e40a2c1
    git -C ext/graspgenx_checkpoints lfs pull
    git -C ext/gripper_descriptions checkout --detach \
      19a03c00d19aeaf052d0f6801f0041982d676e8a
    ~/.local/bin/uv sync --extra serve

    printf '%s  %s\n' \
      8b55f31cdb8340a573b4df27b027c15cff326bd6debcb389bf631d2aaab7ac44 \
      ext/graspgenx_checkpoints/release/gen/epoch_736.pth \
      cbf3f3bdb2e4c03fca8486ed24de0e6a8a859e6bd22bce2f1434a610335abd3e \
      ext/graspgenx_checkpoints/release/dis/epoch_1056.pth \
      357cc0b70064c24cf0f3aa3325a7b0839823f8cc1e3a30bfe0cea0f9d2c41127 \
      ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/robotiq_2f_140/config.json \
      | sha256sum -c -

Start its ZMQ service on loopback:

    ~/.local/bin/uv run python client-server/graspgenx_server.py \
      --config ext/graspgenx_checkpoints/release \
      --assets_dir assets \
      --default_gripper robotiq_2f_140 \
      --host 127.0.0.1 \
      --port 5556

The lightweight wire protocol is one-frame msgpack over ZMQ REQ/REP with
msgpack-numpy arrays. It has no authentication or encryption. Keep it on
loopback or an authenticated tunnel. The current repository does not yet
implement the capture resolver/ZMQ proposal source that converts infer_object
responses into team8-graspgenx-proposals-v1; do not present the service-only
smoke as an end-to-end planner result. Verify the source, checkpoint, and gripper
pins and hashes listed in AI_SELECTION.md. A future source must also bind the
capture digest, capture time, input frame, and an application-supplied
calibration ID/hash before accepting proposals.

## Cube and screw scope

Start with the existing cube path to prove plugin wiring. Then add the supplied
ISO 10642 M6×30 screw as a separate object family. The simulation reference is
30 mm long with a 12 mm countersunk head, 6 mm shaft, and a configured shaft
grasp point 16 mm from the head. The shaft axis and head/tip polarity must be
represented correctly; do not treat a screw as a cube.

The separate screw demo proves only a configured-pose Isaac physics sequence.
It does not prove camera-derived screw targeting, full collision-free planning,
physical gripper fit, retention, or the real screw's friction. The current
shared executor is still primarily top-down XYZ plus yaw. A required arbitrary
6-DoF/side grasp is a lead-owned interface change, not something to hide inside
a score.

## Recommended implementation order

1. **Reproduce the baseline.** Load a saved cube scene and show the legacy
   top-down candidate/plan remains identical with the plugin profile unset.
2. **Generate alternatives geometrically.** Add a small explicit set of cube
   yaw/approach candidates; for screws, add shaft-region candidates consistent
   with the catalog and supported executor.
3. **Reject known failures first.** Check gripper/finger fit, approach corridor,
   neighboring objects, reach, held-object lift/transport/placement clearance,
   uncertainty and scene freshness before scoring.
4. **Freeze one feature schema and success label.** Use only information
   available before action and the same extractor for data collection/runtime.
5. **Start with a simple scorer.** Empirical success rate, logistic regression,
   or a small random forest is easier to calibrate and debug than a deep model.
6. **Select highest success.** Use `HighestSuccessSelector` with deterministic
   tie-breaking. Leave predicted time fields `None`.
7. **Evaluate on matched scenes.** Compare the geometric baseline and
   highest-success selector using the same feasible candidates and recovery
   rules. Split related layouts/sequences together.
8. **Hand off reproducible fixtures.** The lead must be able to reproduce a
   selected and rejected decision without the training dataset.

## Manageable ideas worth testing

- For a cube, compare a few jaw/yaw alignments rather than sampling hundreds of
  arbitrary poses. The useful question is whether measured failure rates differ.
- For a screw, compare several grasp positions along the exposed 6 mm shaft and
  reject the head/tip or obstructed regions geometrically.
- Add perception-quality, clearance and required-opening features before using
  image embeddings. Simple physical features may explain most failures.
- Calibrate probabilities on held-out layouts. If a 0.8 score succeeds only
  half the time, ranking confidence is misleading even when accuracy looks good.
- If one feasible grasp dominates every test, keep the simple rule and report
  that finding instead of forcing an AI model into the demo.

Avoid reinforcement learning, learned world models, end-to-end robot control,
or unconstrained grasp synthesis until the small candidate set and labels are
trustworthy.

## Required deliverables

- pipeline factory and independent profile in this directory;
- versioned candidate generator, feasibility checker, feature schema, scorer
  and highest-success selector IDs;
- cube and screw geometry/gripper assumptions with CAD/model versions;
- complete-success label definition and reproducible collection procedure;
- saved accepted/rejected candidate fixtures with reasons/metrics;
- immutable model artifact hash and safe retrieval/loading notes;
- probability calibration, candidate-ranking and complete-retention results;
- matched geometric-baseline versus highest-success results;
- failure cases: no candidate, blocked approach, stale/moved target, slip/drop,
  wrong object/destination, collision/contact and unresolved episode;
- no committed weights, datasets, runs, secrets or personal paths.

## Done criteria for team handoff

The grasp component is ready for lead integration only when:

1. its profile constructs `GraspPlanningPipeline` without shared-file edits;
2. deterministic rejection always occurs before learned scoring;
3. no stale, wrong-target, infeasible or zero-probability candidate executes;
4. highest-success selection and ties are deterministic and auditable;
5. claimed cube/screw performance measures complete retained placement;
6. collection and runtime feature extraction are identical and versioned;
7. the profile-unset baseline and shared tests still pass;
8. the full handoff checklist in `AGENTS.md` is complete.

If progress stalls, show the AI one scene, the generated/rejected candidates,
their features and the observed outcome. Ask for the smallest experiment that
would determine whether the problem is geometry, perception, labels or scoring.
