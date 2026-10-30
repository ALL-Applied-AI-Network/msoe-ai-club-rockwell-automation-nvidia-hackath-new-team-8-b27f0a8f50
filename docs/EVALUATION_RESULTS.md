# Shared evaluation results

Use `docs/schemas/evaluation-result.v1.schema.json` for machine-readable vision,
grasping, and complete-relocation results. The goal is to let both teams and
the lead compare evidence without copying numbers by hand or silently treating
an unmeasured value as zero.

The format is an **evaluation report**, not a runtime robot command and not a
replacement for raw evidence. Keep private hardware recordings, request
credentials, generated datasets, and model weights outside Git. A checked-in
report may contain public aggregate evidence and opaque run IDs, but never a
private results URL, `.request.json` value, token, or personal absolute path.

## What v1 records

- immutable project, profile, component, calibration, catalog, and source-run
  provenance;
- the evaluation split and backend (`offline_replay`, `isaac`, or `hardware`);
- vision counts by object class, raw pose-error samples, and measured inference
  latency;
- every generated grasp candidate, deterministic accept/reject result and
  rejection reason;
- selected candidate probability, whether its final outcome was actually
  observed, and optional ranking correctness tied to explicit comparative
  evidence;
- one primary episode outcome plus contact, intervention, retry, verification,
  and measured timing evidence.

`complete_relocation_success` means the intended object was acquired, retained
through transport, placed correctly, and released when commanded. Camera-only
verification must stay labeled `camera`; it is not force/contact evidence.

## Required definitions and invariants

- `true_positives + dropouts == reference_instances`. A dropout is an
  evaluation-eligible reference object that the provider did not return.
- Pose error is recorded only when evaluator truth exists. Orientation error
  must already account for the catalog symmetry named by `symmetry_id`.
- `inference_latency_ms` contains at most one sample per evaluated frame. Zero
  evaluated frames require zero vision counts, pose samples, and latencies.
- Accepted and rejected candidates exactly partition
  `generated_candidate_ids`. A learned score cannot move a deterministically
  rejected candidate into the accepted set.
- `selected_candidate_outcome_observed` is true only when the selected
  candidate's complete grasp outcome was observed. It cannot be true without a
  selected candidate, or for `cancelled`, `unresolved`, or `not_executed`.
  When true, `selected_candidate_complete_grasp_success` is a required boolean;
  when false, that label must be null.
- `selected_candidate_complete_grasp_success` means the intended object was
  acquired and retained until the commanded release. It is deliberately
  separate from the episode's task outcome. For example, a
  `wrong_destination` episode may still be a successful grasp if the executor
  carried the intended object without dropping or slipping and released it on
  command. Do not turn every task/executor error into a negative grasp label.
- Non-null `ranking_correct` requires at least two accepted candidates and a
  non-null `ranking_evidence_id` identifying the matched/counterfactual
  evidence. Otherwise both ranking fields are null.
- A probability may be `null` for a geometric baseline. Never invent a model
  probability to make calibration charts nonempty.
- Missing timing fields mean **not measured**. They remain absent and the
  summarizer reports `n=0`/`n/a`; missing is never converted to zero.
- `not_executed` episodes are excluded from the complete-relocation success-rate
  denominator. Cancellation and unresolved outcomes remain in that denominator.
- `contact_count` counts unintended contacts, not the intended finger/object
  contact required to grasp. It and `operator_intervention_count` may coexist
  with any primary outcome and therefore are recorded separately.
- Every terminal executed outcome (`complete_relocation_success`,
  `failed_lift`, `drop_or_slip`, or `wrong_destination`) requires non-null
  target track/class IDs, a selected accepted grasp candidate,
  `selected_candidate_outcome_observed=true`, and at least one verification
  method. This prevents a failed episode with no executed candidate from being
  counted as scorer evidence. Complete relocation additionally requires the
  explicit grasp-success label to be true; `failed_lift` and `drop_or_slip`
  require it to be false. `wrong_destination` may use either label because a
  sound grasp can still be followed by an executor/destination error.
  Cancelled, unresolved, and unexecuted episodes remain unobserved for scorer
  calibration.
- Every `evaluation_id` and `episode_id` must be unique in an aggregate, and
  source-run IDs must not overlap between reports. This prevents accidental
  double counting.
- JSON object keys must be unique; the loader rejects duplicate keys instead of
  silently accepting the last value. It also rejects non-standard numeric
  constants such as `NaN` and `Infinity`.
- `source_run_ids` contains at least one opaque traceable run identifier.
  `calibration_ids` or `components` may be empty only when none applied to that
  evaluation; leave them empty rather than inventing identifiers, and explain
  the condition alongside retained raw evidence.

Each episode also has a required `condition_id`. Reuse the same `condition_id`
only for policy trials that intentionally replay the same layout, arrival
sequence, physics seed, or physical test condition. Give each trial its own
unique `episode_id`. The summarizer exposes condition IDs for matching, but it
does not claim a paired-policy result by itself.

## Validate and summarize

From the repository root:

```bash
uv run --project hardware/fanuc_cell \
  python scripts/summarize_evaluation.py \
  tests/fixtures/evaluation/cube_evaluation.v1.json
```

The default output is readable text. Also save the aggregate JSON used by a
table, plot, or presentation:

```bash
uv run --project hardware/fanuc_cell \
  python scripts/summarize_evaluation.py \
  path/to/result-a.json path/to/result-b.json \
  --json-output path/to/summary.json
```

Use `--json` to print JSON instead of text. The script has no third-party
runtime dependency. It validates the important cross-field invariants in
addition to the formal JSON Schema.

By default, multiple reports must have exactly the same project commit, split,
backend, integration profile, object catalog, calibration IDs, and component
versions. This prevents a plausible-looking rate from silently combining
incomparable runs. For exploratory, descriptive aggregation only, opt in with:

```bash
uv run --project hardware/fanuc_cell \
  python scripts/summarize_evaluation.py \
  tests/fixtures/evaluation/cube_evaluation.v1.json \
  tests/fixtures/evaluation/screw_observation_evaluation.v1.json \
  --allow-mixed
```

Both text and JSON output then carry a prominent mixed-configuration warning.
Do not present that aggregate as a matched policy comparison. The JSON summary
also lists all source splits in `provenance.splits`. `--json-output` must name a
different file from every input report; the CLI refuses to overwrite evidence.

## Aggregate metric behavior

Counts are summed. Recall and dropout rate are recomputed from summed counts.
Pose, latency, and timing means/p95/max are computed from supplied raw samples;
p95 uses the nearest-rank definition. The selected-candidate Brier score is
computed only when `selected_candidate_outcome_observed` is true, the explicit
`selected_candidate_complete_grasp_success` label is present, and the selected
candidate has a probability. It never derives the grasp label from the task
outcome status. Cancelled, unresolved, and unexecuted candidates are therefore
not silently labeled as failures. Five fixed probability bins provide a small
calibration view. Ranking accuracy includes only non-null `ranking_correct`
records backed by `ranking_evidence_id`.

These aggregates do not prove statistical significance. Keep related layouts
and arrival sequences in one split, compare policies on matched conditions, and
report the sample count beside every metric. The checked-in files under
`tests/fixtures/evaluation/` are synthetic contract examples, not project
performance claims.

## Adding a new result

1. Copy the closest fixture and give the evaluation a unique ID.
2. Replace the all-zero fixture Git SHA with the exact 40-character commit.
3. Record the active branch/profile/component and calibration versions.
4. Add only measurements that were actually collected.
5. Validate with the summarizer and retain the raw evidence outside Git.
6. Have the lead review the split, outcome definition, and claimed comparison
   before using the summary in the final presentation.

If a field or metric is unclear, ask the project AI to explain what observation
would establish it before filling it in.
