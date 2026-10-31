# Matched Isaac collection procedure

1. Work from a clean committed `grasping` branch and run the full Brev setup
   for that exact commit. The preparer refuses dirty relevant source paths and
   records the commit plus deterministic hashes of copied source trees.
2. Create fresh external cases:

   `python -m team_plugins.grasping.prepare_screw_trials /workspace/grasp-experiments/smoke-v1`

3. For each case, use a new run directory and its case root as
   `ROCKWELL_REPO_ROOT`. Run the mixed demo with `--physics-only` for bulk
   data. The preparer preserves the validated five-object order and adjusts
   placement by the same axial delta, so 12/16/20 mm candidates share the same
   initial screw and final head pose without a later cube confounding the label.
4. A candidate label is positive only when its event has both
   `pickup_verified` and `placement_verified`, and the final scene report
   passes for that same item. An exception before an event exists is unresolved.
   The mixed smoke applies one global offset to both screws, so its result is a
   whole-run policy comparison, not an independent per-screw training label.
   Do not train the selector from these smoke labels until a target-isolated
   runner exists; retain the sequential-confound metadata meanwhile.
5. Preserve the pre-action feature row, `group_id`, `candidate_id`,
   candidate/rejection evidence, commit, case manifest, simulator/contact
   settings, and report path for each episode.
   Do not place generated cases, runs, videos, datasets, or models in Git.
6. Group related layouts into train or held-out test before training. Never
   randomly split candidates or adjacent frames from one rollout. The trainer
   rejects any `group_id` that crosses splits and reports group-level
   top-candidate success and ranking accuracy.

Initial smoke scope is one matched layout for all three screw offsets. The next
data phase varies object orientation, clear/head/middle/tip obstruction,
camera-pose error, and supported friction/contact settings. Cubes require a
grasping-owned experimental runner or lead-reviewed executor hook because the
existing camera demo currently ignores the selected plan's grasp pose.
