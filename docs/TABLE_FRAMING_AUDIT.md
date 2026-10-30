# Table-camera framing audit

The checked-in camera calibration reproduces its own USD transform, but a
paired physical frame and seed-7 simulation still disagree on the visible
table silhouette. In the 640x360 comparison, the simulated rear table edge is
about 26 pixels lower. Its left rear corner is about 45 pixels farther right,
and the simulated rear edge is about 43 pixels shorter.

This is too large to explain with the training-data camera jitter
(±1 mm, ±0.15 degrees and ±0.3% focal scale). It does **not**, by itself,
identify the cause. Plausible causes include table extents/origin, archive
resize or crop, stale camera calibration, or physical mount movement.

Run the host-only audit from the repository root:

```powershell
python scripts/audit_table_framing.py `
  --workspace-root .. `
  --require-source-files `
  --output ../HardwareResults/SimRealism/table_framing_audit.json
```

The external PNGs are read and SHA-256 verified; they are never modified or
committed. Landmark definitions, source hashes, uncertainty, and diagnostic
thresholds live in `config/table_framing_landmarks.json`.

Do not move the simulated camera merely to make these two corners agree. First
project all four surveyed table corners using the native physical capture,
verify whether the archive was cropped, and check the current camera mount.
Only then change the specific geometry or calibration source shown to be wrong.
The audit deliberately reports `automatic_correction_permitted: false`.

The unresolved 10 mm versus 16 mm cube-size question is separate from this
table-edge discrepancy.
