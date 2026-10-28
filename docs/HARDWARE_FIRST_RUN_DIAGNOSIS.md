# First hardware-run diagnosis and supervised retry

This note diagnoses the first recorded physical-cell session. It contains no
request token or private results link. The evidence is useful, but it does not
authorize unattended robot motion.

## What prevented motion

Two different gates stopped the arm:

1. The first client invocation was correctly in the checked-in **hold-only**
   mode. Both physical-motion interlocks were closed, so the policy repeated a
   fresh measured pose instead of running perception or planning.
2. Later motion-enabled attempts did run perception, but never produced a
   complete plan. Across 186 logged camera-scene events, the policy reported
   either only red (99 events) or no pickable colour (87 events). There was no
   `plan_selected` or `phase_action` event, so no motion command was generated.

The saved robot-state dataset independently shows an essentially stationary
TCP. This agrees with the policy logs: the controller did not ignore a valid
move; the policy did not send one.

## Camera finding

The recorded red, yellow, and blue cube pixels are already inside the
checked-in HSV ranges. Widening those ranges would add false positives from the
robot/cell without addressing the observed failure.

The stronger explanation is a cube-size profile mismatch. The original
hardware policy used the simulator's 10 mm known-size gate, whose accepted
upper edge is 14 mm. An offline replay of the saved table video measured median
silhouettes of approximately 16–17 mm.

Using every tenth recorded frame:

| Hardware vision profile | Frames with exactly one pickable red, yellow, and blue | Longest consecutive sampled run |
|---|---:|---:|
| 10 mm | 0 | 0 |
| 16 mm | 27 | 25 |

The generated reports are kept with the local hardware evidence as
`camera-audit-10mm.json` and `camera-audit-16mm.json`; they are intentionally
not committed as training data.

This is strong diagnostic evidence, not a physical measurement. The recording
is a resized/compressed 640×360 copy of a 1920×1080 observation. Measure the
placed cube or have the operator identify it before choosing 10 or 16 mm.

## Changes made

- Hardware perception now accepts an **explicit** 10 mm or 16 mm profile.
  Isaac and the shared simulation configuration remain 10 mm.
- Motion waits for three consistent complete camera observations by default.
- The phase watchdog now budgets calls from the measured travel distance. A
  one-row action limit no longer times out merely because a safe long approach
  needs more than 20 calls; the no-progress latch remains active.
- `virtual-table` destinations use the existing fixed red/yellow/blue XY
  coordinates but lower to the near-table set-down height. No physical bucket
  is required or detected.
- Virtual-table completion requires a fresh positive camera detection at the
  destination, rather than treating disappearance alone as success.
- Virtual-table outcome verification discards projected colour fragments
  outside the official TCP XY/radial workspace, then positively rechecks every
  completed colour at its destination. The recorded initial scene contained
  hundreds of such red/yellow cell-background fragments; without this paired
  filter/check, a successful red placement would have been reported as a
  failure. Bin mode retains outside-workspace evidence because it lacks the
  same positive destination view.
- `supervised_run.py` clears inherited arming variables and enables only the
  requested mode. This avoids persistent shell exports and mistyped multiline
  commands.
- `audit_table_capture.py` provides an offline, explainable 10/16 mm replay.

No HSV thresholds were changed, because the recorded cube colors already pass.

## Important mechanical uncertainty

The configured custom-finger hook opening is 15.99 mm. A nominal 16 mm cube is
therefore not nominally smaller than that gap. Rounded cube edges or the actual
hook profile might make the grasp possible, but software cannot establish that
from video. The policy permits a 16 mm **hover** inspection, but blocks 16 mm
descent unless the installed fingers and real cube fit are explicitly checked.

The code still does not prove full-arm, finger, hose, held-object, or swept-path
collision clearance. The first retry must therefore stop at hover.

## Supervised retry sequence

Run these from `hardware/fanuc_cell` during an approved slot with the assigned
operator. `supervised_run.py` delegates to the official client; it is not a
safety controller.

1. Inspect the effective settings without contacting the cell:

   ```bash
   uv run supervised_run.py hover --cube-mm 16 --dry-run
   ```

   Replace `16` with `10` if the physical cube is measured/confirmed as 10 mm.

2. Repeat the stationary connection diagnostic:

   ```bash
   uv run supervised_run.py hold
   ```

3. With the operator present and the table path clear, run **hover only**:

   ```bash
   uv run supervised_run.py hover --cube-mm 16
   ```

   This may move to the planned high approach but cannot descend. Stop if the
   detected coordinates, visible path, TCP, installed fingers, or clearance do
   not match expectations.

4. Do not use `sort` until the hover is correct, the cube size and mechanical
   grasp fit are confirmed, UTOOL1 is verified at the fingertips, and the
   operator approves the complete path. The intentionally verbose form is:

   ```bash
   uv run supervised_run.py sort \
     --destination virtual-table \
     --cube-mm 16 \
     --confirm-grasp-fit
   ```

The current policy expects exactly one red, yellow, and blue cube in the
configured pickup region for three stable observations. Cube positions may
vary inside that region, but their centres must remain at least 30 mm apart for
this hardware policy. Keep the three fixed virtual destination footprints
clear before starting; they are at base-frame XY red `(0.24, -0.19)`, yellow
`(0.24, 0.00)`, and blue `(0.24, 0.19)` metres.

Use `Ctrl+C`, not `Ctrl+Z`, to stop the client. `Ctrl+Z` merely suspends it and
can leave an old process or exported environment setting behind.

The launcher deliberately does not ask whether an operator is present: that is
already guaranteed by the event workflow and a command-line checkbox cannot
verify it. `--confirm-grasp-fit` remains only because it materially controls
whether the policy permits 16 mm contact descent.

For a localhost-only official protocol check, use:

```bash
uv run supervised_run.py hover --cube-mm 16 --mock --trials 1 --trial-s 2
```

The current approved request still describes 10 mm cubes and physical bins.
Its remaining trials do not match the no-bucket/16 mm setup unless the operator
and organizers explicitly agree to that changed setup. A fresh formal
submission also cannot be made from the checked-in YAML until its team/contact
TODOs and placement description are corrected.

## Offline camera audit

```bash
uv run audit_table_capture.py /path/to/table.mp4 \
  --cube-mm 16 \
  --every 10 \
  --json camera-audit-16mm.json
```

Prefer an original uncropped 1920×1080 PNG for pose validation. A result-video
replay can expose likely HSV/size failures, but cannot certify millimetre camera
accuracy, grasp fit, or safe motion.
