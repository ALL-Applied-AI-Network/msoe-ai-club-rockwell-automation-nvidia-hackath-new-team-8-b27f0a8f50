# Real-camera evidence for synthetic appearance

The downloaded hardware archives are useful evidence, not ready-made training
data. They contain synchronized 640x360 table, side, and wrist video plus robot
state, but they also contain setup activity and occasional hands, laptops, and
phones. They do not contain object masks, object poses, camera exposure values,
or grasp-success labels.

`config/hardware_frame_curation.json` is therefore **default-deny**. Only frames
inside its explicit `appearance_fit` split may be used to fit simulator
appearance, and any overlap with `excluded_human_or_setup` fails validation.
The review sampled every 20 seconds, so run a finer privacy review before model
training or publication. Do not use a real frame as an Isaac background: it can
contain unlabeled cubes, robot parts, cables, or people that conflict with the
simulated geometry.

`config/real_camera_profiles.json` records archive hashes, the fit/held-out
split, native camera resolutions, broad compressed-video RGB summaries, and the
observed 640x360/10 Hz/YUV420 deployment format. Its provisional targets sample
750 frames per camera from the retained ranges; a separate 339-frame episode
remains held out. Those samples are not certified human-free because the
contact-sheet review was only every 20 seconds; fine-scrub every image before
model training or publication. The table/side neutral medians reproduce within
roughly 1--2 digital levels on the held-out episode. These are aggregate
display-RGB targets, not a photometric calibration or proof that every retained
frame is human-free. The 10 mm label in the archive metadata also does not
resolve the project's 10 mm versus 16 mm physical-cube question.

## Deployment-like previews

Always retain native lossless Isaac RGB first. A separate preview can then be
created without changing the raw image:

```python
from sorting_core import (
    create_deployment_preview,
    load_camera_appearance_profiles,
)

profiles = load_camera_appearance_profiles()
preview = create_deployment_preview(
    native_rgb,
    profiles,
    camera_id="wrist",
    episode_seed=20261029,
    frame_index=42,
)
```

The preview is resized to 640x360, its neutral median is moved toward the
camera-specific curated target with bounded and recorded RGB gains, and it is
round-tripped through planar I420 to approximate the principal chroma loss in
the hardware export. It does not claim AV1 bitstream equivalence or physical
camera-response calibration. Wrist previews can additionally receive a seeded
blue-hose occluder when `add_wrist_occlusion=True`; this stress augmentation is
off by default because the twin already renders fixed hose geometry. The
returned `occluder_mask` covers the synthetic cable-and-shadow geometry before
I420 conversion; chroma spill may extend beyond it. Callers must update
visibility labels conservatively or store the preview as unlabeled robustness imagery. Table
and side views never receive that synthetic hose.

The simulator already contains geometric blue/orange/black cables. Image-space
augmentation represents the observed variability that fixed curve geometry
cannot reproduce; it is not flexible-body physics and must remain opt-in.
