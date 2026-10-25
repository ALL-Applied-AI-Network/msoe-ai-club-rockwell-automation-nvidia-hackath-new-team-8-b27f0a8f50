# ManipBench reference assets

This directory preserves the exact files supplied by a project member on
2026-10-24. It is intentionally separate from `digital-twin-setup/`, whose
contents remain byte-for-byte identical to the supplied starter.

`scripts/setup_brev.sh` verifies `MANIFEST.json`, exports this directory from
the committed Git revision, and installs it at the path exported as
`ROCKWELL_REFERENCE_ASSETS`. The setup also copies the verified tabletop image
to the disposable runtime twin as `assets/mat_print.png`. The starter still has
`MAT_MARKINGS = False`, so merely installing the image does not enable the
photo-derived tabletop appearance.

The STEP, STL, and USDA files are reference/source geometry. Setup does not
silently replace the starter's validated geometry with them. Any conversion,
collision-mesh generation, scale change, or scene substitution must be reviewed
and tested separately in Isaac Sim.

## Inventory and provenance

All source URLs were supplied by the project member. SHA-256 hashes and byte
sizes are pinned in `MANIFEST.json`.

| Asset | Files | Source | Notes |
| --- | --- | --- | --- |
| Work surface | `table/manipbench-table-surface.png` | <https://hackathon.sidclab.org/files/manipbench-table-surface.png> | 3128 x 2260 RGB image. The pixels alone do not establish robot-frame alignment or physical scale. |
| SMC MHZL2-16D2 gripper | `models/SMC-MHZL2-16D2.step` | <https://hackathon.sidclab.org/files/SMC-MHZL2-16D2.step> | Vendor body model; STEP declares millimetres. The starter currently uses its supplied pre-tessellated gripper data. |
| ISO 10642 M6 x 30 screw | `models/ISO10642-M6x30-countersunk-screw.step`, `.stl` | <https://hackathon.sidclab.org/files/ISO10642-M6x30-countersunk-screw.step>, <https://hackathon.sidclab.org/files/ISO10642-M6x30-countersunk-screw.stl> | 90-degree countersunk head and 4 mm hex socket; threads are not modelled. STEP declares millimetres; STL is documented by the source as millimetres. |
| 50 x 50 x 55 mm bin | `models/manipbench-bin-50x50x55.step`, `.stl` | <https://hackathon.sidclab.org/files/manipbench-bin-50x50x55.step>, <https://hackathon.sidclab.org/files/manipbench-bin-50x50x55.stl> | 35 x 35 mm pocket, approximately 25 mm deep. STEP declares millimetres; STL is documented by the source as millimetres. |
| 10 mm cube | `models/manipbench-cube-10mm.step`, `.stl` | <https://hackathon.sidclab.org/files/manipbench-cube-10mm.step>, <https://hackathon.sidclab.org/files/manipbench-cube-10mm.stl> | Bottom-face-centre origin. STEP declares millimetres; STL is documented by the source as millimetres. |
| 16 mm cube | `models/manipbench-cube-16mm.step`, `.stl` | <https://hackathon.sidclab.org/files/manipbench-cube-16mm.step>, <https://hackathon.sidclab.org/files/manipbench-cube-16mm.stl> | Bottom-face-centre origin and 1 mm rounded edges. STEP declares millimetres; STL is documented by the source as millimetres. |
| Vial | `models/csi-vial.usda`, `.stl` | <https://hackathon.sidclab.org/files/csi-vial.usda>, <https://hackathon.sidclab.org/files/csi-vial.stl> | Source dimensions: diameter 27.18 x 203 mm. USDA is Z-up with `metersPerUnit = 0.0254`; STL is documented as millimetres. The USDA is byte-identical to the starter's `assets/usd_assets/vial.usda`. Setup does not change the starter's smaller configured dimensions. |
| Vial cap | `models/csi-vial-cap.usda`, `.stl` | <https://hackathon.sidclab.org/files/csi-vial-cap.usda>, <https://hackathon.sidclab.org/files/csi-vial-cap.stl> | Source dimensions: 31.75 mm flange and 15.7 mm height. USDA is Z-up with `metersPerUnit = 0.0254`; STL is documented as millimetres. The USDA is byte-identical to the starter's `assets/usd_assets/vial_cap.usda`. |

## Licensing and redistribution

The project owner confirmed on 2026-10-24 that redistribution of these supplied
files in this repository is approved. That approval does not transfer ownership:
retain the source attribution above and do not claim these files as team-created.

## Integrity rule

Do not edit these binary/source assets in place. When intentionally replacing
an asset, record its provenance and update its byte size and SHA-256 in
`MANIFEST.json`; setup must fail when a committed file does not match that
manifest.
