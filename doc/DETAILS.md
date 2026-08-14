# pose-dataset-studio — full documentation

> This is the detailed reference behind the top-level [README](../README.md): per-script CLI
> options, every change applied against the original ODT code (with measurements), known
> limitations, and the registration troubleshooting guide. Section numbers (§1–§6) are what
> the code comments refer to.

Customized [ObjectDatasetTools (ODT)](https://github.com/F2Wang/ObjectDatasetTools) pipeline used to build the **custom RGB-D training datasets** for:

> Oh, I., Jang, G., Song, J., Son, M., Kim, D., Yun, J., Ko, K., *"A mixed reality-based remote collaboration framework using improved pose estimation"*, **Computers in Industry** 174 (2026) 104414.

It records RGB-D sequences with an **Azure Kinect**, computes ground-truth 6-DoF poses from **ArUco markers**, reconstructs the scene, and generates the labels/masks/meshes required to train the improved singleshotpose network in [`../../structured_sspose`](../../structured_sspose).

> **Where is RoI-PCA?** The paper's RoI-PCA color-space augmentation is applied **online during network training** (see `structured_sspose/image.py` / `dataset.py`), *not* as an offline dataset step. The PCA analysis / visualization / figure-generation tools that were used to develop and evaluate the method — like the offline occlusion/gamma/rotation augmentation script — are **not part of this release** (see §2, "Not shipped in this release").

---

## 1. Core pipeline

The numbered scripts **1–6** are the complete, self-sufficient path from camera to a singleshotpose-ready RGB-D dataset (images + depth + masks + label files + train/test lists).

```
(0) print & place ArUco markers (DICT_6X6_250, IDs 1-13) around the object
 │
 ▼
1_record_azurekinect.py        RGB-D capture (Azure Kinect, 720p)
 │   → custom/<obj>/JPEGImages/*.jpg, depth/*.png (16-bit), intrinsics.json
 ▼
2_compute_gt_poses.py          ArUco + ICP pose graph (Open3D global optimization)
 │   → custom/<obj>/transforms.npy, log.txt
 ▼
3_register_scene.py            merge all frames into one registered point cloud
 │   → custom/<obj>/registeredScene.ply
 ▼
3a_segment_object_masks.py     name the object once (text / click / browser picker);
 │                             SAM 2 propagates the mask over the whole sequence
 │   → custom/<obj>/objmask/*.png, objmask/seg_meta.json, seg_preview/*.jpg
 ▼
3b_segment_object_cloud.py     mask-guided 3D crop: every frame votes on every point
 │   → custom/<obj>/object.ply  (ASCII PLY)
 │
 │   (fallback, and what this replaces: segment registeredScene.ply by hand in
 │    CloudCompare/MeshLab and save it as object.ply — see §7)
 ▼
4_create_label_files.py        ball-pivoting mesh + OBB + 2D projection
 │   → labels/*.txt, mask/*.png, transforms/*.npy,
 │     <obj>.ply (object-frame ASCII mesh, centroid on the origin — sspose loads this),
 │     mesh_world.ply (the same mesh in camera-0 coords; overlay it on
 │                     registeredScene.ply to check the reconstruction)
 ▼
5_create_config2split.py       train/test split + object diameter + the cfg
 │   → train.txt, test.txt, training_range.txt, <obj>.data
 ▼
6_inspect_labels.py            visual QA: mask + 3D bounding box overlay
```

**Every step writes only inside the sequence folder.** When step 5 finishes, `custom/<obj>/` is a self-contained deliverable — frames, depth, labels, masks, the object mesh, the splits and the `.data` cfg — which you then move into singleshotpose's data root as one unit:

```bash
# after step 5
copy custom/<obj>  ->  <singleshotpose>/custom/<obj>
cd <singleshotpose>
python 1_train_baseline.py --datacfg custom/<obj>/<obj>.data --modelcfg cfg/yolo-pose.cfg
```

The paths written inside the cfg use the `--sspose-root` prefix (default `custom/`), i.e. they describe where the folder *will* sit after the move, so they are not expected to resolve while it is still here. Verified by simulating the move: all four cfg paths, all 529 image entries and `MeshPly` resolve from the new location.

Step 5 needs nothing looked up by hand: `diam` is measured from the mesh step 4 wrote, `fx/fy/u0/v0` come from `intrinsics.json`, `width/height` from an actual JPEG, and `name`/`backup`/`train`/`valid`/`tr_range`/`mesh` are derived from the dataset name.

### Command-line interface (all numbered scripts)

Every step shares the same base interface — **no more editing hard-coded folder names**:

```bash
python <step>.py <dataset>           # a folder next to the scripts: ./<dataset>/
python <step>.py <path/to/sequence>  # direct path (relative or absolute):
                                     # the folder is used exactly where it is
python <step>.py <name> --data-root <parent>   # if you keep sequences under
                                               # one parent folder
python <step>.py --help          # full option list per step
python <step>.py                 # interactive: scans the data root and
                                 # offers a numbered menu of sequence folders
```

**`--data-root` defaults to the current directory, not to `custom/`.** This toolchain never assumes, creates or moves anything into a `custom/` layer: `custom/` is *singleshotpose's* data-root name, and having a second one here is what makes every path ambiguous. Your sequence folder stays where you put it. The only place `custom/` legitimately appears is `5_create_config2split.py --sspose-root` — the prefix written *into* the generated cfg, describing where the folder will live once you move it under singleshotpose. That is a different knob and it keeps its default.

A bare name is looked up under the data root and then as a folder relative to the current directory (with the default root those are the same place). Every step writes its products **into the sequence folder**, which is what keeps that folder a self-contained thing you can hand to `structured_sspose` in one move. Steps 2 and 3 accept `--out-dir` to redirect theirs, and step 5 `--cfg-dir` for the cfg, but the later steps look for their inputs in the sequence folder — so leave those alone unless you have a reason.

**Steps 2 and 3 need nothing but the folder.** Everything that used to require tuning is now measured from the data: the capture layout (camera orbiting vs turntable), the marker-corner depth filter, the point-merge radius, and the crop that removes the re-posed background. Each prints what it chose and why, and every choice is still overridable. `python 2_compute_gt_poses.py mycup` followed by `python 3_register_scene.py mycup` is the whole of it.

Each step validates its prerequisites before running and, if something is missing, tells you **which earlier step produces it**, e.g.:

```
'mycup/' is missing required inputs:
  transforms.npy  <- produced by 2_compute_gt_poses.py
```

If the folder itself cannot be found, the message names what was tried and lists the sequence folders it *can* see (a folder counts as a sequence if it contains `JPEGImages`, `intrinsics.json` or `transforms.npy` — which is also what the interactive menu offers, so sibling folders like `utils/` never show up).

Step-specific options (defaults = the paper's values, from `config/registrationParameters.py`):

| Step | Options |
|---|---|
| `1_record_azurekinect.py` | `--countdown 5`, `--depth-scale 0.001` |
| `2_compute_gt_poses.py` | `--out-dir`, `--capture-mode auto`, `--marker-ransac odometry`, `--corner-depth-tol 0.05`, `--max-loop-closures 1`, `--label-interval 1`, `--k-neighbors 10`, `--voxel-size 0.001`, `--icp-method {point-to-plane,colored-icp}` — see §6 |
| `3_register_scene.py` | `--out-dir`, `--voxel-r` (**auto** from the data), `--crop-margin 0.15`, `--no-crop`, `--min-votes 2`, `--reconstruction-interval 8`, `--label-interval 1` — see §6 |
| `3a_segment_object_masks.py` | prompt: `--prompt TEXT`, `--pick` (+`--pick-port 8765`, `--pick-host 127.0.0.1`, `--pick-open`), `--click X,Y[,L]` (repeatable), `--box X0,Y0,X1,Y1`; `--seed-frame 0`, `--stride 1`, `--seed-only`; detection `--box-threshold 0.30`, `--text-threshold 0.25`, `--detection-index 0`; tracking `--verify-stride 30`, `--verify-iou 0.5`, `--reanchor`, `--max-reanchors 5`; masks `--keep-markers`, `--marker-dilate 3`, `--no-largest-component`, `--min-area-frac 2e-4`; models `--sam2-model`, `--detector-model`, `--device auto`, `--no-offload-video`, `--offload-state`; `--preview-stride 30`, `--out-dir` — see §7 |
| `3b_segment_object_cloud.py` | `--source {auto,scene,depth}`, `--input-scene registeredScene.ply`, `--output object.ply`, `--mask-dir objmask`, `--interval 8`, `--label-interval 1`, `--depth-tol 0.03`, `--erode 2`, `--min-observations 3`, `--min-ratio 0.6`, `--voxel 0.002`, `--no-cluster-filter`, `--cluster-eps 0.01`, `--min-cluster-points 50`, `--no-outlier-removal`, `--min-votes 2` (depth source), `--preview-frames 4` — see §7 |
| `4_create_label_files.py` | `--class-label 0`, `--sample-points 100000`, `--input-cloud object.ply`, `--label-interval 1` |
| `5_create_config2split.py` | `--split {interleave,eval,random}`, `--train-every 5`, `--train-frac 0.2`, `--seed 0`, `--sspose-root custom/`, `--cfg-dir` (default: the sequence folder), `--no-cfg`, `--name`, `--backup`, `--mesh` (auto), `--gpus 0` |
| `6_inspect_labels.py` | `--delay 30` (ms; 0 = key-step; `q` quits), `--save-dir <dir>` |

### Manual steps that cannot be skipped
1. **Naming the object.** Somebody has to say *which* object in the scene is the object. That is now one phrase (`3a --prompt "white mug"`), one click (`3a --pick`), or the CloudCompare cut as before — but the cut itself, the part that used to decide the bounding-box fit, is no longer manual. See §7.
2. **Label format check:** singleshotpose expects the mesh re-saved as **ASCII PLY**.

Steps 3a/3b need `sam2` and `transformers` installed (see §3); the manual route needs neither.

### Core-pipeline sufficiency check (vs the baseline ODT README)

Verified against the workflow in `check_ref/ObjectDatasetTools-master/README.md`; the numbered scripts alone produce a complete RGB-D + labels dataset:

| Baseline ODT step | This repo | Output |
|---|---|---|
| 1. Preparation (print ArUco 1-13) | same (manual) | — |
| 2. `record.py` / other cameras | `1_record_azurekinect.py` — or drop your own `JPEGImages/`, `depth/`, `intrinsics.json` in; step 2 validates them | RGB-D frames + intrinsics |
| 3. `compute_gt_poses.py` | `2_compute_gt_poses.py` | `transforms.npy` |
| 4. `register_scene.py` | `3_register_scene.py` | `registeredScene.ply` |
| 5. manual processing | `3a_segment_object_masks.py` + `3b_segment_object_cloud.py` (manual cut still supported) → `object.ply` | segmented object |
| 6. `create_label_files.py` | `4_create_label_files.py` | `labels/`, `mask/`, `transforms/`, meshes |
| — `makeTrainTestfiles.py` + `getmeshscale.py` + writing `cfg/<obj>.data` by hand | `5_create_config2split.py` | `train.txt`, `test.txt`, `training_range.txt`, `<obj>.data` |
| — `inspectMasks.py` | `6_inspect_labels.py` | visual QA |

---

## 2. File inventory

### Core pipeline (numbered)

| File | Origin vs ODT baseline | Description |
|---|---|---|
| `1_record_azurekinect.py` | **rewritten** (`record2.py`) | RealSense backend replaced with `pykinect_azure` (K4A). 720p BGRA color + WFOV 2×2-binned depth transformed into the color frame. Writes 16-bit depth PNG + `intrinsics.json` (width/height taken from the actual captured frame). Space bar = pause, `q` = stop. |
| `2_compute_gt_poses.py` | **modified** (`compute_gt_poses.py`) | Same ArUco-RANSAC + ICP-fallback pose graph. Adds `log.txt` edge logging, `%06d` filenames, loop-closure candidates spread over the whole sequence (`K_NEIGHBORS` reinterpreted). |
| `3_register_scene.py` | modified (`register_scene.py`) | Filename/config adaptation only; algorithm unchanged. |
| `3a_segment_object_masks.py` | **new (this release)** | Replaces the *selection* half of the manual CloudCompare step. One prompt — free text (Grounding DINO), clicks/box, or the browser picker — on one seed frame; SAM 2's video predictor propagates the mask forwards and backwards over the sequence. Subtracts the ArUco quads and speckle from every mask, verifies the track against the text prompt every `--verify-stride` frames (and repairs it with `--reanchor`), and writes `objmask/*.png`, `objmask/seg_meta.json` (prompt, models, per-frame areas, warnings) and `seg_preview/` including a contact sheet. See §7. |
| `3b_segment_object_cloud.py` | **new (this release)** | Replaces the *cutting* half. Turns `objmask/` + the step-2 poses into `object.ply`: each frame projects every point, abstains where the point is occluded, and otherwise votes object/background by the mask; points kept on `--min-observations` and `--min-ratio`, then cluster/outlier filtered and downsampled to `--voxel` (2 mm, what the manual subsample did). `--source depth` fuses the masked depth instead of cropping the step-3 scene, reusing that script's own vote merge. Prints the resulting OBB in cm and reprojects the cloud + box onto real frames. See §7. |
| `4_create_label_files.py` | **heavily modified** (`create_label_files.py`) | Input is now a *point cloud* (`object.ply`, `--input-cloud`); runs **Open3D ball-pivoting** surface reconstruction, then OBB alignment, 100k-point mask sampling (10× denser than baseline), projects 9 keypoints (mesh centroid + 8 AABB corners) → `labels/*.txt` (singleshotpose format), `mask/*.png`, and writes the object mesh with its centroid on the origin as **`<dataset>.ply` in ASCII** — the exact path/format/frame `structured_sspose` loads. Vertex colours from the input cloud are carried through (columns `6:9`; see §5). Class id set via `--class-label`. |
| `5_create_config2split.py` | **new (this release)** — merges `makeTrainTestfiles.py`, `getmeshscale.py` and the hand-written cfg | Enumerates `mask/` (so augmented frames are included) and splits it: `interleave` (every `--train-every`th labeled frame trains, default 5 → 20 %, the baseline behaviour), `eval` (no training frames — for a pure evaluation sequence, replacing `7b_make_eval_files.py`), or `random` (`--train-frac`, seeded). Measures the object diameter from the mesh step 4 wrote, and writes a complete `cfg/<obj>.data`: `diam` from the mesh, `fx/fy/u0/v0` from `intrinsics.json`, `width/height` from a real JPEG (warning if they disagree with `intrinsics.json`, which would mean skewed labels), and `name`/`backup`/`train`/`valid`/`tr_range`/`mesh` from the dataset name. No comments are written into the `.data` file — singleshotpose's parser does `key, value = line.split('=')` on every non-empty line and would crash on one. |
| `6_inspect_labels.py` | modified (`inspectMasks.py`) | Draws mask overlay + 3D-BB wireframe + centroid for every frame; can save the overlays with `--save-dir`. |

### Libraries (imported by the pipeline — do not rename)

| File | Origin | Description |
|---|---|---|
| `utils/registration.py` | baseline (`xrange` patched, open3d shim, `match_ransac_robust` added in Pass 4) | Rigid-alignment estimators — `match_ransac_robust` (step 2's default), `match_ransac` (the baseline least-squares one), `icp`, and the `rigid_transform_3D` SVD they share. Used by **step 2 only**. |
| `config/registrationParameters.py` | **tuned** | Paper values: `VOXEL_R=0.0002`, `K_NEIGHBORS=10`, `RECONSTRUCTION_INTERVAL=8`, `LABEL_INTERVAL=1`, point-to-plane ICP. Every numbered script can override these per run via CLI options. |
| `utils/camera.py`, `utils/ply.py`, `utils/plane.py` | baseline (unchanged) | Depth→point-cloud conversion, PLY writer, plane fitting. |
| `utils/cli.py` | **new (this release)** | Shared CLI: common `dataset` / `--data-root` arguments, interactive dataset picker, prerequisite validation with which-step-produces-it hints. |
| `utils/segmentation.py` | **new (this release)** | The two promptable models behind step 3a — Grounding DINO (text → boxes, `transformers`, pure PyTorch so nothing compiles locally) and SAM 2.1 (box/clicks → mask, plus the video predictor that propagates it) — with the mask post-processing they share. Opens no window: every entry point takes and returns arrays, which is what lets 3a run headless. Used by **step 3a only**. |
| `utils/picker.py` | **new (this release)** | The `--pick` page: one self-contained HTML page served by the standard library's `http.server` on localhost. Chosen over an OpenCV window because this pipeline is normally driven over SSH on a display-less GPU box, where `cv2.imshow` needs an X server and X forwarding is slow at 720p; a forwarded port works from any client, and locally the URL simply opens. Used by **step 3a only**. |

### Not shipped in this release

The working copies also carried the scripts below. None of them is referenced by the core
pipeline or by `structured_sspose`, and none is needed to reproduce
the paper's dataset, so they are not part of this repo. The paper's offline-augmentation /
RoI-PCA / debug-viewer extras from the original working copies are likewise not part of this
release. The scripts are listed here because earlier
drafts of this README documented them, and because §4 still refers to some of them when
recording what was fixed where. The originals are still in `odt_ref/` and
`proposed/FINAL_ODT_codes/` if any is ever wanted back — under the author's original names
(`++depth.py`, `++depth2pcd.py`, `++depthdiff_mask.py`, `+5_CDE_normal*.py`, `nameChanger.py`,
`apply_obb_mesh.py`, `+7_getmeshscale.py`, …), not the `extra_*` names above, which this
cleanup assigned.

| Dropped | What it was |
|---|---|
| `extra_depth_completion_normalmap.py`, `extra_depth_normalmap_rgbd.py`, `extra_depth_colored_depthmap.py`, `extra_depth_to_pointcloud.py`, `extra_depth_rmse_masked.py` | depth repair via the known 3D model, normal maps from depth gradients, jet-colormap depth figures, depth → `.xyz` clouds, masked depth RMSE. Exploratory work on the depth-dropout problem §6 diagnoses; the conclusion there is radiometric (raise the object's IR return), not algorithmic. |
| `extra_files_extract_every_nth.py`, `extra_files_renumber_inplace.py`, `extra_files_copy_to_yolo_tree.py`, `extra_files_labels_with_id_offset.py`, `extra_files_trainlist_with_id_offset.py`, `extra_files_testname_list.py` | frame/label renumbering and export helpers for merging several captures into one dataset, plus a YOLO-tree exporter. Step 5 covers the single-sequence case; the offset variants duplicated `4_`/`7_` with an index shift. |
| `extra_viz_normalmap_overlay.py` | overlaid the precomputed normal maps on RGB + wireframe — the viewer for the dropped depth scripts' output. |
| `extra_get_mesh_scale.py` | standalone object-diameter check; folded into `5_create_config2split.py` (with the convex-hull `pdist` fix — §4, pass 1, item 4). |
| `extra_lib_obb_for_pointcloud.py` | vendored trimesh internals to compute an OBB on a raw point cloud; unused by the pipeline. |
| `extra_pca_roipca_preview.py`, `extra_pca_wholeimage_preview.py`, `extra_pca_roi_stats.py` | the RoI-PCA / Fancy PCA previews and the statistics behind Fig. 5. Their defects, and the merged comparison tool that fixed them, are recorded in §4, pass 9. None of the three ran (one machine's absolute path; the whole-image one also had a dataset-specific `id + 16` offset). |
| `extra_pca_similarity_metric.py`, `extra_pca_similarity_experiments.py` | a prototype PCA-based scalar image descriptor and RBF similarity between images (5-D: whole image + 4 quadrants), plus the experiment harness around it with SSIM baselines. Dropped rather than replaced: they answered a different question from the paper's, needed `comp/*.jpg` images that were never part of the dataset, and their `fancy_pca` computed `sort_perm` without ever applying it to `eig_vecs`, so each eigenvector was multiplied by the wrong eigenvalue. |
| `extra_calib_chessboard.py` | standalone OpenCV chessboard (4×6) intrinsic calibration. Step 1 writes `intrinsics.json` from the Azure Kinect SDK's own factory calibration, so nothing in the pipeline needed it. |
| `registration.py`'s `feature_registration` | SIFT keypoint matching → depth lookup → rigid fit. Not a file but a function, removed in Pass 10: nothing called it, and it needed `cv2.xfeatures2d`, absent from the pinned opencv build. |
| `extra_aug_*.py` (4), `extra_debug_*.py` (3), `extra_viz_ar_overlay.py` + `MeshDean.py` | superseded during the cleanup by three merged replacement tools — see §4 passes 6, 7 and 8; those tools are themselves not shipped in this release. |

---

## 3. Environment

### Quick setup with uv (recommended, verified)

From the **repo root** (`sspose_improved/`):

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv -r pose-dataset-studio/requirements.txt
uv pip install --python .venv --no-deps pykinect-azure   # step 1 (recording) only
uv pip install --python .venv sam2 transformers          # steps 3a/3b only
# run scripts with the venv python, e.g.:
.venv/Scripts/python pose-dataset-studio/2_compute_gt_poses.py --help
```

Verified combination (Windows, Python 3.10.20): `numpy 1.26.4`, `opencv-contrib-python 4.6.0.66`, `open3d 0.19.0`, `trimesh`, `scipy`, `pykdtree`, `pypng`, `tqdm` (+ `pykinect-azure` for step 1). Verified end to end in a fresh venv installed from `requirements.txt` alone: steps 2–6 ran to completion on a 40-frame capture, reproducing the documented auto merge radius (0.0010 m) and object diameter (0.286384 m).

Version pins that matter (see `requirements.txt` for the full rationale):
- **`opencv-contrib-python==4.6.0.66`** — the configuration everything was verified with; the ArUco shims (`2_`, `utils/markers.py`) cover both the legacy and the ≥4.7 API, so newer versions should also work. Do **not** install `opencv-python` next to it (same `cv2` package — they overwrite each other; this is also why `pykinect-azure` is installed `--no-deps`).
- **`numpy<2`** — the opencv 4.6 wheels are built against numpy 1.x.
- **`open3d==0.19.0`** — the module-move shim resolves `o3d.pipelines.registration`; `open3d ≤ 0.9` (the paper's original configuration) also works via the same shim's first branch.

### Optional: the automatic segmentation stack (steps 3a/3b)

`sam2` + `transformers` (which pulls in `torch`). Nothing is compiled against the local CUDA toolkit: Grounding DINO comes from `transformers` rather than the original repository precisely to avoid its custom CUDA extension, and SAM 2's optional extension is not required. The two checkpoints — `facebook/sam2.1-hiera-large` (~900 MB) and `IDEA-Research/grounding-dino-base` (~700 MB) — download from the Hugging Face hub on first use and are cached in `~/.cache/huggingface`; `--sam2-model` / `--detector-model` take the smaller variants (`sam2.1-hiera-small`, `grounding-dino-tiny`) or any local path.

A GPU is not required (`--device cpu` works) but changes the runtime by roughly two orders of magnitude. Measured on one A100-40GB, 930 frames of 1280×720 at `--stride 8` (117 frames): ~90 s end to end including model load, mask writing, previews and the 38-frame verification pass. VRAM stays small with the default `--offload-video` (the decoded frames live in CPU RAM, ~13 MB per frame at 1024²); `--no-offload-video` is faster if the whole sequence fits in VRAM.

### Non-pip prerequisites
- **Azure Kinect Sensor SDK** (step 1 only; `pykinect_azure` loads its DLLs at runtime)
- CloudCompare or MeshLab — only for the manual segmentation route (steps 3a/3b replace it; see §7)

---

## 4. Changes applied in this release (vs the original working copies)

All sources are preserved untouched in `odt_ref/` and `proposed/FINAL_ODT_codes/`. The copies here received the following **minimal, documented fixes**:

### Pass 1 — verification & bug fixes
1. **OpenCV 3/4 compatibility** — `_, contours, _ = cv2.findContours(...)` → `contours = cv2.findContours(...)[-2]` in 7 files (`4_`, `extra_aug_occlusion_gamma`, `extra_aug_gamma_only`, `extra_aug_occlusion_preview`, `extra_files_labels_with_id_offset`, `extra_pca_roipca_preview`, `extra_pca_roi_stats`). Four of the seven no longer ship — the `extra_aug_*` three were merged into `ext_augmentation.py` in Pass 6 and `extra_files_labels_with_id_offset.py` was dropped (see §2, "Not shipped in this release").
2. **Operator-precedence bug** — `str("%04d" % i * LABEL_INTERVAL)` (string *repetition*) → `str("%04d" % (i * LABEL_INTERVAL))` in `4_create_label_files.py` and (before it was dropped) `extra_files_labels_with_id_offset.py`. A no-op at the paper's `LABEL_INTERVAL=1`, but wrong for any other interval.
3. **Python 2 leftovers** — `xrange` → `range` in `utils/registration.py` and (before it was deleted, see Pass 7) `extra_debug_aruco_check.py`.
4. **Object-diameter fix** — `6_get_mesh_scale.py` (now folded into `5_create_config2split.py`) compared only *consecutive* vertex pairs (underestimates the diameter); now computes the true max pairwise distance via convex hull + `pdist`. **If your existing `.data` `diam` values came from the old script, re-check them.**

### Pass 2 — CLI restructuring & legacy-API hardening
5. **Renumbering** — the offline augmentation steps were pulled out of the core pipeline: `5_augment_occlusion_gamma.py` → `extra_aug_occlusion_gamma.py`, `5b_augment_gamma_only.py` → `extra_aug_gamma_only.py`; the remaining steps were renumbered `6→5`, `7→6`, `8→7`, `8b→7b`.
6. **argparse everywhere** — every numbered script and both `extra_aug_*` scripts now take `dataset` / `--data-root` (plus step-specific options, see §1) instead of hard-coded folder names at the bottom of the file. Omitting `dataset` opens an interactive numbered menu; prerequisites are validated with which-step-produces-it hints (`utils/cli.py`, new).
7. **ArUco API shim** — `2_compute_gt_poses.py` works with both the legacy (`Dictionary_get`, OpenCV ≤ 4.6) and the new (`ArucoDetector`, OpenCV ≥ 4.7) ArUco APIs.
8. **open3d module-move shim** — `2_`, `3_`, `utils/registration.py` use `o3d.registration` when present, else `o3d.pipelines.registration` (open3d ≥ 0.10). Replaces `from open3d import *`.
9. **Precedence bug, remaining instances** — the same string-repetition bug as #2 was fixed in **102** more spots across `extra_aug_occlusion_gamma.py` (66) and `extra_aug_gamma_only.py` (36).
10. **Mask-filename condition fix** (`extra_aug_gamma_only.py`) — the `>= 10000` checks for the 4/6-digit mask name tested indices `+6…+11` while writing indices `+0…+5`; now aligned (only mattered when a sequence crossed 10 000 frames).
11. **intrinsics.json correctness** (`1_record_azurekinect.py`) — width/height now taken from the actual captured frame instead of hard-coded `1280×720`; `depth_scale` is a documented CLI option (`0.001` = K4A millimeters) instead of an unexplained RealSense constant.
12. **Silent failure removal** — the split scripts (then `7_make_train_test_files.py` / `7b`, now `5_create_config2split.py`) no longer wrap the whole body in `try/except: pass`; missing inputs now abort with a message. `training_range.txt` file handle is now closed. The mesh-scale code no longer swallows load errors.

### Pass 3 — post-refactor review fixes (agent-reviewed + end-to-end runtime-tested)

The whole numbered pipeline was executed end-to-end against a synthetic 12-frame ArUco dataset (steps 2→3→4→5→6→7/7b, absolute `--data-root`, label geometry verified against known ground truth). Fixes from that review:

13. **Double `LABEL_INTERVAL` in 7/7b removed** — frame ids parsed from mask filenames already encode the interval; the extra `* LABEL_INTERVAL` corrupted train/test lists for any interval ≠ 1 (missing frames listed, labeled frames dropped, split ratio drifted). The redundant `--label-interval` option was removed from both scripts; the train split now runs on sequence index, keeping the exact 1-in-N ratio.
14. **`colored-icp` fixed for open3d ≥ 0.12** (`utils/registration.py`) — the criteria object was passed in the slot that newer open3d reserves for the estimation method (guaranteed `TypeError` under the pinned 0.19). Now tries the old signature and falls back to `TransformationEstimationForColoredICP` + criteria. `2_`'s registration clouds also normalize colors to [0,1] (required by colored-ICP; ignored by point-to-plane). Unit-tested under 0.19: both methods recover a known 2 mm translation.
15. **Interval validation in `3_register_scene.py`** — `--reconstruction-interval` must be a positive multiple of `--label-interval`; other combinations previously produced *silently* wrong geometry (segments paired with the wrong poses, or all poses = frame 0).
16. **`1_record_azurekinect.py` hardening** — unchecked depth-capture failures could crash mid-recording (depth `ret` now checked); `q`-to-quit was consumed by a second `cv2.waitKey` call ~50 % of the time (single key read now; `q` also quits from pause); the redundant double depth write (cv2 + pypng to the same file) was removed.
17. **`4_create_label_files.py`** — a first frame with no mask contour no longer aborts with `NameError` (empty mask + warning instead); mask filenames now switch to `%06d` at frame ≥ 10 000, matching what the label inspector expects.
18. **`utils/cli.py`** — the interactive picker exits with a clean message instead of an `EOFError` traceback when stdin is closed (piped/CI runs).
19. **`2_compute_gt_poses.py` performance** — marker features (ArUco corners lifted to 3D) are now computed **once per frame and cached** instead of re-loading both images and re-detecting markers for every frame *pair* (~20× redundant at `K_NEIGHBORS=10`), and 16-bit depth PNGs are read with cv2 instead of pure-python pypng (~4×). Verified on a real 529-frame Azure-Kinect sequence: **22 m 47 s → 15 m 47 s**, with bit-equivalent output (identical 574 marker + 102 ICP edge set; poses differ ≤ 4e-15). Remaining cost is point-cloud machinery — per-frame `load_pcd` (~0.9 s: full-frame conversion + 1 mm voxel downsample + normals, ≈ 49 %), per-edge information matrix (~0.4 s, ≈ 27 %), and per-ICP-fallback registration (~2.7 s, ≈ 29 %); raising `--voxel-size` shrinks all three at some accuracy cost.

### Pass 4 — registration accuracy (see §6 for the full diagnosis)

Investigated on a real 529-frame Azure Kinect sequence against a depth-free, bundle-adjusted marker-board reference. The pipeline was recovering only **63 %** of the object's true rotation, drifting to **150° / 890 mm** by the end of the sequence — the "one frame is visibly misaligned" symptom is in fact *progressive* drift affecting every frame past ~8.

**Verified outcome on that sequence** (`--voxel-size 0.001`, all other defaults):

| | before | after |
|---|---|---|
| total rotation recovered | 226.2° = **62.9 %** of reference | 361.4° = **100.4 %** |
| loop-closure residual (frame 528 → 0) | 152.0° / **897 mm** | 0.82° / **8.7 mm** |
| per-edge rotation error vs reference | median 0.356°, max 3.05° | median **0.092°**, max 0.55° |
| ICP fallbacks | 102 | **0** |
| marker-edge residual | 1.18 mm accepted / 27.1 mm rejected | 0.7–2.4 mm |
| pose-graph edges | 676 | 945 |
| wall time | 15 m 47 s | **14 m 40 s** |

A **103× reduction** in end-of-sequence drift, at slightly *less* runtime despite 40 % more edges. The final loop residual (8.7 mm) is below the reference's own (23.2 mm), because the pose graph's loop closures correct drift that the reference's open chain does not.

20. **Marker-corner depth filter** (`2_`, `--corner-depth-tol`, default 0.05 m) — 6.7 % of ArUco corners read depth 5 cm to 3.6 m *behind* the marker plane (background bleeding through at the marker border; 95.9 % of gross errors are farther, not nearer). Sitting at 1.5–5× the lever arm of a clean corner, they drag the unweighted SVD toward the identity: a systematic **23 % under-rotation on every edge** that accumulates linearly instead of averaging out. A corner is now dropped when its depth deviates from its own marker's median corner depth by more than the tolerance. This single filter recovers 98 % of the true rotation. *(A neighbourhood median — the intuitive fix — does **not** work: a mixed-depth corner has a contaminated neighbourhood too, and it flags only 0.114 % of corners.)*
21. **Real RANSAC for the marker fit** (`registration.match_ransac_robust`, `--marker-ransac`, default `odometry`) — replaces the single least-squares SVD on odometry edges. **Inlier threshold 3 mm, not 10 mm**: the threshold must be below the inter-frame corner motion (3.0 mm here), otherwise the identity transform is itself a ~100 %-inlier hypothesis, max-consensus cannot discriminate, and the refit collapses back to least squares (measured: 77 % of the rotation at 10 mm vs 99 % at 3 mm). The `min_inliers` / `min_inlier_ratio` gate is mandatory — without it a 3-point sample fits itself and all 2948 candidate pairs get accepted.
22. **Degenerate-fit guard** (`registration.MIN_MATCH_POINTS = 6`) — `match_ransac` accepted fits from as few as **2** correspondences, where the SVD is rank-deficient and the trimmed-residual test is vacuous (`k = int(2*0.7) = 1`, and an exact 2-point fit scores ~0). Measured: an accepted 2-point fit was off by **801 mm / 76°**, trusted as a *certain* odometry edge. Fewer than 6 surviving correspondences now fall through to ICP.
23. **ICP-fallback warning** (`2_`) — a full-frame ICP only measures the *camera's* motion if the camera is what moved. In this sequence the camera was static and the board rotated on a turntable, so ICP locked onto the static background and returned ≈identity, discarding 98.4 % of the rotation on the 102 edges that used it (**50.5 % of the total deficit**). The script now counts these edges and prints an explicit warning describing the hazard. See §6 for the capture-protocol implication.
24. **Diagnostics in `log.txt`** — the log is opened `'w'` instead of `'a'` (three runs had silently accumulated into one file, making the current run's edge counts unreadable), starts with a header recording every parameter used, and each edge line now carries `n=<correspondences> rmse=<trimmed residual> inliers=<count> reason=<why>`. Previously a bad edge was indistinguishable from a good one in the log — the original author had resorted to a commented-out `input()` pause to inspect fallbacks by hand (`++_CiI/1_DataAnnotation/+2_compute_gt_poses_240617.py:245`).
25. **Loop-closure cap** (`--max-loop-closures`, default 1) — with the marker path fixed, nearly every candidate pair now registers successfully (7.4 accepted loop closures per source frame, vs the handful that used to survive), and each accepted edge costs a point-cloud load plus an information matrix. Capping keeps the runtime at its previous level; candidates are rotated by source frame so short- *and* long-range closures are both represented. `0` gives a pure odometry chain.
26. **Automatic capture-mode detection** (`2_`, `--capture-mode`, default `auto`) — the user no longer has to know or declare whether the camera orbited a static object or the object turned on a turntable; it is measured from the background depth (§6). In `static-camera` mode the ICP fallback is cropped to the marker/object region. Validated on three sequences: the real turntable capture (2.0 mm background change → `static-camera`) and two synthesised camera sweeps at 6 px and 1 px per frame (475 mm and 47 mm → `moving-camera`), plus a forced-fallback run confirming the crop path executes.
27. **Data-derived merge radius + background crop + `--out-dir`** (`3_`) — `--voxel-r` now defaults to one depth pixel at the marker board's range instead of a fixed 0.2 mm, which is what made the object survive the vote filter (§6); the result is cropped to the marker board so the re-posed background's rings and bowl are gone; `--min-votes` exposes the previously hard-coded `vote > 1`; and both steps 2 and 3 accept `--out-dir`. Verified: the auto rule independently reproduces the hand-swept optimum (0.0010 m), and the crop leaves the point count inside its boundary bit-identical.
28. **Step 4 now satisfies the singleshotpose contract without manual steps** — audited against the original `create_label_files.py` and against what `structured_sspose` actually loads:
    - **The mesh is written under the name and format the consumer expects.** `cfg/<obj>.data` names `custom/<obj>/<obj>.ply` and `MeshPly.py` opens it in *text* mode, but step 4 wrote `test_mesh_after.ply` in *binary* — so every dataset needed a hand rename plus a re-save through MeshLab (`check_ref/.../README.md` documents that ritual). The origin-centred mesh is now exported as **`<dataset>.ply` with `encoding='ascii'`**. Verified: the real sspose reader parses it, whereas the old binary export died with a `UnicodeDecodeError`.
    - **`test_mesh.ply` renamed to `mesh_world.ply`.** It is the *world-frame* mesh and was byte-size-identical to the OBB-aligned one, so renaming the wrong file into `<obj>.ply` silently produced garbage poses with no error.
    - **The exported mesh now has its centroid on the origin, which is what makes the baseline's label convention self-consistent.** The invariant the two repos must share is: *the mesh file's origin and label keypoint 0 denote the same physical point.* The baseline writes keypoint 0 as the projection of `mesh.centroid` (`check_ref/.../create_label_files.py`), while singleshotpose builds its first model point as `np.zeros((3,1))` (`structured_sspose/3_eval.py`, `1_train_baseline.py`, `2_train_roipca.py`) — so those agree **only if the centroid is the origin**. `apply_obb()` leaves the *AABB* centred instead, so out of the box they disagreed by the centroid offset: on `dig_camera` 27.0 mm = **24.4 px**, biasing every `solvePnP` pose and hence the translation/rotation and 5 cm-5° metrics.

      Step 4 now applies `mesh.apply_translation(-centroid)` after the OBB alignment and folds it into `Tform`. This is exactly the manual "move it to 0,0,0 after step 4" step that datasets in this project were built with, so **the label convention is unchanged** — regenerated labels are byte-identical to what the baseline logic produced (verified: max 0.0000 px over all 529 frames of `dig_camera`) — while the 9 keypoints now also match what singleshotpose reads, to **0.001 px**. Translating is safe because it shifts all nine model points equally and the pose absorbs it; the 8 corners stay exact.

      Note that the exported mesh's AABB centre is therefore *not* at the origin (it is offset by `+centroid`), which is intended: singleshotpose derives the corners from the AABB wherever it is, and only keypoint 0 is tied to the origin. Older datasets carrying a hand-shifted mesh with baseline labels are already in this configuration and need no rework.
    - **`Moved_Input_points.ply` dropped.** Nothing in either repo ever read it, and it was broken: `estimate_normals()` ran *before* the OBB transform and only the points were replaced, so the stored normals were left un-rotated (measured `mean|cos| = 0.479` against correct normals, only 1.8 % within 0.9) — which is why it rendered as nonsense in mesh viewers.
    - **Step 4 prints the cfg block it already knows** (`mesh`, `diam`, `fx`, `fy`, `u0`, `v0`, `width`, `height`); step 5 then writes that block to `cfg/<obj>.data` outright, so nothing is transcribed by hand. Both resolve the mesh as `<dataset>.ply`, falling back to `test_mesh_after.ply` for older datasets — the filename is never asked of the user.
29. **New `5_create_config2split.py`; label inspection moved to `6_`.** The three tail-end scripts (`6_get_mesh_scale`, `7_make_train_test_files`, `7b_make_eval_files`) plus the hand-written `cfg/<obj>.data` are now one step that takes only the sequence folder. The split strategy is an option instead of a hard-coded `i % 5` (`interleave` / `eval` / `random`), the mesh filename is resolved automatically rather than asked for, `width`/`height` come from a real JPEG and are cross-checked against `intrinsics.json`, and the `.data` file is written **into the sequence folder** so that folder stays a self-contained deliverable — nothing is copied into the other repo on the user's behalf. Verified by simulating the hand-off (sequence folder placed under a stand-in singleshotpose data root): the cfg parses with singleshotpose's own `read_data_cfg`, exposes all 14 keys the training/eval scripts read, every numeric field converts, `MeshPly` loads the mesh (36 143 vertices), and all four cfg paths plus all 529 image entries resolve from the new location.
30. **Lazy normals + bounded cloud cache** (`2_`) — surface normals are only needed by the point-to-plane ICP fallback, so they are no longer computed for every frame (about half of `load_pcd`'s cost). The point-cloud cache is now a dict holding only the frames still needed, instead of a 529-slot list that accumulated every loop-closure target (tens of GB on a long sequence).

All 44 Python files pass `py_compile`; `utils/cli.py` is covered by a 6-case unit test plus piped-stdin runs; the estimator guards and `_select_targets` have their own unit tests; runtime coverage per script: 2/3/4/5/6/7/7b executed successfully on the synthetic dataset and step 2 on the real 529-frame sequence (step 1 requires physical hardware — `--help` only).

### Pass 5 — vertex colours restored to the exported mesh

31. **`<dataset>.ply` and `mesh_world.ply` now carry per-vertex colour** (`4_create_label_files.py`). The upstream ODT exported the segmented cloud directly and its `<name>.ply` was `x y z r g b`. When ball-pivoting triangulation was introduced (in the paper author's working copies — `proposed/FINAL_ODT_codes/+4_create_label_files.py:148`, every `odt_ref/` variant, `++_CiI/`), the mesh was rebuilt with `trimesh.Trimesh(vertices, faces, vertex_normals=...)` and **`vertex_colors` was simply never passed**, so the exported mesh came out geometry-only. Open3D's ball pivoting does carry the input cloud's colours onto the mesh vertices — they were being discarded one line later.

    The colours are now forwarded (`(pointcloud.vertex_colors * 255).astype(np.uint8)`); a colourless input cloud still produces the old geometry-only mesh, with a printed note.

    **This is provably additive.** Passing `vertex_colors` changes nothing about the geometry: on `dig_camera` the vertex array (36 143) and face array (42 735) come out `np.array_equal` with and without it, so the AABB corners are bit-identical (max delta **0.0 m**) and `diam` is unchanged (**0.286384** either way, matching the existing `dig_camera.data`). `custom/dig_camera/`'s two meshes were regenerated in place and re-verified against the previous files on exactly those quantities; `labels/`, `mask/` and `transforms/` were deliberately **not** regenerated, because they cannot change and because `mesh_copy.sample()` is unseeded (re-running step 4 would rewrite all 529 masks with different random samples for no benefit).

    **What this fixes, and what it does not.** Nothing in the training/eval path was affected either way — `1_train_baseline.py`, `2_train_roipca.py` and `3_eval.py` use only `mesh.vertices`. The colour is consumed by the AR/MR overlays in `structured_sspose` (`extra_ar_*`, `extra_mr_*`, `extra_viz_*`, `4_eval_crossmesh.py`), which pass `MeshPly.colors` straight into `uint8` pixels in `projection.py`. Those still need a reader change — see the channel-layout note below and `structured_sspose/README.md` §7.

### Pass 6 — offline augmentation merged into one script

*(the tool this pass produced is not shipped in this release)*

32. **`ext_augmentation.py` replaces all four `extra_aug_*` scripts.** Three of them (`extra_aug_occlusion_gamma.py`, `extra_aug_gamma_only.py`, `extra_aug_occlusion_preview.py`) were near-copies of one another — the gamma-only one still carried an uncalled `OCCaugmentation`, and the preview one never called its own `cv2.imwrite`s and blocked on a `cv2.waitKey(0)` with no `imshow` in front of it. The fourth, `extra_aug_rotation.py`, was a repurposed copy of the mask inspector (its docstring still said `inspectMasks.py`) with `LINEMOD/TASPROJECT/1_5/` hard-coded. The merged script takes `--variants` (any combination of `occlusion`, `gamma`, `rotation`; default all three, 18× per frame) and `--action save|preview`, on the same `dataset` / `--data-root` interface as the numbered steps.

    Four behavioural changes, all deliberate: output moved out of the source folders into `aug_*` (the old scripts appended into `JPEGImages/`/`mask/`/`labels/` and a second run silently overwrote the first); the split lists are extended into `aug_train.txt`/`aug_test.txt`/`aug_training_range.txt`, which moves the step from *before* step 5 to *after* it; **γ = 1.0 was dropped** (identity transform — it wrote an exact duplicate of the source frame, so gamma is now 5 variants, not 6); and `depth/` is no longer duplicated, since singleshotpose reads no depth at all (~2.6 GB of writes avoided on `dig_camera`).

    Rotation gained what the old script lacked: the out-of-frame guard, the `x_range`/`y_range` update, nearest-neighbour mask warping, and no 16-bit depth corruption (the old one read depth with a plain `cv2.imread`, silently flattening it to 8-bit before writing it back). Its string-repetition filename bug (`str("%06d" % (n + id) * LABEL_INTERVAL)`) is gone with it.

    **What was checked against singleshotpose before relying on it:** `transforms/` is read by nothing (all `transforms` hits are `torchvision.transforms`), so rotation not updating the 6-DoF pose is harmless; label fields 19/20 (`x_range`/`y_range`) are never indexed by `region_loss.py` and are **overwritten with `1.0, 1.0`** by `3_eval.py:77`, so they exist only to pad the row to the 21 columns `fill_truth_detection` reshapes to; and `change_background` (`image.py:278`) replaces every non-mask pixel during training, which is what makes a rotation's black corner wedges a non-issue there.

    Verified on `dig_camera`: masks are `{0, 255}` as assumed; the 6 occlusion variants change only masked object pixels (grey 128) and erase exactly that much of the mask (26 829 → 17 022…23 528 px); the 5 gamma variants change the whole frame and leave the mask untouched; **no variant is a duplicate of its source**; occlusion and gamma labels are byte-identical copies; rotated labels round-trip through the inverse rotation to within **0.0013 px** (the label format's 8-character truncation) and a 180° rotation reproduces `x_range`/`y_range` exactly; `aug_mask` uses ODT's 4-digit padding, which is what `.replace('/00', '/')` reconstructs. The out-of-frame guard never fires on `dig_camera` (small, centred object) and was therefore tested separately against synthetic objects: one placed 90 px from the left edge keeps only the 180° variant. The occlusion inner loop was also vectorised (the original ran a per-pixel Python double loop over the bounding box). Trial outputs were deleted afterwards — `dig_camera/` currently has no `aug_*` files.

    One portability fix while testing: em-dashes in the script's `print` output crashed with `UnicodeEncodeError` under the cp949 console codepage whenever stdout was redirected to a file or pipe. Runtime strings in this script are now ASCII. **The same latent bug is still present in `2_`, `3_`, `4_`, `5_` and `utils/cli.py`** — it only bites on a redirect, which is why it has not shown up in interactive runs.

### Pass 7 — the debug extras replaced by one registration viewer

*(the tool this pass produced is not shipped in this release)*

33. **`ext_registration.py` replaces all three `extra_debug_*` scripts.** None of the three ran: each carried a different dataset's path (`LINEMOD/*/`, `custom/temp02/`, `D:/TASDATASET/8`), and the ArUco one also indexed frames as `0.jpg` where this toolchain writes `000000.jpg`.

    - `extra_debug_aruco_check.py` (upstream ODT's `aruco.py`, differing only in `xrange` → `range`) drew a box on each detected marker so you could judge a sequence before running step 2. **Deleted as redundant:** step 2's `log.txt` now records `n=` (matched marker corners), `rmse=`, `inliers=` and `reason=` — including `too_few_common_markers` — per edge, which answers the same question with numbers instead of a box.
    - `extra_debug_zoom_crop.py` (upstream `inspectMasks.py` plus a crop) showed a 500 px window around the label centroid. **Deleted:** it had a dataset-specific `id = id + 16` offset baked in, defined a `zoomImage()` it never called, and crashed on any object near a frame edge. `6_inspect_labels.py` covers the same check; add `--zoom` there if the magnification is ever wanted.
    - `extra_debug_registration_dump.py` wrote one ASCII `.xyz` per frame with a per-point Python loop — hours and tens of GB for a 529-frame sequence — into `test/`, i.e. **outside the sequence folder**, which invariant 1 forbids. Its purpose (step through frames and find where registration breaks) is kept and done three better ways in the replacement.

    Verified against `dig_camera`: the log parser recovers **945 edges (528 odometry + 417 loop closures, 0 ICP fallbacks)**, matching what Pass 4 measured, with RMSE min/median/max **0.60 / 0.90 / 9.86 mm** — the worst edges are all long-range loop closures, as expected. The interactive modes need a display and were checked by exercising their geometry construction directly rather than opening a window.

34. **`utils/markers.py` (new)** — the ArUco API shim and `marker_region_world()` were lifted out of `3_register_scene.py` so the viewer crops to *exactly* the region step 3 crops to. A crop that differs between the tool you diagnose with and the tool you build with is worse than no tool. Verified bit-identical before and after the extraction on `dig_camera`: centre `(-0.042459, 0.079662, 0.5925)`, radius `0.384259` either way.

    This was prompted by the first `--mode dump` run reproducing the artefact §6 describes: uncropped, the merge came out **10.3 × 5.9 × 8.4 m with 89 % of its points more than 1 m from the rotation axis** — the static room counter-rotated into rings and a bowl. Confirmed as the known artefact rather than a new fault by recovering the rotation axis from the poses themselves: frame 0 → 66/132/264/396 gives 43.1° / 88.1° / 179.2° / 91.0° about a **consistent** axis (0.03, 0.84, 0.54), i.e. a clean full turn, and frame 0 → 528 closes to 0.8°. With the crop on, the dump lands at 0.79 × 0.71 × 0.67 m against `registeredScene.ply`'s 0.76 × 0.55 × 0.65 m, centres agreeing to 4 mm.

### Pass 8 — the AR overlay rebuilt on the pipeline's own outputs

*(the tool this pass produced is not shipped in this release; the e2e front-ends' `qa_preview/` overlays cover the same end-to-end visual check)*

35. **`ext_ar_overlay.py` replaces `extra_viz_ar_overlay.py`.** The old renderer could not run
    here at all: it read a NeRF-style `transforms.json`, an `info.data` and an `extracted/`
    folder, none of which any step of this pipeline produces, and it carried one capture's
    absolute Windows paths plus a 4×4 pose matrix as a source literal. The replacement takes
    only the sequence folder, like every other script: mesh from `<dataset>.ply`, pose from
    `transforms/<id>.npy`, intrinsics from `<dataset>.data` if step 5 has run and
    `intrinsics.json` otherwise (it **warns** when the two disagree, since the labels come from
    the JSON), and the 3D box from `labels/<id>.txt`. Options: `--mesh`, `--stride`,
    `--point-size`, `--alpha`, `--bbox {labels,none}`, `--mask`, `--delay`, `--save-dir`,
    `--no-show`.

    Three behavioural fixes beyond the path cleanup:

    - **Colour channels parsed, not guessed.** `MeshDean.py` (its mesh reader, deleted with it)
      took vertex colour from PLY columns `3:6`. Step 4 writes
      `x y z  nx ny nz  red green blue alpha`, so `3:6` are the *normals* — the AR object was
      painted with normal vectors, which live in −1…1 and land on a uint8 pixel as noise. The
      new loader reads the header by property name, so upstream ODT's plain `x y z r g b`
      meshes load too and the bug cannot return. This is the same defect §5 records against
      `structured_sspose/MeshPly.py`.
    - **Nearest surface wins.** The old renderer assigned pixels in vertex order, so whichever
      vertex came last in the file painted over the rest and back faces punched through the
      front. Points are now composited far-to-near, i.e. a z-buffer by construction.
    - **Vectorised.** `projectionAR()` looped over every vertex in Python for every frame —
      36 143 vertices × 529 frames is 19 M interpreted iterations. The projection is array
      maths now, and the script prints its own timing.

    Reading the result: the wireframe comes from `labels/` (the numbers the network trains on)
    while the skin comes from `transforms/`, so they are two views of the same data — a
    wireframe that hugs the object while the skin drifts, or the reverse, means step 4's two
    outputs no longer agree and it should be re-run. **Not verified against a display in this
    environment** (`cv2.imshow` needs one); only `--help` and the import path have been
    exercised. `--save-dir --no-show` is the headless route.

### Pass 9 — the PCA previews merged into one comparison tool

*(the tool this pass produced is not shipped in this release)*

36. **`ext_pca_preview.py` replaces `extra_pca_roipca_preview.py`, `extra_pca_wholeimage_preview.py` and `extra_pca_roi_stats.py`.** All three carried the same `C:/Users/USER/PycharmProjects/JCDE/SWMR/LINEMOD/ape` path and could not run; the whole-image one also had a dataset-specific `id = id + 16` offset (the defect that removed `extra_debug_zoom_crop.py` in Pass 7) and called `cv2.imshow` from inside its own `fancy_pca`. Each kept a private copy of `fancy_pca` with a different amplification compiled into it (×1, ×1000, ×2000); there is no such number here, see below.

    Three measurement bugs, each of which changed what the figures showed:

    - **The ADIFF maps were wrong.** They were computed as `abs(img - augmented)` on uint8 arrays, where the subtraction wraps modulo 256: a true difference of 4 was displayed as **252**. Computed in int16 here, with the true maximum printed on the panel.
    - **The RoI box could index negatively.** `x -= 10` with no clamp wraps to the far side of the image for an object near the top or left edge. Clipped to the frame here. The ×1.2 / −10 px box those scripts used is *not* what training does either — `image.py`'s `imageIntensityModifier_PCA` defaults to `scalefactor=1.0, reloc=0`, i.e. the mask bounding box as-is, which is what this script uses.
    - **Two of the five copies mispaired the eigen-decomposition.** `extra_pca_similarity_metric.py` and `extra_pca_similarity_experiments.py` compute `sort_perm` but never apply it to `eig_vecs`, so eigenvector *i* is multiplied by the eigenvalue of index 2−*i*. The replacement sorts both together.

    `extra_pca_roi_stats.py`'s numbers are kept as `--stats-out`: per-frame RoI area ratio and per-channel means, raw vs augmented, as one TSV **inside the sequence folder**. The original appended to three files in the current working directory, so a second run silently doubled every series.

    **No amplification knob.** Fig. 6's caption says the *added values* are "multiplied by 1000 for clear presentation" — the paper amplifies what the figure draws, not the augmentation. Each difference panel here is simply brightened to its own maximum and prints the **true** maximum in grey levels, so the effect is visible without any number to choose and nothing is silently exaggerated. An earlier draft of this script exposed the perturbation strength and the display gain as two separate options; both were removed as knobs nobody should have to reason about.

    **This script applies Eq. (5); `structured_sspose/image.py` does not.** Eq. (5) is defined on an image normalized to 0–1, so the offset has to be converted to 0–255 units before it is added to the pixels. `image.py` computes the offset from the 0–1 data and adds it to the unnormalized 0–255 pixels, which is 255× smaller. Measured on `dig_camera` (53 frames at `--stride 10`): Eq. (5) moves pixels by a per-frame max of **0–10 levels** (median 3); the un-converted version moves them by **0 or 1 level**, and the 1 is `astype(np.uint8)` truncating rather than rounding, not a colour shift. This script does the conversion; **`image.py` itself is untouched** (see §5). An earlier draft carried an `--as-published` flag to switch between the two — it was removed, because reproducing a defect is not what a preview tool is for and the measurement is recorded here instead.

    Output is deterministic: alpha is seeded from the frame number, so a rerun reproduces the same panels.

    The script deliberately does **not** replace the background. Training does (`change_background_wPCA` composites a random VOC image over every non-mask pixel *after* augmenting), and an earlier draft offered a `--bg-dir` for it, but it obscures the very thing the preview is for: with the background swapped, even whole-image Fancy PCA leaves a difference only on the object silhouette, so the two arms look identical. The comparison the paper's Fig. 4 draws is on the captured frame, which is what this shows.

### Pass 10 — `registration.py` moved into `utils/`, dead code removed

37. **`registration.py` → `utils/registration.py`.** It was the only library left at the top level, sitting among the numbered scripts as though it were one of them, and its name collides with `ext_registration.py` (an unrelated viewer) when read in a file listing. `2_compute_gt_poses.py` now imports it as `from utils.registration import icp, match_ransac, match_ransac_robust`.

38. **Dead imports and dead code removed.** `3_register_scene.py` imported four names from it and called **none** — step 3 reads the poses step 2 already wrote and never registers anything, so the whole import line is gone. `2_compute_gt_poses.py` imported five and called three; the import now names exactly what it uses.

    `feature_registration` (SIFT keypoint matching → depth lookup → rigid fit, ~90 lines) was deleted: nothing in the repo called it, and it needed `cv2.xfeatures2d`, which the pinned `opencv-contrib-python==4.6.0.66` build does not expose — it would have raised on the first call. Removing it also dropped the module's last `cv2` use, so the import went with it. `rigid_transform_3D` stays: it is no longer imported by any script, but `match_ransac` and `match_ransac_robust` build on it internally.

    Verified after the move: all six numbered scripts and all four `ext_*` extras still reach their CLI, and the estimators still behave — on 30 correspondences with one gross outlier injected, `match_ransac_robust` recovers the rotation to 6.7e-16 with 29 inliers while `match_ransac` correctly rejects the pair (`rmse_too_high`).

## 5. Known limitations (left as-is, by design)

- **PLY channel layout is positional, not header-parsed.** Step 4 writes `x y z  nx ny nz  red green blue alpha` (trimesh's fixed order; colours in columns **6:9** as uchar 0–255). None of the PLY readers in this project parse the header — they index columns by position, so the layout has to be matched by hand:

  | Reader | Reads colour from | Against this file |
  |---|---|---|
  | `check_ref/.../MeshPly.py` (upstream singleshotpose) | `6:9`, ÷255 | correct |
  | `structured_sspose/multi/MeshPly.py` | `6:9` | correct |
  | `structured_sspose/MeshPly.py` | `3:6` | **reads the normals as colour** (silently: no exception, values land in −1…1 instead of 0…255) |

  `structured_sspose/MeshPly.py` was written by the paper's authors for the baseline's colour-only `x y z r g b` cloud. Adjust its channels before trusting any colour-dependent output there. Column 0:3 is the same in every case, so vertices, the 3D bounding box, `diam` and every metric are unaffected by the mismatch.

- In `4_create_label_files.py`, a frame whose mask has no contour reuses the previous frame's contour (first frame: empty mask + warning); the per-frame image is read with index `i` (not `i × LABEL_INTERVAL`) — harmless at the paper's `LABEL_INTERVAL=1`; and projections with z ≤ 0 (object behind the camera) write garbage corner values without warning.
- Mask files are named `%04d` below frame 10 000 and `%06d` above; per-frame `transforms/*.npy` are unpadded (inherited from the paper's datasets; all consumers in this repo and `structured_sspose` handle these).
- A bare dataset name is resolved against the *current working directory* (`--data-root` defaults to `.`), so run the scripts from wherever the sequence folder sits — or pass it as a path, which always works from anywhere. Note this is unrelated to the prefix written into `train.txt`/`test.txt`/`<name>.data`: that one is `5_create_config2split.py --sspose-root` and must be what singleshotpose resolves against from *its* working directory.
- `6_inspect_labels.py` needs a display even with `--save-dir` (`cv2.imshow` is unconditional).
- Steps 3a/3b fetch their two checkpoints from the Hugging Face hub the first time they run. On a machine without internet access, copy a populated `~/.cache/huggingface` over, or point `--sam2-model` / `--detector-model` at local paths. Everything after the first run is offline.
- `3a --pick` serves its page on `127.0.0.1` and expects you to forward the port; `--pick-host 0.0.0.0` exposes it to the network instead, with no authentication of any kind — only do that on a network you trust.
- Every-run outputs (`log.txt`, meshes) overwrite without asking.

---

## 6. Registration quality: why `registeredScene.ply` ghosts, and how to check

If step 3 produces a point cloud with doubled/smeared structure, this section is the diagnosis. It was established by measuring a real 529-frame Azure Kinect sequence against an independent, **depth-free** reference: the ArUco board's pose recovered from 2D corner observations alone by bundle adjustment (which recovers the sequence's full 360° turn and closes the loop to ~1°, so it is trustworthy as a yardstick).

### The symptom is drift, not a few bad frames

The instinctive reading — "one frame is registered wrong" — was wrong on this data. The error is **monotone accumulated drift**: the solved poses delivered only **63 %** of the true rotation per edge, so the discrepancy grows from 1.8 mm at frame 1 to 479 mm by frame 528 (150° of rotation error). Every reconstruction segment past frame ~8 ghosts, progressively; the four frames that a smoothness check flags as "jumps" are merely the worst of a uniformly biased set.

Note also that a nearest-neighbour or ICP consistency check **cannot** find these frames when the target is a flat marker board: point-to-plane, point-to-point and colored ICP all agreed with each other to <1.6 mm while all reporting ~0.01° where the truth was 0.69°. In-plane rotation of a planar object is invisible to point-cloud alignment. Use the marker corners, not the cloud, to validate poses.

### Two causes, each about half of the error

| cause | share | mechanism |
|---|---|---|
| Marker-corner depth outliers contaminating a non-robust fit | 49.5 % | 6.7 % of corners read 5 cm–3.6 m *behind* the marker plane (background bleed at marker borders — corner 5×5 depth-std p99 is 906 mm versus 6.85 mm at marker centres). At 1.5–5× the lever arm they pull the unweighted SVD toward the identity, producing a sign-consistent **23 % under-rotation** on every edge. |
| The ICP fallback measuring the wrong object | 50.5 % | The 102 edges whose marker fit was rejected fell through to a full-frame ICP, which delivered **1.6 %** of the rotation actually present. |

What is **not** a significant cause, despite being the obvious suspect: `depth == 0` at marker corners. It affects 12.6 % of corners, but they are simply skipped, and the *failing* pairs actually had fewer zeros than the succeeding ones (18.3 vs 23.2 per pair). Lens distortion, RGB↔depth misalignment and integer (non-subpixel) corner coordinates together account for 0–1 %.

### Capture protocol: the pipeline detects it for you

Both layouts occur in practice and you do **not** have to tell the pipeline which one you used — step 2 detects it (`--capture-mode auto`, the default) and prints what it found:

```
Capture mode: static-camera (background depth changes by 2.0 mm between distant frames; < 5 mm means the camera did not move)
```

The test compares the **depth of the background** (everything away from the markers) between widely separated frames. A static camera leaves it unchanged; a moving camera changes it systematically. Depth is used rather than colour because it ignores lighting, exposure and JPEG artifacts, and distant pairs rather than neighbours because a slow sweep looks static frame-to-frame. Measured separation: **2–3 mm** for a real turntable capture versus **47 mm** for a camera creeping at just 1 px per frame — a wide margin around the 5 mm threshold. Override with `--capture-mode moving-camera|static-camera` if the detection is ever wrong.

**Why it matters.** A full-frame ICP is only meaningful if the camera is what moved: then every point in view shares one rigid motion, which is exactly ICP's assumption. With a static camera and a turning object the frame contains **two** motions — the object and the (unmoving) room — and since the room dominates by point count, ICP returns the room's motion, ≈identity, silently deleting that edge's real rotation. In `static-camera` mode the ICP fallback is therefore restricted to the marker/object region.

**Honest limitation, measured.** That crop does *not* rescue a **planar** marker board: cropped ICP recovered 15.09° of a true 24.7° sweep over 40 frames, versus 15.10° uncropped — no improvement. In-plane rotation of a plane is geometrically unobservable, so no point-cloud method can see it (point-to-plane, point-to-point and colored ICP were all measured returning ~0.01° where the truth was 0.69°). On a turntable capture the **markers are the only reliable signal**, which is why the corner filter and the robust estimator matter so much there. The crop is kept because it is correct in principle and does help a non-planar object.

**If you have the choice, move the camera around a static object** — the layout ODT was designed for. A turntable capture works well (this repo's own datasets were made that way, and after the Pass 4 fixes it reaches 100.4 % rotation recovery with zero fallbacks), but on such a sequence every ICP fallback is a corrupted edge. Watch the printed warning and the `icp_fallback=` count on the last line of `log.txt`.

### How to check your own sequence

1. Run step 2 and read the tail of `log.txt`: the last line reports `edges= marker= icp_fallback=`. **`icp_fallback` should be 0, or near it.** Any non-zero count on a static-camera capture is suspect.
2. Scan the per-edge `rmse=` values. Healthy marker edges on this hardware sit at **0.7–2.4 mm**. Before the fixes, accepted edges averaged 1.18 mm but rejected ones 27 mm — and edges that were 1.4–3.1° wrong were being accepted at 3.4–9.8 mm, just under the old 10 mm tolerance.
3. Check `reason=`: `ok` (least-squares), `ok_ransac` (robust fit), `too_few_points` (<6 usable corners — depth is failing at the markers), `too_few_inliers`, `rmse_too_high`.
4. For a visual check, run step 3 and look at `registeredScene.ply` — the e2e dashboard's result view (§8) has a built-in 3D viewer for it, or open it in CloudCompare. Drift shows up as the doubled/smeared structure this section opened with. Step 3b's cloud-and-box reprojections onto real frames, step 3a's `seg_preview/` sheets and the dashboard's `qa_preview/` overlays are further end-to-end checks.

### The object comes out as a sparse shell while the board looks perfect

A separate failure, diagnosed on the same sequence: `registeredScene.ply` had a razor-sharp marker board and table but only **15 125 points (1.3 %) on the object** — a broken outline that shattered into 17 clusters at a 6 mm gap threshold. Two causes compound:

1. **The object barely returns depth.** Inside the black DSLR's silhouette, **68 % of pixels return no depth of the object** (45 % read zero, 24 % see the board straight through it) versus **7 %** dropout on the white board. This is radiometric, not geometric — proved by the control: the board's *own black ArUco squares* drop out at 30 % and the table's dark pixels at 40 %, the same physics. The board survives because only 6 % of its area is that dark, while 88 % of the object's footprint is dark or mid-grey. Where the object *does* return depth it is only 1.4× noisier than the board (0.53 vs 0.38 mm), so the deficit is **missing samples, not noisy ones**.
2. **The vote radius is far tighter than the sampling.** `post_process` confirms a point only when a later segment lands within `2.5 × VOXEL_R` = **0.5 mm** of it. But the depth pixel pitch at 0.64 m is **1.05 mm** and the frame-to-frame scatter of the same world point is **1.56 mm** — so the radius is ~3× tighter than the scatter it must absorb, only 47 % of object samples vote at all, and `--min-votes 2` amplifies a 3.2× mean-vote deficit into a **6.4× survival deficit**: 97.5 % of the object's merged points deleted, versus 84 % of the board's.

Ruled out by measurement: visibility (every patch is seen by ~14 of 66 segments; only 0.04 % by fewer than 3), near-range clipping (object at 0.53–0.83 m, sensor valid from 0.40 m), motion blur (1.88 mm of surface travel per frame, ≪ one integration), flying pixels (0.18 %).

**Fix — the merge radius is now derived from the data, so plain defaults handle this:**

```bash
python 3_register_scene.py <dataset>                              # auto radius
python 3_register_scene.py <dataset> --reconstruction-interval 4  # 2x the object points, 3.4x the time
```

`--voxel-r` defaults to **one depth pixel at the marker board's range** (`range / fx`), making the vote tolerance 2.5 pixels — matched to the sensor's actual sampling instead of a fixed constant. On this sequence that resolves to 0.0010 m, exactly the value found best by sweeping it by hand. Step 3 prints the value it chose, and warns if a manually passed `--voxel-r` is finer than the sampling.

> **Side effect to know about (handled by default):** widening the vote radius also lets the *background* survive. The merge re-poses the **whole** scene by the board's motion, so in a turntable capture the static room is counter-rotated and sweeps into surfaces of revolution — concentric rings and a bowl-shaped shell around the object, extending 3.5 m out. Surfaces perpendicular to the rotation axis (table top, floor) map onto themselves under that rotation, so they confirm each other and survive at *any* vote setting; the wider radius simply lets much more of them through. Step 3 therefore now crops the result to the marker board (`--crop-margin 0.15`, i.e. board radius + 15 cm; `--no-crop` restores the original behaviour). Measured on this sequence: 3 230 287 → 2 261 816 points, the extent shrinking from 3.54 m to 0.52 m, with the point count inside every radius up to the crop boundary **bit-identical** — the object is untouched. The crop radius is derived from frame 0's marker corners with the same per-marker depth filter as step 2; without it, a single marker corner reading the background at 3.6 m inflated the radius to 2.34 m and cropped nothing.

| merge radius | object points | object surface coverage | board plane rms | wall time |
|---|---|---|---|---|
| 0.0002 (the paper's fixed value) | 15 125 | 1.0× | 1.74 mm | 190 s |
| **0.001 (what auto picks)** | **100 359** | **3.3×** | 1.90 mm | **142 s** |
| 0.001 + `--reconstruction-interval 4` | 203 703 | 4.5× | 1.96 mm | 487 s |
| 0.0005 | 88 545 | 2.6× | 1.80 mm | 151 s |
| 0.002 | 69 049 | 3.6× | 2.02 mm | 130 s |
| `--min-votes 0` (no filtering at all) | 633 043 | 6.9× | 1.83 mm | 1.46 GB file ⚠ |

The auto radius gives **6.6× the object points in less time than the old fixed value**, because deduplicating to a 1 mm grid also shrinks the merge. The board is barely affected (+9 % plane rms). `--min-votes 0` is not the answer: it disables noise rejection entirely and writes a 1.46 GB ASCII file.

**Then, before step 4:** ball-pivoting uses `radius = 1.5 × mean_nn`, which at ~1.2 mm spacing lands *inside* the 1.5–1.8 mm noise shell, so the ball falls through and the mesh shatters into thousands of components. In CloudCompare, **subsample the segmented object to 2 mm spacing** (Edit ▸ Subsample ▸ Space, 0.002) before saving `object.ply`. With that, step 4 auto-picks a ~3.3 mm ball and the largest component covers **93 %** of the object (121 900 mm² of a ~110 000 mm² object) instead of 20 %.

**If you re-capture,** the only change that matters is raising the object's IR return — matte light-grey spray or a matte white paper drape for the geometry pass takes dropout from 68 % toward the board's 7 %. More frames and better poses do not help. Secondary: move the camera to ~0.5 m (halves the pixel pitch) and add a second, higher viewpoint so the top faces are not always at grazing incidence.

### If it is still drifting

The remaining lever is the RANSAC inlier threshold, which must stay **below your sequence's inter-frame corner motion**. Faster camera motion tolerates a larger value; a very slow capture needs a smaller one (`registration.match_ransac_robust(inlier_dist=...)`, currently 3 mm). With the corner filter plus a correctly sized threshold, the measured per-edge rotation error was **0.09°** and the pure odometry chain — no pose graph at all — closed the sequence's loop to 0.96° / 12.2 mm, a ~150× improvement over the original 150° / 890 mm.

The principled long-term fix, not implemented here because it changes what the pipeline is, would be to drop depth from the pose step entirely and solve the board pose by 2D PnP / bundle adjustment over the marker corners. That is what the reference used, and it reaches the same 0.09° per edge without reading depth at the corners at all.

---

## 7. Automatic object segmentation (steps 3a/3b)

### The problem with the manual cut

Everything the pipeline produces after step 3 is derived from `object.ply`: step 4 fits an oriented bounding box to it, puts its centroid on the origin, and projects the eight corners of that box into every frame as the label keypoints. So the hand cut in CloudCompare is not a cosmetic step — it *defines* the ground truth. Two ways it goes wrong, both invisible at the time:

- **Cutting too wide.** A slab of table top or a few marker-board points left attached extend the OBB. Every label keypoint in every frame moves, consistently, and the network learns the wrong box.
- **Cutting too tight.** Shaving the object's silhouette loses geometry, shrinks `diam` in the cfg (which drives the evaluation threshold), and biases the box the other way.

Neither shows up in a viewer, because the wrong cloud looks exactly as clean as the right one. And the cut is made **once, from one viewpoint**, on a merged cloud in which the object and the table touch.

### What replaces it

The decision is moved from "where do I drag the selection box" to "which object is it", and then measured from the data:

```
3a: one prompt on one frame  ──►  a mask of that object in every frame
3b: every frame votes on every point  ──►  object.ply
```

**3a — Grounding DINO + SAM 2.** A free-text phrase becomes a box (Grounding DINO is open-vocabulary; the object does not have to be a class anything was trained on), the box becomes a pixel-accurate mask (SAM 2.1), and SAM 2's video predictor propagates that mask across the sequence with its memory bank — forwards *and* backwards from the seed frame, so the seed can be any frame where the object is large and unoccluded rather than necessarily frame 0.

Design choices worth stating, because the obvious alternatives are worse here:

- *Propagation, not per-frame detection.* Detecting independently in every frame and matching the detections (Hungarian/IoU across frames, optionally with dense point tracking) is what a multi-object, objects-entering-and-leaving pipeline needs. This capture is one static object filmed from a moving camera: SAM 2's own memory is both simpler and more accurate, and it does not depend on the text prompt firing in every single frame.
- *`transformers`' Grounding DINO, not the original repository.* The original ships a custom CUDA extension that has to be compiled against the local toolkit. The `transformers` implementation is pure PyTorch, so `uv pip install transformers` is the whole installation on any machine.
- *The ArUco quads are subtracted from every mask.* They are the one region we can positively identify as *not* the object, and a mask that leaks a few millimetres onto the board drags board points into the cloud, which is exactly the "cutting too wide" failure above.
- *The track is checked, not trusted.* Every `--verify-stride` frames the text prompt is re-run and its box compared with the propagated mask (IoU). Disagreements are reported by frame number; `--reanchor` re-prompts SAM 2 at the first bad frame and re-propagates from there, up to `--max-reanchors` times. Empty masks and masks far off the median area are reported too, and `seg_meta.json` keeps all of it.

**3b — the vote.** For each frame and each candidate point (poses come from step 2, so the projection is exact):

| Situation | What the frame does |
|---|---|
| point projects outside the image, or behind the camera | nothing |
| measured depth at that pixel is 0 (no return) | nothing |
| point is **further** than the measured depth by more than `--depth-tol` | **abstains** — it is occluded in this view, which is not evidence against it |
| point lies on the measured surface, inside the mask | votes **object** |
| point lies on the measured surface, outside the mask | votes **background** |

A point is kept if at least `--min-observations` frames voted object *and* at least `--min-ratio` of the frames that voted on it agreed. The occlusion abstention is what makes this work at all: without it, every frame in which the object hides a table point would vote that table point away — and, worse, every frame in which the object's own far side is hidden would vote the object away.

Then: largest connected blob (`--cluster-eps`), statistical outlier removal, and a downsample to `--voxel` 2 mm — the same 2 mm the manual instructions asked you to do in CloudCompare, and which §6 shows step 4's ball pivoting depends on.

Two sources, both supported:

- **`--source scene`** (default when `registeredScene.ply` exists) crops step 3's merged cloud. The result is exactly the subset of points a perfect hand cut would have kept.
- **`--source depth`** skips step 3 entirely and fuses only the masked depth of each frame, reusing `3_register_scene.py`'s own vote merge (imported by path, so the two cannot drift apart). Denser on the object, because step 3's scene-wide filter and marker-board crop are tuned for a whole room rather than for one object — and much cheaper, since it never touches the ~95 % of each frame that is not the object.

### Measured on the `box` sequence (930 frames, 1280×720, Azure Kinect)

3a, `--prompt "white product box" --stride 8 --reanchor`, one A100-40GB: 117 frames segmented in **~90 s** wall clock including model load, the 38-frame verification pass, mask writing and previews (frame encoding 5 s, propagation 9 s, verification 9 s). Every checked frame agreed with the text prompt — **mean IoU 0.94, worst 0.88** — zero empty masks, zero area outliers, and no repair was triggered.

3b, both sources, defaults:

| | scene crop | depth fusion |
|---|---|---|
| input points | 2 216 003 (registeredScene) | 1 158 455 (fused masked depth) |
| survived the vote | 785 721 (35.5 %) | 725 933 (62.7 %) |
| after 2 mm downsample + outlier removal | **491 922** | **479 118** |
| oriented bounding box | **46.7 × 26.5 × 13.1 cm** | **47.2 × 28.1 × 13.4 cm** |

The two routes share only the masks and the poses — one crops a merged cloud, the other rebuilds one from raw depth — and they agree to **≤1.6 cm on every axis**. That agreement is the useful check: a segmentation error large enough to matter would move them apart.

### When it fails, and what to do

- **Prompt finds nothing / the wrong thing.** Lower `--box-threshold`, reword (colour + material + a spatial cue works better than a bare noun), or inspect the printed candidate list and pass `--detection-index`. `--seed-only` iterates on this in seconds instead of minutes.
- **The track drifts onto the table or a neighbouring object.** The verification pass names the frames. Re-seed from a better frame (`--seed-frame`), add a background click (`--click X,Y,0`), or run `--reanchor`.
- **`object.ply` comes out sparse or hollow.** That is a depth problem, not a mask problem — see §6: dark and glossy surfaces return no depth at all, and no mask invents points the sensor never measured. `--source depth` recovers some of it; a matte coating recovers the rest.
- **The OBB is visibly inflated.** Something survived the vote. Raise `--min-ratio`, raise `--erode` (silhouette pixels mix object and background depth), or raise `--min-observations`.
- **Nothing survives the vote.** Usually `--depth-tol` is smaller than the disagreement between the poses and the depth. The message says so, and prints how many points no frame ever saw on a surface.

The manual route is untouched and still documented: if the object defeats all of the above, cut it in CloudCompare and step 4 will not know the difference.

---

## 8. The studio (`run_e2e_web.py`)

The numbered scripts stay the single source of truth — each one still runs on
its own exactly as documented above. What was missing was the glue: a full
dataset build is six commands in the right order, with the only human decision
(what the object *is*, for step 3a) buried in the middle, and re-runs after a
change re-doing work that was already up to date.

`run_e2e_web.py` drives the whole pipeline from one browser page, served by the
standard library's `http.server` — the same recipe as the 3a picker, and for
the same reason: this pipeline is normally driven over SSH on a display-less
GPU box (`ssh -N -L 8770:localhost:8770 <host>` from your machine, or `--open`
locally). The engine lives in `utils/pipeline.py` — the stage registry,
skip/stale logic, subprocess runner, prompt handling and preview collection —
so the page stays presentation-only (`utils/e2e_page.py`).

Step 6 never opens a window here: the same overlays are rendered headlessly
into `qa_preview/` and played back in the page.

### One argument, three modes, prompts only when needed

Both take just the sequence folder, like every other script. `--mode` selects
what to run:

- `whole` (default) — steps 2→6;
- `label` — steps 2→5 (everything but the visual QA);
- any comma list of stages — `--mode step3a,step3b`.

Before running anything they **plan**: a stage is skipped when all of its
outputs exist and none is older than any of its inputs, re-run otherwise, and
the plan cascades (if step 4 will re-run this run, step 5 is stale by
implication even though its files look newer *now*). The plan is shown with
the reason per stage — `objmask is newer than the outputs`, `labels will be
regenerated this run` — before anything starts, and `--dry-run` prints it and
exits. `--force` re-runs everything selected; `--force-from step3b` re-runs
from there onward.

The object prompt is requested **only when the plan says step 3a will actually
run** — a sequence whose masks are fresh never asks. All three ways of
specifying it work in both front-ends and on both command lines
(`--prompt "white product box"`, `--click X,Y[,L]` repeatable, `--box
X0,Y0,X1,Y1`, plus `--seed-frame N`), and map 1:1 onto 3a's own CLI — the
orchestrator has no private hooks into 3a. A text detection's box is only ever
*drawn* in the preview, never adopted as the prompt, so text runs keep
`--reanchor` and the periodic re-verification available.

`--viz off|preview|full` controls feedback: `off` is text only, `preview`
shows each stage's key images as it finishes (3a's seed/contact sheet, 3b's
cloud reprojections, the QA overlays), `full` additionally plays the QA
overlays through at the end.

Everything else a stage accepts passes through untouched:

```bash
python run_e2e_web.py mycup --extra 'step3a: --stride 8 --reanchor' --extra 'step2: --icp-method colored-icp'
```

### Reproducibility

Every run writes `<sequence>/e2e_log/run_<stamp>/` containing `manifest.json`
(dataset, steps, prompt, per-stage durations and results, argv) and one
`<stage>.log` per stage; each stage's exact command is echoed shlex-quoted
before it runs, so any stage can be reproduced from a terminal by copy-paste.
Inapplicable requests are downgraded with a printed reason (e.g. `--extra
'step3a: --reanchor'` without a text prompt is dropped and says so);
impossible ones fail before anything runs.

### Limits

- One run at a time per server (the web page refuses a second Run while one
  is active). The picker (`3a --pick`) and the dashboard are separate servers
  on separate ports (8765 / 8770).
- The web server binds `127.0.0.1` and has no authentication, same policy as
  the picker: forward the port, do not expose it.
- The live mask preview needs the optional sam2/transformers stack; without
  it, clicks and boxes still work — they are simply passed to 3a unpreviewed.
- Step 1 (recording) is not orchestrated: it needs the Kinect attached and has
  its own interactive flow.

---

## 9. GPU acceleration (`--gpu`, inside steps 2, 3 and 3b)

Steps 2 and 3 dominate the pipeline's wall clock (the box sequence: ~20 min +
~18 min, versus ~90 s for the SAM-based step 3a). Each step is **one script
with two paths**: the CPU implementation is the default and remains the
reference, `--gpu N` selects the accelerated one, and both write the same
outputs and print a `[timing]` breakdown. The studio passes `--gpu` through
when you start it with `--gpu`.

Requires Open3D's CUDA build (the pinned `open3d==0.19.0` wheel on Linux ships
it; a script asked for `--gpu` without a usable CUDA device says so and exits
rather than falling back silently).

### Where the time actually went (box: 930 frames, 1280×720, A100 + 48-core host)

**Step 2** spends almost nothing on pose estimation itself — the ArUco corners
give each edge's transform nearly for free. The per-frame ~1.3 s went into
*weighing* those edges: every edge needs both endpoint clouds decoded,
back-projected and voxel-downsampled (and the loop-closure target was loaded
and thrown away every time, because the cache only kept the odometry pair),
then a CPU nearest-neighbour pass builds the information matrix.

**Step 3** decoded each 16-bit depth PNG with `pypng` — a pure-Python decoder —
and then rebuilt a KD-tree over the *growing* merged cloud for every segment
(3 s/it at segment 20, 13 s/it at segment 110: the classic quadratic
growing-array pattern; the merge alone was 922 s of the 1092 s).

### What changed

| bottleneck | replacement | where |
|---|---|---|
| loop-closure clouds re-decoded per edge | LRU cache of downsampled clouds on the GPU (~7 MB each; 1367 loads vs 2090 cache hits on box) | step 2 |
| CPU decode → back-project → `voxel_down_sample` | threaded JPEG/PNG decode, vectorised back-projection, CUDA `voxel_down_sample` | step 2 |
| information matrix (CPU NN pass per edge) | `o3d.t.pipelines.registration.get_information_matrix` on CUDA — verified to match the legacy CPU function to ~1e-7 relative on identical inputs | step 2 |
| marker detection, serially per frame | precomputed for all frames on a thread pool (~9 s for 930) | step 2 |
| ICP fallback (markers failed) | **unchanged**: converts the cached clouds to legacy CPU and calls the original `utils.registration.icp` — bit-identical path, it fired on 3 of 1857 edges | step 2 |
| `pypng` frame decode | `cv2.imread(IMREAD_UNCHANGED)` on threads (identical pixels — steps 2/3b already decode the same files this way) | step 3 |
| growing-array KD-tree per segment | Open3D CUDA `NearestNeighborSearch.hybrid_search` (nearest neighbour within the vote radius); merge/vote rules kept bit-for-bit, including the one-vote-per-unique-neighbour fancy-indexing semantics | step 3 |

### Measured (box, single A100, `/usr/bin/time` wall clock)

| | CPU original | GPU version | speed-up |
|---|---|---|---|
| step 2 | 1212.5 s (1.304 s/frame) | **198.5 s (0.213 s/frame)** | **6.1×** |
| step 3 | 1091.6 s (9.41 s/segment) | **375.8 s (3.24 s/segment)** | **2.9×** |
| 2 + 3 together | 2304 s (38.4 min) | **574 s (9.6 min)** | **4.0×** |

GPU step 2's remaining wall clock is almost entirely frame decode +
downsample (~200 thread-seconds over 8 threads); `--io-threads` raises the
overlap if the disks keep up. GPU step 3's remaining cost is the CUDA index
rebuild over the accumulated cloud per segment — still growing with scene
size, but ~30× cheaper per rebuild than the CPU tree.

### Accuracy: what the speed-up costs

Nothing measurable. Two independent checks, plus a control:

- **Control** — re-running the CPU original reproduces the round-2
  `transforms.npy` *exactly* (max pose delta 0.0000° / 0.000 mm), so any GPU
  difference is real, not rerun noise.
- **Step 2, GPU vs CPU (930 poses)**: rotation median 0.0006°, p95 0.008°,
  max **0.0197°**; translation median 0.005 mm, p95 0.11 mm, max **0.27 mm**.
  The only source is the CUDA vs legacy `voxel_down_sample` grid details
  feeding the edge weights; the deltas sit far below sensor depth noise and
  the 2 mm downsample every consumer applies.
- **Step 3, GPU vs CPU merged scene**: 2,215,997 vs 2,216,003 points
  (**6 of 2.2 M differ**, boundary ties at the merge radius); cloud-to-cloud
  distance median 0.0000 mm, p99 0.001 mm; oriented bounding boxes identical
  to 0.1 cm in every axis.

### Running them

```bash
python 2_compute_gt_poses.py mycup --gpu 0      # same script, GPU path
python 3_register_scene.py   mycup --gpu 0
python 3b_segment_object_cloud.py mycup --gpu 0
python run_e2e_web.py        mycup --gpu 0      # the studio, GPU throughout
```

`--gpu` takes an optional device index (bare `--gpu` = device 0); omit it and
the CPU path runs. The GPU paths add `--io-threads` (and `--chunk` in 3b),
all defaulting to `auto` — sized from the CPU count and free GPU memory, so
there is nothing to tune by hand. Step 4 needs no flag at all: its rewrite is
CPU-side vectorisation and is simply the implementation now.

### Second optimisation pass (deeper profile, same guarantees)

The first pass left three misconceptions that a cProfile of the *optimised*
scripts exposed; fixing them, plus accelerating two more steps, is another ~4×
on top:

1. **Step 2's real remaining bottleneck was not the GPU work but
   `match_ransac_robust`** — 158 s of the 198 s wall: 200 RANSAC hypotheses
   per odometry edge, each a separate Python `np.matrix` SVD fit. The GPU path
   now
   draws the *same 200 index triplets from the same seeded generator in the
   same order* and runs centroids/H/SVD/residuals as one batch —
   **output-identical on all 929 edges of box** (verified transform-for-
   transform), 104.8 s → 9.6 s for the estimator itself.
2. **The back-projection was numpy-bound** (71 ms of the 110 ms per frame).
   It now runs in torch on the GPU (upload raw depth+JPEG, not unpacked
   xyz+rgb, so the transfer shrinks too); decode threads (`auto` from the CPU
   count) feed it, and the cloud cache sizes itself from free GPU memory
   (`auto`), which drops loads to the 930-frame lower bound.
3. **Step 3's merge was dominated by implementation overhead**, not by the
   NN search: per-segment `concatenate` re-copied the whole accumulated cloud
   (~22 GB over the run), everything round-tripped to numpy, and the index was
   rebuilt from scratch every segment. The merged cloud now lives in one
   preallocated GPU buffer, votes/distances stay on the GPU (zero-copy dlpack
   views), and the index is rebuilt only when the un-indexed tail exceeds
   max(1 M, 25 %) — queries take the min over (main index, tail index), which
   is *exactly* the nearest neighbour, not an approximation. Verified: the
   full 116-segment A/B against the first-pass algorithm produces
   **bit-identical accumulation and votes**; the script is deterministic
   across runs.

Two more steps joined the `--gpu` toggle:

- **step 3b** — the per-frame vote loop as torch
  tensor ops (project → gather depth/mask → abstain/object/background as
  boolean masks). Vote loop ~87× (96 s → 1.1 s); wall 137 s → 45 s. All the
  post-vote filtering is shared with the CPU path in the same file, and
  `--source depth` reuses step 3's merge by path-import (so the vote-merge
  exists exactly once).
- **step 4** — no GPU at all; it was pure
  algorithmic waste: it re-sampled the same mesh 930 times (sample once,
  transform per frame — the baseline is itself RNG-nondeterministic, so this
  is statistically identical), rasterised the mask with 100 000 Python
  `cv2.circle` calls per frame (→ plot points + dilate with the *exact*
  11×11 circle kernel, unit-tested pixel-identical, border handled by
  padding), and decoded a JPEG per frame only to read its shape. Label loop
  ~100×; `--workers auto` splits frames across processes.

### Measured, second pass (box, one A100, quiet machine)

| stage | CPU original | accelerated | speed-up | equivalence (vs CPU, with a CPU-rerun noise floor) |
|---|---|---|---|---|
| step 2 | 1212.5 s (1.304 s/frame) | **92.2 s (0.099 s/frame)** | **13.2×** | rot ≤ 0.004°, trans ≤ 0.045 mm over 930 poses (CPU rerun floor: exactly 0); RANSAC edges transform-identical |
| step 3 | 1091.6 s (9.41 s/segment) | **61.5 s (0.53 s/segment)** | **17.8×** | ±0.005 % of 2.2 M points, c2c p99 0.055 mm; downstream object OBB within 0.1 cm per axis |
| step 3b | 137.2 s | **45.3 s** | **3.0×** | 12 of 2.2 M vote flips; object OBB Δ ≤ 0.13 cm (scene), identical (depth) |
| step 4 | 609.1 s (0.65 s/frame) | **76.8 s (0.083 s/frame)** | **7.9×** | mask IoU vs CPU = the CPU's own rerun noise floor (0.9925, equal to 5 decimals); poses/transforms/meshes bit-identical |
| steps 5+6 | 4.6 s | (unchanged) | — | — |

The studio's `--gpu` flag passes `--gpu` down to steps 2/3/3b (step 4 is
always the fast implementation),
picks the single CUDA device with the most free memory once per run
(`nvidia-smi`), and grants it to each child via `CUDA_VISIBLE_DEVICES`;
a per-stage `--gpu`/`--device` given through `--extra` overrides it. Step 3a is additionally run **in
parallel with the step2→step3 branch** (it depends only on `JPEGImages`;
both share the one GPU fine) — `--no-parallel` restores strict order.
The whole-label-path wall time appears in the run manifest of each run.

Measured end to end on box (930 frames, raw inputs → train-ready folder,
label mode, `--gpu`, one A100): **289 s wall** — step3a (80 s, stride 8,
re-verified, all frames ok) fully hidden under step2→step3 (166 s), then
3b 40 s + step4 74 s + step5 3 s. The same path on the CPU originals sums to
~52 minutes: **~11× end to end**, with `diam` agreeing with the reference
run to 4 µm (0.523145 vs 0.523141) and the object cloud to 9 points in 492 k.

### Mask confirmation (added after first real use)

Two rules were added to both front-ends after user feedback:

1. **The prompt panel is available whenever step 3a is part of the selected
   mode** — not only when 3a is about to run. Fresh masks show their current
   `seed.jpg` / `contact.jpg` next to the picker, so the existing mask state
   is visible before deciding anything; picking a new prompt on fresh masks
   automatically re-runs from step 3a (3a→3b→4→5), announced in the log.
2. **No unseen mask ever propagates.** Any interactively picked prompt must
   be previewed (the SAM mask drawn over the frame) and explicitly accepted —
   the web page's "use this mask" button — before Run is allowed; editing the prompt in any way (click, box, text,
   frame change) revokes the confirmation. What runs is exactly the confirmed
   spec. Command-line prompts (`--prompt/--click/--box`, and `--yes`
   automation) are treated as explicit decisions and skip the gate.

### GPU selection: `--gpu`

The front-ends take a single `--gpu` argument and nothing else GPU-related:

| value | meaning |
|---|---|
| *(omitted)* | CPU only — original numbered scripts, live preview on CPU |
| `--gpu 0` | GPU 0 hosts everything: the live mask preview and every accelerated stage |
| `--gpu 0,1` | as above, plus step 3a runs on GPU 1 while it overlaps steps 2–3 |
| bare `--gpu` | like `--gpu N` with the freest device chosen once at startup |

Multi-GPU honesty: the second index buys only the removal of the one
device-sharing window (step 3a ∥ steps 2–3), measured at a few percent of
the end-to-end wall time — the rest of the pipeline is sequential or
CPU-bound, so more GPUs do not help further. Per-stage overrides remain
possible with `--extra 'step2: --gpu 3'`. `--gpus` is accepted as an alias spelling; a bare `--gpu` (the old flag)
means 'auto'.

### The result 3D viewer

When a run finishes, the dashboard's result view splits in two: previews
(masks, 6-DoF overlays, QA images) on the left, and a **3D viewer on the
right** showing the sequence's `.ply` files — the final `<ds>.ply` mesh by
default, with `object.ply` / `registeredScene.ply` / `mesh_world.ply`
selectable. Orbit by dragging, zoom with the wheel, pan with shift- or
right-drag, double-click to reset.

No dependencies were added: the viewer is ~150 lines of raw WebGL inside the
page, and the server packs each cloud as `uint32 count + float32 xyz +
uint8 rgb` (the 87 MB scene becomes a 6 MB transfer at the 400 k-point
subsample cap). The first load of a file parses the ASCII PLY server-side
(~3-12 s); it is cached against the file's mtime afterwards.
