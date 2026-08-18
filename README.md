# pose-dataset-studio

**Record → click the object once → train-ready 6-DoF pose dataset.**

An RGB-D dataset studio that fully automates [ObjectDatasetTools](https://github.com/F2Wang/ObjectDatasetTools)-style dataset creation: **Grounding DINO + SAM 2** replace the manual point-cloud segmentation, every heavy step has a GPU path (**~5 min instead of ~52** for a 930-frame sequence), and a **browser dashboard** runs the whole pipeline from one command.

This is the extended successor of [dataset-toolchain](https://github.com/inyoungoh-cde/mr-remote-collaboration/tree/main/dataset-toolchain), built for:

> Oh, I., Jang, G., Song, J., Son, M., Kim, D., Yun, J., Ko, K., *"A mixed reality-based remote collaboration framework using improved pose estimation"*, **Computers in Industry** 174 (2026) 104414. [[link]](https://www.sciencedirect.com/science/article/pii/S0166361525001794)

<img src="doc/pick_to_mask.gif" width="640"/>

*One prompt on one frame — SAM 2 carries the mask through the whole orbit.*

<img src="doc/labels_result.gif" width="420"/> <img src="doc/object_3d.gif" width="300"/>

*What you get: a projected 3D box + pixel mask on every frame, and the reconstructed object.*

## Quick start

Python 3.10. CPU pipeline runs on Windows and Linux; the auto-segmentation / GPU stack targets Linux + CUDA.

```bash
pip install -r requirements.txt                     # core pipeline + studio
pip install "sam2==1.1.0" "transformers==5.15.0"    # auto-segmentation + GPU paths
pip install --no-deps pykinect-azure                # recording only (needs Azure Kinect SDK)
```

The two segmentation checkpoints (~1.6 GB) download from the Hugging Face hub on first run and are cached afterwards.

No Kinect on hand? [`dig_camera.zip`](https://github.com/inyoungoh-cde/pose-dataset-studio/releases/tag/pose-dataset-studio-sample-v1) is a real 529-frame capture, ready for the pipeline:

```bash
unzip dig_camera.zip -d pose-dataset-studio/
cd pose-dataset-studio
python run_e2e_web.py dig_camera --gpu
```

That one command opens the studio in your browser: click the object once (live SAM mask preview), press run, and it takes the sequence all the way to a train-ready folder — labels, masks, mesh, train/test split and cfg, laid out exactly as [singleshotpose](https://github.com/microsoft/singleshotpose) expects. On a headless server, forward the port first: `ssh -N -L 8770:localhost:8770 <host>`.

<img src="doc/studio_pick.jpg" width="820"/>

*Pick the object, see the SAM mask instantly — nothing runs until you confirm it.*

<img src="doc/studio_run.jpg" width="820"/>

*While it runs: per-stage progress, previews as they appear.*

<img src="doc/studio_result.jpg" width="820"/>

*When it finishes: QA overlays and the reconstructed object in a 3D viewer.*

The studio plans before it runs — a stage only executes when its outputs are missing or stale, so re-running redoes exactly what changed. `--gpu` picks the freest CUDA device (`--gpu 0` to pin one); omit it to stay CPU-only.

## What's new vs dataset-toolchain

The numbered scripts and the sequence-folder layout are the same as [dataset-toolchain](https://github.com/inyoungoh-cde/mr-remote-collaboration/tree/main/dataset-toolchain) — marker-board preparation and the step-by-step reference live there. This repo adds:

| | |
|---|---|
| `3a_segment_object_masks.py` | Name the object once — `--prompt "white mug"`, `--pick` (browser picker), or `--click X,Y` / `--box` — and Grounding DINO finds it, SAM 2 carries the mask through the whole sequence, forwards and backwards from the seed frame. Replaces the manual segmentation step entirely. |
| `3b_segment_object_cloud.py` | Every frame votes on every 3D point of the registered scene: occluded views abstain, wrong masks are outvoted → `object.ply` with no hand cutting. |
| `run_e2e_web.py` | The studio: one command, browser UI, stage planning, parallel execution, previews, 3D result viewer. |
| `--gpu` on steps 2 / 3 / 3b | GPU path inside the same script — CPU stays the default and produces equivalent output. |

Two quick sanity checks after a run: `seg_preview/contact.jpg` shows the tracked mask on the whole sequence at a glance, and step 3b prints the object's bounding box in centimetres — compare it with a ruler and you know immediately whether the segmentation was right.

<img src="doc/auto_segment_object.jpg" width="720"/>

## Notes

- Prefer moving the camera around a static object; a turntable also works (the pipeline auto-detects which one you did).
- Dark or glossy objects return little depth — a matte coating improves reconstruction more than extra frames do. Segmentation cannot invent points the sensor never measured.

## Citation

If you use this toolchain, please cite:

```bibtex
@article{oh2026mixed,
  title   = {A mixed reality-based remote collaboration framework using improved pose estimation},
  author  = {Oh, Inyoung and Jang, Gilsang and Song, Jinho and Son, Moongu and Kim, Daewoon and Yun, Junsang and Ko, Kwanghee},
  journal = {Computers in Industry},
  volume  = {174},
  pages   = {104414},
  year    = {2026},
  doi     = {10.1016/j.compind.2025.104414}
}
```

## License

[MIT](LICENSE). Derived from [ObjectDatasetTools](https://github.com/F2Wang/ObjectDatasetTools) (MIT, © 2019 s0972456), whose copyright notice is retained in the LICENSE file. Built to train [singleshotpose](https://github.com/microsoft/singleshotpose) (MIT, Microsoft). The automatic segmentation uses [SAM 2](https://github.com/facebookresearch/sam2) (Apache-2.0, Meta) and [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) via [transformers](https://github.com/huggingface/transformers) (Apache-2.0) as pip dependencies — nothing from either project is vendored into this repository.
