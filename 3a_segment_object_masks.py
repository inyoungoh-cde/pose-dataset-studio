"""
3a_segment_object_masks.py
--------------------------

Say which object you want -- once -- and get a pixel mask of it in every frame.

This is the first half of the automatic replacement for the manual CloudCompare
step. It never touches 3D: it produces `objmask/`, a per-frame binary mask of
the target object, which `3b_segment_object_cloud.py` then turns into
`object.ply` using the poses from step 2.

The object is named in one of three ways, and none of them needs a display:

    # text, fully scriptable -- Grounding DINO is open-vocabulary, so the
    # phrase does not have to name a class anything was trained on
    python 3a_segment_object_masks.py box --prompt "white product box"

    # a browser page you click on, served over an SSH tunnel (see --pick)
    python 3a_segment_object_masks.py box --pick

    # coordinates you already know (from --pick, or from any image viewer)
    python 3a_segment_object_masks.py box --click 660,300 --click 520,250,0
    python 3a_segment_object_masks.py box --box 491,199,848,401

Whatever the prompt, only the seed frame is prompted; SAM 2's video predictor
carries the mask through the rest of the sequence with its memory bank, both
forwards and backwards from the seed. Add `--reanchor` to have the text prompt
re-checked periodically and the track repaired if it has drifted.

Outputs, all inside the sequence folder:

    objmask/%06d.png    binary mask per frame (255 = object)
    objmask/seg_meta.json   the prompt, the models, per-frame areas, warnings
    seg_preview/*.jpg   overlays every --preview-stride frames + contact.jpg
"""

import glob
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

from utils.cli import build_parser, resolve_dataset
from utils import segmentation as seg


def frame_ids(path):
    """Sorted frame numbers present in JPEGImages."""
    files = glob.glob1(os.path.join(path, "JPEGImages"), "*.jpg")
    ids = []
    for f in files:
        stem = os.path.splitext(f)[0]
        if stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def read_frame(path, fid):
    img = cv2.imread(os.path.join(path, "JPEGImages", "%06d.jpg" % fid))
    if img is None:
        sys.exit("Could not read %sJPEGImages/%06d.jpg" % (path, fid))
    return img


def parse_click(values):
    """--click x,y[,label] (repeatable) -> (points (n,2), labels (n,))."""
    pts, labels = [], []
    for v in values or []:
        parts = [p for p in v.replace(";", ",").split(",") if p.strip() != ""]
        if len(parts) not in (2, 3):
            sys.exit("--click expects x,y or x,y,label (1 = object, 0 = background), got %r" % v)
        pts.append([float(parts[0]), float(parts[1])])
        labels.append(int(parts[2]) if len(parts) == 3 else 1)
    return (np.array(pts, dtype=np.float32), np.array(labels, dtype=np.int32)) if pts else (None, None)


def parse_box(value):
    if not value:
        return None
    parts = [p for p in value.replace(";", ",").split(",") if p.strip() != ""]
    if len(parts) != 4:
        sys.exit("--box expects x0,y0,x1,y1, got %r" % value)
    x0, y0, x1, y1 = (float(p) for p in parts)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def clean_mask(mask, bgr, args):
    """Speckle removal + marker subtraction, applied to every propagated mask."""
    if args.largest_component:
        mask = seg.largest_component(mask)
    if not args.keep_markers:
        mask = mask & ~seg.marker_mask(bgr, args.marker_dilate)
        if args.largest_component:
            mask = seg.largest_component(mask)
    return mask


def contact_sheet(previews, cols=4, width=1600):
    """One image summarising the whole sequence, for a quick look over SSH."""
    if not previews:
        return None
    rows = int(np.ceil(len(previews) / float(cols)))
    tw = width // cols
    th = int(round(tw * previews[0][1].shape[0] / float(previews[0][1].shape[1])))
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, (fid, img) in enumerate(previews):
        thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, "%06d" % fid, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(thumb, "%06d" % fid, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb
    return sheet


def build_frame_dir(path, frames, tmp):
    """SAM 2 loads a video from a folder of frames named <number>.jpg.

    With --stride 1 that is JPEGImages itself; otherwise a folder of symlinks
    renumbered 0..n-1 is handed over instead, so only the frames we actually
    want are ever encoded.
    """
    src = os.path.join(path, "JPEGImages")
    if frames == list(range(len(frames))) and len(frames) == len(frame_ids(path)):
        return src
    for k, fid in enumerate(frames):
        dst = os.path.join(tmp, "%06d.jpg" % k)
        try:
            os.symlink(os.path.abspath(os.path.join(src, "%06d.jpg" % fid)), dst)
        except (OSError, NotImplementedError):
            shutil.copyfile(os.path.join(src, "%06d.jpg" % fid), dst)
    return tmp


def main():
    parser = build_parser(
        "Text- or click-prompted object masks for the whole sequence (SAM 2 + Grounding DINO)")
    prompt = parser.add_argument_group("what to segment (pick one)")
    prompt.add_argument("--prompt", default=None,
                        help="free-text description of the object, e.g. 'white product box'. "
                             "Open-vocabulary: it does not have to be a trained class name")
    prompt.add_argument("--click", action="append", default=None, metavar="X,Y[,L]",
                        help="click coordinate on the seed frame; L=1 (default) means the "
                             "object, L=0 means push this region out of the mask. Repeatable")
    prompt.add_argument("--box", default=None, metavar="X0,Y0,X1,Y1",
                        help="box around the object on the seed frame, in pixels")
    prompt.add_argument("--pick", action="store_true",
                        help="choose interactively in a browser page served on localhost -- "
                             "works over SSH with 'ssh -N -L 8765:localhost:8765 host', and "
                             "locally by just opening the URL. Clicks, box and text prompt are "
                             "all available there, with the mask redrawn after every action")
    parser.add_argument("--pick-port", type=int, default=8765, help="port for --pick")
    parser.add_argument("--pick-host", default="127.0.0.1",
                        help="interface for --pick. Localhost by default: forward the port "
                             "rather than exposing the page to the network")
    parser.add_argument("--pick-open", action="store_true",
                        help="also launch a local browser (only useful if this machine has one)")

    parser.add_argument("--seed-frame", type=int, default=0,
                        help="the one frame the prompt refers to. The mask is propagated both "
                             "forwards and backwards from it, so a mid-sequence frame where the "
                             "object is large and unoccluded is often the best choice")
    parser.add_argument("--stride", type=int, default=1,
                        help="segment every Nth frame. 1 = every frame (needed if you want the "
                             "masks for anything else); step 3b only needs the frames it fuses, "
                             "so a stride matching its --interval is enough and is N times "
                             "faster and lighter on memory")
    parser.add_argument("--seed-only", action="store_true",
                        help="stop after the seed frame's mask and write only its preview -- "
                             "the fast way to iterate on a prompt before committing to the "
                             "whole sequence")

    parser.add_argument("--box-threshold", type=float, default=0.30,
                        help="Grounding DINO box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Grounding DINO token confidence threshold")
    parser.add_argument("--detection-index", type=int, default=0,
                        help="which text detection to use, 0 = highest scoring. All candidates "
                             "are printed and drawn into seg_preview/seed.jpg")

    parser.add_argument("--reanchor", action="store_true",
                        help="re-run the text prompt every --verify-stride frames and, where "
                             "the propagated mask no longer matches the detection, re-prompt "
                             "SAM 2 at that frame and re-propagate from there. Needs --prompt")
    parser.add_argument("--verify-stride", type=int, default=30,
                        help="check the track against the text prompt every N frames "
                             "(0 disables the check). Reporting only unless --reanchor")
    parser.add_argument("--verify-iou", type=float, default=0.5,
                        help="report (or, with --reanchor, repair) a frame whose propagated "
                             "mask overlaps the detected box by less than this")
    parser.add_argument("--max-reanchors", type=int, default=5,
                        help="give up after this many repairs, rather than fighting a prompt "
                             "that does not describe the object well")

    parser.add_argument("--keep-markers", action="store_true",
                        help="do not subtract the ArUco markers from the mask. They are the one "
                             "region we know is not the object, and a mask that leaks onto the "
                             "board drags the board into the object cloud")
    parser.add_argument("--marker-dilate", type=int, default=3,
                        help="dilate the subtracted marker quads by this many pixels")
    parser.add_argument("--no-largest-component", dest="largest_component",
                        action="store_false", help="keep every blob, not just the biggest one")
    parser.add_argument("--min-area-frac", type=float, default=2e-4,
                        help="a mask smaller than this fraction of the image counts as lost and "
                             "is reported (and written empty)")

    parser.add_argument("--sam2-model", default=seg.DEFAULT_SAM2, help="SAM 2 checkpoint on the HF hub")
    parser.add_argument("--detector-model", default=seg.DEFAULT_DETECTOR,
                        help="Grounding DINO checkpoint on the HF hub")
    parser.add_argument("--device", default="auto", help="cuda / cuda:1 / cpu / mps / auto")
    parser.add_argument("--offload-video", dest="offload_video", action="store_true", default=True,
                        help="keep the decoded frames in CPU RAM instead of VRAM")
    parser.add_argument("--no-offload-video", dest="offload_video", action="store_false",
                        help="keep them in VRAM (faster; ~13 MB of VRAM per frame at 720p)")
    parser.add_argument("--offload-state", action="store_true",
                        help="also keep SAM 2's memory bank on the CPU (slower, for long "
                             "sequences on a small GPU)")
    parser.add_argument("--preview-stride", type=int, default=30,
                        help="write an overlay preview every N frames (0 = none)")
    parser.add_argument("--out-dir", default=None,
                        help="where to write objmask/ and seg_preview/ (default: the sequence "
                             "folder, which is where step 3b expects them)")
    args = parser.parse_args()

    dataset, path = resolve_dataset(args, require=("JPEGImages",))
    out_path = path if args.out_dir is None else args.out_dir.rstrip("/") + "/"
    mask_dir = os.path.join(out_path, "objmask")
    prev_dir = os.path.join(out_path, "seg_preview")
    for d in (mask_dir, prev_dir):
        if not os.path.isdir(d):
            os.makedirs(d)

    ids = frame_ids(path)
    if not ids:
        sys.exit("No frames found in %sJPEGImages." % path)
    if args.stride < 1:
        sys.exit("--stride must be >= 1")
    frames = ids[::args.stride]
    if args.seed_frame not in frames:
        nearest = min(frames, key=lambda f: abs(f - args.seed_frame))
        print("Seed frame %d is not on the stride grid; using frame %d instead."
              % (args.seed_frame, nearest))
        args.seed_frame = nearest
    seed_k = frames.index(args.seed_frame)

    have_prompt = bool(args.prompt or args.click or args.box or args.pick)
    if not have_prompt:
        sys.exit("Tell the script which object to segment:\n"
                 "  --prompt \"white product box\"   text (open-vocabulary, scriptable)\n"
                 "  --pick                         click it in a browser page (works over SSH)\n"
                 "  --click X,Y / --box X0,Y0,X1,Y1  coordinates you already have\n"
                 "Run with --seed-only first if you want to check the prompt cheaply.")
    if args.reanchor and not args.prompt:
        sys.exit("--reanchor re-runs the text prompt, so it needs --prompt.")

    device = seg.pick_device(args.device)
    seg.enable_fast_math(device)
    print("Device: %s | SAM 2: %s" % (device, args.sam2_model))

    import torch

    seed_bgr = read_frame(path, args.seed_frame)
    seed_rgb = cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2RGB)
    h, w = seed_bgr.shape[:2]

    detector = None

    def get_detector():
        nonlocal detector
        if detector is None:
            print("Loading Grounding DINO (%s) ..." % args.detector_model)
            detector = seg.TextDetector(args.detector_model, device)
        return detector

    print("Loading SAM 2 image predictor ...")
    image_seg = seg.ImageSegmenter(args.sam2_model, device)

    def detect_box(rgb, text, index=0, quiet=False):
        dets = get_detector().detect(rgb, text, args.box_threshold, args.text_threshold)
        if not quiet:
            if not dets:
                print("Grounding DINO found nothing for %r at threshold %.2f."
                      % (text, args.box_threshold))
            else:
                print("Grounding DINO candidates for %r:" % text)
                for i, d in enumerate(dets):
                    print("  [%d] score %.3f  label %-24s box %s%s"
                          % (i, d["score"], d["label"],
                             " ".join("%7.1f" % v for v in d["box"]),
                             "   <- used" if i == index else ""))
        if not dets:
            return None, dets
        if index >= len(dets):
            sys.exit("--detection-index %d but only %d candidate(s) were found."
                     % (index, len(dets)))
        return dets[index]["box"], dets

    # ---------------------------------------------------------------- prompt
    spec = {"frame": args.seed_frame, "points": [], "box": None, "text": args.prompt or ""}

    if args.pick:
        from utils.picker import pick_prompt

        # The page steps through frames by position, so that a sequence with
        # gaps in its numbering cannot ask for a frame that does not exist.
        def frame_at(pos):
            return ids[max(0, min(len(ids) - 1, int(pos)))]

        def preview_fn(req):
            fid = frame_at(req.get("frame", 0))
            bgr = read_frame(path, fid)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            box = req.get("box")
            pts = req.get("points") or []
            text = (req.get("text") or "").strip()
            if text and not pts and not box:
                box, dets = detect_box(rgb, text, args.detection_index, quiet=True)
                if box is None:
                    return np.zeros((bgr.shape[0], bgr.shape[1]), bool), \
                           "no detection for %r -- lower --box-threshold or click instead" % text, None
                note = "detection: score %.3f (%d candidate(s))" % (dets[args.detection_index]["score"],
                                                                    len(dets))
            else:
                note = ""
            image_seg.set_image(rgb, image_id=fid)
            points = np.array([[p[0], p[1]] for p in pts], np.float32) if pts else None
            labels = np.array([int(p[2]) for p in pts], np.int32) if pts else None
            mask, score = image_seg.segment(box=box, points=points, labels=labels)
            mask = clean_mask(mask, bgr, args)
            note = ("%s | mask %.2f%% of the image, SAM score %.3f"
                    % (note or "prompt", 100.0 * mask.mean(), score))
            return mask, note, box

        picked = pick_prompt(lambda pos: read_frame(path, frame_at(pos)), len(ids), preview_fn,
                             seed_frame=ids.index(args.seed_frame),
                             step=max(1, len(ids) // 20),
                             host=args.pick_host, port=args.pick_port,
                             open_browser=args.pick_open)
        if picked is None:
            sys.exit("Nothing picked -- aborted.")
        spec = {"frame": frame_at(picked.get("frame", 0)),
                "points": picked.get("points") or [],
                "box": picked.get("box"),
                "text": (picked.get("text") or "").strip()}
        if spec["frame"] not in frames:
            nearest = min(frames, key=lambda f: abs(f - spec["frame"]))
            print("Picked frame %d is not on the stride grid; seeding from %d."
                  % (spec["frame"], nearest))
            spec["frame"] = nearest
        args.seed_frame = spec["frame"]
        seed_k = frames.index(args.seed_frame)
        seed_bgr = read_frame(path, args.seed_frame)
        seed_rgb = cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2RGB)
        if not args.prompt and spec["text"]:
            args.prompt = spec["text"]
    else:
        pts, labels = parse_click(args.click)
        spec["points"] = ([[float(p[0]), float(p[1]), int(l)] for p, l in zip(pts, labels)]
                          if pts is not None else [])
        spec["box"] = list(parse_box(args.box)) if args.box else None

    seed_points = (np.array([[p[0], p[1]] for p in spec["points"]], np.float32)
                   if spec["points"] else None)
    seed_labels = (np.array([int(p[2]) for p in spec["points"]], np.int32)
                   if spec["points"] else None)
    seed_box = tuple(spec["box"]) if spec["box"] else None

    if seed_box is None and seed_points is None:
        if not spec["text"]:
            sys.exit("No usable prompt.")
        seed_box, _ = detect_box(seed_rgb, spec["text"], args.detection_index)
        if seed_box is None:
            sys.exit("No detection to seed from. Lower --box-threshold, reword --prompt, "
                     "or use --pick / --click.")

    # ------------------------------------------------------------ seed check
    image_seg.set_image(seed_rgb, image_id=args.seed_frame)
    seed_mask, seed_score = image_seg.segment(box=seed_box, points=seed_points,
                                              labels=seed_labels)
    seed_mask = clean_mask(seed_mask, seed_bgr, args)
    print("Seed frame %06d: mask covers %.2f%% of the image (SAM score %.3f)"
          % (args.seed_frame, 100.0 * seed_mask.mean(), seed_score))
    seed_prev = seg.overlay(seed_bgr, seed_mask, box=seed_box,
                            points=seed_points, labels=seed_labels,
                            text="seed %06d" % args.seed_frame)
    cv2.imwrite(os.path.join(prev_dir, "seed.jpg"), seed_prev)
    print("Seed preview: %s" % os.path.join(prev_dir, "seed.jpg"))
    if seed_mask.mean() < args.min_area_frac:
        sys.exit("The seed mask is essentially empty -- fix the prompt before propagating.")

    if args.seed_only:
        cv2.imwrite(os.path.join(mask_dir, "%06d.png" % args.seed_frame),
                    (seed_mask.astype(np.uint8) * 255))
        print("--seed-only: stopped after the seed frame. Re-run without it to propagate.")
        return

    # ----------------------------------------------------------- propagation
    print("Loading SAM 2 video predictor ...")
    video = seg.load_video_predictor(args.sam2_model, device)

    tmpdir = tempfile.mkdtemp(prefix="odt_frames_")
    try:
        frame_dir = build_frame_dir(path, frames, tmpdir)
        print("Encoding %d frames (%s) ..."
              % (len(frames), "every frame" if args.stride == 1 else "stride %d" % args.stride))
        with torch.inference_mode(), seg.Autocast(device):
            state = video.init_state(video_path=frame_dir,
                                     offload_video_to_cpu=args.offload_video,
                                     offload_state_to_cpu=args.offload_state)
            masks = {}

            def prompt_at(k, box=None, points=None, labels=None):
                video.add_new_points_or_box(state, frame_idx=k, obj_id=1, box=box,
                                            points=points, labels=labels)

            def propagate(start_k, reverse=False):
                total = (start_k + 1) if reverse else (len(frames) - start_k)
                bar = tqdm(total=total, desc="propagate %s" % ("<-" if reverse else "->"))
                for k, _ids, logits in video.propagate_in_video(state, start_frame_idx=start_k,
                                                                reverse=reverse):
                    masks[k] = (logits[0] > 0.0).cpu().numpy()[0]
                    bar.update(1)
                bar.close()

            prompt_at(seed_k, box=np.array(seed_box, np.float32) if seed_box is not None else None,
                      points=seed_points, labels=seed_labels)
            propagate(seed_k, reverse=False)
            if seed_k > 0:
                propagate(seed_k, reverse=True)

            # -------------------------------------------------- verify/repair
            checks, repairs = [], []
            if args.verify_stride > 0 and spec["text"]:
                check_ks = [k for k in range(0, len(frames), max(1, args.verify_stride // args.stride))
                            if k != seed_k]
                print("Checking the track against the text prompt on %d frames ..." % len(check_ks))
                attempts = 0
                while True:
                    bad = []
                    checks = []
                    for k in tqdm(check_ks, desc="verify"):
                        bgr = read_frame(path, frames[k])
                        det_box, _ = detect_box(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                                spec["text"], args.detection_index, quiet=True)
                        mask_box = seg.bbox_of(masks.get(k, np.zeros((h, w), bool)))
                        score = seg.iou(det_box, mask_box) if det_box is not None else None
                        checks.append({"frame": frames[k], "iou": score})
                        if score is not None and score < args.verify_iou:
                            bad.append((k, det_box, score))
                    if not bad:
                        print("  track agrees with the text prompt on every checked frame.")
                        break
                    print("  %d checked frame(s) disagree with the prompt (IoU < %.2f): %s"
                          % (len(bad), args.verify_iou,
                             ", ".join("%06d(%.2f)" % (frames[k], s) for k, _, s in bad[:10])
                             + (" ..." if len(bad) > 10 else "")))
                    if not args.reanchor or attempts >= args.max_reanchors:
                        if not args.reanchor:
                            print("  (reporting only -- pass --reanchor to repair these, or "
                                  "check seg_preview/ first)")
                        else:
                            print("  giving up after %d repairs." % attempts)
                        break
                    k, det_box, score = bad[0]
                    print("  repairing from frame %06d (IoU %.2f) and re-propagating forward ..."
                          % (frames[k], score))
                    prompt_at(k, box=np.array(det_box, np.float32))
                    propagate(k, reverse=False)
                    repairs.append({"frame": frames[k], "iou_before": score})
                    attempts += 1

        # ------------------------------------------------------------ write
        print("Writing masks to %s ..." % mask_dir)
        areas, empty, previews = {}, [], []
        min_area = args.min_area_frac * h * w
        for k, fid in enumerate(tqdm(frames, desc="masks")):
            bgr = read_frame(path, fid)
            m = masks.get(k)
            m = np.zeros((h, w), bool) if m is None else clean_mask(m, bgr, args)
            if m.sum() < min_area:
                if m.any():
                    m[:] = False
                empty.append(fid)
            areas[fid] = int(m.sum())
            cv2.imwrite(os.path.join(mask_dir, "%06d.png" % fid), m.astype(np.uint8) * 255)
            if args.preview_stride and fid % args.preview_stride == 0:
                prev = seg.overlay(bgr, m, text="%06d  %.2f%%" % (fid, 100.0 * m.mean()))
                cv2.imwrite(os.path.join(prev_dir, "%06d.jpg" % fid), prev)
                previews.append((fid, prev))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    sheet = contact_sheet(previews)
    if sheet is not None:
        cv2.imwrite(os.path.join(prev_dir, "contact.jpg"), sheet)

    a = np.array([areas[f] for f in frames], dtype=np.float64)
    med = float(np.median(a[a > 0])) if (a > 0).any() else 0.0
    wobbly = [f for f in frames
              if med > 0 and areas[f] > 0 and (areas[f] < 0.25 * med or areas[f] > 4.0 * med)]
    print("\nMask area: median %.2f%% of the image, min %.2f%%, max %.2f%%"
          % (100.0 * med / (h * w), 100.0 * a.min() / (h * w), 100.0 * a.max() / (h * w)))
    if empty:
        print("WARNING: %d frame(s) ended up with an empty mask: %s%s"
              % (len(empty), ", ".join("%06d" % f for f in empty[:12]),
                 " ..." if len(empty) > 12 else ""))
        print("  Step 3b simply ignores those frames; if there are many, seed from a different "
              "frame\n  (--seed-frame) or add a background click where the track leaks.")
    if wobbly:
        print("WARNING: %d frame(s) are more than 4x off the median mask area: %s%s"
              % (len(wobbly), ", ".join("%06d" % f for f in wobbly[:12]),
                 " ..." if len(wobbly) > 12 else ""))
        print("  That usually means the track jumped to another surface. Look at those frames "
              "in\n  seg_preview/ before running 3b.")

    meta = {"dataset": dataset, "prompt": spec, "seed_frame": args.seed_frame,
            "stride": args.stride, "frames": frames,
            "models": {"sam2": args.sam2_model, "detector": args.detector_model,
                       "device": device},
            "params": {"box_threshold": args.box_threshold,
                       "text_threshold": args.text_threshold,
                       "keep_markers": args.keep_markers,
                       "marker_dilate": args.marker_dilate,
                       "largest_component": args.largest_component,
                       "min_area_frac": args.min_area_frac},
            "areas": {str(f): areas[f] for f in frames},
            "empty_frames": empty, "area_outliers": wobbly,
            "verify": checks, "repairs": repairs}
    with open(os.path.join(mask_dir, "seg_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print("\nWrote %d masks to %s" % (len(frames), mask_dir))
    if previews:
        print("Previews: %s (contact.jpg is the whole sequence at a glance)" % prev_dir)
        print("  over SSH:  python -m http.server -d %s 8000   then open localhost:8000"
              % prev_dir)
    print("Next: python 3b_segment_object_cloud.py %s" % (args.dataset or dataset))


if __name__ == "__main__":
    main()
