"""
utils/pipeline.py
-----------------

Orchestration core behind `run_e2e_web.py` (the studio).

The front-end is a thin shell over this module -- the stage registry, the
skip/stale logic, the subprocess runner, the prompt handling and the preview
collection all live here, so the page (`utils/e2e_page.py`) stays pure
presentation.

Design (borrowed from the e2e drivers in KIST/3drecon/3DGS and
KIST/reloc/baseline_loma, keeping what worked there and fixing what did not):

* every stage is one of the existing numbered scripts, run as a subprocess of
  `sys.executable` with an argv list (never a shell string), cwd pinned to
  this folder so `from utils...` imports resolve;
* a stage is *skipped* when all of its outputs exist and none is older than
  any of its inputs -- so a re-run after a new 3a pass automatically re-runs
  3b/4/5 and nothing else (`--force` / `--force-from` override);
* every run writes `<sequence>/e2e_log/run_<stamp>/` with a `manifest.json`
  (argv, plan, per-stage durations, results) and one `<stage>.log` per stage,
  and echoes each child command shlex-quoted, so any stage can be re-run by
  hand with copy-paste;
* impossible requests fail early with the exact command that fixes them;
  merely inapplicable ones are auto-downgraded with a printed reason.

The prompt for step 3a is a plain dict ("spec"):

    {"text": str, "points": [[x, y, label], ...], "box": [x0,y0,x1,y1]|None,
     "frame": int|None}

which maps 1:1 onto 3a's own CLI (`--prompt/--click/--box/--seed-frame`), so
the orchestrator needs no private hooks into 3a.
"""

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable  # children inherit the venv this was started with


# --------------------------------------------------------------------------
# Stage registry
# --------------------------------------------------------------------------

class Stage(object):
    def __init__(self, key, script, title, requires=(), soft_requires=(),
                 produces=(), needs_prompt=False, needs_window=False, note=""):
        self.key = key
        self.script = script
        self.title = title
        self.requires = tuple(requires)            # must exist before the stage runs
        self.soft_requires = tuple(soft_requires)  # used if present (staleness only)
        self.produces = tuple(produces)            # '{ds}' expands to the dataset name
        self.needs_prompt = needs_prompt           # step 3a: needs an object prompt
        self.needs_window = needs_window           # step 6: opens a cv2 window
        self.note = note


STAGES = {}
for _s in (
    Stage("step2", "2_compute_gt_poses.py", "GT poses (ArUco + ICP pose graph)",
          requires=("intrinsics.json", "JPEGImages", "depth"),
          produces=("transforms.npy",)),
    Stage("step3", "3_register_scene.py", "Scene registration (merged cloud)",
          requires=("intrinsics.json", "transforms.npy", "JPEGImages", "depth"),
          produces=("registeredScene.ply",)),
    Stage("step3a", "3a_segment_object_masks.py", "Object masks (SAM 2 propagation)",
          requires=("JPEGImages",),
          produces=("objmask",), needs_prompt=True),
    Stage("step3b", "3b_segment_object_cloud.py", "Object cloud (mask-guided crop)",
          requires=("intrinsics.json", "transforms.npy", "JPEGImages", "depth", "objmask"),
          soft_requires=("registeredScene.ply",),
          produces=("object.ply",)),
    Stage("step4", "4_create_label_files.py", "Labels / masks / mesh",
          requires=("intrinsics.json", "transforms.npy", "JPEGImages", "object.ply"),
          produces=("labels", "mask", "transforms", "{ds}.ply")),
    Stage("step5", "5_create_config2split.py", "Train/test split + .data cfg",
          requires=("intrinsics.json", "JPEGImages", "labels"),
          produces=("train.txt", "test.txt", "{ds}.data")),
    Stage("step6", "6_inspect_labels.py", "Visual QA",
          requires=("JPEGImages", "mask", "labels"),
          produces=(), needs_window=True,
          note="headless front-ends render the same overlays to qa_preview/ instead"),
):
    STAGES[_s.key] = _s

STAGE_ORDER = list(STAGES)  # insertion order == pipeline order

# How each stage that can use a GPU spells it. There is one script per step;
# the GPU path is a flag inside it, so the front-end only has
# to pass the right argument -- never a different file. The child always sees
# a single device (CUDA_VISIBLE_DEVICES, set in _cmd_for), hence index 0.
GPU_ARGS = {"step2": ["--gpu", "0"],
            "step3": ["--gpu", "0"],
            "step3b": ["--gpu", "0"],
            "step3a": ["--device", "cuda:0"]}


def pick_gpu():
    """Index of the CUDA device with the most free memory, or None.

    One device is chosen once per run and every stage is pointed at it --
    the pipeline is single-GPU by design (stages already share it fine).
    """
    try:
        import subprocess as sp
        out = sp.run(["nvidia-smi", "--query-gpu=index,memory.free",
                      "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=10)
        rows = [r.split(",") for r in out.stdout.strip().splitlines() if r.strip()]
        if not rows:
            return None
        return int(max(rows, key=lambda r: int(r[1]))[0])
    except Exception:
        return None


def parse_gpus(spec):
    """--gpu value -> ordered list of device indices, or None (= CPU only).

    '0' -> [0]; '0,1' -> [0, 1]; 'auto' (also what a bare '--gpu' means) ->
    [the freest device] (None when there is no CUDA at all); None/'' -> None.
    The FIRST index is the app's main device (live-preview models + every
    accelerated stage); a SECOND index is used for exactly one thing --
    step 3a runs there while it overlaps steps 2-3, removing the only GPU
    contention in the pipeline.
    """
    if spec in (None, "", "none", False):
        return None
    if spec is True:  # a legacy store_true-style value
        spec = "auto"
    s = str(spec).strip().lower()
    if s == "auto":
        k = pick_gpu()
        return [k] if k is not None else None
    try:
        idx = [int(v) for v in s.replace(";", ",").split(",") if v.strip() != ""]
    except ValueError:
        raise ValueError("--gpu expects indices like '0' or '0,1' (or 'auto'), "
                         "got %r" % spec)
    out = []
    for k in idx:
        if k not in out:
            out.append(k)
    return out or None


def gpus_from_args(args):
    """Resolve the front-ends' --gpu argument to a device list (or None)."""
    return parse_gpus(getattr(args, "gpu", None))


# Stages with no dependency between them, runnable side by side: step3a needs
# only JPEGImages, while step2->step3 is its own chain. On one GPU the SAM
# propagation and the (largely CPU/IO-bound) registration overlap fine.
PARALLEL_BRANCHES = (("step2", "step3"), ("step3a",))

MODES = {
    "whole": STAGE_ORDER,                                  # 2 .. 6
    "label": [k for k in STAGE_ORDER if k != "step6"],     # 2 .. 5
}


def resolve_steps(mode):
    """'whole' | 'label' | comma list of stage keys -> ordered stage keys."""
    mode = (mode or "whole").strip().lower()
    if mode in MODES:
        return list(MODES[mode])
    keys = [m.strip() for m in mode.split(",") if m.strip()]
    bad = [k for k in keys if k not in STAGES]
    if bad:
        raise ValueError("unknown step(s) %s -- valid: %s, or a mode name (%s)"
                         % (", ".join(bad), ", ".join(STAGE_ORDER), ", ".join(MODES)))
    return [k for k in STAGE_ORDER if k in keys]  # always pipeline order


# --------------------------------------------------------------------------
# Freshness / planning
# --------------------------------------------------------------------------

def _expand(name, dataset):
    return name.replace("{ds}", dataset)


def _mtime(p):
    """mtime of a file, or of the newest entry in a (flat) directory.

    A directory's own mtime only moves when entries are added or removed --
    step 4 overwriting all 930 files in labels/ leaves the directory mtime
    untouched, which would make everything downstream look forever stale.
    """
    try:
        st = os.stat(p)
    except OSError:
        return None
    if not os.path.isdir(p):
        return st.st_mtime
    newest = st.st_mtime
    try:
        with os.scandir(p) as it:
            for e in it:
                try:
                    t = e.stat().st_mtime
                    if t > newest:
                        newest = t
                except OSError:
                    pass
    except OSError:
        pass
    return newest


def stage_status(stage, path, dataset):
    """One of 'fresh' / 'stale' / 'pending' / 'blocked', with a human reason.

    fresh   -- outputs exist, none older than any existing input -> skippable
    stale   -- outputs exist but an input is newer -> should re-run
    pending -- outputs missing, inputs all present -> should run
    blocked -- a required input is missing -> cannot run (yet)
    """
    ins, missing_in = [], []  # ins: (mtime, name)
    for r in stage.requires:
        t = _mtime(os.path.join(path, _expand(r, dataset)))
        (ins.append((t, r)) if t is not None else missing_in.append(r))
    for r in stage.soft_requires:
        t = _mtime(os.path.join(path, _expand(r, dataset)))
        if t is not None:
            ins.append((t, r))

    if missing_in:
        return {"state": "blocked", "reason": "missing input: " + ", ".join(missing_in)}
    if not stage.produces:  # QA stages: nothing to be fresh about
        return {"state": "pending", "reason": "no outputs -- always runs when selected"}

    outs, missing_out = [], []
    for pnm in stage.produces:
        t = _mtime(os.path.join(path, _expand(pnm, dataset)))
        (outs.append(t) if t is not None else missing_out.append(_expand(pnm, dataset)))
    if missing_out:
        return {"state": "pending", "reason": "not yet produced: " + ", ".join(missing_out)}
    if ins and min(outs) < max(ins)[0]:
        return {"state": "stale", "reason": "%s is newer than the outputs" % max(ins)[1]}
    return {"state": "fresh", "reason": "outputs up to date"}


def make_plan(path, dataset, steps, force=False, force_from=None):
    """[{key, title, status, reason, will_run}] in pipeline order.

    A 'blocked' stage becomes runnable when an earlier planned stage will
    produce the missing input, so `--mode whole` on a bare capture plans
    cleanly even though nothing exists yet.
    """
    plan = []
    will_exist = set()    # inputs produced by earlier planned stages
    will_refresh = set()  # inputs an earlier planned stage will REWRITE this run
    forcing = False
    for key in steps:
        st = STAGES[key]
        if force or (force_from and key == force_from):
            forcing = True
        s = stage_status(st, path, dataset)
        state, reason = s["state"], s["reason"]
        if state == "blocked":
            missing = [r for r in st.requires
                       if _mtime(os.path.join(path, _expand(r, dataset))) is None]
            if all(m in will_exist for m in missing):
                state, reason = "pending", "inputs produced by an earlier stage this run"
        elif state == "fresh":
            # cascade: an earlier stage re-running this run makes us stale too
            touched = [r for r in list(st.requires) + list(st.soft_requires)
                       if _expand(r, dataset) in will_refresh]
            if touched:
                state, reason = "stale", "%s will be regenerated this run" % touched[0]
        will_run = state in ("pending", "stale") or (forcing and state != "blocked")
        if state == "fresh" and forcing:
            reason = "forced re-run"
        plan.append({"key": key, "title": st.title, "script": st.script,
                     "state": state, "reason": reason, "will_run": will_run,
                     "needs_prompt": st.needs_prompt, "needs_window": st.needs_window})
        if will_run or state == "fresh":
            will_exist.update(_expand(p, dataset) for p in st.produces)
        if will_run:
            will_refresh.update(_expand(p, dataset) for p in st.produces)
    return plan


def plan_needs_prompt(plan, path):
    """True when step 3a will actually run, so the UI should ask for a prompt.

    (This is the whole point of asking *after* planning: a sequence whose
    masks are already fresh never bothers the user with a prompt.)
    """
    return any(p["needs_prompt"] and p["will_run"] for p in plan)


# --------------------------------------------------------------------------
# Prompt spec <-> 3a CLI args
# --------------------------------------------------------------------------

def normalize_spec(text=None, clicks=None, box=None, frame=None):
    spec = {"text": (text or "").strip(), "points": [], "box": None, "frame": frame}
    for c in clicks or []:
        if isinstance(c, str):
            parts = [p for p in c.replace(";", ",").split(",") if p.strip()]
            c = [float(parts[0]), float(parts[1]),
                 int(parts[2]) if len(parts) > 2 else 1]
        spec["points"].append([float(c[0]), float(c[1]),
                               int(c[2]) if len(c) > 2 else 1])
    if box:
        if isinstance(box, str):
            box = [float(p) for p in box.replace(";", ",").split(",") if p.strip()]
        if len(box) == 4:
            spec["box"] = [float(v) for v in box]
    return spec


def spec_is_empty(spec):
    return not (spec and (spec.get("text") or spec.get("points") or spec.get("box")))


def prompt_args(spec):
    """spec dict -> the exact 3a CLI arguments."""
    a = []
    if spec.get("text"):
        a += ["--prompt", spec["text"]]
    for p in spec.get("points") or []:
        a += ["--click", "%.1f,%.1f,%d" % (p[0], p[1], int(p[2]) if len(p) > 2 else 1)]
    if spec.get("box"):
        a += ["--box", ",".join("%.1f" % v for v in spec["box"])]
    if spec.get("frame") is not None:
        a += ["--seed-frame", str(int(spec["frame"]))]
    return a


def parse_extra(items):
    """['step3a: --stride 8 --reanchor', ...] -> {'step3a': ['--stride','8',...]}."""
    extra = {}
    for item in items or []:
        if ":" not in item:
            raise ValueError("--extra expects 'stepX: <args>', got %r" % item)
        key, rest = item.split(":", 1)
        key = key.strip()
        if key not in STAGES:
            raise ValueError("--extra: unknown stage %r (valid: %s)"
                             % (key, ", ".join(STAGE_ORDER)))
        extra.setdefault(key, []).extend(shlex.split(rest))
    return extra


# --------------------------------------------------------------------------
# Sequence helpers shared by both front-ends
# --------------------------------------------------------------------------

def frame_ids(path):
    files = glob.glob1(os.path.join(path, "JPEGImages"), "*.jpg")
    return sorted(int(os.path.splitext(f)[0]) for f in files
                  if os.path.splitext(f)[0].isdigit())


def read_frame(path, fid):
    img = cv2.imread(os.path.join(path, "JPEGImages", "%06d.jpg" % fid))
    if img is None:
        raise IOError("cannot read %s/JPEGImages/%06d.jpg" % (path.rstrip("/"), fid))
    return img


def _read_ply_vertex_count(fname):
    """Vertex count from a PLY header (ASCII header even in binary files)."""
    try:
        with open(fname, "rb") as f:
            head = f.read(4096).decode("ascii", "replace")
        for line in head.splitlines():
            if line.startswith("element vertex"):
                return int(line.split()[-1])
    except Exception:
        pass
    return None


def stage_summary(key, path, dataset):
    """One line about the stage's *outputs*, printed after it finishes."""
    try:
        if key == "step2":
            T = np.load(os.path.join(path, "transforms.npy"), mmap_mode="r")
            return "%d poses -> transforms.npy" % len(T)
        if key == "step3":
            f = os.path.join(path, "registeredScene.ply")
            n = _read_ply_vertex_count(f)
            return "registeredScene.ply: %s points, %.0f MB" % (
                "{:,}".format(n) if n else "?", os.path.getsize(f) / 1e6)
        if key == "step3a":
            meta = json.load(open(os.path.join(path, "objmask", "seg_meta.json")))
            n = len(meta.get("frames", []))
            bad = len(meta.get("empty_frames", [])) + len(meta.get("area_outliers", []))
            ver = meta.get("verify", [])
            vfail = sum(1 for v in ver if not v.get("ok", True))
            s = "%d masks (stride %s)" % (n, meta.get("stride", "?"))
            s += ", %d empty/outlier" % bad if bad else ", no empty masks"
            if ver:
                s += ", verified %d frame(s)%s" % (len(ver),
                                                   " -- %d FAILED" % vfail if vfail else " all ok")
            return s
        if key == "step3b":
            f = os.path.join(path, "object.ply")
            n = _read_ply_vertex_count(f)
            return "object.ply: %s points" % ("{:,}".format(n) if n else "?")
        if key == "step4":
            n = len(glob.glob1(os.path.join(path, "labels"), "*.txt"))
            mesh = os.path.join(path, dataset + ".ply")
            v = _read_ply_vertex_count(mesh)
            return "%d label files, %s.ply (%s vertices)" % (
                n, dataset, "{:,}".format(v) if v else "?")
        if key == "step5":
            cfg = os.path.join(path, dataset + ".data")
            diam = ""
            for line in open(cfg):
                if line.startswith("diam"):
                    diam = line.split("=")[1].strip()
            ntr = sum(1 for _ in open(os.path.join(path, "train.txt")))
            nte = sum(1 for _ in open(os.path.join(path, "test.txt")))
            return "train %d / test %d, diam %s -> %s.data" % (ntr, nte, diam, dataset)
        if key == "step6":
            qa = os.path.join(path, "qa_preview")
            if os.path.isdir(qa):
                return "%d QA overlays in qa_preview/" % len(glob.glob1(qa, "*.jpg"))
            return "inspected in a window"
    except Exception as exc:
        return "(summary unavailable: %s)" % exc
    return ""


# --------------------------------------------------------------------------
# QA overlays, headless (the drawing half of 6_inspect_labels.py)
# --------------------------------------------------------------------------

_BOX_EDGES = ((1, 2), (1, 3), (2, 4), (3, 4), (1, 5), (3, 7),
              (5, 7), (2, 6), (4, 8), (6, 8), (5, 6), (7, 8))


def label_overlay(path, fid):
    """One frame with its mask tint + projected 3D box, or None."""
    img = cv2.imread(os.path.join(path, "JPEGImages", "%06d.jpg" % fid))
    if img is None:
        return None
    mask_name = ("%04d.png" if fid < 10000 else "%06d.png") % fid
    m = cv2.imread(os.path.join(path, "mask", mask_name))
    if m is not None:
        cv2.addWeighted(m, 0.4, img, 0.6, 0, img)
    labelfile = os.path.join(path, "labels", "%06d.txt" % fid)
    if os.path.exists(labelfile):
        with open(labelfile) as fp:
            vals = [float(v) for v in fp.read().split()]
        if len(vals) >= 19:
            h, w = img.shape[:2]
            pts = [(int(vals[2 * i + 1] * w), int(vals[2 * i + 2] * h)) for i in range(9)]
            cv2.circle(img, pts[0], 3, (255, 0, 255), -1)
            for a, b in _BOX_EDGES:
                cv2.line(img, pts[a], pts[b], (255, 0, 0), 3)
    return img


def render_label_overlays(path, dataset, out_dir=None, max_frames=32):
    """Write an evenly-strided set of QA overlays; returns their paths."""
    out_dir = out_dir or os.path.join(path, "qa_preview")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    ids = sorted(int(f[:-4]) for f in glob.glob1(os.path.join(path, "labels"), "*.txt")
                 if f[:-4].isdigit())
    if not ids:
        return []
    stride = max(1, len(ids) // max_frames)
    written = []
    for fid in ids[::stride]:
        img = label_overlay(path, fid)
        if img is None:
            continue
        out = os.path.join(out_dir, "%06d.jpg" % fid)
        cv2.imwrite(out, img)
        written.append(out)
    return written


def stage_previews(key, path, dataset):
    """Paths of the preview images a finished stage left on disk."""
    prev = os.path.join(path, "seg_preview")
    if key == "step3a":
        seeds = [os.path.join(prev, n) for n in ("seed.jpg", "contact.jpg")]
        return [p for p in seeds if os.path.exists(p)]
    if key == "step3b":
        return sorted(glob.glob(os.path.join(prev, "object_*.jpg")))
    if key in ("step4", "step6"):
        qa = os.path.join(path, "qa_preview")
        return sorted(glob.glob(os.path.join(qa, "*.jpg")))
    return []


# --------------------------------------------------------------------------
# Live prompt preview (optional; used by both front-ends' interactive pickers)
# --------------------------------------------------------------------------

class LivePrompter(object):
    """Lazy Grounding-DINO + SAM2 image preview, mirroring 3a's picker preview.

    Loading is deferred until the first preview so front-ends stay instant
    when the user types a text prompt and never asks for a live mask. If
    torch/sam2 are not installed, preview() degrades to a friendly note and
    the click/box coordinates are simply passed to 3a untested.
    """

    def __init__(self, sam2_model=None, detector_model=None, device="auto"):
        from utils import segmentation as seg
        self.seg = seg
        self.sam2_model = sam2_model or seg.DEFAULT_SAM2
        self.detector_model = detector_model or seg.DEFAULT_DETECTOR
        self.device_spec = device
        self._image_seg = None
        self._detector = None
        self._lock = threading.Lock()
        self._available = None
        self.error = None
        self.status = "cold"      # cold -> loading -> ready | unavailable
        self.device = None        # resolved device string, once loaded
        self.load_seconds = None
        self._load_started = None

    def available(self):
        if self._available is None:
            try:
                import torch  # noqa: F401
                import sam2   # noqa: F401
                self._available = True
            except Exception as exc:
                self.error = "live preview unavailable (%s) -- prompts still work, " \
                             "they are just not previewed" % exc
                self._available = False
        return self._available

    def _ensure(self, want_detector):
        device = self.seg.pick_device(self.device_spec)
        self.device = device
        if self._image_seg is None or (want_detector and self._detector is None):
            self.status = "loading"
            self._load_started = self._load_started or time.time()
        if self._image_seg is None:
            self._image_seg = self.seg.ImageSegmenter(self.sam2_model, device)
        if want_detector and self._detector is None:
            self._detector = self.seg.TextDetector(self.detector_model, device)
        if self.status != "ready":
            self.status = "ready"
            self.load_seconds = time.time() - self._load_started

    def warmup(self, background=True):
        """Load both models now (in a daemon thread by default) instead of on
        the first click -- the first mask otherwise arrives ~20 s late, which
        reads as 'broken' rather than 'loading'. Also encodes one dummy frame
        so the first real click pays no CUDA warm-up either.
        """
        def _load():
            # even the availability probe imports torch (~8 s), so everything
            # happens on this thread -- the caller must never be delayed
            if not self.available():
                self.status = "unavailable"
                return
            try:
                with self._lock:
                    self._ensure(want_detector=True)
                    dummy = np.zeros((720, 1280, 3), np.uint8)
                    self._image_seg.set_image(dummy, image_id="__warmup__")
                    self._image_seg.segment(points=np.array([[640.0, 360.0]], np.float32),
                                            labels=np.array([1], np.int32))
            except Exception as exc:
                self.status = "unavailable"
                self.error = "model load failed: %s" % exc

        self.status = "loading"
        self._load_started = time.time()
        if background:
            threading.Thread(target=_load, daemon=True).start()
        else:
            _load()

    def state(self):
        """Dict for UIs: {'status', 'device', 'seconds', 'error'}."""
        elapsed = (time.time() - self._load_started) if (
            self.status == "loading" and self._load_started) else self.load_seconds
        return {"status": self.status, "device": self.device,
                "seconds": round(elapsed, 1) if elapsed else None,
                "error": self.error}

    def preview(self, bgr, spec, image_id=None, clean=True):
        """(mask bool|None, note str, box|None) for a prompt spec on one frame."""
        if not self.available():
            return None, self.error, None
        with self._lock:  # the models are not re-entrant
            text = (spec.get("text") or "").strip()
            pts = spec.get("points") or []
            box = spec.get("box")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            note = ""
            self._ensure(want_detector=bool(text and not pts and not box))
            if text and not pts and not box:
                dets = self._detector.detect(rgb, text)
                if not dets:
                    return None, "no detection for %r -- click instead, or reword" % text, None
                box = dets[0]["box"]
                note = "detection score %.3f (%d candidate(s)) | " % (dets[0]["score"], len(dets))
            self._image_seg.set_image(rgb, image_id=image_id)
            points = np.array([[p[0], p[1]] for p in pts], np.float32) if pts else None
            labels = np.array([int(p[2]) for p in pts], np.int32) if pts else None
            mask, score = self._image_seg.segment(box=box, points=points, labels=labels)
            if clean:
                mask = self.seg.largest_component(mask)
                mask = mask & ~self.seg.marker_mask(bgr)
                mask = self.seg.largest_component(mask)
            note += "mask %.2f%% of the image, SAM score %.3f" % (100.0 * mask.mean(), score)
            return mask, note, box

    def overlay(self, bgr, mask, spec, note=None):
        pts = spec.get("points") or []
        return self.seg.overlay(
            bgr, mask, box=spec.get("box"),
            points=[[p[0], p[1]] for p in pts] or None,
            labels=[int(p[2]) for p in pts] or None,
            text=note)


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------

class PipelineRun(object):
    """One end-to-end run. Synchronous; front-ends thread it themselves.

    on_event(dict) receives, in order:
        {'type': 'plan',        'plan': [...]}
        {'type': 'stage_skip',  'key', 'reason'}
        {'type': 'stage_start', 'key', 'cmd', 'index', 'total'}
        {'type': 'line',        'key', 'text', 'cr': bool}   # cr = tqdm-style overwrite
        {'type': 'stage_done',  'key', 'seconds', 'summary', 'previews': [...]}
        {'type': 'stage_fail',  'key', 'seconds', 'returncode'}
        {'type': 'done',        'ok', 'seconds', 'rows': [...], 'log_dir'}
    """

    def __init__(self, path, dataset, steps, prompt_spec=None, extra=None,
                 viz="preview", force=False, force_from=None,
                 qa_headless=False, gpus=None, parallel=True, on_event=None):
        self.path = path if path.endswith("/") else path + "/"
        self.dataset = dataset
        self.steps = steps
        self.spec = prompt_spec or {}
        self.extra = extra or {}
        self.viz = viz
        self.force = force
        self.force_from = force_from
        self.qa_headless = qa_headless
        self.gpus = list(gpus) if gpus else None  # e.g. [0] or [0, 1]; None = CPU
        self.use_gpu = bool(self.gpus)
        self.parallel = parallel
        self.on_event = on_event or (lambda e: None)
        self.cancel = threading.Event()
        self._procs = set()
        self._proc_lock = threading.Lock()
        self.log_dir = None

    # -- internals --------------------------------------------------------

    def _emit(self, **e):
        try:
            self.on_event(e)
        except Exception:
            pass  # a broken UI callback must not kill the pipeline

    def _cmd_for(self, key):
        st = STAGES[key]
        cmd = [PY, os.path.join(PIPELINE_DIR, st.script), self.path.rstrip("/")]
        if key == "step3a" and not spec_is_empty(self.spec):
            cmd += prompt_args(self.spec)
        if key == "step6" and self.qa_headless:
            return None  # replaced by render_label_overlays()
        extra = list(self.extra.get(key, []))
        if key == "step3a" and "--reanchor" in extra and not self.spec.get("text"):
            extra.remove("--reanchor")
            self._emit(type="line", key=key, cr=False,
                       text="[e2e] --reanchor needs a text prompt -> dropped for this run")
        # every GPU-capable stage runs on gpus[0]; with a second GPU listed,
        # step 3a alone moves there (it overlaps steps 2-3, the one place two
        # stages compete for the device). A user-supplied --gpu/--device via
        # --extra wins. The physical GPU is granted via CUDA_VISIBLE_DEVICES
        # (otherwise torch/open3d still open small contexts on every other
        # device) and the child addresses it as device 0 of its visible set.
        env = None
        if (self.gpus and key in GPU_ARGS
                and not ({"--gpu", "--device"} & set(extra))):
            phys = self.gpus[-1] if (key == "step3a" and len(self.gpus) > 1) \
                else self.gpus[0]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(phys))
            cmd += GPU_ARGS[key]
        cmd += extra
        return cmd, env

    _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    # 'desc:  42%|####      | 391/930 [00:34<00:44, 12.03it/s]'
    _TQDM_RE = re.compile(r"^(.*?):?\s*(\d{1,3})%\|[^|]*\|\s*(\S+)\s*(\[[^\]]*\])?\s*$")

    def _handle_line(self, key, text, cr, prefix, log_f, logged):
        """One decoded child line -> a 'progress' or 'line' event.

        tqdm bars are the reason this exists: their refreshes arrive as \\r
        updates and ANSI cursor moves (nested bars use ESC[A), and treating
        each refresh as a log line floods the UI. They become 'progress'
        events -- percent + a compact text -- which the front-ends render as
        a progress bar in the stage's card, not in the log. Only real lines
        reach the log.
        """
        text = self._ANSI_RE.sub("", text).strip("\r")
        if not text.strip():
            return
        m = self._TQDM_RE.match(text.strip())
        if m:
            desc = (m.group(1) or "").strip() or "working"
            pct = min(100, int(m.group(2)))
            detail = m.group(3) + (" " + m.group(4) if m.group(4) else "")
            self._emit(type="progress", key=key, pct=pct,
                       text="%s %s" % (desc, detail))
            # keep the log file readable: one line per 10 % per bar
            slot = (desc, pct // 10)
            if slot not in logged:
                logged.add(slot)
                log_f.write("%s %d%% %s\n" % (desc, pct, detail))
            return
        log_f.write(text + "\n")
        self._emit(type="line", key=key, cr=False,
                   text=(prefix + text) if prefix else text)

    def _stream(self, key, cmd, log_f, tag=False, env=None):
        """Run one child, teeing interleaved stdout+stderr to the log + events.

        tag=True prefixes lines with the stage key -- used when two stages
        stream at once so the merged log stays attributable.
        """
        proc = subprocess.Popen(cmd, cwd=PIPELINE_DIR, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        with self._proc_lock:
            self._procs.add(proc)
        prefix = ("[%s] " % key) if tag else ""
        logged = set()  # per-stream: (bar desc, decile) already written to the log
        buf = b""
        while True:
            # read1: return as soon as *any* output arrives (read() would wait
            # for a full 1024 bytes and make the live log stutter)
            chunk = proc.stdout.read1(1024)
            if not chunk:
                break
            if self.cancel.is_set():
                proc.terminate()
            buf += chunk
            while True:
                # split on whichever of \n / \r comes first, keeping the kind
                i_n, i_r = buf.find(b"\n"), buf.find(b"\r")
                if i_n < 0 and i_r < 0:
                    break
                if i_r >= 0 and (i_n < 0 or i_r < i_n):
                    line, buf = buf[:i_r], buf[i_r + 1:]
                    if buf[:1] == b"\n":  # \r\n is a plain newline
                        buf = buf[1:]
                    cr = True
                else:
                    line, buf = buf[:i_n], buf[i_n + 1:]
                    cr = False
                self._handle_line(key, line.decode("utf-8", "replace"),
                                  cr, prefix, log_f, logged)
        if buf.strip():
            self._handle_line(key, buf.decode("utf-8", "replace"),
                              False, prefix, log_f, logged)
        proc.wait()
        with self._proc_lock:
            self._procs.discard(proc)
        return proc.returncode

    # -- public -----------------------------------------------------------

    def request_cancel(self):
        self.cancel.set()
        with self._proc_lock:
            procs = list(self._procs)
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass

    def _exec_stage(self, p, index, total, tag=False):
        """Run one plan entry; returns its manifest row."""
        key = p["key"]
        if self.cancel.is_set():
            return {"key": key, "result": "cancelled"}
        if not p["will_run"]:
            self._emit(type="stage_skip", key=key, reason=p["reason"])
            return {"key": key, "result": "skipped", "reason": p["reason"]}

        cmd, env = self._cmd_for(key)
        t0 = time.time()
        if cmd is None:  # headless QA
            self._emit(type="stage_start", key=key, cmd="(built-in) render QA overlays",
                       index=index, total=total)
            try:
                outs = render_label_overlays(self.path, self.dataset)
                rc = 0 if outs else 1
            except Exception as exc:
                self._emit(type="line", key=key, text="QA render failed: %s" % exc, cr=False)
                rc = 1
        else:
            shown = " ".join(shlex.quote(c) for c in cmd)
            self._emit(type="stage_start", key=key, cmd=shown, index=index, total=total)
            with open(os.path.join(self.log_dir, key + ".log"), "w") as log_f:
                log_f.write("$ %s\n\n" % shown)
                rc = self._stream(key, cmd, log_f, tag=tag, env=env)

        dt = time.time() - t0
        if rc != 0:
            result = "cancelled" if self.cancel.is_set() else "failed"
            self._emit(type="stage_fail", key=key, seconds=dt, returncode=rc)
            return {"key": key, "result": result, "seconds": dt, "returncode": rc}
        summary = stage_summary(key, self.path, self.dataset)
        previews = stage_previews(key, self.path, self.dataset) if self.viz != "off" else []
        self._emit(type="stage_done", key=key, seconds=dt,
                   summary=summary, previews=previews)
        return {"key": key, "result": "ok", "seconds": dt, "summary": summary}

    def run(self):
        t_all = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.path, "e2e_log", "run_" + stamp)
        os.makedirs(self.log_dir)

        plan = make_plan(self.path, self.dataset, self.steps,
                         force=self.force, force_from=self.force_from)
        self._emit(type="plan", plan=plan)
        if self.gpus:
            if len(self.gpus) > 1:
                self._emit(type="line", key="e2e", cr=False,
                           text="[e2e] GPUs %s: accelerated stages on GPU %d; step3a "
                                "on GPU %d while the branches overlap"
                                % (self.gpus, self.gpus[0], self.gpus[-1]))
            else:
                self._emit(type="line", key="e2e", cr=False,
                           text="[e2e] single-GPU run: everything pinned to GPU %d"
                                % self.gpus[0])

        manifest = {"dataset": self.dataset, "path": os.path.abspath(self.path),
                    "steps": self.steps, "viz": self.viz, "force": self.force,
                    "force_from": self.force_from, "argv": sys.argv,
                    "gpus": self.gpus,
                    "prompt": self.spec if not spec_is_empty(self.spec) else None,
                    "extra": {k: " ".join(v) for k, v in self.extra.items()},
                    "started": stamp, "stages": []}

        rows = {}   # key -> row, ordered into the manifest at the end
        state = {"ok": True}

        def run_seq(entries, tag=False):
            for p in entries:
                row = self._exec_stage(p, list(plan).index(p), len(plan), tag=tag)
                rows[p["key"]] = row
                if row["result"] not in ("ok", "skipped"):
                    state["ok"] = False
                    return

        branch_a = [p for p in plan if p["key"] in PARALLEL_BRANCHES[0]]
        branch_b = [p for p in plan if p["key"] in PARALLEL_BRANCHES[1]]
        rest = [p for p in plan if p not in branch_a and p not in branch_b]
        both_busy = (any(p["will_run"] for p in branch_a)
                     and any(p["will_run"] for p in branch_b))

        if self.parallel and both_busy:
            self._emit(type="line", key="e2e", cr=False,
                       text="[e2e] step3a is independent of step2/step3 -> "
                            "running the two branches in parallel")
            ta = threading.Thread(target=run_seq, args=(branch_a, True), daemon=True)
            tb = threading.Thread(target=run_seq, args=(branch_b, True), daemon=True)
            ta.start(), tb.start()
            ta.join(), tb.join()
            if state["ok"] and not self.cancel.is_set():
                run_seq(rest)
        else:
            run_seq(plan)  # plan order == pipeline order

        ordered = [rows[p["key"]] for p in plan if p["key"] in rows]
        manifest["stages"] = ordered
        manifest["ok"] = state["ok"]
        manifest["total_seconds"] = time.time() - t_all
        with open(os.path.join(self.log_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        self._emit(type="done", ok=state["ok"], seconds=manifest["total_seconds"],
                   rows=ordered, log_dir=self.log_dir)
        return manifest


# --------------------------------------------------------------------------
# Shared CLI plumbing for the two front-ends
# --------------------------------------------------------------------------

def add_e2e_args(parser):
    parser.add_argument("--mode", default="whole",
                        help="what to run: 'whole' (steps 2-6), 'label' (2-5), or a comma "
                             "list of steps, e.g. 'step3a,step3b' (choices: %s)"
                             % ", ".join(STAGE_ORDER))
    parser.add_argument("--viz", choices=["off", "preview", "full"], default="preview",
                        help="visual feedback while running: 'off' = text only, 'preview' = "
                             "key images per stage, 'full' = every preview + QA playback")
    parser.add_argument("--prompt", default=None,
                        help="step 3a text prompt, e.g. \"white product box\"")
    parser.add_argument("--click", action="append", default=None, metavar="X,Y[,L]",
                        help="step 3a seed click (repeatable; L: 1=object 0=background)")
    parser.add_argument("--box", default=None, metavar="X0,Y0,X1,Y1",
                        help="step 3a seed box")
    parser.add_argument("--seed-frame", type=int, default=None,
                        help="frame the step-3a prompt refers to")
    parser.add_argument("--extra", action="append", default=None, metavar="STEP:ARGS",
                        help="pass extra args to one stage, repeatable -- e.g. "
                             "--extra 'step3a: --stride 8 --reanchor'")
    parser.add_argument("--gpu", "--gpus", dest="gpu", nargs="?", const="auto",
                        default=None, metavar="N[,M]",
                        help="GPU indices. '--gpu 0': the GPU path of steps 2/3/3b "
                             "AND the live mask "
                             "preview run on GPU 0. '--gpu 0,1': additionally, step 3a "
                             "gets GPU 1 to itself while it overlaps steps 2-3 (the one "
                             "place two stages share a device; saves a few percent). "
                             "Bare '--gpu' = 'auto' (the freest device). Omit entirely "
                             "= CPU only. Output is measured-equivalent either way")
    parser.add_argument("--no-parallel", action="store_true",
                        help="do not overlap step3a with step2/step3 (they are "
                             "independent and normally run side by side)")
    parser.add_argument("--force", action="store_true",
                        help="re-run every selected stage even if its outputs are fresh")
    parser.add_argument("--force-from", default=None, metavar="STEP",
                        help="re-run from this stage onward even where fresh (e.g. step3b)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and each command without running anything")
    return parser


def spec_from_args(args):
    return normalize_spec(text=args.prompt, clicks=args.click, box=args.box,
                          frame=args.seed_frame)


def print_plan(plan, file=sys.stdout):
    w = max(len(p["title"]) for p in plan) if plan else 0
    for p in plan:
        mark = {"fresh": " ", "stale": "!", "pending": ">", "blocked": "x"}[p["state"]]
        act = "RUN " if p["will_run"] else "skip"
        print(" %s %-7s %-*s  %-8s %s" % (mark, p["key"], w, p["title"],
                                          act, p["reason"]), file=file)
