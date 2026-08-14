"""
run_e2e_web.py
--------------

The studio: the whole dataset pipeline (steps 2-6) driven from one browser
page -- pick the object, watch it run, inspect the result in 3D.

One page, served by the standard library's `http.server` -- the same recipe as
the 3a picker, so it works identically on a headless GPU server (forward the
port once: `ssh -N -L 8770:localhost:8770 <host>`) and on a local machine
(`--open` launches the browser). No X server, nothing to install.

    ../.venv/bin/python run_e2e_web.py                # pick everything in the page
    ../.venv/bin/python run_e2e_web.py box --gpu 0    # pre-select a sequence, use GPU 0
    ../.venv/bin/python run_e2e_web.py box --mode label --prompt "white product box"

What the page does:

* shows every pipeline stage as a card with its live status -- fresh / stale /
  pending / blocked -- and *why*, so you see what a run will actually do
  before you press Run (the same skip/stale logic re-runs exactly the stages
  whose inputs changed, nothing else);
* mode picker: whole (2-6) / label (2-5) / any subset of stages;
* the object prompt for step 3a is asked for **only when step 3a will
  actually run**: type what the object is, click it (left = object,
  right = not the object), or drag a box -- with a live SAM mask preview when
  the segmentation stack is installed;
* turns each stage's tqdm output into a real progress bar on that stage's
  card, keeping the raw log folded away at the bottom;
* per-stage preview images as they appear (viz=preview), and a play-through
  of the final QA overlays (viz=full). Step 6 never opens a window here --
  the same overlays are rendered headlessly into qa_preview/ and played in
  the page;
* when the run finishes: previews on the left, and the resulting .ply files
  in a WebGL viewer (orbit / zoom / pan) on the right.

Every run also writes <sequence>/e2e_log/run_<stamp>/ with one log per stage
and a manifest.json, and each stage's exact command is echoed, so anything
can be reproduced from a terminal alone.
"""

import argparse
import base64
import collections
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from utils import pipeline as pl
from utils.cli import list_datasets, normalize_root

DEFAULT_PORT = 8770


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

from utils.e2e_page import PAGE as _PAGE  # the dashboard HTML/JS lives there


def _jpeg_data_url(bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode preview JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------

class E2EServer(object):
    def __init__(self, args):
        self.args = args
        self.root = normalize_root(args.data_root)
        if self.root == "./":
            self.root = ""
        # --gpu decides everything: gpus[0] hosts the live-preview models and
        # every accelerated stage; a second index is used only to give step 3a
        # its own device while it overlaps steps 2-3. No --gpu = CPU only.
        self.gpus = pl.gpus_from_args(args)
        if self.gpus:
            # confine THIS process to its one GPU before torch ever loads --
            # without this, torch/open3d open small contexts on every device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpus[0])
        self.prompter = pl.LivePrompter(device="cuda:0" if self.gpus else "cpu")
        self.prompter.warmup()  # load SAM2 + Grounding DINO now, not on first click
        self.events = collections.deque(maxlen=4000)  # (seq, event)
        self.seq = 0
        self.run = None          # active PipelineRun
        self.run_thread = None
        self.lock = threading.Lock()

    # -- helpers -----------------------------------------------------------

    def seq_path(self, dataset):
        """Resolve a dataset name/path exactly like the numbered scripts do."""
        dataset = (dataset or "").replace("\\", "/").rstrip("/")
        for cand in (self.root + dataset + "/", dataset + "/"):
            if os.path.isdir(cand):
                return dataset.rsplit("/", 1)[-1], cand
        raise ValueError("sequence folder not found: %r" % dataset)

    def push(self, event):
        with self.lock:
            self.seq += 1
            self.events.append((self.seq, event))

    def datasets(self):
        found = list_datasets(self.root or ".")
        # a dataset given as a path may live outside the root
        if self.args.dataset and self.args.dataset not in found:
            found.insert(0, self.args.dataset)
        return found

    # -- API ---------------------------------------------------------------

    def api_state(self, req):
        return {"datasets": self.datasets(),
                "dataset": self.args.dataset or (self.datasets() or [""])[0],
                "mode": self.args.mode, "viz": self.args.viz,
                "prompt_text": self.args.prompt or "",
                "models": self.prompter.state()}

    def api_models(self, req):
        return self.prompter.state()

    def api_plan(self, req):
        dataset, path = self.seq_path(req.get("dataset"))
        steps = pl.resolve_steps(req.get("mode"))
        plan = pl.make_plan(path, dataset, steps, force=bool(req.get("force")),
                            force_from=req.get("force_from") or self.args.force_from)
        ids = pl.frame_ids(path)
        needs = pl.plan_needs_prompt(plan, path)
        # the panel is offered whenever step 3a is part of the selection at
        # all -- fresh masks can still be inspected and re-picked (a new
        # prompt then re-runs 3a and everything downstream automatically)
        st3a = next((p for p in plan if p["key"] == "step3a"), None)
        can_prompt = st3a is not None
        mask_state = st3a["state"] if st3a else "absent"
        existing = [os.path.relpath(p, path) for p in
                    pl.stage_previews("step3a", path, dataset)]
        note = "plan: %d to run, %d fresh" % (sum(p["will_run"] for p in plan),
                                              sum(not p["will_run"] for p in plan))
        return {"plan": plan, "needs_prompt": needs, "can_prompt": can_prompt,
                "mask_state": mask_state, "existing_previews": existing,
                "n_frames": len(ids),
                "suggest_frame": ids[len(ids) // 2] if ids else 0, "note": note}

    def api_frame(self, req):
        dataset, path = self.seq_path(req.get("dataset"))
        ids = pl.frame_ids(path)
        if not ids:
            return {"error": "no frames in %sJPEGImages" % path}
        want = int(req.get("frame", 0))
        fid = min(ids, key=lambda f: abs(f - want))
        return {"frame": fid, "n_frames": len(ids),
                "image": _jpeg_data_url(pl.read_frame(path, fid))}

    def api_preview(self, req):
        dataset, path = self.seq_path(req.get("dataset"))
        fid = int(req.get("frame", 0))
        bgr = pl.read_frame(path, fid)
        spec = pl.normalize_spec(text=req.get("text"), clicks=req.get("points"),
                                 box=req.get("box"), frame=fid)
        mask, note, box = self.prompter.preview(bgr, spec, image_id=(path, fid))
        if mask is None:
            return {"note": note}
        # a detected box is drawn for feedback but never returned as the prompt
        img = self.prompter.overlay(bgr, mask, dict(spec, box=box or spec.get("box")))
        return {"image": _jpeg_data_url(img), "note": note}

    def api_run(self, req):
        with self.lock:
            if self.run_thread and self.run_thread.is_alive():
                return {"error": "a run is already in progress"}
        dataset, path = self.seq_path(req.get("dataset"))
        steps = pl.resolve_steps(req.get("mode"))
        spec = pl.normalize_spec(text=(req.get("prompt") or {}).get("text"),
                                 clicks=(req.get("prompt") or {}).get("points"),
                                 box=(req.get("prompt") or {}).get("box"),
                                 frame=(req.get("prompt") or {}).get("frame"))
        plan = pl.make_plan(path, dataset, steps, force=bool(req.get("force")),
                            force_from=self.args.force_from)
        if pl.plan_needs_prompt(plan, path) and pl.spec_is_empty(spec):
            return {"error": "step 3a will run but no object prompt was given -- "
                             "type what the object is, or click it in the frame"}
        # a fresh mask set + a new prompt = the user wants the masks redone:
        # re-run from 3a so the pick actually takes effect
        force_from = self.args.force_from
        st3a = next((p for p in plan if p["key"] == "step3a"), None)
        if (not pl.spec_is_empty(spec) and st3a is not None
                and not st3a["will_run"] and not force_from):
            force_from = "step3a"
            self.push({"type": "line", "key": "e2e", "cr": False,
                       "text": "[e2e] new object prompt on fresh masks -> "
                               "re-running from step3a"})
        extra = pl.parse_extra(self.args.extra)
        self.run = pl.PipelineRun(
            path, dataset, steps, prompt_spec=spec, extra=extra,
            viz=req.get("viz", self.args.viz), force=bool(req.get("force")),
            force_from=force_from, qa_headless=True,
            gpus=self.gpus, parallel=not self.args.no_parallel,
            on_event=self.push_event)
        self.run_thread = threading.Thread(target=self.run.run, daemon=True)
        self.run_thread.start()
        return {"ok": True}

    def push_event(self, e):
        # preview paths -> paths relative to the sequence folder, for /img
        if e.get("previews"):
            base = os.path.abspath(self.run.path)
            e = dict(e, previews=[os.path.relpath(os.path.abspath(p), base)
                                  for p in e["previews"]])
        self.push(e)

    def api_events(self, req):
        since = int(req.get("since", 0))
        with self.lock:
            out = [dict(e, _seq=s) for s, e in self.events if s > since]
            nxt = self.seq
        return {"events": out, "next": nxt}

    def api_cancel(self, req):
        if self.run:
            self.run.request_cancel()
        return {"ok": True}

    def api_clouds(self, req):
        """The .ply files of a sequence, for the result 3D viewer."""
        dataset, path = self.seq_path(req.get("dataset"))
        out = []
        for f in sorted(os.listdir(path)):
            if f.endswith(".ply"):
                out.append({"file": f,
                            "mb": round(os.path.getsize(os.path.join(path, f)) / 1e6, 1)})
        # the final object mesh first -- it is what the run was for
        out.sort(key=lambda c: (c["file"] != dataset + ".ply", c["file"]))
        return {"clouds": out}

    # -- static ------------------------------------------------------------

    _cloud_cache = {}  # (fullpath, mtime, n) -> packed bytes

    def cloud_bytes(self, dataset, rel, max_points=400000):
        """A .ply packed for WebGL: uint32 count, xyz float32, rgb uint8."""
        import struct
        import numpy as np
        import open3d as o3d
        _, path = self.seq_path(dataset)
        full = os.path.abspath(os.path.join(path, rel))
        if not full.startswith(os.path.abspath(path) + os.sep) or not full.endswith(".ply"):
            raise ValueError("not a .ply inside the sequence folder")
        key = (full, os.path.getmtime(full), max_points)
        if key in self._cloud_cache:
            return self._cloud_cache[key]
        pcd = o3d.io.read_point_cloud(full)
        pts = np.asarray(pcd.points, dtype=np.float32)
        cols = np.asarray(pcd.colors)
        if not len(pts):  # a mesh: take its vertices
            mesh = o3d.io.read_triangle_mesh(full)
            pts = np.asarray(mesh.vertices, dtype=np.float32)
            cols = np.asarray(mesh.vertex_colors)
        if not len(pts):
            raise ValueError("no vertices in %s" % rel)
        if len(pts) > max_points:  # uniform subsample keeps the shape
            sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
            pts = pts[sel]
            cols = cols[sel] if len(cols) else cols
        rgb = ((cols * 255).clip(0, 255).astype(np.uint8) if len(cols)
               else np.full((len(pts), 3), 180, np.uint8))
        data = struct.pack("<I", len(pts)) + pts.tobytes() + rgb.tobytes()
        self._cloud_cache.clear()  # keep at most one cloud in memory
        self._cloud_cache[key] = data
        return data

    def image_bytes(self, dataset, rel):
        _, path = self.seq_path(dataset)
        full = os.path.abspath(os.path.join(path, rel))
        if not full.startswith(os.path.abspath(path) + os.sep):
            raise ValueError("path escapes the sequence folder")
        with open(full, "rb") as f:
            return f.read(), ("image/png" if full.endswith(".png") else "image/jpeg")


def serve(args):
    core = E2EServer(args)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json", code=200):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = self.path.split("?")[0]
            if route in ("/", "/index.html"):
                self._send(_PAGE, "text/html; charset=utf-8")
            elif route == "/img":
                try:
                    from urllib.parse import parse_qs, urlparse
                    q = parse_qs(urlparse(self.path).query)
                    data, ctype = core.image_bytes(q["ds"][0], q["f"][0])
                    self._send(data, ctype)
                except Exception as exc:
                    self._send(json.dumps({"error": str(exc)}), code=404)
            elif route == "/cloud":
                try:
                    from urllib.parse import parse_qs, urlparse
                    q = parse_qs(urlparse(self.path).query)
                    n = int(q.get("n", ["400000"])[0])
                    data = core.cloud_bytes(q["ds"][0], q["f"][0], max_points=n)
                    self._send(data, "application/octet-stream")
                except Exception as exc:
                    self._send(json.dumps({"error": str(exc)}), code=404)
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            route = self.path.split("?")[0].replace("/api/", "api_")
            fn = getattr(core, route, None)
            if fn is None:
                self.send_error(404)
                return
            try:
                self._send(json.dumps(fn(req)))
            except Exception as exc:  # a broken request must not kill the server
                self._send(json.dumps({"error": "%s: %s" % (type(exc).__name__, exc)}))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://localhost:%d/" % server.server_port
    print("pose-dataset-studio dashboard: %s" % url)
    if core.gpus and len(core.gpus) > 1:
        print("GPUs: %s -- stages + preview on GPU %d, step3a on GPU %d during overlap"
              % (core.gpus, core.gpus[0], core.gpus[-1]))
    elif core.gpus:
        print("GPU: pinned to GPU %d (preview models + all accelerated stages)"
              % core.gpus[0])
    else:
        print("GPU: none requested (--gpu N to enable) -- CPU paths only")
    print("  headless server: on YOUR machine run")
    print("      ssh -N -L %d:localhost:%d <this-host>" % (server.server_port,
                                                           server.server_port))
    print("      then open %s in your browser" % url)
    print("  local machine:   just open that URL (--open does it for you)")
    print("Ctrl-C stops the server.")
    if args.open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        if core.run:
            core.run.request_cancel()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Browser-driven end-to-end pipeline (steps 2-6): plan, prompt, "
                    "run, watch -- all in one page. Works over SSH without X.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", nargs="?", default=None,
                        help="sequence folder to pre-select in the page (optional)")
    parser.add_argument("--data-root", default=".",
                        help="parent folder the page scans for sequence folders")
    pl.add_e2e_args(parser)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (keep loopback; forward the port over SSH)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true",
                        help="open the local browser (local machines only)")
    args = parser.parse_args()

    try:                             # validate early, fail with a message
        pl.resolve_steps(args.mode)
        pl.parse_extra(args.extra)
        pl.gpus_from_args(args)
    except ValueError as exc:
        sys.exit(str(exc))

    if args.dry_run:
        if not args.dataset:
            sys.exit("--dry-run needs a dataset")
        root = normalize_root(args.data_root)
        if root == "./":
            root = ""
        ds = args.dataset.replace("\\", "/").rstrip("/")
        for cand in (root + ds + "/", ds + "/"):
            if os.path.isdir(cand):
                pl.print_plan(pl.make_plan(cand, ds.rsplit("/", 1)[-1],
                                           pl.resolve_steps(args.mode),
                                           force=args.force, force_from=args.force_from))
                return
        sys.exit("sequence folder not found: %r" % args.dataset)

    serve(args)


if __name__ == "__main__":
    main()
