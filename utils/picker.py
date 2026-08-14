"""
utils/picker.py
---------------

A one-page, dependency-free picker for choosing the object to segment.

Why a browser page and not an OpenCV window: this pipeline is normally driven
over SSH on a machine with GPUs and no display, where `cv2.imshow` needs an X
server, X forwarding is slow at 720p and Wayland/macOS clients often cannot do
it at all. The page is served by Python's own `http.server` on localhost, so:

* **headless server** -- forward the port once
  (`ssh -N -L 8765:localhost:8765 user@host`) and open the URL in the browser
  on your own machine. Nothing is installed on either side.
* **local workstation** -- the same URL just opens; `--pick-open` launches the
  browser for you.

The page shows one frame of the sequence. Click the object (right-click to
push a wrong region back out), or drag a box, or type what it is and let
Grounding DINO find it; the mask is re-computed after every interaction and
drawn back over the frame, so what you accept is exactly what will be
propagated.

`pick_prompt()` blocks until the page posts Accept or Cancel and returns the
accepted prompt, which the caller then feeds to SAM 2's video predictor.
"""

import base64
import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

_PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>pose-dataset-studio - pick the object</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 14px/1.5 system-ui, sans-serif; background: #14161a; color: #e8eaed; }
  header { padding: 10px 16px; background: #1d2026; border-bottom: 1px solid #2b2f38;
           display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  #wrap { padding: 12px 16px 24px; }
  canvas { max-width: 100%; border: 1px solid #2b2f38; border-radius: 6px;
           cursor: crosshair; display: block; }
  button, input, select { font: inherit; background: #262a33; color: #e8eaed;
           border: 1px solid #3a4050; border-radius: 6px; padding: 6px 10px; }
  button:hover { background: #313743; }
  button.go { background: #2f6f43; border-color: #3d8b56; }
  button.stop { background: #6f2f36; border-color: #8b3d46; }
  #status { color: #9aa3b2; margin-left: auto; }
  .hint { color: #9aa3b2; padding: 8px 16px 0; }
  b { color: #fff; }
</style>
<header>
  <span>frame</span>
  <button id="prev">&#9664;</button>
  <input id="frame" type="number" value="__FRAME__" min="0" max="__NFRAMES_MAX__" style="width:7em">
  <button id="next">&#9654;</button>
  <select id="mode">
    <option value="point">click points</option>
    <option value="box">drag a box</option>
  </select>
  <input id="text" placeholder="or describe it: white product box" style="width:22em">
  <button id="detect">detect</button>
  <button id="undo">undo</button>
  <button id="clear">clear</button>
  <button id="accept" class="go">accept &amp; propagate</button>
  <button id="cancel" class="stop">cancel</button>
  <span id="status">ready</span>
</header>
<p class="hint">Left-click = part of the object &middot; right-click = <b>not</b> the object &middot;
   drag in box mode &middot; the mask updates after every action.</p>
<div id="wrap"><canvas id="cv"></canvas></div>
<script>
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const st = document.getElementById('status');
let img = new Image(), pts = [], box = null, drag = null, frame = __FRAME__;
let overlay = null, busy = false;

function say(s) { st.textContent = s; }
function draw() {
  const src = overlay || img;
  if (!src.width) return;
  cv.width = src.width; cv.height = src.height;
  ctx.drawImage(src, 0, 0);
  if (box) { ctx.strokeStyle = '#ffb020'; ctx.lineWidth = 3; ctx.strokeRect(box[0], box[1], box[2]-box[0], box[3]-box[1]); }
  for (const p of pts) {
    ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, 7);
    ctx.fillStyle = p[2] ? '#22c55e' : '#ef4444'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
  }
}
function at(e) {
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height];
}
async function post(path, body) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
                              body: JSON.stringify(body)});
  return await r.json();
}
async function loadFrame(i) {
  say('loading frame ' + i + ' ...');
  const r = await post('/api/frame', {frame: i});
  frame = r.frame; document.getElementById('frame').value = frame;
  overlay = null; pts = []; box = null;
  img = new Image(); img.onload = () => { draw(); say('frame ' + frame); }; img.src = r.image;
}
async function refresh(spec) {
  if (busy) return; busy = true; say('segmenting ...');
  try {
    const r = await post('/api/preview', spec);
    if (r.error) { say(r.error); busy = false; return; }
    if (r.box) box = r.box;
    const o = new Image();
    o.onload = () => { overlay = o; draw(); say(r.note); busy = false; };
    o.src = r.image;
  } catch (err) { say('' + err); busy = false; }
}
function previewPrompt() {
  if (!pts.length && !box) { overlay = null; draw(); say('cleared'); return; }
  refresh({frame: frame, points: pts, box: box});
}
cv.addEventListener('mousedown', e => {
  if (document.getElementById('mode').value !== 'box') return;
  drag = at(e); box = null; e.preventDefault();
});
cv.addEventListener('mousemove', e => {
  if (!drag) return;
  const p = at(e);
  box = [Math.min(drag[0], p[0]), Math.min(drag[1], p[1]), Math.max(drag[0], p[0]), Math.max(drag[1], p[1])];
  draw();
});
cv.addEventListener('mouseup', e => { if (drag) { drag = null; previewPrompt(); } });
cv.addEventListener('click', e => {
  if (document.getElementById('mode').value !== 'point') return;
  pts.push([...at(e), 1]); draw(); previewPrompt();
});
cv.addEventListener('contextmenu', e => {
  e.preventDefault();
  if (document.getElementById('mode').value !== 'point') return;
  pts.push([...at(e), 0]); draw(); previewPrompt();
});
document.getElementById('detect').onclick = () => {
  const t = document.getElementById('text').value.trim();
  if (!t) { say('type what the object is first'); return; }
  pts = []; box = null; refresh({frame: frame, text: t});
};
document.getElementById('undo').onclick = () => { pts.pop(); previewPrompt(); };
document.getElementById('clear').onclick = () => { pts = []; box = null; overlay = null; draw(); say('cleared'); };
document.getElementById('prev').onclick = () => loadFrame(Math.max(0, frame - __STEP__));
document.getElementById('next').onclick = () => loadFrame(frame + __STEP__);
document.getElementById('frame').onchange = e => loadFrame(parseInt(e.target.value || '0', 10));
document.getElementById('accept').onclick = async () => {
  const t = document.getElementById('text').value.trim();
  const r = await post('/api/accept', {frame: frame, points: pts, box: box, text: (!pts.length && !box) ? t : ''});
  say(r.ok ? 'accepted - go back to the terminal' : (r.error || 'nothing to accept'));
};
document.getElementById('cancel').onclick = async () => {
  await post('/api/cancel', {}); say('cancelled - go back to the terminal');
};
loadFrame(frame);
</script>
"""


def _jpeg_data_url(bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode preview JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _local_hostname():
    try:
        return socket.gethostname()
    except Exception:
        return "the server"


def pick_prompt(frame_provider, n_frames, preview_fn, seed_frame=0, step=10,
                host="127.0.0.1", port=8765, open_browser=False):
    """Serve the picker and block until the page accepts or cancels.

    frame_provider : callable(i) -> BGR frame
    n_frames       : how many frames the sequence has (bounds the frame stepper)
    preview_fn     : callable(spec) -> (mask bool (h,w), note str, box or None)
                     spec is {'frame': int, 'points': [[x,y,label],...],
                              'box': [x0,y0,x1,y1] or None, 'text': str}
    Returns the accepted spec, or None if the page cancelled.
    """
    state = {"result": None, "done": threading.Event()}
    lock = threading.Lock()  # the models are not re-entrant

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the terminal readable
            pass

        def _send(self, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            page = (_PAGE.replace("__FRAME__", str(seed_frame))
                         .replace("__NFRAMES_MAX__", str(max(0, n_frames - 1)))
                         .replace("__STEP__", str(step)))
            self._send(page, "text/html; charset=utf-8")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            route = self.path.split("?")[0]
            try:
                if route == "/api/frame":
                    i = max(0, min(n_frames - 1, int(req.get("frame", 0))))
                    self._send(json.dumps({"frame": i,
                                           "image": _jpeg_data_url(frame_provider(i))}))
                elif route == "/api/preview":
                    with lock:
                        mask, note, box = preview_fn(req)
                    frame = frame_provider(int(req.get("frame", 0)))
                    from utils.segmentation import overlay as _ov
                    img = _ov(frame, mask, box=box)
                    self._send(json.dumps({"image": _jpeg_data_url(img), "note": note,
                                           "box": list(box) if box is not None else None}))
                elif route == "/api/accept":
                    if not req.get("points") and not req.get("box") and not req.get("text"):
                        self._send(json.dumps({"ok": False, "error": "no prompt given"}))
                        return
                    state["result"] = req
                    self._send(json.dumps({"ok": True}))
                    state["done"].set()
                elif route == "/api/cancel":
                    state["result"] = None
                    self._send(json.dumps({"ok": True}))
                    state["done"].set()
                else:
                    self.send_error(404)
            except Exception as exc:  # a broken prompt must not kill the server
                self._send(json.dumps({"error": "%s: %s" % (type(exc).__name__, exc)}))

    server = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d/" % ("localhost" if host in ("127.0.0.1", "0.0.0.0") else host,
                             server.server_port)
    print("\nPicker running at %s" % url)
    print("  headless server: on YOUR machine run")
    print("      ssh -N -L %d:localhost:%d %s" % (server.server_port, server.server_port,
                                                  _local_hostname()))
    print("      then open %s in your browser" % url)
    print("  local machine:   just open that URL (--pick-open does it for you)")
    print("Waiting for the page to accept a prompt (Ctrl-C to abort) ...")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        while not state["done"].wait(0.25):
            pass
    except KeyboardInterrupt:
        print("\nAborted.")
        state["result"] = None
    finally:
        server.shutdown()
        server.server_close()
    return state["result"]
