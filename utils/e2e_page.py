"""
utils/e2e_page.py
-----------------

The single-page dashboard served by `run_e2e_web.py`. Pure presentation --
every API it talks to lives in run_e2e_web.py, and the events it renders come
from utils/pipeline.py.

The page is *phase-aware*, which is what makes it feel like a tool rather
than a log viewer:

  setup    the object picker is the main event: frame canvas, click/box/text
           prompt, live SAM mask, explicit "use this mask" confirmation.
  running  the picker collapses to a one-line chip; the stage cards carry
           real progress bars (tqdm output is parsed server-side into
           percent events, never shown as raw lines), the newest preview
           image is front and centre, and the raw log sits folded at the
           bottom for when it is actually wanted.
  done     the run summary and the preview gallery (with playback) take
           over; the picker chip offers re-opening for the next run.
"""

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>pose-dataset-studio</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 system-ui, sans-serif; background: #14161a; color: #e8eaed; }
  header { padding: 10px 16px; background: #1d2026; border-bottom: 1px solid #2b2f38;
           display: flex; gap: 10px; align-items: center; flex-wrap: wrap; position: sticky; top: 0; z-index: 5; }
  main { display: grid; grid-template-columns: 340px 1fr; gap: 14px; padding: 14px 16px; align-items: start; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  h3 { margin: 4px 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: #9aa3b2; }
  button, input, select { font: inherit; background: #262a33; color: #e8eaed;
           border: 1px solid #3a4050; border-radius: 6px; padding: 6px 10px; }
  button:hover { background: #313743; } button:disabled { opacity: .45; }
  button.go { background: #2f6f43; border-color: #3d8b56; font-weight: 600; }
  button.stop { background: #6f2f36; border-color: #8b3d46; }
  #status { color: #9aa3b2; margin-left: auto; }
  .card { background: #1b1e24; border: 1px solid #2b2f38; border-radius: 8px; padding: 8px 12px; margin-bottom: 8px; cursor: default; }
  .card.excluded { opacity: .38; }
  .card .top { display: flex; gap: 10px; align-items: baseline; }
  .card .key { font-family: ui-monospace, monospace; color: #9aa3b2; width: 56px; }
  .card .title { flex: 1; }
  .chip { font-size: 11px; padding: 1px 8px; border-radius: 10px; border: 1px solid; white-space: nowrap; }
  .chip.fresh   { color: #7dd3a0; border-color: #2f6f43; }
  .chip.stale   { color: #ffd479; border-color: #8a6d1d; }
  .chip.pending { color: #93c5fd; border-color: #33517a; }
  .chip.blocked { color: #f0a3a3; border-color: #8b3d46; }
  .chip.running { color: #93c5fd; border-color: #33517a; animation: pulse 1.2s infinite; }
  .chip.ok      { color: #7dd3a0; border-color: #2f6f43; }
  .chip.failed  { color: #f0a3a3; border-color: #8b3d46; }
  .chip.skipped { color: #9aa3b2; border-color: #3a4050; }
  @keyframes pulse { 50% { opacity: .4; } }
  .card .why { display: block; font-size: 12px; color: #9aa3b2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card .sum { display: block; font-size: 12px; color: #cbd5e1; }
  .pbar { height: 6px; background: #262a33; border-radius: 3px; margin-top: 6px; overflow: hidden; }
  .pbar > div { height: 100%; background: #3d8b56; border-radius: 3px; transition: width .3s; }
  .ptext { font-size: 11px; color: #93c5fd; font-family: ui-monospace, monospace; }

  .panel { background: #1b1e24; border: 1px solid #2b2f38; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  #prompt-box { border-color: #8a6d1d; }
  #prompt-chip { display: flex; gap: 10px; align-items: center; }
  canvas { max-width: 100%; border: 1px solid #2b2f38; border-radius: 6px; cursor: crosshair; display: block; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 6px 0; }
  .hint { font-size: 12px; color: #9aa3b2; }
  b { color: #fff; }

  #live-now { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }
  #live-img { max-width: 100%; border: 1px solid #2b2f38; border-radius: 8px; display: block; }
  #gallery { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  #gallery img { height: 96px; border: 1px solid #2b2f38; border-radius: 6px; cursor: pointer; }
  #summary table { border-collapse: collapse; font-size: 13px; width: 100%; }
  #summary td, #summary th { border: 1px solid #2b2f38; padding: 4px 10px; text-align: left; }
  #results-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
  @media (max-width: 1200px) { #results-cols { grid-template-columns: 1fr; } }
  #glcv { width: 100%; height: 480px; background: #101216; border: 1px solid #2b2f38;
          border-radius: 8px; cursor: grab; display: block; }
  #results-left #gallery img { height: 88px; }
  details#logbox { margin-top: 12px; }
  details#logbox summary { cursor: pointer; color: #9aa3b2; font-size: 13px; }
  #log { background: #101216; border: 1px solid #2b2f38; border-radius: 8px; padding: 10px 12px; margin-top: 8px;
         font: 12px/1.45 ui-monospace, monospace; max-height: 300px; overflow-y: auto; white-space: pre-wrap; }
  #log .cmd { color: #ffd479; }
  #big { position: fixed; inset: 0; background: rgba(10,11,14,.92); display: none;
         align-items: center; justify-content: center; z-index: 20; flex-direction: column; gap: 10px; }
  #big img { max-width: 94vw; max-height: 86vh; border-radius: 8px; }
  #big .bar { color: #9aa3b2; }

  /* phase visibility: the page shows what the current phase needs, only */
  main.phase-setup   #live-box, main.phase-setup   #results-box, main.phase-setup #prompt-chip { display: none; }
  main.phase-running #prompt-box, main.phase-running #results-box { display: none; }
  main.phase-done    #prompt-box, main.phase-done    #live-box { display: none; }
</style>

<header>
  <b>pose-dataset-studio</b>
  <select id="dataset"></select>
  <select id="mode">
    <option value="whole">whole (2 &rarr; 6)</option>
    <option value="label">label (2 &rarr; 5)</option>
    <option value="custom">custom (click cards)</option>
  </select>
  <select id="viz">
    <option value="preview">viz: preview</option>
    <option value="full">viz: full</option>
    <option value="off">viz: off</option>
  </select>
  <label class="hint"><input type="checkbox" id="force"> force</label>
  <button id="run" class="go">Run</button>
  <button id="cancel" class="stop" disabled>Cancel</button>
  <span id="status">loading ...</span>
</header>

<main id="main" class="phase-setup">
<section>
  <h3>Pipeline</h3>
  <div id="cards"></div>
  <div class="hint" id="cards-hint">A stage runs only when its outputs are missing or older than
  its inputs; in <b>custom</b> mode click a card to include/exclude it.</div>
  <div id="summary"></div>
</section>

<section>
  <div id="prompt-chip" class="panel">
    <span id="chip-text" class="hint">object prompt</span>
    <span id="chip-state" class="chip ok" style="display:none">confirmed</span>
    <button id="chip-edit" style="margin-left:auto">edit prompt</button>
  </div>

  <div id="prompt-box" class="panel">
    <h3 id="ptitle">Step 3a needs to know the object</h3>
    <div class="row" id="pexisting-row" style="display:none">
      <span class="hint">current masks:</span><div id="pexisting" style="display:flex;gap:6px"></div>
    </div>
    <div class="row">
      <input id="ptext" placeholder='describe it: "white product box"' style="width: 24em">
      <button id="pdetect">detect</button>
      <span class="hint">or click it below &middot; left = object, right = <b>not</b> the object &middot; shift+drag = box</span>
    </div>
    <div class="row">
      <span class="hint">seed frame</span>
      <button id="pprev">&#9664;</button>
      <input id="pframe" type="number" value="0" min="0" style="width: 7em">
      <button id="pnext">&#9654;</button>
      <button id="pundo">undo</button>
      <button id="pclear">clear</button>
      <button id="pconfirm" class="go" disabled>&#10003; use this mask</button>
      <span id="pconfirmed" class="chip ok" style="display:none"></span>
      <span id="pmodels" class="chip pending">models: starting &hellip;</span>
    </div>
    <div class="row"><span id="pstatus" class="hint"></span></div>
    <canvas id="pcv"></canvas>
  </div>

  <div id="live-box" class="panel">
    <div id="live-now">
      <h3 style="margin:0">Now running</h3>
      <span id="live-stage" class="chip running">-</span>
      <span id="live-elapsed" class="hint"></span>
    </div>
    <img id="live-img" style="display:none">
    <div class="hint" id="live-note">stage previews appear here as they are produced</div>
  </div>

  <div id="results-box" class="panel">
    <div id="results-cols">
      <div id="results-left">
        <div class="row">
          <h3 style="margin:0">Previews</h3>
          <button id="play" disabled>&#9654; play overlays</button>
          <span class="hint" id="gallery-note"></span>
        </div>
        <!-- the gallery node is moved in here when a run finishes -->
      </div>
      <div id="results-right">
        <div class="row">
          <h3 style="margin:0">3D</h3>
          <select id="cloud-file"></select>
          <button id="cloud-reset">reset view</button>
          <input id="cloud-size" type="range" min="1" max="6" step="0.5" value="2"
                 title="point size" style="width:90px">
          <span class="hint" id="cloud-info"></span>
        </div>
        <canvas id="glcv"></canvas>
        <div class="hint">drag = orbit &middot; wheel = zoom &middot; shift+drag (or right-drag) = pan &middot; double-click = reset</div>
      </div>
    </div>
  </div>

  <div id="gallery"></div>

  <details id="logbox">
    <summary>raw log &middot; <span id="logcount">0</span> lines (progress bars live in the stage cards, not here)</summary>
    <div id="log"></div>
  </details>
</section>
</main>

<div id="big"><img><div class="bar"></div></div>

<script>
const $ = id => document.getElementById(id);
const state = { plan: [], excluded: new Set(), running: false, since: 0,
                previews: [], nframes: 1, frame: 0, pts: [], box: null,
                drag: null, img: new Image(), overlay: null,
                confirmedSpec: null, needsPrompt: false, models: {status: 'cold'},
                runStart: 0, cliSteps: null };

function say(s) { $('status').textContent = s; }
async function post(path, body) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
                               body: JSON.stringify(body || {})});
  return await r.json();
}
function setPhase(p) { $('main').className = 'phase-' + p; }

/* ---------------- plan cards (with live progress bars) ---------------- */
function chip(cls, txt) { return `<span class="chip ${cls}">${txt || cls}</span>`; }
function renderCards() {
  const custom = $('mode').value === 'custom';
  $('cards').innerHTML = state.plan.map(p => {
    const ex = custom && state.excluded.has(p.key);
    const live = p.live || {};
    const c = live.chip || (p.will_run ? (p.state === 'stale' ? 'stale' : 'pending') : p.state);
    const prog = (live.chip === 'running' && live.progress) ? live.progress : null;
    return `<div class="card ${ex ? 'excluded' : ''}" data-key="${p.key}" ${custom ? 'style="cursor:pointer"' : ''}>
      <div class="top"><span class="key">${p.key}</span>
        <span class="title">${p.title}</span>${chip(c, live.chipText)}</div>
      ${prog ? `<div class="pbar"><div style="width:${prog.pct}%"></div></div>
                <span class="ptext">${prog.pct}%  ${prog.text}</span>`
             : `<span class="why">${live.summary || live.why || p.reason}</span>`}
    </div>`;
  }).join('');
  if (custom && !state.running) for (const el of $('cards').children)
    el.onclick = () => { const k = el.dataset.key;
      state.excluded.has(k) ? state.excluded.delete(k) : state.excluded.add(k);
      refreshPlan(); };
}
function liveCard(key, patch) {
  const p = state.plan.find(x => x.key === key);
  if (p) { p.live = Object.assign(p.live || {}, patch); renderCards(); }
}
function currentSteps() {
  const m = $('mode').value;
  if (m !== 'custom') return m;
  if (!state.plan.length) return state.cliSteps || 'whole';
  const keys = state.plan.map(p => p.key).filter(k => !state.excluded.has(k));
  return keys.join(',') || 'whole';
}
async function refreshPlan() {
  const ds = $('dataset').value;
  if (!ds) return;
  const r = await post('/api/plan', {dataset: ds, mode: currentSteps(), force: $('force').checked});
  if (r.error) { say(r.error); return; }
  state.plan = r.plan;
  state.needsPrompt = r.needs_prompt;
  renderCards();
  $('prompt-box').style.display = r.can_prompt ? '' : 'none';
  $('ptitle').textContent = r.needs_prompt
    ? 'Pick the object for step 3a -- check the mask, then confirm it'
    : (r.mask_state === 'fresh'
       ? 'Masks are up to date -- pick a new prompt only to redo them (re-runs 3a→3b→4→5)'
       : 'Step 3a object prompt');
  const ex = r.existing_previews || [];
  $('pexisting-row').style.display = ex.length ? 'flex' : 'none';
  $('pexisting').innerHTML = ex.map(f => {
    const src = '/img?ds=' + encodeURIComponent(ds) + '&f=' + encodeURIComponent(f);
    return `<img src="${src}" title="${f}" style="height:64px;border:1px solid #2b2f38;border-radius:4px;cursor:pointer" onclick="window.open('${src}')">`;
  }).join('');
  if (r.can_prompt && !state.img.width) loadFrame(r.suggest_frame || 0);
  state.nframes = r.n_frames || 1;
  say(r.note || 'ready');
}

/* ---------------- model preload status ---------------- */
function renderModels() {
  const m = state.models, el = $('pmodels');
  if (m.status === 'ready') { el.className = 'chip ok';
    el.textContent = 'models ready · ' + (m.device || ''); }
  else if (m.status === 'loading') { el.className = 'chip pending';
    el.textContent = 'models loading ... ' + (m.seconds ? m.seconds + 's' : ''); }
  else if (m.status === 'unavailable') { el.className = 'chip failed';
    el.textContent = 'no live preview'; el.title = m.error || ''; }
  else { el.className = 'chip pending'; el.textContent = 'models: starting ...'; }
}
async function pollModels() {
  try { state.models = await post('/api/models'); } catch (e) {}
  renderModels();
  if (state.models.status !== 'ready' && state.models.status !== 'unavailable')
    setTimeout(pollModels, 1500);
}

/* ---------------- prompt canvas + confirmation ---------------- */
const pcv = $('pcv'), pctx = pcv.getContext('2d');
function pdraw() {
  const src = state.overlay || state.img;
  if (!src.width) return;
  pcv.width = src.width; pcv.height = src.height;
  pctx.drawImage(src, 0, 0);
  if (state.box) { pctx.strokeStyle = '#ffb020'; pctx.lineWidth = 3;
    pctx.strokeRect(state.box[0], state.box[1], state.box[2]-state.box[0], state.box[3]-state.box[1]); }
  for (const p of state.pts) {
    pctx.beginPath(); pctx.arc(p[0], p[1], 7, 0, 7);
    pctx.fillStyle = p[2] ? '#22c55e' : '#ef4444'; pctx.fill();
    pctx.strokeStyle = '#fff'; pctx.lineWidth = 2; pctx.stroke();
  }
}
function pat(e) { const r = pcv.getBoundingClientRect();
  return [(e.clientX - r.left) * pcv.width / r.width, (e.clientY - r.top) * pcv.height / r.height]; }
async function loadFrame(i) {
  // no-op when we are already on that frame: the frame input fires a second
  // 'change' when it loses focus to a canvas click, and reloading here would
  // silently wipe the click that was just placed
  if (i === state.frame && state.img.width) return;
  const r = await post('/api/frame', {dataset: $('dataset').value, frame: i});
  if (r.error) { $('pstatus').textContent = r.error; return; }
  state.frame = r.frame; $('pframe').value = r.frame; $('pframe').max = r.n_frames - 1;
  state.nframes = r.n_frames;
  state.overlay = null; state.pts = []; state.box = null;
  resetConfirm('');
  state.img = new Image(); state.img.onload = pdraw; state.img.src = r.image;
}
function resetConfirm(reason) {
  if (state.confirmedSpec) $('pstatus').textContent = reason ?? 'prompt changed -- preview and confirm again';
  state.confirmedSpec = null;
  $('pconfirm').disabled = !state.overlay;
  $('pconfirmed').style.display = 'none';
}
let busy = false;
async function livePreview(spec) {
  if (busy) return; busy = true;
  $('pstatus').textContent = state.models.status === 'ready'
    ? 'segmenting ...'
    : 'models are still loading -- your prompt is queued, the mask will appear as soon as they are ready';
  try {
    const r = await post('/api/preview', Object.assign({dataset: $('dataset').value}, spec));
    if (r.error) { $('pstatus').textContent = r.error; busy = false; return; }
    if (r.image) { const o = new Image();
      o.onload = () => { state.overlay = o; pdraw();
        $('pstatus').textContent = r.note + '  -- happy with it? press "use this mask"';
        $('pconfirm').disabled = false; busy = false; };
      o.src = r.image; }
    else { $('pstatus').textContent = r.note || ''; busy = false; }
  } catch (err) { $('pstatus').textContent = '' + err; busy = false; }
}
function promptSpec() {
  return { frame: state.frame, points: state.pts, box: state.box,
           text: (!state.pts.length && !state.box) ? $('ptext').value.trim() : '' };
}
function previewNow() {
  if (!state.pts.length && !state.box) { state.overlay = null; pdraw(); return; }
  livePreview({frame: state.frame, points: state.pts, box: state.box});
}
pcv.addEventListener('mousedown', e => { if (e.button !== 0 || e.shiftKey === false) return;
  state.drag = pat(e); state.box = null; resetConfirm(); e.preventDefault(); });
pcv.addEventListener('mousemove', e => { if (!state.drag) return;
  const p = pat(e);
  state.box = [Math.min(state.drag[0], p[0]), Math.min(state.drag[1], p[1]),
               Math.max(state.drag[0], p[0]), Math.max(state.drag[1], p[1])];
  pdraw(); });
pcv.addEventListener('mouseup', e => { if (state.drag) { state.drag = null; previewNow(); } });
pcv.addEventListener('click', e => { if (e.shiftKey) return;
  state.pts.push([...pat(e), 1]); resetConfirm(); pdraw(); previewNow(); });
pcv.addEventListener('contextmenu', e => { e.preventDefault();
  state.pts.push([...pat(e), 0]); resetConfirm(); pdraw(); previewNow(); });
$('pdetect').onclick = () => { const t = $('ptext').value.trim();
  if (!t) { $('pstatus').textContent = 'type what the object is first'; return; }
  state.pts = []; state.box = null; resetConfirm(); livePreview({frame: state.frame, text: t}); };
$('ptext').oninput = () => resetConfirm();
$('pundo').onclick = () => { state.pts.pop(); resetConfirm(); pdraw(); previewNow(); };
$('pclear').onclick = () => { state.pts = []; state.box = null; state.overlay = null;
  resetConfirm(''); pdraw(); $('pstatus').textContent = 'cleared'; };
$('pconfirm').onclick = () => {
  if (!state.overlay) return;
  state.confirmedSpec = promptSpec();
  $('pconfirmed').textContent = 'mask confirmed @ frame ' + state.frame;
  $('pconfirmed').style.display = 'inline';
  $('pstatus').textContent = 'confirmed -- Run will propagate exactly this mask';
};
$('pprev').onclick = () => loadFrame(Math.max(0, state.frame - Math.max(1, Math.floor(state.nframes / 20))));
$('pnext').onclick = () => loadFrame(Math.min(state.nframes - 1, state.frame + Math.max(1, Math.floor(state.nframes / 20))));
$('pframe').onchange = e => loadFrame(parseInt(e.target.value || '0', 10));
$('chip-edit').onclick = () => { if (!state.running) setPhase('setup'); };

/* ---------------- run / events ---------------- */
function addLog(text, cls) {
  const el = document.createElement('div');
  if (cls) el.className = cls;
  el.textContent = text;
  $('log').appendChild(el);
  while ($('log').children.length > 500) $('log').removeChild($('log').firstChild);
  $('logcount').textContent = $('log').children.length;
  if ($('logbox').open) $('log').scrollTop = $('log').scrollHeight;
}
function describeSpec(s) {
  if (!s) return 'no prompt (masks already fresh)';
  if (s.text) return `"${s.text}" @ frame ${s.frame}`;
  const n = (s.points || []).length;
  return (n ? n + ' click(s)' : 'box') + ' @ frame ' + s.frame;
}
$('run').onclick = async () => {
  const ds = $('dataset').value;
  if (!ds) { say('choose a sequence first'); return; }
  const panelVisible = $('prompt-box').style.display !== 'none';
  const live = promptSpec();
  const hasSpec = !!(live.text || live.points.length || live.box);
  if (panelVisible && !state.confirmedSpec) {
    if (hasSpec) { say('check the mask first: preview it, then press "use this mask"'); return; }
    if (state.needsPrompt) { say('step 3a needs a confirmed mask -- click the object (or type it), check the preview, then "use this mask"'); return; }
  }
  const spec = state.confirmedSpec || (hasSpec ? live : null);
  const r = await post('/api/run', { dataset: ds, mode: currentSteps(), viz: $('viz').value,
                                     force: $('force').checked, prompt: spec || {} });
  if (r.error) { say(r.error); return; }
  state.running = true; state.since = 0; state.previews = []; state.runStart = Date.now();
  for (const p of state.plan) p.live = null;
  $('logbox').before($('gallery'));   // gallery back under the live view while running
  $('gallery').innerHTML = ''; $('summary').innerHTML = ''; $('log').innerHTML = '';
  $('live-img').style.display = 'none'; $('live-note').style.display = '';
  $('chip-text').textContent = 'object: ' + describeSpec(state.confirmedSpec);
  $('chip-state').style.display = state.confirmedSpec ? 'inline' : 'none';
  $('chip-edit').disabled = true;
  $('run').disabled = true; $('cancel').disabled = false; $('play').disabled = true;
  setPhase('running');
  say('running ...');
  tick(); poll();
};
$('cancel').onclick = () => post('/api/cancel');
function tick() {
  if (!state.running) return;
  $('live-elapsed').textContent = fmtSec((Date.now() - state.runStart) / 1000) + ' elapsed';
  setTimeout(tick, 1000);
}
async function poll() {
  if (!state.running) return;
  const r = await post('/api/events', {since: state.since});
  state.since = r.next;
  for (const e of r.events) handleEvent(e);
  if (state.running) setTimeout(poll, 600);
}
function handleEvent(e) {
  if (e.type === 'plan') { state.plan = e.plan; renderCards(); }
  else if (e.type === 'stage_skip') liveCard(e.key, {chip: 'skipped', why: e.reason});
  else if (e.type === 'stage_start') {
    liveCard(e.key, {chip: 'running', why: 'running ...'});
    $('live-stage').textContent = runningStages().join(' + ') || e.key;
    addLog('$ ' + e.cmd, 'cmd');
  }
  else if (e.type === 'progress') liveCard(e.key, {progress: {pct: e.pct, text: e.text}});
  else if (e.type === 'line') addLog(e.text);
  else if (e.type === 'stage_done') {
    liveCard(e.key, {chip: 'ok', chipText: 'ok ' + fmtSec(e.seconds), progress: null,
                     summary: e.summary});
    $('live-stage').textContent = runningStages().join(' + ') || 'finishing ...';
    for (const p of e.previews || []) addPreview(p);
  }
  else if (e.type === 'stage_fail') {
    liveCard(e.key, {chip: 'failed', chipText: 'failed (rc ' + e.returncode + ')', progress: null});
    addLog('stage failed -- open the raw log below for details', 'cmd');
    $('logbox').open = true;
  }
  else if (e.type === 'done') {
    state.running = false;
    $('run').disabled = false; $('cancel').disabled = true; $('chip-edit').disabled = false;
    say(e.ok ? 'finished in ' + fmtSec(e.seconds) : 'stopped after ' + fmtSec(e.seconds));
    renderSummary(e.rows, e.log_dir);
    if (state.previews.length) $('play').disabled = false;
    $('gallery-note').textContent = state.previews.length + ' preview image(s)';
    setPhase('done');
    $('results-left').appendChild($('gallery'));   // previews left, 3D right
    showResultClouds();
    refreshPlan();
  }
}
function runningStages() {
  return state.plan.filter(p => p.live && p.live.chip === 'running').map(p => p.key);
}
function fmtSec(s) { return s >= 90 ? (s / 60).toFixed(1) + 'm' : s.toFixed(1) + 's'; }
function renderSummary(rows, logDir) {
  let html = '<h3>Run summary</h3><table><tr><th>stage</th><th>result</th><th>time</th><th></th></tr>';
  for (const r of rows)
    html += `<tr><td>${r.key}</td><td>${r.result}</td><td>${r.seconds ? fmtSec(r.seconds) : ''}</td>` +
            `<td>${r.summary || r.reason || ''}</td></tr>`;
  html += `</table><div class="hint">logs: ${logDir}</div>`;
  $('summary').innerHTML = html;
}

/* ---------------- previews: live image + gallery + viewer ---------------- */
function imgUrl(relpath) {
  return '/img?ds=' + encodeURIComponent($('dataset').value) + '&f=' + encodeURIComponent(relpath);
}
function addPreview(relpath) {
  if (state.previews.includes(relpath)) return;
  state.previews.push(relpath);
  $('live-img').src = imgUrl(relpath) + '&t=' + Date.now();
  $('live-img').style.display = ''; $('live-note').style.display = 'none';
  const img = document.createElement('img');
  img.src = imgUrl(relpath);
  img.title = relpath;
  img.onclick = () => showBig(state.previews.indexOf(relpath));
  $('gallery').appendChild(img);
}
let bigIdx = 0, playTimer = null;
function showBig(i) {
  bigIdx = (i + state.previews.length) % state.previews.length;
  const f = state.previews[bigIdx];
  $('big').style.display = 'flex';
  $('big').querySelector('img').src = imgUrl(f);
  $('big').querySelector('.bar').textContent =
    (bigIdx + 1) + ' / ' + state.previews.length + '  ' + f + '   (arrows step, esc/click closes, space plays)';
}
function stopPlay() { if (playTimer) { clearInterval(playTimer); playTimer = null; } }
$('big').onclick = () => { stopPlay(); $('big').style.display = 'none'; };
$('play').onclick = () => { if (state.previews.length) { showBig(0); startPlay(); } };
function startPlay() { stopPlay(); playTimer = setInterval(() => showBig(bigIdx + 1), 180); }
document.addEventListener('keydown', e => {
  if ($('big').style.display !== 'flex') return;
  if (e.key === 'Escape') { stopPlay(); $('big').style.display = 'none'; }
  else if (e.key === 'ArrowRight') { stopPlay(); showBig(bigIdx + 1); }
  else if (e.key === 'ArrowLeft') { stopPlay(); showBig(bigIdx - 1); }
  else if (e.key === ' ') { e.preventDefault(); playTimer ? stopPlay() : startPlay(); }
});

/* ---------------- 3D point-cloud viewer (self-contained WebGL) ---------------- */
const glv = { gl: null, prog: null, n: 0, yaw: 0.6, pitch: 0.35, dist: 1.5,
              target: [0, 0, 0], home: null, size: 2, drag: null, dirty: true };
function glvInit() {
  if (glv.gl) return true;
  const cv = $('glcv');
  const gl = cv.getContext('webgl', {antialias: false});
  if (!gl) { $('cloud-info').textContent = 'WebGL unavailable in this browser'; return false; }
  const vs = `attribute vec3 p; attribute vec3 c; uniform mat4 mvp; uniform float ps;
              varying vec3 vc; void main(){ gl_Position = mvp*vec4(p,1.0); gl_PointSize = ps; vc = c; }`;
  const fs = `precision mediump float; varying vec3 vc; void main(){ gl_FragColor = vec4(vc,1.0); }`;
  function sh(type, src) { const s = gl.createShader(type); gl.shaderSource(s, src);
    gl.compileShader(s); return s; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog); gl.useProgram(prog);
  glv.gl = gl; glv.prog = prog;
  glv.pbuf = gl.createBuffer(); glv.cbuf = gl.createBuffer();
  glv.aP = gl.getAttribLocation(prog, 'p'); glv.aC = gl.getAttribLocation(prog, 'c');
  glv.uM = gl.getUniformLocation(prog, 'mvp'); glv.uS = gl.getUniformLocation(prog, 'ps');
  gl.enable(gl.DEPTH_TEST);
  // controls: drag orbit / shift+drag or right-drag pan / wheel zoom / dblclick reset
  cv.addEventListener('mousedown', e => { glv.drag = {x: e.clientX, y: e.clientY,
    pan: e.shiftKey || e.button === 2}; e.preventDefault(); });
  window.addEventListener('mousemove', e => {
    if (!glv.drag) return;
    const dx = e.clientX - glv.drag.x, dy = e.clientY - glv.drag.y;
    glv.drag.x = e.clientX; glv.drag.y = e.clientY;
    if (glv.drag.pan) {
      const s = glv.dist * 0.0012, cy = Math.cos(glv.yaw), sy = Math.sin(glv.yaw);
      glv.target[0] -= (dx * cy) * s; glv.target[2] -= (dx * -sy) * s;
      glv.target[1] += dy * s;
    } else { glv.yaw += dx * 0.008; glv.pitch = Math.max(-1.55, Math.min(1.55, glv.pitch + dy * 0.008)); }
    glv.dirty = true;
  });
  window.addEventListener('mouseup', () => glv.drag = null);
  cv.addEventListener('contextmenu', e => e.preventDefault());
  cv.addEventListener('wheel', e => { e.preventDefault();
    glv.dist *= Math.pow(1.0015, e.deltaY); glv.dirty = true; }, {passive: false});
  cv.addEventListener('dblclick', () => { if (glv.home) { Object.assign(glv, glv.home);
    glv.target = glv.home.target.slice(); glv.dirty = true; } });
  $('cloud-size').oninput = e => { glv.size = +e.target.value; glv.dirty = true; };
  $('cloud-reset').onclick = () => $('glcv').dispatchEvent(new Event('dblclick'));
  requestAnimationFrame(glvFrame);
  return true;
}
function glvFrame() {
  const cv = $('glcv'), gl = glv.gl;
  if (gl && cv.offsetParent !== null) {  // visible
    const w = cv.clientWidth * (window.devicePixelRatio || 1) | 0,
          h = cv.clientHeight * (window.devicePixelRatio || 1) | 0;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; glv.dirty = true; }
    if (glv.dirty && glv.n) { glvDraw(); glv.dirty = false; }
  }
  requestAnimationFrame(glvFrame);
}
function glvDraw() {
  const gl = glv.gl, cv = $('glcv');
  gl.viewport(0, 0, cv.width, cv.height);
  gl.clearColor(0.063, 0.07, 0.086, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  // eye from yaw/pitch/dist around target
  const cp = Math.cos(glv.pitch), t = glv.target;
  const eye = [t[0] + glv.dist * cp * Math.sin(glv.yaw),
               t[1] + glv.dist * Math.sin(glv.pitch),
               t[2] + glv.dist * cp * Math.cos(glv.yaw)];
  // lookAt
  function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
  function nrm(a){const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];}
  function crs(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
  function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
  const f = nrm(sub(t, eye)), r = nrm(crs(f, [0,1,0])), u = crs(r, f);
  const view = [r[0],u[0],-f[0],0, r[1],u[1],-f[1],0, r[2],u[2],-f[2],0,
                -dot(r,eye),-dot(u,eye),dot(f,eye),1];
  const asp = cv.width / cv.height, fy = 1 / Math.tan(0.4), zn = glv.dist * 0.01, zf = glv.dist * 40;
  const proj = [fy/asp,0,0,0, 0,fy,0,0, 0,0,(zf+zn)/(zn-zf),-1, 0,0,2*zf*zn/(zn-zf),0];
  const m = new Float32Array(16);  // proj * view
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0; for (let k = 0; k < 4; k++) s += proj[k*4+j] * view[i*4+k];
    m[i*4+j] = s;
  }
  gl.uniformMatrix4fv(glv.uM, false, m);
  gl.uniform1f(glv.uS, glv.size * (window.devicePixelRatio || 1));
  gl.bindBuffer(gl.ARRAY_BUFFER, glv.pbuf);
  gl.enableVertexAttribArray(glv.aP); gl.vertexAttribPointer(glv.aP, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, glv.cbuf);
  gl.enableVertexAttribArray(glv.aC); gl.vertexAttribPointer(glv.aC, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.drawArrays(gl.POINTS, 0, glv.n);
}
async function loadCloud(file) {
  if (!glvInit()) return;
  $('cloud-info').textContent = 'loading ' + file + ' ...';
  try {
    const t0 = performance.now();
    const r = await fetch('/cloud?ds=' + encodeURIComponent($('dataset').value) +
                          '&f=' + encodeURIComponent(file));
    if (!r.ok) { $('cloud-info').textContent = 'load failed'; return; }
    const buf = await r.arrayBuffer();
    const n = new Uint32Array(buf, 0, 1)[0];
    const pts = new Float32Array(buf, 4, n * 3);
    const rgb = new Uint8Array(buf, 4 + n * 12, n * 3);
    const gl = glv.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, glv.pbuf); gl.bufferData(gl.ARRAY_BUFFER, pts, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, glv.cbuf); gl.bufferData(gl.ARRAY_BUFFER, rgb, gl.STATIC_DRAW);
    glv.n = n;
    // frame the cloud: centre on its bbox, back off by its extent
    let mn = [1e9,1e9,1e9], mx = [-1e9,-1e9,-1e9];
    for (let i = 0; i < n * 3; i += 3) for (let k = 0; k < 3; k++) {
      const v = pts[i + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v;
    }
    glv.target = [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2];
    glv.dist = 1.6 * Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 0.05);
    glv.yaw = 0.6; glv.pitch = 0.35;
    glv.home = {yaw: glv.yaw, pitch: glv.pitch, dist: glv.dist, target: glv.target.slice()};
    glv.dirty = true;
    $('cloud-info').textContent = n.toLocaleString() + ' pts · ' +
      ((performance.now() - t0) / 1000).toFixed(1) + 's';
  } catch (err) { $('cloud-info').textContent = '' + err; }
}
async function showResultClouds() {
  const r = await post('/api/clouds', {dataset: $('dataset').value});
  const clouds = (r.clouds || []);
  $('cloud-file').innerHTML = clouds.map(c =>
    `<option value="${c.file}">${c.file} (${c.mb} MB)</option>`).join('');
  $('cloud-file').onchange = e => loadCloud(e.target.value);
  if (clouds.length) loadCloud(clouds[0].file);
  else $('cloud-info').textContent = 'no .ply in the sequence folder yet';
}

/* ---------------- boot ---------------- */
$('mode').onchange = refreshPlan;
$('force').onchange = refreshPlan;
$('dataset').onchange = () => { state.img = new Image(); state.pts = []; state.box = null;
  resetConfirm(''); setPhase('setup'); refreshPlan(); };
(async function boot() {
  const r = await post('/api/state');
  $('dataset').innerHTML = r.datasets.map(d =>
    `<option ${d === r.dataset ? 'selected' : ''}>${d}</option>`).join('');
  if (r.mode in {whole:1,label:1}) $('mode').value = r.mode;
  else { $('mode').value = 'custom'; state.cliSteps = r.mode; }
  $('viz').value = r.viz;
  if (r.prompt_text) $('ptext').value = r.prompt_text;
  state.models = r.models || {status: 'cold'};
  renderModels();
  pollModels();
  await refreshPlan();
})();
</script>
"""
