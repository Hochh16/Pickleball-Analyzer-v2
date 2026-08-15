/* Pickleball Analyzer — setup wizard (Phase 1) front end.
   Vanilla JS single-page wizard: Video -> Court -> Players -> You -> Review. */

'use strict';

// ---------------------------------------------------------------- constants
const STEPS = ['video', 'court', 'players', 'you', 'review', 'run'];

const CORNER = '#ff4d4d', KU = '#2ab7ff', KO = '#ffab2e';
const POINTS = [
  { label: 'Court corner — bottom LEFT',        color: CORNER },
  { label: 'Court corner — bottom RIGHT',       color: CORNER },
  { label: 'Court corner — top RIGHT',          color: CORNER },
  { label: 'Court corner — top LEFT',           color: CORNER },
  { label: 'User kitchen line — LEFT end',      color: KU },
  { label: 'User kitchen line — RIGHT end',     color: KU },
  { label: 'Opponent kitchen line — LEFT end',  color: KO },
  { label: 'Opponent kitchen line — RIGHT end', color: KO },
];
const FRAME_MAXW = 1600;

// ---------------------------------------------------------------- state
const S = {
  step: 'video',
  reachedIdx: 0,            // furthest step reached — earlier steps stay clickable
  driveSync: false,         // Google Drive for Desktop auto-sync available (from /api/config)
  session: null,
  startingCorner: 'left',   // set visually on the "You" step; default until then
  court: { frameIdx: 0, markFrame: null, points: new Array(8).fill(null), img: null, imgFrame: -1 },
  calib: null,
  courtConfirmed: false,
  you: { frameIdx: 0, img: null, imgFrame: -1 },
};

// ---------------------------------------------------------------- helpers
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const el = (id) => document.getElementById(id);

function toast(msg, isErr) {
  const t = el('toast');
  t.textContent = msg;
  t.className = 'toast' + (isErr ? ' err' : '');
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, isErr ? 5200 : 2600);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = j.detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res;
}
const jsonPost = (path, body) =>
  api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

function fmtDuration(sec) {
  sec = Math.round(sec || 0);
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

// ---------------------------------------------------------------- routing
// A step is reachable if it's at/behind the furthest step we've reached, and
// (for anything past Video) we have a loaded session. Lets the user click the
// step bar to jump back and re-edit without a linear back-button trail.
function canGoto(i) {
  return i <= S.reachedIdx && (i === 0 || !!S.session);
}

function goto(step) {
  S.step = step;
  STEPS.forEach((s) => { const p = el('panel-' + s); if (p) p.hidden = (s !== step); });
  const idxCur = STEPS.indexOf(step);
  S.reachedIdx = Math.max(S.reachedIdx, idxCur);
  $$('#stepnav li').forEach((li) => {
    const i = STEPS.indexOf(li.dataset.step);
    li.classList.toggle('is-active', i === idxCur);
    li.classList.toggle('is-done', i < idxCur);
    li.classList.toggle('is-clickable', canGoto(i));
    const num = li.querySelector('.num');
    num.innerHTML = i < idxCur ? '' : String(i + 1);
  });
  if (step === 'court') enterCourt();
  if (step === 'you') enterYou();
  if (step === 'review') enterReview();
  if (step === 'run') enterRun();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ================================================================ STEP 1: VIDEO
async function initVideoStep() {
  el('refreshVideos').addEventListener('click', loadVideos);
  // Always show the folder path first, even if the listing later fails.
  try {
    const cfg = await api('/api/config');
    el('videosDir').textContent = cfg.videos_dir;
    S.driveSync = !!cfg.drive_sync;
  } catch (e) { /* fall back to whatever /api/videos returns */ }
  loadVideos();
  loadExistingSessions();
}

async function loadVideos() {
  const list = el('videoList');
  try {
    const data = await api('/api/videos');
    if (data.dir) el('videosDir').textContent = data.dir;
    list.innerHTML = '';
    if (!data.videos.length) {
      list.appendChild(Object.assign(document.createElement('div'), {
        className: 'empty',
        textContent: 'No videos here yet. Copy your clip into the folder shown above, then press Refresh.',
      }));
      return;
    }
    data.videos.forEach((v) => {
      const size = v.size_mb >= 1024 ? (v.size_mb / 1024).toFixed(1) + ' GB' : v.size_mb + ' MB';
      list.appendChild(rowEl('video', '🎬', v.name, size, () => pickLocal(v.path)));
    });
  } catch (e) {
    list.innerHTML = '';
    const msg = document.createElement('div');
    msg.className = 'empty';
    msg.textContent = 'Could not read the videos folder. If you just updated the app, restart the server, then press Refresh.';
    list.appendChild(msg);
  }
}

function rowEl(kind, icon, name, size, onClick) {
  const row = document.createElement('div');
  row.className = 'row ' + kind;
  row.innerHTML = `<span class="ic">${icon}</span><span class="nm"></span><span class="sz"></span>`;
  row.querySelector('.nm').textContent = name;
  row.querySelector('.sz').textContent = size;
  row.addEventListener('click', onClick);
  return row;
}

async function pickLocal(path) {
  try {
    toast('Opening video…');
    const session = await jsonPost('/api/sessions', { path });  // backend derives a good name
    onSessionReady(session);
  } catch (e) { toast('Could not open that video: ' + e.message, true); }
}

async function loadExistingSessions() {
  try {
    const { sessions } = await api('/api/sessions');
    if (!sessions.length) return;
    el('existingWrap').hidden = false;
    const list = el('sessionList');
    list.innerHTML = '';
    sessions.forEach((s) => {
      const card = document.createElement('div');
      card.className = 'session-card';
      const steps = s.steps || {};
      const pill = (k, label) => `<span class="pill ${steps[k] ? 'done' : ''}">${label}</span>`;
      card.innerHTML =
        `<div class="sc-name"></div>
         <div class="sc-meta">${s.video.frame_width}×${s.video.frame_height} · ${fmtDuration(s.video.duration_sec)}</div>
         <div class="sc-steps">${pill('calibration', 'Court')}${pill('roster', 'Players')}</div>`;
      card.querySelector('.sc-name').textContent = s.name;
      card.addEventListener('click', () => onSessionReady(s));
      list.appendChild(card);
    });
  } catch (e) { /* library is best-effort */ }
}

async function onSessionReady(session) {
  S.session = session;
  S.courtConfirmed = !!(session.steps && session.steps.calibration);
  toast(`Loaded “${session.name}”`);
  // Ask whose report this is before anything else, once per video. Deciding up front is
  // what lets a finished analysis join the cumulative report by itself, instead of the
  // operator having to come back for a second step after a ~100-minute run.
  if (session.collection_id === undefined) {
    await askWhoseReport(session);
    return;
  }
  const configured = !!(session.steps && session.steps.calibration && session.steps.roster);
  if (!configured) { goto('court'); return; }
  // Fully configured session: hydrate the wizard from disk, unlock every step,
  // and jump straight to the run (or review if nothing is running yet) — no
  // re-marking the court just to get back to your analysis.
  S.reachedIdx = STEPS.length - 1;
  try {
    const sum = await api(`/api/sessions/${session.id}/summary`);
    if (sum.roster && sum.roster.handedness) {
      const h = sum.roster.handedness;
      el('handUser').value = h.user || 'right';
      el('handPartner').value = h.partner || 'unknown';
      el('handOppA').value = h.opp_a || 'unknown';
      el('handOppB').value = h.opp_b || 'unknown';
    }
    const ui = sum.calibration && sum.calibration.user_inputs;
    if (ui && ui.user_starting_corner) S.startingCorner = ui.user_starting_corner;
  } catch (e) { /* hydration is best-effort */ }
  try {
    const run = await api(`/api/sessions/${session.id}/run`);
    goto(run.phase && run.phase !== 'idle' ? 'run' : 'review');
  } catch (e) { goto('review'); }
}

// ================================================================ STEP 2: COURT
let courtCanvas, courtCtx, loupeCanvas, loupeCtx;

function initCourtStep() {
  courtCanvas = el('courtCanvas'); courtCtx = courtCanvas.getContext('2d');
  loupeCanvas = el('loupe'); loupeCtx = loupeCanvas.getContext('2d');
  loupeCanvas.width = 150; loupeCanvas.height = 150;

  courtCanvas.addEventListener('click', onCourtClick);
  courtCanvas.addEventListener('mousemove', onCourtMove);
  courtCanvas.addEventListener('mouseleave', () => { loupeCanvas.hidden = true; });

  el('undoBtn').addEventListener('click', () => { undoLastPoint(); });
  el('clearBtn').addEventListener('click', () => { clearPoints(); });
  el('frameSlider').addEventListener('input', (e) => setCourtFrame(parseInt(e.target.value, 10)));
  el('frameBack').addEventListener('click', () => setCourtFrame(S.court.frameIdx - 1));
  el('frameFwd').addEventListener('click', () => setCourtFrame(S.court.frameIdx + 1));
  el('calibrateBtn').addEventListener('click', runCalibrate);
  el('confirmCourtBtn').addEventListener('click', () => { el('calibResult').hidden = true; S.courtConfirmed = true; goto('players'); });
  el('redoBtn').addEventListener('click', () => { el('calibResult').hidden = true; window.scrollTo({ top: 0, behavior: 'smooth' }); });
}

function hideCalibResult() { const r = el('calibResult'); if (r) r.hidden = true; }

function enterCourt() {
  const v = S.session.video;
  const slider = el('frameSlider');
  slider.max = Math.max(0, v.frame_count - 1);
  // default to a frame a little into the clip (players/serve less likely to block corners at start)
  if (S.court.imgFrame < 0) {
    S.court.frameIdx = Math.min(Math.floor((v.frame_count || 1) * 0.05), Math.max(0, v.frame_count - 1));
    slider.value = S.court.frameIdx;
  }
  renderPointList();
  loadCourtFrame();
  updateCalibButton();
}

function setCourtFrame(idx) {
  const max = Math.max(0, (S.session.video.frame_count || 1) - 1);
  idx = Math.max(0, Math.min(max, idx));
  S.court.frameIdx = idx;
  el('frameSlider').value = idx;
  loadCourtFrame();
}

function loadCourtFrame() {
  const s = S.session, idx = S.court.frameIdx;
  el('frameLabel').textContent = `${idx} / ${Math.max(0, s.video.frame_count - 1)}`;
  const img = new Image();
  img.onload = () => {
    S.court.img = img; S.court.imgFrame = idx;
    courtCanvas.width = img.naturalWidth;
    courtCanvas.height = img.naturalHeight;
    drawCourt();
  };
  img.onerror = () => toast('Could not load that frame', true);
  img.src = `/api/sessions/${s.id}/frame/${idx}?maxw=${FRAME_MAXW}`;
}

// scale between source pixels and the served (canvas) image
function servedScale() {
  return S.court.img ? (S.court.img.naturalWidth / S.session.video.frame_width) : 1;
}

function drawCourt() {
  if (!S.court.img) return;
  const ctx = courtCtx, sc = servedScale();
  ctx.drawImage(S.court.img, 0, 0);
  const P = S.court.points.map((p) => p ? [p[0] * sc, p[1] * sc] : null);

  const line = (a, b, color) => {
    if (!P[a] || !P[b]) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(P[a][0], P[a][1]); ctx.lineTo(P[b][0], P[b][1]); ctx.stroke();
  };
  // court rectangle
  line(0, 1, CORNER); line(1, 2, CORNER); line(2, 3, CORNER); line(3, 0, CORNER);
  line(4, 5, KU); line(6, 7, KO);

  P.forEach((p, i) => {
    if (!p) return;
    ctx.fillStyle = POINTS[i].color;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 13px sans-serif';
    ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.lineWidth = 3;
    ctx.strokeText(String(i + 1), p[0] + 9, p[1] - 8);
    ctx.fillText(String(i + 1), p[0] + 9, p[1] - 8);
  });
}

function canvasToSource(e) {
  const rect = courtCanvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (courtCanvas.width / rect.width);
  const cy = (e.clientY - rect.top) * (courtCanvas.height / rect.height);
  const sc = servedScale();
  return [cx / sc, cy / sc];  // source pixels
}

function nextPointIdx() {
  return S.court.points.findIndex((p) => p === null);
}

function onCourtClick(e) {
  const idx = nextPointIdx();
  if (idx === -1) { toast('All 8 points are placed — use Undo to change one.'); return; }
  const [sx, sy] = canvasToSource(e);
  S.court.points[idx] = [sx, sy];
  if (idx === 0) S.court.markFrame = S.court.frameIdx;
  hideCalibResult();
  drawCourt(); renderPointList(); updateCalibButton();
}

function onCourtMove(e) {
  if (!S.court.img) return;
  const rect = courtCanvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (courtCanvas.width / rect.width);
  const cy = (e.clientY - rect.top) * (courtCanvas.height / rect.height);
  // magnifier
  const z = 3.2, size = 150, half = size / 2;
  loupeCtx.clearRect(0, 0, size, size);
  loupeCtx.imageSmoothingEnabled = false;
  const srcSize = size / z;
  loupeCtx.drawImage(S.court.img, cx - srcSize / 2, cy - srcSize / 2, srcSize, srcSize, 0, 0, size, size);
  loupeCtx.strokeStyle = '#00e0ff'; loupeCtx.lineWidth = 1;
  loupeCtx.beginPath(); loupeCtx.moveTo(half, 0); loupeCtx.lineTo(half, size);
  loupeCtx.moveTo(0, half); loupeCtx.lineTo(size, half); loupeCtx.stroke();
  loupeCanvas.hidden = false;
  // position near cursor but inside wrap, avoiding the cursor itself
  const wrap = el('canvasWrap').getBoundingClientRect();
  let lx = e.clientX - wrap.left + 20, ly = e.clientY - wrap.top + 20;
  if (lx + size > wrap.width) lx = e.clientX - wrap.left - size - 20;
  if (ly + size > wrap.height) ly = e.clientY - wrap.top - size - 20;
  loupeCanvas.style.left = Math.max(0, lx) + 'px';
  loupeCanvas.style.top = Math.max(0, ly) + 'px';
}

function undoLastPoint() {
  for (let i = S.court.points.length - 1; i >= 0; i--) {
    if (S.court.points[i] !== null) { S.court.points[i] = null; break; }
  }
  if (nextPointIdx() === 0) S.court.markFrame = null;
  hideCalibResult();
  drawCourt(); renderPointList(); updateCalibButton();
}
function clearPoints() {
  S.court.points = new Array(8).fill(null);
  S.court.markFrame = null;
  hideCalibResult();
  drawCourt(); renderPointList(); updateCalibButton();
}

function renderPointList() {
  const ol = el('pointList'); ol.innerHTML = '';
  const nextIdx = nextPointIdx();
  POINTS.forEach((p, i) => {
    const li = document.createElement('li');
    const set = S.court.points[i] !== null;
    li.className = (set ? 'set' : '') + (i === nextIdx ? ' next' : '');
    li.innerHTML = `<span class="dot" style="background:${p.color}"></span><span>${i + 1}. ${p.label}</span>`;
    ol.appendChild(li);
  });
  // prompt bar
  if (nextIdx === -1) {
    el('promptSwatch').style.background = 'var(--ok)';
    el('promptText').textContent = 'All points placed. Press “Check calibration” →';
    el('promptCount').textContent = '8 / 8';
  } else {
    el('promptSwatch').style.background = POINTS[nextIdx].color;
    el('promptText').textContent = 'Click: ' + POINTS[nextIdx].label;
    el('promptCount').textContent = `${S.court.points.filter(Boolean).length} / 8`;
  }
}

function updateCalibButton() {
  el('calibrateBtn').disabled = nextPointIdx() !== -1;
}

async function runCalibrate() {
  const pts = S.court.points;
  if (pts.some((p) => p === null)) { toast('Mark all 8 points first', true); return; }
  const btn = el('calibrateBtn');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>Checking…';
  const payload = {
    court_corners_image: pts.slice(0, 4),
    kitchen_line_user_image: pts.slice(4, 6),
    kitchen_line_opponent_image: pts.slice(6, 8),
    // The analyzed player is always on the near baseline: the camera protocol puts
    // the camera in the corner nearest their start, and the court marking treats
    // points 5-6 as the user's (near/bottom) kitchen line. Asked once, on the "You"
    // step (which SIDE). Stage 2.5 v1 only supports the user on the near baseline.
    user_baseline: 'near',
    // Handedness is collected on the Players step (roster.json is authoritative for
    // Stage 6). Calibration still needs the field, so send a placeholder now; the
    // backend patches court.json.dominant_hand from the roster when Players saves.
    dominant_hand: 'right',
    user_starting_corner: S.startingCorner,   // confirmed visually on the "You" step
    frame_used_for_calibration: S.court.markFrame ?? S.court.frameIdx,
  };
  try {
    const res = await jsonPost(`/api/sessions/${S.session.id}/calibrate`, payload);
    S.calib = res;
    showCalibResult(res);
  } catch (e) {
    toast('Calibration failed: ' + e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = 'Check calibration';
  }
}

function showCalibResult(res) {
  el('previewImg').src = 'data:image/jpeg;base64,' + res.preview_jpeg_base64;
  const v = res.validation;
  const meta = el('previewMeta');
  const cls = (val, warn) => val <= warn ? 'good' : 'bad';
  meta.innerHTML =
    `<div class="metric"><span class="k">Corner fit (RMSE)</span><span class="v ${cls(v.homography_rmse_pixels, 5)}">${v.homography_rmse_pixels.toFixed(1)} px</span></div>
     <div class="metric"><span class="k">Your kitchen line</span><span class="v ${cls(v.kitchen_projection_error_user_px, 10)}">${v.kitchen_projection_error_user_px.toFixed(1)} px off</span></div>
     <div class="metric"><span class="k">Opponent kitchen line</span><span class="v ${cls(v.kitchen_projection_error_opponent_px, 10)}">${v.kitchen_projection_error_opponent_px.toFixed(1)} px off</span></div>`;
  if (v.warnings && v.warnings.length) {
    const w = document.createElement('div'); w.className = 'warns';
    v.warnings.forEach((msg) => {
      const d = document.createElement('div'); d.className = 'warn-item'; d.textContent = '⚠ ' + msg; w.appendChild(d);
    });
    meta.appendChild(w);
  } else {
    const d = document.createElement('div'); d.className = 'metric'; d.style.marginTop = '10px';
    d.innerHTML = '<span class="k">Warnings</span><span class="v good">none</span>';
    meta.appendChild(d);
  }
  const r = el('calibResult');
  r.hidden = false;
  r.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ================================================================ STEP 3: PLAYERS
function initPlayersStep() {
  el('playersNext').addEventListener('click', savePlayers);
  $$('[data-goto]').forEach((b) => b.addEventListener('click', () => goto(b.dataset.goto)));
}

async function savePlayers() {
  const body = {
    user: el('handUser').value,
    partner: el('handPartner').value,
    opp_a: el('handOppA').value,
    opp_b: el('handOppB').value,
  };
  const btn = el('playersNext');
  btn.disabled = true;
  try {
    await jsonPost(`/api/sessions/${S.session.id}/roster`, body);
    toast('Players saved');
    goto('you');
  } catch (e) { toast('Could not save players: ' + e.message, true); }
  finally { btn.disabled = false; }
}

// ================================================================ STEP 4: YOU (which side)
let youCanvas, youCtx;
function initYouStep() {
  youCanvas = el('youCanvas'); youCtx = youCanvas.getContext('2d');
  el('youSlider').addEventListener('input', (e) => setYouFrame(parseInt(e.target.value, 10)));
  el('youBack').addEventListener('click', () => setYouFrame(S.you.frameIdx - 1));
  el('youFwd').addEventListener('click', () => setYouFrame(S.you.frameIdx + 1));
  ['cardLeft', 'cardRight'].forEach((id) =>
    el(id).addEventListener('click', () => pickCorner(el(id).dataset.corner)));
  el('youNext').addEventListener('click', saveSide);
}

function enterYou() {
  const v = S.session.video;
  el('youSlider').max = Math.max(0, v.frame_count - 1);
  if (S.you.imgFrame < 0) { S.you.frameIdx = Math.floor((v.frame_count || 1) * 0.1); el('youSlider').value = S.you.frameIdx; }
  loadYouFrame();
  reflectCorner();
}
function setYouFrame(idx) {
  const max = Math.max(0, (S.session.video.frame_count || 1) - 1);
  idx = Math.max(0, Math.min(max, idx));
  S.you.frameIdx = idx; el('youSlider').value = idx; loadYouFrame();
}
function loadYouFrame() {
  const s = S.session, idx = S.you.frameIdx;
  el('youFrameLabel').textContent = `${idx} / ${Math.max(0, s.video.frame_count - 1)}`;
  const img = new Image();
  img.onload = () => { S.you.img = img; S.you.imgFrame = idx; youCanvas.width = img.naturalWidth; youCanvas.height = img.naturalHeight; youCtx.drawImage(img, 0, 0); };
  img.onerror = () => toast('Could not load that frame', true);
  img.src = `/api/sessions/${s.id}/frame/${idx}?maxw=${FRAME_MAXW}`;
}
function pickCorner(corner) {
  S.startingCorner = corner;
  reflectCorner();
}
function reflectCorner() {
  ['Left', 'Right'].forEach((side) => {
    el('card' + side).classList.toggle('selected', S.startingCorner === side.toLowerCase());
  });
}
async function saveSide() {
  const btn = el('youNext'); btn.disabled = true;
  try {
    // persist the starting side into markers.json + court.json (used by Stage 2/2.5)
    await jsonPost(`/api/sessions/${S.session.id}/starting-corner`, { corner: S.startingCorner });
    toast(`Starting side: ${S.startingCorner}`);
    goto('review');
  } catch (e) { toast('Could not save: ' + e.message, true); }
  finally { btn.disabled = false; }
}

// ================================================================ STEP 5: REVIEW
function initReviewStep() {
  el('finishBtn').addEventListener('click', async () => {
    el('finishBtn').disabled = true;
    try {
      await api(`/api/sessions/${S.session.id}/run`, { method: 'POST' });
      goto('run');
    } catch (e) {
      toast('Could not start analysis: ' + e.message, true);
      el('finishBtn').disabled = false;
    }
  });
}

async function enterReview() {
  let sum;
  try { sum = await api(`/api/sessions/${S.session.id}/summary`); }
  catch (e) { toast('Could not load summary: ' + e.message, true); return; }
  const grid = el('reviewGrid');
  const cal = sum.calibration, roster = sum.roster, v = S.session.video;
  const handLabel = (h) => ({ right: 'Right', left: 'Left', unknown: 'Not sure' }[h] || h);

  const card = (title, rows) =>
    `<div class="review-card"><h3>${title}</h3>${rows.map(([k, val]) => `<div class="kv"><span class="k">${k}</span><span class="v">${val}</span></div>`).join('')}</div>`;

  const calBadge = cal
    ? `<span class="badge ok">✓ done</span>`
    : `<span class="badge skip">not set</span>`;

  grid.innerHTML =
    card('Video', [
      ['Name', esc(S.session.name)],
      ['Resolution', `${v.frame_width}×${v.frame_height}`],
      ['Length', `${fmtDuration(v.duration_sec)} · ${v.frame_count} frames`],
      ['FPS', v.fps.toFixed(0)],
    ]) +
    card('Court calibration', cal ? [
      ['Status', calBadge],
      ['Marked on frame', String(cal.frame_used_for_calibration)],
      ['Corner fit (RMSE)', cal.validation.homography_rmse_pixels.toFixed(1) + ' px'],
      ['Warnings', String((cal.validation.warnings || []).length)],
    ] : [['Status', calBadge]]) +
    card('Players', roster ? [
      ['You', handLabel(roster.handedness.user)],
      ['Partner', handLabel(roster.handedness.partner)],
      ['Opponent A', handLabel(roster.handedness.opp_a)],
      ['Opponent B', handLabel(roster.handedness.opp_b)],
    ] : [['Status', '<span class="badge skip">not set</span>']]) +
    card('Your side', [
      ['Starting side', S.startingCorner === 'right' ? 'Right' : 'Left'],
      ['Baseline', cal ? (cal.user_inputs.user_baseline === 'far' ? 'Far' : 'Near') : '—'],
    ]);
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ================================================================ STEP 6: RUN
let runES = null;         // EventSource
let visionBusy = false;

function initRunStep() {
  el('visionUploadBtn').addEventListener('click', () => el('visionInput').click());
  el('visionInput').addEventListener('change', () => { if (el('visionInput').files.length) uploadVision(el('visionInput').files); });
  el('retryBtn').addEventListener('click', async () => {
    try { await api(`/api/sessions/${S.session.id}/run`, { method: 'POST' }); enterRun(); }
    catch (e) { toast('Retry failed: ' + e.message, true); }
  });
}

// When Google Drive for Desktop is configured, the app pushes the clip to Drive
// and auto-ingests the results — rewrite the hand-off card to that (no manual
// download; upload stays as a fallback).
function applyHandoffMode() {
  if (!S.driveSync) return;
  el('handoffIntro').textContent =
    'Vision runs on Colab’s GPU. Your clip is synced to Google Drive automatically — run the notebook and the results import themselves.';
  el('handoffSteps').innerHTML =
    '<li><b>Open the Colab notebook</b> and choose <b>Runtime → Run all</b> (GPU runtime). Your clip is already on Drive — nothing to upload or edit.</li>' +
    '<li>Leave this page open — the steps on the left tick off as Colab works, and analysis <b>resumes automatically</b> when it finishes.</li>';
  el('bundleDownloadBtn').hidden = true;   // bundle is auto-pushed to Drive
  // manual upload becomes a de-emphasized fallback (auto-ingest is the path)
  const ub = el('visionUploadBtn');
  ub.classList.remove('primary');
  ub.classList.add('ghost');
  ub.textContent = 'Upload outputs manually (fallback)';
  el('visionNote').textContent = '⏳ Waiting for Colab results (auto-syncing from Google Drive)…';
}

function enterRun() {
  // serve via /files/ so the report's relative <video src="annotated_web.mp4"> resolves
  el('viewReportBtn').href = `/api/sessions/${S.session.id}/files/report.html`;
  el('viewVideoBtn').href = `/api/sessions/${S.session.id}/files/video.mp4`;
  el('bundleDownloadBtn').href = `/api/sessions/${S.session.id}/vision-input.zip`;
  applyHandoffMode();
  // (re)connect the live stream
  if (runES) { runES.close(); runES = null; }
  runES = new EventSource(`/api/sessions/${S.session.id}/run/stream`);
  runES.onmessage = (ev) => { try { renderRun(JSON.parse(ev.data)); } catch (e) {} };
  runES.onerror = () => { /* browser auto-reconnects */ };
}

const STEP_ICON = { pending: '○', running: '<span class="spinner"></span>', done: '✓', failed: '✕', skipped: '–', waiting: '⏸' };

function renderRun(job) {
  // steps checklist
  const wrap = el('runSteps');
  wrap.innerHTML = '';
  (job.steps || []).forEach((s) => {
    const row = document.createElement('div');
    row.className = 'run-step ' + s.status;
    let dur = '';
    if (s.started_at && s.ended_at) dur = `${Math.round(s.ended_at - s.started_at)}s`;
    row.innerHTML =
      `<span class="rs-icon">${STEP_ICON[s.status] || '○'}</span>` +
      `<span class="rs-label"></span><span class="rs-dur">${dur}</span>`;
    row.querySelector('.rs-label').textContent = s.label;
    wrap.appendChild(row);
  });

  // log tail
  const log = el('runLog');
  log.textContent = (job.log || []).join('\n');
  log.scrollTop = log.scrollHeight;

  // side cards
  el('visionHandoff').hidden = job.phase !== 'vision';
  el('runDone').hidden = job.phase !== 'done';
  if (job.phase === 'done') showCollectStep();
  el('runFail').hidden = job.phase !== 'failed';
  if (job.phase === 'failed') el('runFailMsg').textContent = job.error || 'A stage failed. See the activity log.';

  // header
  const titles = {
    prepare: 'Preparing your video', vision: 'Vision runs on a GPU',
    post: 'Finishing your analysis', done: 'Your report is ready',
    failed: 'Analysis stopped', idle: 'Analyzing your match',
  };
  el('runTitle').textContent = titles[job.phase] || 'Analyzing your match';
  if (job.phase === 'done' && runES) { runES.close(); runES = null; }
}

async function uploadVision(files) {
  if (visionBusy) return;
  files = Array.from(files);
  const note = el('visionNote');
  const hasZip = files.some((f) => f.name.toLowerCase().endsWith('.zip'));
  const hasBall = files.some((f) => f.name === 'ball.parquet');
  if (!hasZip && !hasBall) { note.textContent = 'Please include ball.parquet and the other output files (or a .zip of them).'; return; }
  visionBusy = true;
  note.textContent = `Uploading ${files.length} file(s)…`;
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  try {
    const res = await api(`/api/sessions/${S.session.id}/vision`, { method: 'POST', body: fd });
    if (res.resumed) {
      note.textContent = 'Received — resuming analysis…';
      el('visionHandoff').hidden = true;
    } else if (!res.have_all_outputs) {
      note.textContent = 'Got ' + res.saved.join(', ') + '. Still missing some of: players.parquet, track_roles.json, poses.parquet, ball.parquet.';
    } else {
      note.textContent = 'Received.';
    }
  } catch (e) {
    note.textContent = 'Upload failed: ' + e.message;
  } finally { visionBusy = false; }
}

// ---------------------------------------------------------------- boot
function boot() {
  initVideoStep();
  initCourtStep();
  initPlayersStep();
  initYouStep();
  initReviewStep();
  initRunStep();
  initCollectStep();
  initCollMgr();
  $$('[data-goto]').forEach((b) => b.addEventListener('click', () => goto(b.dataset.goto)));
  $$('#stepnav li').forEach((li) => li.addEventListener('click', () => {
    const i = STEPS.indexOf(li.dataset.step);
    if (canGoto(i)) goto(li.dataset.step);
  }));
  goto('video');
}
document.addEventListener('DOMContentLoaded', boot);

// ---------------------------------------------------------------------------
// Cumulative reports (collections)
//
// NOTHING IS PRE-SELECTED. Adding a video to the wrong player's cumulative report is
// not something the app can detect, and un-picking it means a rebuild plus explaining
// why their numbers moved. One extra click is cheaper than that, every time.
// ---------------------------------------------------------------------------

let collectLoaded = false;

async function showCollectStep() {
  const card = el('collectCard');
  if (!card || collectLoaded) return;
  collectLoaded = true;
  // The report was chosen with the video, so this is a confirmation, not a question.
  // Falling back to asking here only matters for videos set up before that existed.
  try {
    const res = await jsonPost(`/api/sessions/${S.session.id}/finalize-collection`, {});
    if (res.collection_id) {
      card.hidden = false;
      const nm = (res.collection && res.collection.name) || res.collection_id;
      finishCollect(res.collection_id,
                    res.added ? `Added to "${nm}".` : `Already part of "${nm}".`);
      return;
    }
    card.hidden = true;   // standalone: nothing to ask
  } catch (e) {
    card.hidden = false;
    el('collectReminder').textContent = e.message || 'Could not update the cumulative report.';
  }
}

function renderCollectList(list) {
  const wrap = el('collectList');
  const open = list.filter((c) => !c.closed_at);
  if (!open.length) {
    wrap.innerHTML = '<p class="small muted">No cumulative reports yet. '
      + 'Start one to track this player across videos.</p>';
    return;
  }
  wrap.innerHTML = '';
  open.forEach((c) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn ghost block collect-pick';
    const n = (c.members || []).length;
    b.innerHTML = `<b>${escapeHtml(c.name)}</b>`
      + `<span class="small muted"> — ${n} video${n === 1 ? '' : 's'}</span>`;
    b.onclick = () => addToCollection(c.id, c.name);
    wrap.appendChild(b);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function addToCollection(cid, name) {
  const wrap = el('collectList');
  wrap.innerHTML = '<p class="small muted">Adding and rebuilding…</p>';
  try {
    await jsonPost(`/api/collections/${cid}/members`,
                   { session_id: S.session.id });
    finishCollect(cid, `Added to "${name}".`);
  } catch (e) {
    // The most likely refusals are meaningful to the operator: an unsupported venue
    // (D4) or the same video twice. Show the server's reason rather than a generic error.
    wrap.innerHTML = `<p class="small err">${escapeHtml(e.message || 'Could not add.')}</p>`;
    toast(e.message || 'Could not add to the cumulative report', true);
  }
}

function finishCollect(cid, msg) {
  el('collectList').innerHTML = '';
  el('collectNewWrap').hidden = true;
  el('collectNewBtn').hidden = true;
  el('collectSkipBtn').hidden = true;
  el('collectDoneMsg').textContent = msg;
  el('viewCollectionBtn').href = `/api/collections/${cid}/files/report.html`;
  el('collectDone').hidden = false;
}

function initCollectStep() {
  const nb = el('collectNewBtn');
  if (!nb) return;
  nb.onclick = () => {
    el('collectNewWrap').hidden = false;
    el('collectNewName').focus();
  };
  el('collectNewGo').onclick = async () => {
    const name = el('collectNewName').value.trim();
    if (!name) { toast('Give the report a name (whose is it?)', true); return; }
    try {
      const c = await jsonPost('/api/collections', { name });
      await addToCollection(c.id, c.name);
    } catch (e) {
      toast(e.message || 'Could not create the cumulative report', true);
    }
  };
  el('collectSkipBtn').onclick = () => {
    el('collectCard').hidden = true;
  };
}


// ---------------------------------------------------------------------------
// Cumulative reports — management view
//
// Reachable any time from the top bar rather than being a wizard step: managing a
// player's running report has nothing to do with setting up one video.
// ---------------------------------------------------------------------------

function showCollectionsPanel(on) {
  $$('.panel').forEach((p) => { p.hidden = true; });
  el('panel-collections').hidden = !on;
  if (on) loadCollMgr(); else goto('video');
}

async function loadCollMgr() {
  const wrap = el('collMgrList');
  wrap.innerHTML = '<p class="small muted">Loading…</p>';
  try {
    const data = await api('/api/collections');
    el('collMgrReminder').textContent = data.reminder || '';
    renderCollMgr(data.collections || [], data.active_id);
  } catch (e) {
    wrap.innerHTML = '<p class="small err">' + escapeHtml(e.message || 'Could not load') + '</p>';
  }
}

function renderCollMgr(list, activeId) {
  const wrap = el('collMgrList');
  if (!list.length) {
    wrap.innerHTML = '<p class="small muted">No cumulative reports yet. Create one on the right.</p>';
    el('collMgrAddCard').hidden = true;
    return;
  }
  wrap.innerHTML = '';
  list.forEach((c) => {
    const n = (c.members || []).length;
    const card = document.createElement('div');
    card.className = 'coll-card' + (c.closed_at ? ' is-closed' : '');
    const members = (c.members || []).map((m) =>
      '<li><span>' + escapeHtml(m.session_id) + '</span>'
      + '<button class="btn ghost xsmall" data-rm="' + escapeHtml(m.session_id)
      + '" data-cid="' + escapeHtml(c.id) + '">Remove</button></li>').join('');
    card.innerHTML =
      '<div class="coll-card-head"><h3>' + escapeHtml(c.name)
      + (c.id === activeId ? ' <span class="badge ok">default</span>' : '')
      + (c.closed_at ? ' <span class="badge">closed</span>' : '')
      + '</h3><span class="small muted">' + n + ' video' + (n === 1 ? '' : 's')
      + '</span></div>'
      + '<ul class="coll-members">' + (members || '<li class="small muted">No videos yet</li>')
      + '</ul><div class="coll-card-actions">'
      + '<a class="btn primary small" target="_blank" href="/api/collections/'
      + encodeURIComponent(c.id) + '/files/report.html">View report →</a>'
      + (c.closed_at
        ? '<button class="btn ghost small" data-reopen="' + escapeHtml(c.id) + '">Resume adding videos</button>'
        : '<button class="btn ghost small" data-add="' + escapeHtml(c.id) + '">Add a video</button>'
        + '<button class="btn ghost small" data-act="' + escapeHtml(c.id) + '">Make default</button>'
        + '<button class="btn ghost small" data-close="' + escapeHtml(c.id) + '">Stop adding videos</button>')
      + '</div>';
    wrap.appendChild(card);
  });

  wrap.querySelectorAll('[data-rm]').forEach((b) => {
    b.onclick = () => collMgrAction('/api/collections/' + b.dataset.cid + '/members/'
                                    + b.dataset.rm, 'DELETE', 'Removed ' + b.dataset.rm);
  });
  wrap.querySelectorAll('[data-act]').forEach((b) => {
    b.onclick = () => collMgrAction('/api/collections/' + b.dataset.act + '/activate',
                                    'POST', 'Set as default');
  });
  wrap.querySelectorAll('[data-reopen]').forEach((b) => {
    b.onclick = () => collMgrAction('/api/collections/' + b.dataset.reopen + '/reopen',
                                    'POST', 'Resumed');
  });
  wrap.querySelectorAll('[data-close]').forEach((b) => {
    b.onclick = () => {
      // This used to be labelled "Close", which reads as closing the panel. The operator
      // pressed it after viewing a report and silently ended their cumulative analysis.
      if (!confirm('Stop adding videos to this report? The report stays and can be '
                   + 'resumed later - this only stops new videos joining it.')) return;
      collMgrAction('/api/collections/' + b.dataset.close + '/close', 'POST',
                    'Stopped adding videos');
    };
  });
  wrap.querySelectorAll('[data-add]').forEach((b) => {
    b.onclick = () => openAddPicker(b.dataset.add);
  });
}

async function collMgrAction(path, method, okMsg) {
  try {
    await api(path, { method });
    toast(okMsg);
    await loadCollMgr();
  } catch (e) {
    toast(e.message || 'That did not work', true);
  }
}

async function openAddPicker(cid) {
  const card = el('collMgrAddCard');
  const list = el('collMgrAddList');
  card.hidden = false;
  list.innerHTML = '<p class="small muted">Loading videos…</p>';
  try {
    const { sessions } = await api('/api/sessions');
    list.innerHTML = '';
    if (!sessions.length) {
      list.innerHTML = '<p class="small muted">No videos set up yet.</p>';
      return;
    }
    sessions.forEach((s) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn ghost block collect-pick';
      b.textContent = s.name || s.id;
      b.onclick = async () => {
        try {
          await jsonPost('/api/collections/' + cid + '/members', { session_id: s.id });
          toast('Added ' + (s.name || s.id));
          card.hidden = true;
          await loadCollMgr();
        } catch (e) {
          // The refusals that actually happen carry meaning: an unsupported venue, the
          // same video twice, or a video that was never analysed. Show the real reason.
          toast(e.message || 'Could not add that video', true);
        }
      };
      list.appendChild(b);
    });
  } catch (e) {
    list.innerHTML = '<p class="small err">' + escapeHtml(e.message || 'Could not load') + '</p>';
  }
}

// ---------------------------------------------------------------------------
// "Whose report is this?" — asked on the video page, with the video.
// ---------------------------------------------------------------------------

async function askWhoseReport(session) {
  const box = el('videoCollect');
  const list = el('videoCollectList');
  box.hidden = false;
  list.innerHTML = '<p class="small muted">Loading…</p>';
  try {
    const data = await api('/api/collections');
    el('videoCollectReminder').textContent = data.reminder || '';
    const open = (data.collections || []).filter((c) => !c.closed_at);
    list.innerHTML = '';
    open.forEach((c) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn ghost block collect-pick';
      const n = (c.members || []).length;
      b.innerHTML = '<b>' + escapeHtml(c.name) + '</b><span class="small muted"> — '
                    + n + ' video' + (n === 1 ? '' : 's') + '</span>';
      b.onclick = () => chooseReport(session, c.id, c.name);
      list.appendChild(b);
    });
    if (!open.length) {
      list.innerHTML = '<p class="small muted">No cumulative reports yet.</p>';
    }
  } catch (e) {
    list.innerHTML = '<p class="small err">' + escapeHtml(e.message || 'Could not load') + '</p>';
  }
  el('videoCollectNewGo').onclick = async () => {
    const name = el('videoCollectNewName').value.trim();
    if (!name) { toast('Whose report is it?', true); return; }
    try {
      const c = await jsonPost('/api/collections', { name });
      await chooseReport(session, c.id, c.name);
    } catch (e) { toast(e.message || 'Could not create', true); }
  };
  el('videoCollectSkip').onclick = () => chooseReport(session, null, null);
}

async function chooseReport(session, cid, name) {
  try {
    const updated = await jsonPost(`/api/sessions/${session.id}/collection`,
                                   { collection_id: cid, player_name: name });
    el('videoCollect').hidden = true;
    toast(cid ? `This video will join "${name}"` : 'Standalone analysis');
    await onSessionReady(updated);
  } catch (e) {
    toast(e.message || 'Could not save that choice', true);
  }
}

function initCollMgr() {
  const open = el('openCollections');
  if (!open) return;
  open.onclick = () => showCollectionsPanel(true);
  el('collMgrBack').onclick = () => showCollectionsPanel(false);
  el('collMgrNewGo').onclick = async () => {
    const name = el('collMgrNewName').value.trim();
    if (!name) { toast('Give the report a name (whose is it?)', true); return; }
    try {
      await jsonPost('/api/collections', { name });
      el('collMgrNewName').value = '';
      toast('Created "' + name + '"');
      await loadCollMgr();
    } catch (e) {
      toast(e.message || 'Could not create', true);
    }
  };
}
