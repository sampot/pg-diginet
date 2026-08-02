import { forward, rasterizeCentered } from "./nn.js";

const pad = document.getElementById("pad");
const pctx = pad.getContext("2d");
const net = document.getElementById("net");
const nctx = net.getContext("2d");
const barsEl = document.getElementById("bars");
const predEl = document.getElementById("pred");
const predWrap = document.getElementById("pred-wrap");
const statusEl = document.getElementById("status");
const heat = document.getElementById("heat");
const hctx = heat.getContext("2d");
const confEl = document.getElementById("conf");
const samplesEl = document.getElementById("samples");

const btnClear = document.getElementById("btn-clear");
const btnPulse = document.getElementById("btn-pulse");
const heatHint = document.getElementById("heat-hint");

/** @type {import("./nn.js").Weights | null} */
let model = null;
/** @type {number | null} */
let selectedHidden = null;

const PAD = 280;

/** @type {Float32Array} */
let targetX = new Float32Array(256);
/** @type {Float32Array} */
let targetH = new Float32Array(24);
/** @type {Float32Array} */
let targetP = new Float32Array(10);
/** @type {Float32Array} */
let showH = new Float32Array(24);
/** @type {Float32Array} */
let showP = new Float32Array(10);

/** @type {{x0:number,y0:number,x1:number,y1:number,t:number,speed:number,r:number,g:number,b:number,a:number}[]} */
let particles = [];
/** Cached edge segments for particles */
let edgeBank = [];

let waveT = 1; // 0→1 forward sweep
let flash = 0;
let bestDigit = -1;
let cssW = 360;
let cssH = 300;
/** @type {{x:number,y:number}[]} */
let hPos = [];
/** @type {{x:number,y:number}[]} */
let oPos = [];
let gx = 0,
  gy = 0,
  grid = 80,
  cell = 5;
let dirtyBars = true;
let lastInfer = 0;

function clearPad() {
  pctx.fillStyle = "#070b14";
  pctx.fillRect(0, 0, pad.width, pad.height);
  // faint grid
  pctx.strokeStyle = "rgba(45,212,191,0.06)";
  pctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const v = (PAD * i) / 4;
    pctx.beginPath();
    pctx.moveTo(v, 0);
    pctx.lineTo(v, PAD);
    pctx.moveTo(0, v);
    pctx.lineTo(PAD, v);
    pctx.stroke();
  }
  selectedHidden = null;
  waveT = 1;
  infer(true);
}

function setupPad() {
  pad.width = PAD;
  pad.height = PAD;
  clearPad();
}

let drawing = false;
let last = null;

function pos(e) {
  const r = pad.getBoundingClientRect();
  const x =
    (("clientX" in e ? e.clientX : e.touches[0].clientX) - r.left) *
    (pad.width / r.width);
  const y =
    (("clientY" in e ? e.clientY : e.touches[0].clientY) - r.top) *
    (pad.height / r.height);
  return { x, y };
}

function stroke(a, b) {
  pctx.save();
  pctx.strokeStyle = "#f8fafc";
  pctx.shadowColor = "rgba(45, 212, 191, 0.65)";
  pctx.shadowBlur = 12;
  pctx.lineWidth = 18;
  pctx.lineCap = "round";
  pctx.lineJoin = "round";
  pctx.beginPath();
  pctx.moveTo(a.x, a.y);
  pctx.lineTo(b.x, b.y);
  pctx.stroke();
  pctx.restore();
}

pad.addEventListener("pointerdown", e => {
  pad.setPointerCapture(e.pointerId);
  drawing = true;
  last = pos(e);
  stroke(last, last);
  infer(false);
});
pad.addEventListener("pointermove", e => {
  if (!drawing) return;
  const p = pos(e);
  stroke(last, p);
  last = p;
  const now = performance.now();
  if (now - lastInfer > 32) {
    lastInfer = now;
    infer(false);
  }
});
pad.addEventListener("pointerup", () => {
  drawing = false;
  last = null;
  waveT = 0;
  flash = 1;
  infer(true);
});
pad.addEventListener("pointercancel", () => {
  drawing = false;
  last = null;
});

btnClear.addEventListener("click", clearPad);
btnPulse.addEventListener("click", () => {
  waveT = 0;
  flash = 1;
  infer(true);
});

function infer(forceWave) {
  if (!model) return;
  targetX = rasterizeCentered(pad, model.inputSize);
  const { h, probs } = forward(model, targetX);
  targetH = h;
  targetP = probs;
  dirtyBars = true;
  if (forceWave) rebuildEdges();
}

function lerpArr(dst, src, k) {
  for (let i = 0; i < dst.length; i++) dst[i] += (src[i] - dst[i]) * k;
}

function rebuildEdges() {
  if (!model) return;
  edgeBank = [];
  const size = model.inputSize;
  let hMax = 0.001;
  for (let j = 0; j < model.hidden; j++) hMax = Math.max(hMax, targetH[j]);

  for (let j = 0; j < model.hidden; j++) {
    if (targetH[j] < 0.05 && selectedHidden !== j) continue;
    const row = model.w1[j];
    const idxs = [];
    for (let i = 0; i < row.length; i++) idxs.push(i);
    idxs.sort((a, b) => Math.abs(row[b]) - Math.abs(row[a]));
    for (const i of idxs.slice(0, 4)) {
      const w = row[i];
      edgeBank.push({
        x0: gx + (i % size) * cell + cell / 2,
        y0: gy + Math.floor(i / size) * cell + cell / 2,
        x1: hPos[j]?.x ?? cssW * 0.48,
        y1: hPos[j]?.y ?? cssH / 2,
        phase: "in",
        sign: w >= 0 ? 1 : -1,
        w: Math.abs(w) * (0.4 + targetH[j] / hMax),
      });
    }
  }

  let best = 0;
  for (let k = 1; k < 10; k++) if (targetP[k] > targetP[best]) best = k;
  for (let j = 0; j < model.hidden; j++) {
    const w = model.w2[best][j];
    const strength = (targetH[j] / hMax) * Math.abs(w);
    if (strength < 0.12) continue;
    edgeBank.push({
      x0: hPos[j]?.x ?? cssW * 0.48,
      y0: hPos[j]?.y ?? cssH / 2,
      x1: oPos[best]?.x ?? cssW * 0.86,
      y1: oPos[best]?.y ?? cssH / 2,
      phase: "out",
      sign: w >= 0 ? 1 : -1,
      w: strength,
    });
  }
}

function spawnParticles(dt) {
  if (!edgeBank.length || waveT < 0.15) return;
  const n = Math.min(3, Math.ceil(edgeBank.length * dt * 8));
  for (let i = 0; i < n; i++) {
    const e = edgeBank[(Math.random() * edgeBank.length) | 0];
    const cool = e.phase === "in";
    particles.push({
      x0: e.x0,
      y0: e.y0,
      x1: e.x1,
      y1: e.y1,
      t: 0,
      speed: 0.55 + Math.random() * 0.9,
      r: cool ? (e.sign > 0 ? 45 : 248) : e.sign > 0 ? 125 : 251,
      g: cool ? (e.sign > 0 ? 212 : 113) : e.sign > 0 ? 211 : 146,
      b: cool ? (e.sign > 0 ? 191 : 113) : e.sign > 0 ? 252 : 60,
      a: 0.35 + Math.min(0.55, e.w),
    });
  }
  if (particles.length > 120) particles.splice(0, particles.length - 120);
}

function renderBars() {
  let best = 0;
  for (let i = 1; i < showP.length; i++) if (showP[i] > showP[best]) best = i;
  const conf = showP[best];
  const show = conf > 0.08;
  bestDigit = show ? best : -1;
  predEl.textContent = show ? String(best) : "—";
  predWrap.classList.toggle("hot", show && conf > 0.45);
  confEl.textContent = show ? `${(conf * 100).toFixed(0)}%` : "";
  confEl.style.opacity = show ? String(0.5 + conf * 0.5) : "0";

  if (!dirtyBars && barsEl.childElementCount === 10) {
    for (let d = 0; d < 10; d++) {
      const row = barsEl.children[d];
      row.classList.toggle("best", d === best && show);
      row.querySelector(".bar-fill").style.width = `${(showP[d] * 100).toFixed(1)}%`;
      row.querySelector(".bar-pct").textContent = `${(showP[d] * 100).toFixed(0)}%`;
    }
    return;
  }
  dirtyBars = false;
  barsEl.innerHTML = "";
  for (let d = 0; d < 10; d++) {
    const row = document.createElement("div");
    row.className = "bar-row" + (d === best && show ? " best" : "");
    row.innerHTML = `<span class="bar-label">${d}</span><div class="bar-track"><div class="bar-fill" style="width:${(showP[d] * 100).toFixed(1)}%"></div></div><span class="bar-pct">${(showP[d] * 100).toFixed(0)}%</span>`;
    barsEl.appendChild(row);
  }
}

function drawHeatmap(neuron) {
  if (!model || neuron == null) {
    hctx.clearRect(0, 0, heat.width, heat.height);
    heatHint.textContent = "點隱藏節點看權重；畫的時候訊號會沿連線流動。";
    return;
  }
  const size = model.inputSize;
  const row = model.w1[neuron];
  let max = 0;
  for (let i = 0; i < row.length; i++) max = Math.max(max, Math.abs(row[i]));
  heat.width = size * 12;
  heat.height = size * 12;
  const c = 12;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const w = row[y * size + x];
      const a = max > 0 ? Math.abs(w) / max : 0;
      hctx.fillStyle =
        w >= 0
          ? `rgba(45, 212, 191, ${0.15 + a * 0.85})`
          : `rgba(248, 113, 113, ${0.15 + a * 0.85})`;
      hctx.fillRect(x * c, y * c, c - 1, c - 1);
    }
  }
  heatHint.textContent = `隱藏 #${neuron}（青正紅負）· 再點一下取消`;
}

function layoutNet() {
  cssW = net.clientWidth || 360;
  cssH = 300;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  net.width = Math.floor(cssW * dpr);
  net.height = Math.floor(cssH * dpr);
  nctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (!model) return;
  const hidden = model.hidden;
  const colIn = cssW * 0.16;
  const colH = cssW * 0.5;
  const colOut = cssW * 0.86;
  grid = Math.min(88, cssW * 0.22);
  gx = colIn - grid / 2;
  gy = (cssH - grid) / 2;
  cell = grid / model.inputSize;

  hPos = [];
  for (let j = 0; j < hidden; j++) {
    const t = hidden === 1 ? 0.5 : j / (hidden - 1);
    // slight arc for depth
    const bulge = Math.sin(t * Math.PI) * 10;
    hPos.push({ x: colH + bulge, y: 32 + t * (cssH - 64) });
  }
  oPos = [];
  for (let k = 0; k < 10; k++) {
    const t = k / 9;
    oPos.push({ x: colOut, y: 32 + t * (cssH - 64) });
  }
  net._hPos = hPos;
}

function drawNetFrame(now) {
  layoutNet();
  nctx.clearRect(0, 0, cssW, cssH);

  // depth backdrop
  const g = nctx.createLinearGradient(0, 0, cssW, cssH);
  g.addColorStop(0, "#070b14");
  g.addColorStop(0.5, "#0c1524");
  g.addColorStop(1, "#070b14");
  nctx.fillStyle = g;
  nctx.fillRect(0, 0, cssW, cssH);

  // soft vignette pulse on flash
  if (flash > 0.01) {
    nctx.fillStyle = `rgba(45,212,191,${flash * 0.07})`;
    nctx.fillRect(0, 0, cssW, cssH);
  }

  if (!model) return;
  const size = model.inputSize;
  const hidden = model.hidden;
  const t = now / 1000;

  // layer gates from wave
  const gateIn = Math.min(1, waveT / 0.35);
  const gateH = Math.max(0, Math.min(1, (waveT - 0.2) / 0.35));
  const gateOut = Math.max(0, Math.min(1, (waveT - 0.45) / 0.4));

  // input grid
  nctx.globalAlpha = 0.35 + gateIn * 0.65;
  for (let y = 0; y < size; y++) {
    for (let i = 0; i < size; i++) {
      const v = targetX[y * size + i];
      const pulse = 0.85 + 0.15 * Math.sin(t * 4 + i * 0.3 + y * 0.2);
      const c = Math.round(v * 255 * pulse);
      nctx.fillStyle = `rgb(${c},${Math.round(c * 0.98)},${Math.round(30 + v * 200)})`;
      nctx.fillRect(gx + i * cell, gy + y * cell, Math.max(1, cell - 0.5), Math.max(1, cell - 0.5));
    }
  }
  nctx.globalAlpha = 1;
  nctx.strokeStyle = `rgba(45,212,191,${0.25 + gateIn * 0.4})`;
  nctx.lineWidth = 1.5;
  nctx.strokeRect(gx - 1, gy - 1, grid + 2, grid + 2);
  nctx.fillStyle = "#9aa3ad";
  nctx.font = "11px sans-serif";
  nctx.textAlign = "center";
  nctx.fillText("輸入", gx + grid / 2, gy - 10);

  let hMax = 0.001;
  for (let j = 0; j < hidden; j++) hMax = Math.max(hMax, showH[j]);

  // static edges (dim)
  nctx.lineWidth = 0.6;
  for (let j = 0; j < hidden; j++) {
    const act = showH[j];
    if (act < 0.04 && selectedHidden !== j) continue;
    if (gateH < 0.05) continue;
    const row = model.w1[j];
    const idxs = [];
    for (let i = 0; i < row.length; i++) idxs.push(i);
    idxs.sort((a, b) => Math.abs(row[b]) - Math.abs(row[a]));
    for (const i of idxs.slice(0, selectedHidden === j ? 28 : 5)) {
      const w = row[i];
      const a =
        Math.min(0.45, 0.06 + Math.abs(w) * 1.8 * (0.25 + act / hMax)) *
        gateH *
        (selectedHidden === j || selectedHidden == null ? 1 : 0.15);
      nctx.strokeStyle =
        w >= 0 ? `rgba(45,212,191,${a})` : `rgba(248,113,113,${a})`;
      nctx.beginPath();
      nctx.moveTo(gx + (i % size) * cell + cell / 2, gy + Math.floor(i / size) * cell + cell / 2);
      nctx.lineTo(hPos[j].x, hPos[j].y);
      nctx.stroke();
    }
  }

  for (let k = 0; k < 10; k++) {
    for (let j = 0; j < hidden; j++) {
      const w = model.w2[k][j];
      const strength = (showH[j] / hMax) * Math.abs(w) * showP[k];
      if (strength < 0.05 && !(selectedHidden === j)) continue;
      if (gateOut < 0.05) continue;
      const a = Math.min(0.4, 0.04 + strength * 0.7) * gateOut;
      nctx.strokeStyle =
        w >= 0 ? `rgba(125,211,252,${a})` : `rgba(251,146,60,${a})`;
      nctx.beginPath();
      nctx.moveTo(hPos[j].x, hPos[j].y);
      nctx.lineTo(oPos[k].x, oPos[k].y);
      nctx.stroke();
    }
  }

  // particles on top
  for (const p of particles) {
    const u = p.t;
    const x = p.x0 + (p.x1 - p.x0) * u;
    const y = p.y0 + (p.y1 - p.y0) * u;
    const fade = u < 0.15 ? u / 0.15 : u > 0.85 ? (1 - u) / 0.15 : 1;
    nctx.beginPath();
    nctx.arc(x, y, 2.2, 0, Math.PI * 2);
    nctx.fillStyle = `rgba(${p.r},${p.g},${p.b},${p.a * fade})`;
    nctx.fill();
    nctx.beginPath();
    nctx.arc(x, y, 5, 0, Math.PI * 2);
    nctx.fillStyle = `rgba(${p.r},${p.g},${p.b},${0.12 * fade})`;
    nctx.fill();
  }

  // hidden nodes
  nctx.fillStyle = "#9aa3ad";
  nctx.fillText(`隱藏×${hidden}`, hPos[0] ? hPos[Math.floor(hidden / 2)].x : cssW / 2, 18);
  for (let j = 0; j < hidden; j++) {
    const p = hPos[j];
    const act = Math.min(1, showH[j] / hMax) * gateH;
    const breathe = 1 + 0.12 * Math.sin(t * 5 + j * 0.4) * act;
    const r = (6.5 + act * 5) * breathe + (selectedHidden === j ? 2 : 0);
    nctx.beginPath();
    nctx.arc(p.x, p.y, r + 6 * act, 0, Math.PI * 2);
    nctx.fillStyle = `rgba(45,212,191,${0.06 + act * 0.18})`;
    nctx.fill();
    nctx.beginPath();
    nctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    nctx.fillStyle = `rgba(45,212,191,${0.18 + act * 0.75})`;
    nctx.fill();
    nctx.strokeStyle =
      selectedHidden === j ? "#fff" : `rgba(255,255,255,${0.25 + act * 0.4})`;
    nctx.lineWidth = selectedHidden === j ? 2 : 1;
    nctx.stroke();
  }

  // outputs
  nctx.fillStyle = "#9aa3ad";
  nctx.fillText("輸出", oPos[0]?.x ?? cssW * 0.86, 18);
  for (let k = 0; k < 10; k++) {
    const p = oPos[k];
    const pr = showP[k] * gateOut;
    const win = k === bestDigit && pr > 0.08;
    const r = 7 + pr * 10 + (win ? 2 + flash * 3 : 0);
    nctx.beginPath();
    nctx.arc(p.x, p.y, r + (win ? 8 + flash * 10 : pr * 6), 0, Math.PI * 2);
    nctx.fillStyle = win
      ? `rgba(56,189,248,${0.1 + pr * 0.25 + flash * 0.15})`
      : `rgba(56,189,248,${0.04 + pr * 0.12})`;
    nctx.fill();
    nctx.beginPath();
    nctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    nctx.fillStyle = `rgba(56,189,248,${0.22 + pr * 0.75})`;
    nctx.fill();
    nctx.fillStyle = win ? "#fff" : "#e8ecf1";
    nctx.font = win ? "bold 12px sans-serif" : "10px sans-serif";
    nctx.fillText(String(k), p.x + 18, p.y + 4);
  }
}

net.addEventListener("click", e => {
  if (!model || !hPos.length) return;
  const r = net.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  let hit = null;
  let best = 16;
  for (let j = 0; j < hPos.length; j++) {
    const d = Math.hypot(hPos[j].x - x, hPos[j].y - y);
    if (d < best) {
      best = d;
      hit = j;
    }
  }
  selectedHidden = hit === selectedHidden ? null : hit;
  drawHeatmap(selectedHidden);
  rebuildEdges();
});

/** Crude stroke demos so visitors see something cool immediately. */
const DEMOS = {
  0: [[0.5, 0.18], [0.72, 0.28], [0.78, 0.5], [0.7, 0.75], [0.5, 0.85], [0.28, 0.72], [0.22, 0.5], [0.3, 0.28], [0.5, 0.18]],
  1: [[0.38, 0.28], [0.52, 0.18], [0.52, 0.85]],
  2: [[0.28, 0.3], [0.5, 0.18], [0.72, 0.28], [0.7, 0.45], [0.4, 0.62], [0.28, 0.85], [0.75, 0.85]],
  3: [[0.3, 0.22], [0.7, 0.22], [0.55, 0.48], [0.72, 0.62], [0.68, 0.8], [0.3, 0.82]],
  4: [[0.65, 0.18], [0.28, 0.58], [0.78, 0.58], [0.62, 0.58], [0.62, 0.85]],
  5: [[0.7, 0.2], [0.32, 0.2], [0.3, 0.48], [0.62, 0.45], [0.75, 0.6], [0.65, 0.82], [0.32, 0.82]],
  6: [[0.65, 0.22], [0.35, 0.35], [0.28, 0.55], [0.35, 0.78], [0.6, 0.85], [0.75, 0.68], [0.6, 0.52], [0.35, 0.55]],
  7: [[0.28, 0.22], [0.78, 0.22], [0.45, 0.85]],
  8: [[0.5, 0.18], [0.72, 0.3], [0.5, 0.48], [0.28, 0.3], [0.5, 0.18], [0.5, 0.48], [0.75, 0.68], [0.5, 0.88], [0.25, 0.68], [0.5, 0.48]],
  9: [[0.65, 0.55], [0.4, 0.2], [0.28, 0.38], [0.4, 0.55], [0.7, 0.4], [0.65, 0.2], [0.55, 0.85]],
};

function paintDemo(digit) {
  clearPad();
  const path = DEMOS[digit];
  if (!path) return;
  let prev = null;
  for (const [nx, ny] of path) {
    const p = { x: nx * PAD, y: ny * PAD };
    if (prev) stroke(prev, p);
    else stroke(p, p);
    prev = p;
  }
  waveT = 0;
  flash = 1;
  infer(true);
}

function buildSamples() {
  samplesEl.innerHTML = "";
  for (let d = 0; d < 10; d++) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = String(d);
    b.title = `示範 ${d}`;
    b.addEventListener("click", () => paintDemo(d));
    samplesEl.appendChild(b);
  }
}

let lastTs = 0;
function frame(ts) {
  const dt = Math.min(0.05, (ts - lastTs) / 1000 || 0.016);
  lastTs = ts;

  lerpArr(showH, targetH, 1 - Math.pow(0.001, dt));
  lerpArr(showP, targetP, 1 - Math.pow(0.001, dt));

  if (waveT < 1) waveT = Math.min(1, waveT + dt * 1.35);
  if (flash > 0) flash = Math.max(0, flash - dt * 1.8);

  spawnParticles(dt);
  for (const p of particles) p.t += p.speed * dt;
  particles = particles.filter(p => p.t < 1);

  if (hPos.length && edgeBank.length === 0 && waveT > 0.2) rebuildEdges();

  drawNetFrame(ts);
  renderBars();
  requestAnimationFrame(frame);
}

async function boot() {
  statusEl.textContent = "載入權重…";
  setupPad();
  buildSamples();
  try {
    const res = await fetch("./weights.json");
    model = await res.json();
    showH = new Float32Array(model.hidden);
    showP = new Float32Array(model.classes);
    targetH = new Float32Array(model.hidden);
    targetP = new Float32Array(model.classes);
    statusEl.textContent = `MLP ${model.inputSize}×${model.inputSize} → ${model.hidden} → 10 · 畫的當下就會「流」過網路`;
    heat.width = model.inputSize * 12;
    heat.height = model.inputSize * 12;
    drawHeatmap(null);
    infer(true);
    requestAnimationFrame(frame);
    // auto demo so first paint isn't empty
    setTimeout(() => paintDemo(5), 280);
  } catch (err) {
    statusEl.textContent = "權重載入失敗";
    console.error(err);
  }
}

window.addEventListener("resize", () => {
  if (model) rebuildEdges();
});

boot();
