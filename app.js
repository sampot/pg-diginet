import { forward, rasterizeCentered } from "./nn.js";

const pad = document.getElementById("pad");
const pctx = pad.getContext("2d");
const net = document.getElementById("net");
const nctx = net.getContext("2d");
const barsEl = document.getElementById("bars");
const predEl = document.getElementById("pred");
const statusEl = document.getElementById("status");
const heat = document.getElementById("heat");
const hctx = heat.getContext("2d");

const btnClear = document.getElementById("btn-clear");
const heatHint = document.getElementById("heat-hint");

/** @type {import("./nn.js").Weights | null} */
let model = null;
/** @type {number | null} */
let selectedHidden = null;

const PAD = 280;

function clearPad() {
  pctx.fillStyle = "#0b1220";
  pctx.fillRect(0, 0, pad.width, pad.height);
  selectedHidden = null;
  predict();
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
  const x = (("clientX" in e ? e.clientX : e.touches[0].clientX) - r.left) * (pad.width / r.width);
  const y = (("clientY" in e ? e.clientY : e.touches[0].clientY) - r.top) * (pad.height / r.height);
  return { x, y };
}

function stroke(a, b) {
  pctx.strokeStyle = "#f8fafc";
  pctx.lineWidth = 18;
  pctx.lineCap = "round";
  pctx.lineJoin = "round";
  pctx.beginPath();
  pctx.moveTo(a.x, a.y);
  pctx.lineTo(b.x, b.y);
  pctx.stroke();
}

pad.addEventListener("pointerdown", e => {
  pad.setPointerCapture(e.pointerId);
  drawing = true;
  last = pos(e);
  stroke(last, last);
});
pad.addEventListener("pointermove", e => {
  if (!drawing) return;
  const p = pos(e);
  stroke(last, p);
  last = p;
});
pad.addEventListener("pointerup", () => {
  drawing = false;
  last = null;
  predict();
});
pad.addEventListener("pointercancel", () => {
  drawing = false;
  last = null;
});

btnClear.addEventListener("click", clearPad);

function renderBars(probs) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  predEl.textContent = Number.isFinite(probs[best]) && probs[best] > 0.05 ? String(best) : "—";
  barsEl.innerHTML = "";
  for (let d = 0; d < 10; d++) {
    const row = document.createElement("div");
    row.className = "bar-row" + (d === best && probs[d] > 0.05 ? " best" : "");
    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = String(d);
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(probs[d] * 100).toFixed(1)}%`;
    const pct = document.createElement("span");
    pct.className = "bar-pct";
    pct.textContent = `${(probs[d] * 100).toFixed(0)}%`;
    track.appendChild(fill);
    row.append(label, track, pct);
    barsEl.appendChild(row);
  }
}

function drawHeatmap(neuron) {
  if (!model || neuron == null) {
    hctx.clearRect(0, 0, heat.width, heat.height);
    heatHint.textContent = "點網路中間的隱藏節點，看它對哪些像素敏感。";
    return;
  }
  const size = model.inputSize;
  const row = model.w1[neuron];
  let max = 0;
  for (let i = 0; i < row.length; i++) max = Math.max(max, Math.abs(row[i]));
  heat.width = size * 12;
  heat.height = size * 12;
  const cell = 12;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const w = row[y * size + x];
      const a = max > 0 ? Math.abs(w) / max : 0;
      if (w >= 0) hctx.fillStyle = `rgba(45, 212, 191, ${0.15 + a * 0.85})`;
      else hctx.fillStyle = `rgba(248, 113, 113, ${0.15 + a * 0.85})`;
      hctx.fillRect(x * cell, y * cell, cell - 1, cell - 1);
    }
  }
  heatHint.textContent = `隱藏單元 #${neuron} 的輸入權重（青＝正、紅＝負）。`;
}

function drawNet(x, h, probs) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = net.clientWidth || 360;
  const cssH = 280;
  net.width = Math.floor(cssW * dpr);
  net.height = Math.floor(cssH * dpr);
  nctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  nctx.clearRect(0, 0, cssW, cssH);
  nctx.fillStyle = "#0b1220";
  nctx.fillRect(0, 0, cssW, cssH);

  if (!model) return;
  const size = model.inputSize;
  const hidden = model.hidden;

  const colIn = cssW * 0.14;
  const colH = cssW * 0.5;
  const colOut = cssW * 0.86;

  // input grid
  const grid = 72;
  const gx = colIn - grid / 2;
  const gy = (cssH - grid) / 2;
  const cell = grid / size;
  for (let y = 0; y < size; y++) {
    for (let i = 0; i < size; i++) {
      const v = x[y * size + i];
      nctx.fillStyle = `rgb(${Math.round(v * 240)},${Math.round(v * 245)},${Math.round(20 + v * 220)})`;
      nctx.fillRect(gx + i * cell, gy + y * cell, cell, cell);
    }
  }
  nctx.strokeStyle = "rgba(255,255,255,0.2)";
  nctx.strokeRect(gx, gy, grid, grid);
  nctx.fillStyle = "#9aa3ad";
  nctx.font = "11px sans-serif";
  nctx.textAlign = "center";
  nctx.fillText("輸入", colIn, gy - 8);

  // hidden positions
  const hPos = [];
  const hR = 7;
  for (let j = 0; j < hidden; j++) {
    const t = hidden === 1 ? 0.5 : j / (hidden - 1);
    hPos.push({ x: colH, y: 28 + t * (cssH - 56) });
  }

  // output positions
  const oPos = [];
  for (let k = 0; k < 10; k++) {
    const t = k / 9;
    oPos.push({ x: colOut, y: 28 + t * (cssH - 56) });
  }

  // edges in→hidden (sample strongest per hidden)
  for (let j = 0; j < hidden; j++) {
    const act = h[j];
    if (act < 0.02 && selectedHidden !== j) continue;
    const row = model.w1[j];
    // pick a few strong weights
    const idxs = [];
    for (let i = 0; i < row.length; i++) idxs.push(i);
    idxs.sort((a, b) => Math.abs(row[b]) - Math.abs(row[a]));
    const top = idxs.slice(0, selectedHidden === j ? 40 : 6);
    for (const i of top) {
      const px = gx + (i % size) * cell + cell / 2;
      const py = gy + Math.floor(i / size) * cell + cell / 2;
      const w = row[i];
      const a = Math.min(0.55, 0.08 + Math.abs(w) * 2 * (0.3 + act));
      nctx.strokeStyle =
        w >= 0 ? `rgba(45, 212, 191, ${a})` : `rgba(248, 113, 113, ${a})`;
      nctx.lineWidth = selectedHidden === j ? 1.2 : 0.7;
      nctx.beginPath();
      nctx.moveTo(px, py);
      nctx.lineTo(hPos[j].x, hPos[j].y);
      nctx.stroke();
    }
  }

  // edges hidden→out
  let hMax = 0.001;
  for (let j = 0; j < hidden; j++) hMax = Math.max(hMax, h[j]);
  for (let k = 0; k < 10; k++) {
    for (let j = 0; j < hidden; j++) {
      const w = model.w2[k][j];
      const strength = (h[j] / hMax) * Math.abs(w);
      if (strength < 0.08 && selectedHidden !== j) continue;
      const a = Math.min(0.5, 0.05 + strength * 0.6);
      nctx.strokeStyle =
        w >= 0 ? `rgba(125, 211, 252, ${a})` : `rgba(251, 146, 60, ${a})`;
      nctx.lineWidth = 0.8;
      nctx.beginPath();
      nctx.moveTo(hPos[j].x, hPos[j].y);
      nctx.lineTo(oPos[k].x, oPos[k].y);
      nctx.stroke();
    }
  }

  // hidden nodes
  for (let j = 0; j < hidden; j++) {
    const p = hPos[j];
    const t = Math.min(1, h[j] / hMax);
    nctx.beginPath();
    nctx.arc(p.x, p.y, hR + (selectedHidden === j ? 2 : 0), 0, Math.PI * 2);
    nctx.fillStyle = `rgba(45, 212, 191, ${0.2 + t * 0.8})`;
    nctx.fill();
    nctx.strokeStyle = selectedHidden === j ? "#fff" : "rgba(255,255,255,0.35)";
    nctx.lineWidth = selectedHidden === j ? 2 : 1;
    nctx.stroke();
  }
  nctx.fillStyle = "#9aa3ad";
  nctx.fillText(`隱藏×${hidden}`, colH, 16);

  // outputs
  for (let k = 0; k < 10; k++) {
    const p = oPos[k];
    const t = probs[k];
    nctx.beginPath();
    nctx.arc(p.x, p.y, 8 + t * 6, 0, Math.PI * 2);
    nctx.fillStyle = `rgba(56, 189, 248, ${0.25 + t * 0.75})`;
    nctx.fill();
    nctx.fillStyle = "#e8ecf1";
    nctx.font = "10px sans-serif";
    nctx.fillText(String(k), p.x + 16, p.y + 3);
  }
  nctx.fillStyle = "#9aa3ad";
  nctx.font = "11px sans-serif";
  nctx.fillText("輸出", colOut, 16);

  // hit regions for click
  net._hPos = hPos;
}

net.addEventListener("click", e => {
  if (!model || !net._hPos) return;
  const r = net.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  let hit = null;
  let best = 14;
  for (let j = 0; j < net._hPos.length; j++) {
    const p = net._hPos[j];
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < best) {
      best = d;
      hit = j;
    }
  }
  selectedHidden = hit === selectedHidden ? null : hit;
  predict();
});

function predict() {
  if (!model) return;
  const x = rasterizeCentered(pad, model.inputSize);
  const { h, probs } = forward(model, x);
  renderBars(probs);
  drawNet(x, h, probs);
  drawHeatmap(selectedHidden);
}

async function boot() {
  statusEl.textContent = "載入權重…";
  setupPad();
  try {
    const res = await fetch("./weights.json");
    model = await res.json();
    statusEl.textContent = `MLP ${model.inputSize}×${model.inputSize} → ${model.hidden} → 10（MNIST 小品權重）`;
    heat.width = model.inputSize * 12;
    heat.height = model.inputSize * 12;
    predict();
  } catch (err) {
    statusEl.textContent = "權重載入失敗";
    console.error(err);
  }
}

window.addEventListener("resize", () => {
  if (model) predict();
});

boot();
