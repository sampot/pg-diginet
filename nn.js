/** Tiny MLP forward pass for digit toy. */

/**
 * @typedef {{ inputSize: number, hidden: number, classes: number, w1: number[][], b1: number[], w2: number[][], b2: number[] }} Weights
 */

/**
 * @param {Weights} model
 * @param {Float32Array|number[]} x flat [0,1], length inputSize²
 */
export function forward(model, x) {
  const { w1, b1, w2, b2, hidden, classes } = model;
  const nIn = model.inputSize * model.inputSize;
  const h = new Float32Array(hidden);
  for (let j = 0; j < hidden; j++) {
    let s = b1[j];
    const row = w1[j];
    for (let i = 0; i < nIn; i++) s += row[i] * x[i];
    h[j] = s > 0 ? s : 0;
  }
  const logits = new Float32Array(classes);
  for (let k = 0; k < classes; k++) {
    let s = b2[k];
    const row = w2[k];
    for (let j = 0; j < hidden; j++) s += row[j] * h[j];
    logits[k] = s;
  }
  return { h, logits, probs: softmax(logits) };
}

/** @param {Float32Array} logits */
function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const e = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    e[i] = Math.exp(logits[i] - max);
    sum += e[i];
  }
  for (let i = 0; i < e.length; i++) e[i] /= sum;
  return e;
}

/**
 * Rasterize drawing canvas → centered SIZE×SIZE like MNIST habit.
 * Source is white ink on black.
 * @param {HTMLCanvasElement} src
 * @param {number} size
 */
export function rasterizeCentered(src, size) {
  const sw = src.width;
  const sh = src.height;
  const sctx = src.getContext("2d", { willReadFrequently: true });
  const img = sctx.getImageData(0, 0, sw, sh);
  const data = img.data;

  let minX = sw,
    minY = sh,
    maxX = -1,
    maxY = -1;
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const a = data[(y * sw + x) * 4];
      if (a > 20) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  const out = new Float32Array(size * size);
  if (maxX < 0) return out;

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const box = Math.max(bw, bh);
  // ~20% padding like many MNIST demos
  const pad = box * 0.2;
  const side = box + pad * 2;
  const tmp = document.createElement("canvas");
  tmp.width = size;
  tmp.height = size;
  const tctx = tmp.getContext("2d");
  tctx.fillStyle = "#000";
  tctx.fillRect(0, 0, size, size);

  const scale = size / side;
  const dx = (size - bw * scale) / 2;
  const dy = (size - bh * scale) / 2;
  tctx.imageSmoothingEnabled = true;
  tctx.drawImage(src, minX, minY, bw, bh, dx, dy, bw * scale, bh * scale);

  const small = tctx.getImageData(0, 0, size, size).data;
  for (let i = 0; i < size * size; i++) {
    out[i] = small[i * 4] / 255;
  }
  return out;
}
