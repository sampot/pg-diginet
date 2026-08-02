#!/usr/bin/env python3
"""Train a tiny MLP on MNIST (16×16) and write weights.json for the SAM."""

from __future__ import annotations

import gzip
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "scripts" / ".mnist-cache"
OUT = ROOT / "weights.json"

SIZE = 16
HIDDEN = 24
CLASSES = 10
EPOCHS = 12
BATCH = 128
LR = 0.08


def download(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists():
        return path
    url = f"https://storage.googleapis.com/cvdf-datasets/mnist/{name}"
    print(f"download {name} …")
    urllib.request.urlretrieve(url, path)
    return path


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return data.astype(np.float32) / 255.0


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).copy()


def downsample(imgs: np.ndarray) -> np.ndarray:
    # 28 → 16 via block average after pad to 32 then 2×2
    pad = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), mode="constant")
    # 32x32 → 16x16
    h, w = pad.shape[1] // 2, pad.shape[2] // 2
    out = pad.reshape(imgs.shape[0], h, 2, w, 2).mean(axis=(2, 4))
    return out.reshape(imgs.shape[0], -1)


def one_hot(y: np.ndarray, k: int = CLASSES) -> np.ndarray:
    m = np.zeros((y.size, k), dtype=np.float32)
    m[np.arange(y.size), y] = 1.0
    return m


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    x_train = downsample(load_images(download("train-images-idx3-ubyte.gz")))
    y_train = load_labels(download("train-labels-idx1-ubyte.gz"))
    x_test = downsample(load_images(download("t10k-images-idx3-ubyte.gz")))
    y_test = load_labels(download("t10k-labels-idx1-ubyte.gz"))

    rng = np.random.default_rng(42)
    n_in = SIZE * SIZE
    w1 = (rng.normal(0, 0.1, (n_in, HIDDEN))).astype(np.float32)
    b1 = np.zeros(HIDDEN, dtype=np.float32)
    w2 = (rng.normal(0, 0.1, (HIDDEN, CLASSES))).astype(np.float32)
    b2 = np.zeros(CLASSES, dtype=np.float32)

    n = x_train.shape[0]
    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        total_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            xb, yb = x_train[idx], one_hot(y_train[idx])
            h = relu(xb @ w1 + b1)
            logits = h @ w2 + b2
            probs = softmax(logits)
            total_loss += float(-(yb * np.log(probs + 1e-8)).sum())

            dlogits = (probs - yb) / xb.shape[0]
            dw2 = h.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ w2.T
            dh *= (h > 0).astype(np.float32)
            dw1 = xb.T @ dh
            db1 = dh.sum(axis=0)

            w2 -= LR * dw2
            b2 -= LR * db2
            w1 -= LR * dw1
            b1 -= LR * db1

        # eval
        h = relu(x_test @ w1 + b1)
        pred = (h @ w2 + b2).argmax(axis=1)
        acc = float((pred == y_test).mean())
        print(f"epoch {epoch + 1}/{EPOCHS} loss≈{total_loss / n:.4f} test_acc={acc:.3f}")

    payload = {
        "inputSize": SIZE,
        "hidden": HIDDEN,
        "classes": CLASSES,
        "w1": w1.T.tolist(),  # [hidden][in] — row-major per neuron for viz
        "b1": b1.tolist(),
        "w2": w2.T.tolist(),  # [out][hidden]
        "b2": b2.tolist(),
        "note": "MNIST downsampled 16×16; toy demo weights, not production.",
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
