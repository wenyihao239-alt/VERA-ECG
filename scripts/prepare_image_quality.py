#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np
import pandas as pd


def features_from_array(img):
    if img is None:
        return np.zeros(7, dtype=np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    lap = float(cv2.Laplacian(img, cv2.CV_64F).var())
    contrast = float(img.std())
    brightness = float(img.mean())
    b = max(1, int(0.05 * min(h, w)))
    border = np.concatenate([img[:b, :].ravel(), img[-b:, :].ravel(), img[:, :b].ravel(), img[:, -b:].ravel()])
    white_ratio = float(np.mean(border > 240))
    dark_ratio = float(np.mean(img < 50))
    edges = cv2.Canny(img, 50, 150)
    edge_density = float(edges.mean() / 255.0)
    return np.array([lap, contrast, brightness, white_ratio, dark_ratio, edge_density, float(h * w)], dtype=np.float32)


def features(path):
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = None
    return features_from_array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(args.data_dir, "meta.csv"))
    for split in ["train", "val", "test"]:
        sub = meta[meta["split"] == split].reset_index(drop=True)
        feats = []
        for ecg_id in sub["ecg_id"].values:
            path = os.path.join(args.render_dir, split, f"{int(ecg_id)}.jpg")
            feats.append(features(path))
        arr = np.stack(feats, axis=0).astype(np.float32) if feats else np.zeros((0, 7), dtype=np.float32)
        np.save(os.path.join(args.out_dir, f"qi_{split}.npy"), arr)
        print(split, arr.shape, flush=True)
    print("done", args.out_dir)


if __name__ == "__main__":
    main()
