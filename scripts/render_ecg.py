#!/usr/bin/env python3
import argparse
import os
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


def render_one(job):
    x, fs, out_path, width, height, dpi = job
    n_leads, T = x.shape
    sec_per_col = 10.0 / 4.0
    col_samples = int(round(sec_per_col * fs))
    y = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(y, [0.5, 99.5])
    span = max(hi - lo, 1.0)
    ylim = (lo - 0.1 * span, hi + 0.1 * span)

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.85], hspace=0.16, wspace=0.06)
    idx = {name: i for i, name in enumerate(LEADS)}
    for r in range(3):
        for c in range(4):
            ax = fig.add_subplot(gs[r, c])
            name = LAYOUT[r][c]
            s = c * col_samples
            e = min(s + col_samples, T)
            t = np.arange(s, e) / fs
            ax.plot(t, y[idx[name], s:e], color="black", linewidth=0.55)
            ax.set_xlim(s / fs, (s + col_samples) / fs)
            ax.set_ylim(*ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, color="#e8e8e8", linewidth=0.35)
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
                spine.set_linewidth(0.3)
            ax.text(0.01, 0.94, name, transform=ax.transAxes, fontsize=5.5, va="top", ha="left", color="#333333")
    ax = fig.add_subplot(gs[3, :])
    t = np.arange(T) / fs
    ax.plot(t, y[idx["II"]], color="black", linewidth=0.55)
    ax.set_xlim(0, T / fs)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color="#e8e8e8", linewidth=0.35)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.3)
    ax.text(0.01, 0.94, "II (rhythm)", transform=ax.transAxes, fontsize=5.5, va="top", ha="left", color="#333333")
    fig.subplots_adjust(left=0.025, right=0.995, top=0.995, bottom=0.02)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, format="jpg", dpi=dpi, pil_kwargs={"quality": 92, "optimize": True})
    plt.close(fig)
    img_std = float(np.std(plt.imread(out_path))) if os.path.exists(out_path) else 0.0
    return out_path, img_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--dpi", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.data_dir, "meta.csv"))
    jobs = []
    rows = []
    for split in ["train", "val", "test"]:
        x = np.load(os.path.join(args.data_dir, f"X_{split}.npy"), mmap_mode="r")
        sub = meta[meta["split"] == split].reset_index(drop=True)
        n = len(sub) if not args.limit else min(args.limit, len(sub))
        for i in range(n):
            ecg_id = int(sub.iloc[i]["ecg_id"])
            out_path = os.path.join(args.out_dir, split, f"{ecg_id}.jpg")
            jobs.append((np.asarray(x[i], dtype=np.float32), 100, out_path, args.width, args.height, args.dpi))
            rows.append({"ecg_id": ecg_id, "split": split, "path": os.path.relpath(out_path, args.out_dir)})
    print(f"rendering {len(jobs)} images -> {args.out_dir}", flush=True)
    with Pool(args.workers) as pool:
        for k, (path, std) in enumerate(pool.imap_unordered(render_one, jobs, chunksize=8)):
            if k % 1000 == 0:
                print(f"{k}/{len(jobs)} last_std={std:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "render_index.csv"), index=False)
    print("done", args.out_dir)


if __name__ == "__main__":
    main()
