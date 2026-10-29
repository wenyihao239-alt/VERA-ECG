#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np


def stats_for_split(x):
    out = []
    for i in range(len(x)):
        rec = np.asarray(x[i], dtype=np.float32)
        std = rec.std(axis=1)
        p95 = np.percentile(rec, 95, axis=1)
        p5 = np.percentile(rec, 5, axis=1)
        out.append(np.concatenate([std, p95, p5], axis=0))
    return np.stack(out, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    names = []
    for stat in ["std", "p95", "p5"]:
        for i in range(12):
            names.append(f"{stat}_lead{i+1}")
    for split in args.splits:
        x = np.load(os.path.join(args.data_dir, f"X_{split}.npy"), mmap_mode="r")
        mv = stats_for_split(x)
        np.save(os.path.join(args.out_dir, f"mv_{split}.npy"), mv)
        print(split, mv.shape, flush=True)
    with open(os.path.join(args.out_dir, "mv_names.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    print("done", args.out_dir)


if __name__ == "__main__":
    main()
