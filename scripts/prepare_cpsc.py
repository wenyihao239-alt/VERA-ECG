#!/usr/bin/env python3
import argparse
import io
import json
import os
import struct
import zlib

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from prepare_ptbxl import quality_features

CPSC_LABELS = ["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]


def iter_local(zip_path, prefix):
    size = os.path.getsize(zip_path)
    with open(zip_path, "rb") as f:
        off = 0
        while off + 30 <= size:
            f.seek(off)
            h = f.read(30)
            if len(h) < 30 or h[:4] != b"PK\x03\x04":
                break
            sig, ver, flag, method, mtime, mdate, crc, csize, usize, fnlen, extralen = struct.unpack("<IHHHHHIIIHH", h)
            name = f.read(fnlen).decode("utf-8", "replace")
            if name.startswith(prefix) or ("label" in name and name.startswith("Fold0/")):
                data_off = off + 30 + fnlen + extralen
                if csize == 0:
                    off = data_off
                    continue
                f.seek(data_off)
                data = f.read(csize)
                if method == 8:
                    data = zlib.decompress(data, -15)
                yield name, data
            off = off + 30 + fnlen + extralen + csize
            if name.startswith("Fold1/"):
                break


def load_signals(zip_path, fold, split):
    import zipfile

    prefix = f"Fold{fold}/{split}/"
    label_name = f"Fold{fold}/{split}_label{fold}.npy"
    names, signals, labels = [], [], None
    try:
        z = zipfile.ZipFile(zip_path)
        for n in z.namelist():
            if n.startswith(prefix) and n.endswith(".npy"):
                names.append(n)
                signals.append(np.load(io.BytesIO(z.read(n))))
            elif n == label_name:
                labels = np.load(io.BytesIO(z.read(n)))
    except Exception:
        pass
    if labels is None:
        names, signals, labels = [], [], None
        for n, data in iter_local(zip_path, prefix):
            if n.startswith(prefix) and n.endswith(".npy"):
                names.append(n)
                signals.append(np.load(io.BytesIO(data)))
            elif n == label_name:
                labels = np.load(io.BytesIO(data))
    if labels is None:
        raise FileNotFoundError(label_name)
    return names, signals, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["test", "train", "valid"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-fs", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    names, signals, labels = load_signals(args.zip, args.fold, args.split)
    if args.limit:
        names = names[:args.limit]
        signals = signals[:args.limit]
        labels = labels[:args.limit]
    X, Q = [], []
    for s in signals:
        x = np.asarray(s, dtype=np.float32)
        if x.shape[0] != 12 and x.shape[1] == 12:
            x = x.T
        if x.shape[0] != 12:
            raise ValueError(f"unexpected shape {x.shape}")
        if args.target_fs == 100:
            x = resample_poly(x, 1, 5, axis=1).astype(np.float32)
        if x.shape[1] < 1000:
            x = np.pad(x, ((0, 0), (0, 1000 - x.shape[1])))
        elif x.shape[1] > 1000:
            st = (x.shape[1] - 1000) // 2
            x = x[:, st:st + 1000]
        Q.append(quality_features(x, args.target_fs))
        X.append(x)
    X = np.stack(X, axis=0).astype(np.float32)
    Q = np.stack(Q, axis=0).astype(np.float32)
    y = np.zeros((len(labels), 9), dtype=np.float32)
    y[np.arange(len(labels)), labels.astype(int)] = 1.0
    np.save(os.path.join(args.out_dir, f"X_{args.split}.npy"), X)
    np.save(os.path.join(args.out_dir, f"y_{args.split}_cpsc.npy"), y)
    np.save(os.path.join(args.out_dir, f"q_{args.split}.npy"), Q)
    pd.DataFrame({"filename": names, "label": labels.astype(int)}).to_csv(os.path.join(args.out_dir, f"meta_{args.split}.csv"), index=False)
    counts = {CPSC_LABELS[i]: int((labels == i).sum()) for i in range(9)}
    print(json.dumps({"n": len(labels), "shape": list(X.shape), "counts": counts}, ensure_ascii=False, indent=2))
    with open(os.path.join(args.out_dir, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(CPSC_LABELS, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()



