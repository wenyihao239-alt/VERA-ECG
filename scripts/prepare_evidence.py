#!/usr/bin/env python3
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

NAMES = ["HR", "PR", "QRS", "QT", "QTc"]


def one(args):
    raw_dir, relpath = args
    import wfdb
    import neurokit2 as nk

    try:
        rec = wfdb.rdrecord(os.path.join(raw_dir, relpath))
        x = np.asarray(rec.p_signal[:, 1], dtype=float)
        fs = float(rec.fs)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0)
        sig, info = nk.ecg_process(x, sampling_rate=fs)
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=float)
        if len(rpeaks) < 2:
            return np.full(5, np.nan, dtype=np.float32)
        _, waves = nk.ecg_delineate(sig["ECG_Clean"], rpeaks.astype(int), sampling_rate=fs, method="dwt")
        p_on = np.asarray(waves.get("ECG_P_Onsets", []), dtype=float)
        r_on = np.asarray(waves.get("ECG_R_Onsets", []), dtype=float)
        r_off = np.asarray(waves.get("ECG_R_Offsets", []), dtype=float)
        t_off = np.asarray(waves.get("ECG_T_Offsets", []), dtype=float)

        def med(v):
            v = v[np.isfinite(v)]
            return float(np.median(v)) if len(v) else np.nan

        rr = np.diff(rpeaks) / fs
        hr = 60.0 / np.median(rr) if len(rr) and np.median(rr) > 0 else np.nan

        n = min(len(p_on), len(r_on), len(r_off), len(t_off))
        pr = (r_on[:n] - p_on[:n]) / fs * 1000.0
        qrs = (r_off[:n] - r_on[:n]) / fs * 1000.0
        qt = (t_off[:n] - r_on[:n]) / fs * 1000.0
        pr_m, qrs_m, qt_m = med(pr), med(qrs), med(qt)
        rr_m = float(np.median(rr)) if len(rr) else np.nan
        qtc = qt_m / np.sqrt(rr_m) if np.isfinite(qt_m) and np.isfinite(rr_m) and rr_m > 0 else np.nan

        vals = np.array([hr, pr_m, qrs_m, qt_m, qtc], dtype=np.float32)
        limits = [(30, 220), (80, 400), (40, 250), (200, 700), (250, 700)]
        for i, (lo, hi) in enumerate(limits):
            if not np.isfinite(vals[i]) or vals[i] < lo or vals[i] > hi:
                vals[i] = np.nan
        return vals
    except Exception:
        return np.full(5, np.nan, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(args.data_dir, "meta.csv"))
    for split in ["train", "val", "test"]:
        sub = meta[meta["split"] == split].reset_index(drop=True)
        if args.limit:
            sub = sub.head(args.limit).copy()
        tasks = [(args.raw_dir, str(p)) for p in sub["filename_hr"].values]
        vals = []
        with Pool(args.workers) as pool:
            for i, v in enumerate(pool.imap(one, tasks, chunksize=16)):
                vals.append(v)
                if (i + 1) % 1000 == 0:
                    print(split, i + 1, len(tasks), flush=True)
        arr = np.stack(vals, axis=0).astype(np.float32)
        mask = np.isfinite(arr).astype(np.float32)
        np.save(os.path.join(args.out_dir, f"evidence_{split}.npy"), arr)
        np.save(os.path.join(args.out_dir, f"evidence_mask_{split}.npy"), mask)
        print(split, arr.shape, "valid", mask.mean(axis=0), flush=True)
    with open(os.path.join(args.out_dir, "evidence_names.json"), "w", encoding="utf-8") as f:
        json.dump(NAMES, f)
    print("done", args.out_dir)


if __name__ == "__main__":
    main()
