#!/usr/bin/env python3
"""Prepare PTB-XL arrays, labels, quality features and metadata.

Output layout (default):
  <out>/X_train.npy, X_val.npy, X_test.npy          float32 [N,12,T]
  <out>/y_*_{super,sub,rhythm,cpsc}.npy              float32 [N,K]
  <out>/q_*.npy                                      float32 [N,8]
  <out>/meta.csv
  <out>/label_names.json
  <out>/splits.json

Split: official strat_fold 1-8 train, 9 val, 10 test. Patient-level by construction.
"""
import argparse
import ast
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
# SCP code -> CPSC2018 shared label. Codes are checked at runtime; missing codes are skipped.
CPSC_MAP = {
    "Normal": ["NORM"],
    "AF": ["AFIB"],
    "I-AVB": ["1AVB"],
    "LBBB": ["CLBBB", "ILBBB"],
    "RBBB": ["CRBBB", "IRBBB"],
    "PAC": ["PAC"],
    "PVC": ["PVC"],
    "STD": ["STD_"],
    "STE": ["STE_"],
}
CPSC_LABELS = ["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]


def parse_codes(s):
    try:
        d = ast.literal_eval(s)
        return {str(k): float(v) for k, v in d.items()}
    except Exception:
        return {}


def quality_features(x, fs):
    """x: [12, T] physical mV. Returns 8 scalar quality features."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    n_leads, T = x.shape
    std = x.std(axis=1)
    ptp = np.ptp(x, axis=1)

    missing = float(np.mean(std < 1e-6))
    flat = float(np.mean(std < 0.01))

    # clipping: fraction of samples at the per-lead min or max
    mx = x.max(axis=1, keepdims=True)
    mn = x.min(axis=1, keepdims=True)
    clip = float(np.mean(np.isclose(x, mx, atol=1e-6) | np.isclose(x, mn, atol=1e-6)))

    # spectral features
    xc = x - x.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    total = spec.sum(axis=1) + 1e-12
    baseline = float(np.mean(spec[:, freqs < 0.5].sum(axis=1) / total))
    hf = float(np.mean(spec[:, freqs > 40.0].sum(axis=1) / total))
    sig_band = spec[:, (freqs >= 0.5) & (freqs <= 40.0)].sum(axis=1)
    noise_band = spec[:, freqs > 40.0].sum(axis=1) + 1e-12
    snr = float(np.mean(10.0 * np.log10((sig_band + 1e-12) / noise_band)))

    # mean off-diagonal correlation
    if n_leads > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(x)
        off = c[~np.eye(n_leads, dtype=bool)]
        lead_corr = float(np.nanmean(off)) if np.isfinite(off).any() else 0.0
    else:
        lead_corr = 0.0

    # kurtosis (excess), mean across leads
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1) + 1e-12
    kurt = float(np.mean(((x - mu) ** 4).mean(axis=1) / (var ** 2) - 3.0))
    amp = float(np.median(ptp))

    return np.array(
        [snr, baseline, hf, missing, flat, clip, lead_corr, kurt, amp],
        dtype=np.float32,
    )


def load_one(args):
    raw_dir, relpath, fs_expect = args
    import wfdb

    path = os.path.join(raw_dir, relpath)
    rec = wfdb.rdrecord(path, physical=True)
    x = np.asarray(rec.p_signal, dtype=np.float32).T  # [12,T]
    if np.isnan(x).any():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[1] < fs_expect * 10:
        pad = fs_expect * 10 - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad)))
    elif x.shape[1] > fs_expect * 10:
        start = (x.shape[1] - fs_expect * 10) // 2
        x = x[:, start : start + fs_expect * 10]
    return x.astype(np.float32), quality_features(x, fs_expect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/mnt/data/wyh/xdt/data/raw/ptbxl")
    ap.add_argument("--out-dir", default="/mnt/data/wyh/xdt/data/processed/ptbxl_100")
    ap.add_argument("--sampling", type=int, default=100, choices=[100, 500])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N records")
    args = ap.parse_args()

    raw = args.raw_dir
    fs = args.sampling
    T = fs * 10
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    df = pd.read_csv(os.path.join(raw, "ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(raw, "scp_statements.csv"), index_col=0)
    df["codes"] = df["scp_codes"].apply(parse_codes)

    def supers(codes):
        out = set()
        for k in codes:
            if k in scp.index and pd.notna(scp.loc[k, "diagnostic"]) and float(scp.loc[k, "diagnostic"]) == 1:
                cls = scp.loc[k, "diagnostic_class"]
                if isinstance(cls, str) and cls:
                    out.add(cls)
        return out

    def subs(codes):
        return {
            k
            for k in codes
            if k in scp.index and pd.notna(scp.loc[k, "diagnostic"]) and float(scp.loc[k, "diagnostic"]) == 1
        }

    def rhythms(codes):
        return {k for k in codes if k in scp.index and pd.notna(scp.loc[k, "rhythm"]) and float(scp.loc[k, "rhythm"]) == 1}

    df["super_set"] = df["codes"].apply(supers)
    df["sub_set"] = df["codes"].apply(subs)
    df["rhythm_set"] = df["codes"].apply(rhythms)

    # Keep records with at least one diagnostic superclass label.
    n0 = len(df)
    df = df[df["super_set"].str.len() > 0].copy()
    df = df.sort_values("ecg_id").reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).copy()
    n = len(df)

    # all diagnostic subcodes and top-23 list
    sub_counter = Counter()
    for s in df["sub_set"]:
        sub_counter.update(s)
    sub23 = [k for k, _ in sub_counter.most_common(23)]
    rhythm_counter = Counter()
    for s in df["rhythm_set"]:
        rhythm_counter.update(s)
    rhythm_labels = [k for k, c in rhythm_counter.items() if c >= 20]

    # labels
    y_super = np.zeros((n, len(SUPER)), dtype=np.float32)
    for i, s in enumerate(df["super_set"]):
        for j, name in enumerate(SUPER):
            if name in s:
                y_super[i, j] = 1.0
    y_sub = np.zeros((n, len(sub23)), dtype=np.float32)
    for i, s in enumerate(df["sub_set"]):
        for j, name in enumerate(sub23):
            if name in s:
                y_sub[i, j] = 1.0
    y_rhythm = np.zeros((n, len(rhythm_labels)), dtype=np.float32)
    for i, s in enumerate(df["rhythm_set"]):
        for j, name in enumerate(rhythm_labels):
            if name in s:
                y_rhythm[i, j] = 1.0
    y_cpsc = np.zeros((n, len(CPSC_LABELS)), dtype=np.float32)
    for i, codes in enumerate(df["codes"]):
        for label, code_list in CPSC_MAP.items():
            if any(code in codes for code in code_list):
                y_cpsc[i, CPSC_LABELS.index(label)] = 1.0

    split = np.where(df["strat_fold"].values <= 8, "train", np.where(df["strat_fold"].values == 9, "val", "test"))
    df["split"] = split

    # memmaps
    arrays = {}
    for name in ["train", "val", "test"]:
        m = split == name
        arrays[name] = np.lib.format.open_memmap(
            os.path.join(out, f"X_{name}.npy"), mode="w+", dtype=np.float32, shape=(int(m.sum()), 12, T)
        )
    q = np.zeros((n, 9), dtype=np.float32)
    ok = np.zeros(n, dtype=np.uint8)

    relpaths = (
        df["filename_lr"].values if args.sampling == 100 else df["filename_hr"].values
    )
    missing = [p for p in relpaths if not os.path.exists(os.path.join(raw, str(p) + ".hea"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} records missing headers, first: {missing[:5]}")
    tasks = [(raw, str(p), fs) for p in relpaths]
    write_idx = {name: 0 for name in ["train", "val", "test"]}
    if args.workers <= 1:
        iterator = map(load_one, tasks)
        pool = None
    else:
        pool = Pool(args.workers, maxtasksperchild=100)
        iterator = pool.imap(load_one, tasks, chunksize=16)
    try:
        for i, (x, qf) in enumerate(iterator):
            sp = split[i]
            arrays[sp][write_idx[sp]] = x
            write_idx[sp] += 1
            q[i] = qf
            ok[i] = 1
            if (i + 1) % 2000 == 0:
                print(f"processed {i+1}/{n}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    for name in arrays:
        arrays[name].flush()

    # save labels / quality split by split
    for name in ["train", "val", "test"]:
        m = split == name
        for tag, arr in [("super", y_super), ("sub", y_sub), ("rhythm", y_rhythm), ("cpsc", y_cpsc)]:
            np.save(os.path.join(out, f"y_{name}_{tag}.npy"), arr[m])
        np.save(os.path.join(out, f"q_{name}.npy"), q[m])
        np.save(os.path.join(out, f"ok_{name}.npy"), ok[m])

    meta_cols = [
        "ecg_id", "patient_id", "age", "sex", "strat_fold", "split",
        "filename_lr", "filename_hr", "heart_axis", "infarction_stadium1",
        "infarction_stadium2", "validated_by_human", "baseline_drift",
        "static_noise", "burst_noise", "electrodes_problems",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    df[meta_cols].to_csv(os.path.join(out, "meta.csv"), index=False)

    names = {
        "super": SUPER,
        "sub": sub23,
        "rhythm": rhythm_labels,
        "cpsc": CPSC_LABELS,
        "q": ["snr_db", "baseline_ratio", "hf_ratio", "missing_ratio", "flat_ratio", "clip_ratio", "lead_corr", "kurtosis", "amplitude_mV"],
    }
    with open(os.path.join(out, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_total_raw": n0,
                "n_used": n,
                "n_excluded_no_super": n0 - n,
                "sampling": fs,
                "T": T,
                "counts": {k: int(v) for k, v in zip(*np.unique(split, return_counts=True))},
                "super_counts": {name: int(y_super[:, j].sum()) for j, name in enumerate(SUPER)},
                "cpsc_counts": {name: int(y_cpsc[:, j].sum()) for j, name in enumerate(CPSC_LABELS)},
                "sub_top23": sub23,
                "rhythm_labels": rhythm_labels,
                "raw_dir": raw,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("done", out)
    print(json.dumps({"n": n, "counts": {k: int(v) for k, v in zip(*np.unique(split, return_counts=True))}}, ensure_ascii=False))
    print("top sub:", sub23)
    print("rhythm:", rhythm_labels)
    print("cpsc counts:", {name: int(y_cpsc[:, j].sum()) for j, name in enumerate(CPSC_LABELS)})


if __name__ == "__main__":
    main()



