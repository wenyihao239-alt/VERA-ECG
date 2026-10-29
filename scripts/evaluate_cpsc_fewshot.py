#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from train_signal import ResNet1d


def feats(model, x, device, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].astype(np.float32)
            mu = xb.mean(axis=2, keepdims=True)
            sd = xb.std(axis=2, keepdims=True) + 1e-6
            xb = (xb - mu) / sd
            xb = torch.from_numpy(xb).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                _, z = model(xb, return_features=True)
            out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def macro_auc(y, p):
    vals = []
    for j in range(y.shape[1]):
        if y[:, j].sum() > 0 and y[:, j].sum() < len(y):
            vals.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(vals))


def macro_ap(y, p):
    vals = []
    for j in range(y.shape[1]):
        if y[:, j].sum() > 0:
            vals.append(average_precision_score(y[:, j], p[:, j]))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cpsc-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    xtr = np.load(os.path.join(args.cpsc_dir, "X_train.npy"))
    ytr = np.load(os.path.join(args.cpsc_dir, "y_train_cpsc.npy"))
    xte = np.load(os.path.join(args.cpsc_dir, "X_test.npy"))
    yte = np.load(os.path.join(args.cpsc_dir, "y_test_cpsc.npy"))
    model = ResNet1d(num_classes=9)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    model.to(device)
    ztr = feats(model, xtr, device)
    zte = feats(model, xte, device)
    ytr_idx = ytr.argmax(axis=1)
    yte_idx = yte.argmax(axis=1)

    model.eval()
    p0s = []
    with torch.no_grad():
        for i in range(0, len(xte), 256):
            xb = xte[i:i + 256].astype(np.float32)
            mu = xb.mean(axis=2, keepdims=True)
            sd = xb.std(axis=2, keepdims=True) + 1e-6
            xb = (xb - mu) / sd
            xb = torch.from_numpy(xb).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(xb)
            p0s.append(torch.sigmoid(logits).float().cpu().numpy())
    p0 = np.concatenate(p0s, axis=0)
    rows = [{"fraction": 0.0, "auc": macro_auc(yte, p0), "ap": macro_ap(yte, p0), "f1": float(f1_score(yte, (p0 >= 0.5).astype(int), average="macro", zero_division=0))}]
    print(rows[0], flush=True)
    cls_counts = np.bincount(ytr_idx, minlength=9)
    for frac in [0.01, 0.05, 0.10, 1.00]:
        aucs, aps, f1s = [], [], []
        for rep in range(args.repeats):
            rng = np.random.default_rng(1000 + rep)
            idx = []
            for c in range(9):
                n = max(1, int(round(frac * cls_counts[c])))
                pool = np.where(ytr_idx == c)[0]
                idx.extend(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
            idx = np.array(idx)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(ztr[idx], ytr_idx[idx])
            p = clf.predict_proba(zte)
            full = np.zeros_like(yte)
            full[:, clf.classes_] = p
            aucs.append(macro_auc(yte, full))
            aps.append(macro_ap(yte, full))
            f1s.append(f1_score(yte, (full >= 0.5).astype(int), average="macro", zero_division=0))
        row = {"fraction": frac, "n": int(len(idx)), "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
               "ap": float(np.mean(aps)), "ap_std": float(np.std(aps)), "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
        rows.append(row)
        print(row, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()


