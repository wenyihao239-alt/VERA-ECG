#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from train_signal import ece

NAMES = ["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]


def sqrtm(C, inv=False):
    C = (C + C.T) / 2
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    if inv:
        w = 1.0 / np.sqrt(w)
    else:
        w = np.sqrt(w)
    return (V * w) @ V.T


def coral(Xs, Xt, reg=1e-3):
    mu_s, mu_t = Xs.mean(0, keepdims=True), Xt.mean(0, keepdims=True)
    Cs = np.cov(Xs, rowvar=False) + reg * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + reg * np.eye(Xs.shape[1])
    Xt_c = Xt - mu_t
    Xt_a = Xt_c @ sqrtm(Ct, inv=True) @ sqrtm(Cs)
    return (Xt_a + mu_s).astype(np.float32)


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


def aurc(y, p):
    pp = np.clip(p, 1e-6, 1 - 1e-6)
    ent = -(pp * np.log(pp) + (1 - pp) * np.log(1 - pp)).mean(1) / np.log(2)
    conf = 1 - ent
    err = 1 - ((p >= 0.5) == y).mean(1)
    order = np.argsort(-conf)
    grid = np.linspace(0.05, 1.0, 20)
    risks = [float(err[order[:max(1, int(c * len(y)))]].mean()) for c in grid]
    return float(np.mean(risks))


def metrics(y, p):
    pred = p.argmax(1); true = y.argmax(1)
    per = {}
    for j, n in enumerate(NAMES):
        if y[:, j].sum() > 0 and y[:, j].sum() < len(y):
            per[n] = {"auc": float(roc_auc_score(y[:, j], p[:, j])), "ap": float(average_precision_score(y[:, j], p[:, j]))}
    return {"macro_auc": macro_auc(y, p), "macro_ap": macro_ap(y, p),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "ece": float(ece(y, p)), "aurc": aurc(y, p), "per_class": per}


def train_head(ztr, ytr, epochs=100, lr=1e-3, device="cpu", balanced=True):
    ycls = ytr.argmax(1)
    counts = np.bincount(ycls, minlength=9)
    w = 1.0 / np.maximum(counts, 1)
    sw = w[ycls]
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True) if balanced else None
    ds = TensorDataset(torch.from_numpy(ztr), torch.from_numpy(ycls).long())
    dl = DataLoader(ds, batch_size=512, sampler=sampler, shuffle=(sampler is None))
    head = nn.Linear(ztr.shape[1], 9).to(device)
    cw = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    cw = (cw / cw.mean()).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(xb), yb)
            loss.backward(); opt.step()
    head.eval()
    return head


def predict(head, x, device):
    with torch.no_grad():
        p = torch.softmax(head(torch.from_numpy(x).to(device)), dim=1).cpu().numpy()
    return p


def em_prior(p, src_prior, iters=30):
    pi = src_prior.copy()
    for _ in range(iters):
        w = p * (pi / np.maximum(src_prior, 1e-8))
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-8)
        pi = w.mean(0)
    return pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ztr = np.load(os.path.join(args.feature_dir, "signal_train.npy")).astype(np.float32)
    ytr = np.load(os.path.join(args.feature_dir, "y_train_cpsc.npy")).astype(np.float32)
    zva = np.load(os.path.join(args.feature_dir, "signal_val.npy")).astype(np.float32)
    yva = np.load(os.path.join(args.feature_dir, "y_val_cpsc.npy")).astype(np.float32)
    mu, sd = ztr.mean(0, keepdims=True), ztr.std(0, keepdims=True) + 1e-6
    ztr = (ztr - mu) / sd; zva = (zva - mu) / sd
    head = train_head(ztr, ytr, device=device)
    src_prior = np.bincount(ytr.argmax(1), minlength=9) / len(ytr)
    rows = []
    for f in range(5):
        zt = np.load(os.path.join(args.feature_dir, f"cpsc_fold{f}.npy")).astype(np.float32)
        yt = np.load(os.path.join(args.feature_dir, f"cpsc_fold{f}_label.npy")).astype(np.float32)
        zt = (zt - mu) / sd
        p0 = predict(head, zt, device)
        p_c = predict(head, coral(ztr, zt), device)
        pi = em_prior(p_c, src_prior)
        p_cp = p_c * (pi / np.maximum(src_prior, 1e-8))
        p_cp = p_cp / np.maximum(p_cp.sum(1, keepdims=True), 1e-8)
        rows.append({"fold": f, "n": int(len(yt)),
                     "zero": metrics(yt, p0), "coral": metrics(yt, p_c), "coral_prior": metrics(yt, p_cp)})
        print("fold", f, "zero", rows[-1]["zero"]["macro_auc"], "coral", rows[-1]["coral"]["macro_auc"],
              "coral_prior", rows[-1]["coral_prior"]["macro_auc"], flush=True)
    mean = {}
    for k in ["zero", "coral", "coral_prior"]:
        mean[k] = {m: float(np.mean([r[k][m] for r in rows])) for m in ["macro_auc", "macro_ap", "macro_f1", "ece", "aurc"]}
        mean[k]["std_auc"] = float(np.std([r[k]["macro_auc"] for r in rows]))
    out = {"folds": rows, "mean": mean}
    print(json.dumps(mean, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

