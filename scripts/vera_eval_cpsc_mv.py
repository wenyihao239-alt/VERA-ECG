#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from train_signal import ece, macro_f1
from train_signal_mv import ResNet1dMV
from vera_prepare_mv import stats_for_split
from vera_mv_utils import stats_hp

NAMES = ["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]


def metrics(y, p):
    aucs, aps, per = [], [], {}
    for j, name in enumerate(NAMES):
        if y[:, j].sum() > 0 and y[:, j].sum() < len(y):
            a = roc_auc_score(y[:, j], p[:, j])
            ap = average_precision_score(y[:, j], p[:, j])
            aucs.append(a); aps.append(ap)
            per[name] = {"auc": float(a), "ap": float(ap), "n_pos": int(y[:, j].sum())}
    return {"macro_auc": float(np.mean(aucs)), "macro_ap": float(np.mean(aps)),
            "macro_f1": macro_f1(y, p, 0.5), "ece": ece(y, p), "per_class": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mv-mode", default="raw", choices=["raw", "zero", "hp", "hp_std"])
    ap.add_argument("--mv-gain", type=float, default=1.0)
    ap.add_argument("--mv-offset", type=float, default=0.0)
    ap.add_argument("--ptbxl-train-x", default=r"C:\Users\YOLO\Documents\心电图\QEG-ECG\data\processed\ptbxl_100\X_train.npy")
    ap.add_argument("--hp-cache", default=r"C:\Users\YOLO\Documents\心电图\QEG-ECG\data\processed\ptbxl_100_mv_hp_stats.npz")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    model = ResNet1dMV(num_classes=9, mv_dim=36)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    mv_mean = np.load(os.path.join(os.path.dirname(args.checkpoint), "mv_mean.npy")).reshape(-1)
    mv_std = np.load(os.path.join(os.path.dirname(args.checkpoint), "mv_std.npy")).reshape(-1)
    hp_mean = hp_std = None
    if args.mv_mode == "hp_std":
        if os.path.exists(args.hp_cache):
            d = np.load(args.hp_cache)
            hp_mean, hp_std = d["mean"], d["std"]
        else:
            x = np.load(args.ptbxl_train_x, mmap_mode="r")
            hp = stats_hp(x)
            hp_mean = hp.mean(0, keepdims=True)
            hp_std = hp.std(0, keepdims=True) + 1e-6
            np.savez_compressed(args.hp_cache, mean=hp_mean, std=hp_std)
    folds = []
    for f in range(5):
        if f == 0:
            d = os.path.join(args.root, "cpsc2018_100")
            split = "test"
        else:
            d = os.path.join(args.root, f"cpsc2018_fold{f}_100")
            split = "valid"
        x = np.load(os.path.join(d, f"X_{split}.npy"))
        y = np.load(os.path.join(d, f"y_{split}_cpsc.npy"))
        xmv = x * args.mv_gain + args.mv_offset
        if args.mv_mode == "zero":
            mv = np.zeros((len(x), 36), dtype=np.float32)
        elif args.mv_mode == "raw":
            mv = (stats_for_split(xmv) - mv_mean) / mv_std
        else:
            h = stats_hp(xmv)
            if args.mv_mode == "hp":
                mv = (h - mv_mean) / mv_std
            else:
                mv = (h - hp_mean) / hp_std
        ps = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                xb = x[i:i + 256].astype(np.float32)
                mu = xb.mean(axis=2, keepdims=True)
                sd = xb.std(axis=2, keepdims=True) + 1e-6
                xb = (xb - mu) / sd
                xb = torch.from_numpy(xb).to(device)
                mb = torch.from_numpy(mv[i:i + 256].astype(np.float32)).to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(xb, mb)
                ps.append(torch.sigmoid(logits).float().cpu().numpy())
        p = np.concatenate(ps, axis=0)
        m = metrics(y, p)
        folds.append({"fold": f, "n": int(len(y)), **m})
        print("fold", f, m["macro_auc"], m["macro_ap"], m["macro_f1"], m["ece"], flush=True)
    mean = {k: float(np.mean([f[k] for f in folds])) for k in ["macro_auc", "macro_ap", "macro_f1", "ece"]}
    out = {"mv_mode": args.mv_mode, "folds": folds, "mean": mean}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(mean, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



