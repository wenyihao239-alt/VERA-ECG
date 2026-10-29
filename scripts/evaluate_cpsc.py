#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from train_signal import InceptionTime, ResNet1d, ece


def load_split(data_dir, split):
    x = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}_cpsc.npy"))
    return x, y


def predict(model, x, device, batch_size=256):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].astype(np.float32)
            mu = xb.mean(axis=2, keepdims=True)
            sd = xb.std(axis=2, keepdims=True) + 1e-6
            xb = (xb - mu) / sd
            xb = torch.from_numpy(xb).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(xb)
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(ps, axis=0)


def metrics(y, p, names):
    aucs, aps = [], []
    per = {}
    for j, name in enumerate(names):
        if y[:, j].sum() > 0 and y[:, j].sum() < len(y):
            auc = roc_auc_score(y[:, j], p[:, j])
            ap = average_precision_score(y[:, j], p[:, j])
            aucs.append(auc)
            aps.append(ap)
            per[name] = {"auc": float(auc), "ap": float(ap), "prevalence": float(y[:, j].mean()), "n_pos": int(y[:, j].sum())}
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_ap": float(np.mean(aps)) if aps else float("nan"),
        "macro_f1": float(f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)),
        "ece": float(ece(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "per_class": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ptbxl-dir", required=True)
    ap.add_argument("--cpsc-dir", required=True)
    ap.add_argument("--model", default="resnet1d", choices=["resnet1d", "inceptiontime"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    names = json.load(open(os.path.join(args.cpsc_dir, "label_names.json"), encoding="utf-8"))
    model = ResNet1d(num_classes=9) if args.model == "resnet1d" else InceptionTime(num_classes=9)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    model.to(device)

    x_in, y_in = load_split(args.ptbxl_dir, "test")
    x_out, y_out = load_split(args.cpsc_dir, "test")
    p_in = predict(model, x_in, device)
    p_out = predict(model, x_out, device)
    out = {
        "checkpoint": args.checkpoint,
        "ptbxl_in_domain": metrics(y_in, p_in, names),
        "cpsc_zero_shot": metrics(y_out, p_out, names),
        "n_ptbxl": int(len(y_in)),
        "n_cpsc": int(len(y_out)),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_class"} for k, v in out.items() if isinstance(v, dict) and "macro_auc" in v}, ensure_ascii=False, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
