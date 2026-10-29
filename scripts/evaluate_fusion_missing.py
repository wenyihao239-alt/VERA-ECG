#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch

from train_fusion import FusionNet, normalize
from train_signal import macro_ap, macro_auc, macro_f1


def load_split(feat_dir, modality, split):
    return np.load(os.path.join(feat_dir, f"{modality}_{split}.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--signal-features", required=True)
    ap.add_argument("--image-features", required=True)
    ap.add_argument("--image-quality", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--runs", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    zs = {s: load_split(args.signal_features, "signal", s) for s in ["train", "val", "test"]}
    zi = {s: load_split(args.image_features, "image", s) for s in ["train", "val", "test"]}
    qs = {s: np.load(os.path.join(args.data_dir, f"q_{s}.npy")) for s in ["train", "val", "test"]}
    qi = {s: np.load(os.path.join(args.image_quality, f"qi_{s}.npy")) for s in ["train", "val", "test"]}
    y = {s: np.load(os.path.join(args.data_dir, f"y_{s}_{args.task}.npy")) for s in ["train", "val", "test"]}

    zs["train"], zs["val"], zs["test"] = normalize(zs["train"], zs["val"], zs["test"])
    zi["train"], zi["val"], zi["test"] = normalize(zi["train"], zi["val"], zi["test"])
    qs["train"], qs["val"], qs["test"] = normalize(qs["train"], qs["val"], qs["test"])
    q_dim = qs["train"].shape[1]
    for s in ["train", "val", "test"]:
        if qi[s].shape[1] < q_dim:
            qi[s] = np.concatenate([qi[s], np.zeros((len(qi[s]), q_dim - qi[s].shape[1]), dtype=np.float32)], axis=1)
        elif qi[s].shape[1] > q_dim:
            qi[s] = qi[s][:, :q_dim]
    qi["train"], qi["val"], qi["test"] = normalize(qi["train"], qi["val"], qi["test"])

    rows = []
    for run in args.runs:
        ckpt_path = os.path.join(run, "best.pt")
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        a = ckpt["args"]
        mode = a["mode"]
        model = FusionNet(mode, zs["test"].shape[1], zi["test"].shape[1], q_dim, y["test"].shape[1]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for cond in ["both", "signal_only", "image_only"]:
            zs_t = torch.from_numpy(zs["test"]).to(device)
            zi_t = torch.from_numpy(zi["test"]).to(device)
            qs_t = torch.from_numpy(qs["test"]).to(device)
            qi_t = torch.from_numpy(qi["test"]).to(device)
            ms = torch.ones(len(zs_t), 1, device=device)
            mi = torch.ones(len(zi_t), 1, device=device)
            if cond == "signal_only":
                mi = torch.zeros_like(mi)
                zi_t = torch.zeros_like(zi_t)
            elif cond == "image_only":
                ms = torch.zeros_like(ms)
                zs_t = torch.zeros_like(zs_t)
            with torch.no_grad():
                logits, aux = model(zs_t, zi_t, qs_t, qi_t, ms, mi)
                p = torch.sigmoid(logits).float().cpu().numpy()
            yy = y["test"]
            rows.append({
                "run": os.path.basename(run),
                "mode": mode,
                "condition": cond,
                "auc": macro_auc(yy, p),
                "ap": macro_ap(yy, p),
                "f1": macro_f1(yy, p, 0.5),
                "gate_mean": float(aux["gate"].mean().cpu()) if "gate" in aux else float("nan"),
            })
            print(rows[-1], flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
